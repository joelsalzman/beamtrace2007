// Wavefront beam tracing on the GPU (after Laine, Karras & Aila, "Megakernels
// considered harmful", HPG 2013, with persistent warps as in Aila & Laine,
// HPG 2009).
//
// Sub-beams are self-contained records (beam/wavefront_core.h) kept in
// queues in global memory. Each round runs four kernels:
//   schedule  (1 thread)  resets the round's counters and admits new roots
//                         while their sub-beam budgets fit in the queues;
//   admit     1 thread = 1 root (a pixel for soft shadows, warps taking 8x4
//                         pixel tiles; a 16x16 tile for primary visibility; a
//                         primary hit beam for point shadows): sets up the
//                         root beam, pre-splits it, queues it;
//   trace     1 thread = 1 queued sub-beam: one leaf per iteration (climb /
//                         descend to it, test its triangles), until it is
//                         finished or a triangle needs a real clip;
//   split     1 thread = 1 sub-beam that needs a clip: clip it; its pieces
//                         form the next round's trace queue.
// Every kernel is a persistent grid. Warps take queue entries with one
// atomicAdd for the whole warp; trace lanes are refilled as soon as their
// sub-beam leaves the loop; appends are warp-aggregated, so siblings stay
// adjacent in the queues.
//
// Memory is bounded without any waiting: a root is admitted only if its
// budget of live sub-beams fits in the queues; a root that would exceed it
// (or overflow a fixed per-thread list) is abandoned and recomputed on the
// CPU. Soft-shadow visibility is accumulated in 64-bit fixed point, so the
// result does not depend on the order in which sub-beams finish.
#include <algorithm>
#include <vector>

#include "beam/wavefront_core.h"
#include "cuda/gpu_common.cuh"
#include "render/point_core.h"
#include "render/soft_core.h"

namespace bt {

struct WfBuffers {
  size_t capacity = 0, rootCap = 0, statCap = 0, outCap = 0;
  WfBeam* tq[2] = {nullptr, nullptr};
  WfBeam* sq = nullptr;
  void* roots = nullptr;  // WfRoot[rootCap]
  int* live = nullptr;
  int* state = nullptr;
  unsigned long long* missFx = nullptr;
  double* G = nullptr;
  unsigned long long* wstats = nullptr;
  void* st = nullptr;  // WfState
  OutBeam* out = nullptr;
  int* outRoot = nullptr;
};

namespace {

constexpr unsigned kFull = 0xffffffffu;
constexpr int kBlock = 128;
constexpr int kWarpsPerBlock = kBlock / 32;
constexpr double kFix = 1099511627776.0;  // 2^40: visible fraction in fixed point

enum WfStat { sKd, sLeaf, sTests, sHits, sSplits, sBeams, sRoots, sPresplit, sFive, sClimbs, sNum };
enum WfKernel { kAdmit, kTrace, kSplit, kNumKernels };
enum RootState { kRootTraced = 0, kRootKilled = 1, kRootNoBeam = 2 };
enum WfApp { kAppSoft = 0, kAppPrimary = 1, kAppPoint = 2 };

struct WfState {
  int next;                 // next root to admit
  int admitBegin, admitCount;
  int count[2];             // trace queue sizes (current / next round)
  int splitCount;
  int head[kNumKernels];    // fetch counters
  int done;
  int killed;               // roots abandoned
  int maxQueue;
  int overflow;             // a queue exceeded its capacity (must not happen)
  int outCount;             // output polygons (primary / point shadows)
  unsigned long long reserved;  // sub-beam slots reserved by roots in flight
  unsigned long long traceItems, splitItems;  // queue entries processed (totals)
};

struct WfRoot {
  BeamQuery q;
  Real ext;
  Real lightArea;  // soft: area of the light polygon (visible fraction = miss / lightArea)
  Vec3 x, ns;      // soft: shading point and normal (exact integration)
  int tri;         // point shadows: the receiver triangle
};

struct Params {
  SceneView sv;
  int R;  // roots
  int budget;
  unsigned long long capacity;
  WfBeam* tq[2];
  WfBeam* sq;
  WfRoot* roots;
  int* live;
  int* state;
  unsigned long long* missFx;
  double* G;
  unsigned long long* wstats;  // [kernel][warp][stat]
  int statWarps;
  WfState* st;
  // image (all applications)
  int W, H;
  BeamPlane camPlane;
  Vec3 eye, fwd, right, down;
  Real halfW, halfH;
  // soft shadows
  int tilesX;
  AreaLight light;
  Real offset;
  bool exact;
  const int* tri;
  const Vec3* X;
  const Vec3* N;
  float *vis, *E, *Eu;
  unsigned char* over;
  // primary visibility
  int tile, tilesW;
  bool cull;
  // point shadows
  Vec3 pointLight;
  const OutBeam* prim;
  // polygon output (primary / point shadows)
  OutBeam* out;
  int* outRoot;
  int outCap;
};

__device__ __forceinline__ int laneId() { return threadIdx.x & 31; }

// Persistent warps: lane 0 claims the next 32 entries for the warp.
__device__ __forceinline__ int fetch32(int* head) {
  int base = 0;
  if (laneId() == 0) base = atomicAdd(head, 32);
  return __shfl_sync(kFull, base, 0);
}

// Warp-aggregated append of n entries per lane; returns this lane's first
// slot. Lanes' blocks are consecutive in lane order.
__device__ __forceinline__ int warpAppend(int* counter, int n) {
  const int lane = laneId();
  int x = n;
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(kFull, x, o);
    if (lane >= o) x += y;
  }
  int base = 0;
  if (lane == 31 && x > 0) base = atomicAdd(counter, x);
  base = __shfl_sync(kFull, base, 31);
  return base + x - n;
}

__device__ void flushStats(const Params& p, int kernel, const TraceStats& ts) {
  unsigned long long v[sNum] = {ts.kdSteps, ts.leafVisits, ts.triTests, ts.hits, ts.splits,
                                ts.beams,   ts.rootBeams,  ts.presplitBeams, ts.fiveSplits, ts.climbs};
  const int warp = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  for (int f = 0; f < sNum; ++f) {
    unsigned long long s = v[f];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(kFull, s, o);
    if (laneId() == 0 && warp < p.statWarps) p.wstats[(size_t(kernel) * p.statWarps + warp) * sNum + f] += s;
  }
}

__device__ void rootDone(const Params& p) {
  atomicAdd(&p.st->reserved, (unsigned long long)(-(long long)p.budget));
}

__device__ void liveSub(const Params& p, int root, int n) {
  if (atomicSub(&p.live[root], n) == n) rootDone(p);
}

__device__ void killRoot(const Params& p, int root) {
  if (atomicExch(&p.state[root], int(kRootKilled)) != int(kRootKilled)) atomicAdd(&p.st->killed, 1);
}

__device__ void makeCtx(const Params& p, int root, BeamCtx& c) {
  c.sv = p.sv;
  const WfRoot& R = p.roots[root];
  c.setup(R.q, R.ext);
}

__device__ void appendPoly(const Params& p, int root, const OutBeam& ob) {
  const int k = atomicAdd(&p.st->outCount, 1);
  if (k >= p.outCap) return;  // the host retries with a larger buffer
  p.out[k] = ob;
  p.outRoot[k] = root;
}

// Finished pieces. Soft shadows: unoccluded area in fixed point (and the
// exact form factor). Primary: visible polygons. Point shadows: occluded
// pieces, projected back into the image.
template <int App>
struct WfOut {
  const Params* p;
  __device__ void emit(int root, const Poly2& poly, int tri, Real area) {
    const WfRoot& R = p->roots[root];
    if (App == kAppSoft) {
      if (tri >= 0) return;  // occluded
      double f = double(area) / double(R.lightArea) * kFix;
      if (f > 0) atomicAdd(&p->missFx[root], (unsigned long long)llrint(f));
      if (p->exact) {
        Vec3 P[4];
        const int n = poly.n < 4 ? poly.n : 4;
        for (int i = 0; i < n; ++i) P[i] = R.q.plane.point(poly.x[i], poly.y[i]);
        atomicAdd(&p->G[root], double(polygonFormFactorG(R.x, R.ns, P, n)));
      }
      return;
    }
    if (tri < 0) return;  // background / lit
    OutBeam ob;
    if (App == kAppPrimary) {
      ob.n = poly.n;
      for (int i = 0; i < 4; ++i) {
        const int k = i < poly.n ? i : poly.n - 1;
        ob.x[i] = poly.x[k];
        ob.y[i] = poly.y[k];
      }
      ob.tri = tri;
      ob.area = area;
    } else if (!pointShadowToImage(p->camPlane, R.q.plane, poly.x, poly.y, poly.n, R.tri, ob)) {
      return;
    }
    appendPoly(*p, root, ob);
  }
  __device__ void dropped(int, Real) {}
};

struct DevPush {
  const Params* p;
  int cur;
  int* count;
  __device__ bool operator()(const WfBeam& r) {
    int slot = atomicAdd(&p->st->count[cur], 1);
    if (slot >= int(p->capacity)) {
      p->st->overflow = 1;
      return false;
    }
    p->tq[cur][slot] = r;
    ++*count;
    return true;
  }
};

// Soft shadows: root k is a pixel in 8x4 tile order (a warp's 32 roots form
// one tile).
__device__ __forceinline__ bool softPixel(const Params& p, int k, int& pix) {
  const int tile = k >> 5, lane = k & 31;
  const int x = (tile % p.tilesX) * 8 + (lane & 7), y = (tile / p.tilesX) * 4 + (lane >> 3);
  pix = y * p.W + x;
  return x < p.W && y < p.H;
}

// Sets up root k and queues its pre-split pieces.
template <int App>
__device__ void admitRoot(const Params& p, int k, int cur, TraceStats& ts) {
  p.live[k] = 0;
  p.state[k] = kRootNoBeam;
  WfRoot R;
  Poly2 poly;
  if (App == kAppSoft) {
    p.missFx[k] = 0;
    p.G[k] = 0;
    int pix;
    if (!softPixel(p, k, pix)) {
      rootDone(p);
      return;
    }
    p.over[pix] = 0;
    const int T = p.tri[pix];
    if (T < 0) {
      p.vis[pix] = -1;
      p.E[pix] = 0;
      p.Eu[pix] = 0;
      rootDone(p);
      return;
    }
    const Vec3 x = p.X[pix], ns = p.N[pix];
    p.Eu[pix] = float(p.light.area() * lightCenterG(p.light, x, ns));
    if (!lightBeamSetup(p.light, x, ns, T, p.offset, R.q, poly, R.lightArea)) {
      p.vis[pix] = 0;
      p.E[pix] = 0;
      rootDone(p);
      return;
    }
    R.x = x;
    R.ns = ns;
  } else if (App == kAppPrimary) {
    const int tx = k % p.tilesW, ty = k / p.tilesW;
    const int x0 = tx * p.tile, y0 = ty * p.tile;
    const int x1 = min(p.W, x0 + p.tile), y1 = min(p.H, y0 + p.tile);
    const Real qx0 = (2 * Real(x0) / Real(p.W) - 1) * p.halfW, qx1 = (2 * Real(x1) / Real(p.W) - 1) * p.halfW;
    const Real qy0 = (2 * Real(y0) / Real(p.H) - 1) * p.halfH, qy1 = (2 * Real(y1) / Real(p.H) - 1) * p.halfH;
    poly.push(qx0, qy0);
    poly.push(qx1, qy0);
    poly.push(qx1, qy1);
    poly.push(qx0, qy1);
    R.q.plane = p.camPlane;
    R.q.mode = BeamMode::Nearest;
    R.q.cullBackfaces = p.cull;
  } else {
    const OutBeam pb = p.prim[k];
    const int T = pb.tri;
    R.tri = T;
    if (!pointShadowSetup(p.eye, p.fwd, p.right, p.down, p.sv.triN[T], p.sv.pos[p.sv.tri3[3 * size_t(T)]], T,
                          p.pointLight, pb.x, pb.y, pb.n, R.q, poly)) {
      appendPoly(p, k, pb);  // the receiver faces away from the light: all in shadow
      rootDone(p);
      return;
    }
  }
  R.ext = poly.extent();
  p.roots[k] = R;
  p.state[k] = kRootTraced;
  BeamCtx c;
  c.sv = p.sv;
  c.setup(R.q, R.ext);
  WfOut<App> out{&p};
  int pushed = 0;
  DevPush push{&p, cur, &pushed};
  const bool ok = wfAdmit(c, poly, k, out, push, ts);
  p.live[k] = pushed;
  if (!ok || pushed > p.budget) killRoot(p, k);
  if (pushed == 0) rootDone(p);
}

__global__ void scheduleKernel(Params p, int cur) {
  WfState* s = p.st;
  const int nxt = cur ^ 1;
  s->traceItems += (unsigned long long)s->count[nxt];  // (the previous round's queues)
  s->splitItems += (unsigned long long)s->splitCount;
  s->count[nxt] = 0;
  s->splitCount = 0;
  for (int k = 0; k < kNumKernels; ++k) s->head[k] = 0;
  const unsigned long long freeSlots = p.capacity > s->reserved ? p.capacity - s->reserved : 0;
  long long n = (long long)(p.R - s->next);
  long long fit = (long long)(freeSlots / (unsigned long long)p.budget);
  if (fit < n) n = fit;
  s->admitBegin = s->next;
  s->admitCount = int(n);
  s->next += int(n);
  s->reserved += (unsigned long long)n * (unsigned long long)p.budget;
  if (s->count[cur] > s->maxQueue) s->maxQueue = s->count[cur];
  s->done = (n == 0 && s->next >= p.R && s->count[cur] == 0) ? 1 : 0;
}

template <int App>
__global__ void __launch_bounds__(kBlock) admitKernel(Params p, int cur) {
  TraceStats ts;
  const int begin = p.st->admitBegin, n = p.st->admitCount;
  for (;;) {
    const int base = fetch32(&p.st->head[kAdmit]);
    if (base >= n) break;
    const int i = base + laneId();
    if (i < n) admitRoot<App>(p, begin + i, cur, ts);
  }
  flushStats(p, kAdmit, ts);
}

// Every iteration each lane advances its sub-beam by one leaf (climb and
// descend to it, test its triangles). Lanes whose sub-beam finished or needs a
// clip take new ones from the queue at once (one atomicAdd per warp for all
// idle lanes), so warps do not wait for their longest sub-beam: Aila & Laine's
// persistent threads with ray replacement. (Measured alternatives: one unit
// of work per iteration ("if-if"), or one triangle test per iteration with
// the cheap steps before it, were 15-60% slower.)
constexpr int kRefill = 1;

// Instrumentation (-DBT_WF_CYCLES): warp cycles per phase of the trace and
// split loops.
#ifdef BT_WF_CYCLES
__device__ unsigned long long g_cyc[8];
#define WF_CYC(k)                     \
  do {                                \
    __syncwarp();                     \
    const long long t_ = clock64();   \
    cyc[k] += t_ - tprev;             \
    tprev = t_;                       \
  } while (0)
#else
#define WF_CYC(k)
#endif

template <int App>
__global__ void __launch_bounds__(kBlock, 4) traceKernel(Params p, int cur) {
  TraceStats ts;
  const int n = p.st->count[cur];
  const WfBeam* q = p.tq[cur];
  const int lane = laneId();
  const unsigned below = (1u << lane) - 1;
  bool active = false, exhausted = false;
  WfBeam r;
  WfWork w;
  WfStepState ss;
  BeamCtx c;
  c.sv = p.sv;
  WfOut<App> out{&p};
#ifdef BT_WF_CYCLES
  long long cyc[4] = {0, 0, 0, 0};
  long long tprev = clock64();
#endif
  for (;;) {
    const unsigned idle = __ballot_sync(kFull, !active);
    if (!exhausted && (__popc(idle) >= kRefill || idle == kFull)) {
      int base = 0;
      if (lane == 0) base = atomicAdd(&p.st->head[kTrace], __popc(idle));
      base = __shfl_sync(kFull, base, 0);
      if (base + __popc(idle) >= n) exhausted = true;
      if (!active) {
        const int i = base + __popc(idle & below);
        if (i < n) {
          r = q[i];
          if (r.n == 0 || p.state[r.root] == kRootKilled) {  // a dropped sliver, or an abandoned root
            liveSub(p, r.root, 1);
          } else {
            const WfRoot& R = p.roots[r.root];
            c.setup(R.q, R.ext);
            wfExpand(c, r, w);
            wfBegin(c, r, ss);
            active = true;
          }
        }
      }
    }
    if (!__any_sync(kFull, active)) {
      if (exhausted) break;
      continue;
    }
    WF_CYC(0);
    // one leaf: climb / descend to it, then test its triangles
    int res = kWfContinue;
    if (active) {
      BT_SIMT(0);
      res = wfAdvance(c, r, w, ss, out, ts);
    }
    WF_CYC(1);
    if (active) {
      while (res == kWfContinue) {
        res = wfTestOne(c, r, w, out, ts);
        if (res != kWfContinue || !wfMoreInLeaf(c, r)) break;
      }
    }
    WF_CYC(2);
    if (active) {
      if (res == kWfFinished) liveSub(p, r.root, 1);
      if (res != kWfContinue) active = false;
    }
    const unsigned splits = __ballot_sync(kFull, res == kWfSplit);
    if (splits) {
      int base = 0;
      if (lane == 0) base = atomicAdd(&p.st->splitCount, __popc(splits));
      base = __shfl_sync(kFull, base, 0);
      if (res == kWfSplit) p.sq[base + __popc(splits & below)] = r;
    }
    WF_CYC(3);
  }
#ifdef BT_WF_CYCLES
  if (lane == 0)
    for (int k = 0; k < 4; ++k) atomicAdd(&g_cyc[k], (unsigned long long)cyc[k]);
#endif
  flushStats(p, kTrace, ts);
}

// Clips each queued sub-beam, reserves its pieces' slots in the next trace
// queue (warp-aggregated) and writes them there directly.
template <int App>
__global__ void __launch_bounds__(kBlock) splitKernel(Params p, int cur) {
  TraceStats ts;
  const int n = p.st->splitCount;
  const int nxt = cur ^ 1;
  WfClipSink sink;
#ifdef BT_WF_CYCLES
  long long cyc[4] = {0, 0, 0, 0};
  long long tprev = clock64();
#endif
  for (;;) {
    const int base = fetch32(&p.st->head[kSplit]);
    if (base >= n) break;
    const int i = base + laneId();
    int slots = 0;
    bool have = false, emit = false, commit = false;
    WfBeam r;
    WfWork w;
    BeamCtx c;
    if (i < n) {
      r = p.sq[i];
      if (p.state[r.root] == kRootKilled) {
        liveSub(p, r.root, 1);
      } else {
        makeCtx(p, r.root, c);
        wfExpand(c, r, w);
        have = true;
      }
    }
    WF_CYC(0);
    bool ok = false;
    if (have) ok = wfSplitClip(c, r, w, true, sink, ts, commit, slots);
    WF_CYC(1);
    if (have) {
      const int root = r.root;
      if (!ok) {
        killRoot(p, root);
        liveSub(p, root, 1);
        slots = 0;
      } else {
        const int old = atomicAdd(&p.live[root], slots - 1);
        if (old + slots - 1 > p.budget) {
          killRoot(p, root);
          liveSub(p, root, slots);
          slots = 0;
        } else {
          emit = true;
          if (slots == 0 && old == 1) rootDone(p);
        }
      }
    }
    const int slot = warpAppend(&p.st->count[nxt], slots);
    if (slots > 0 && slot + slots > int(p.capacity)) {
      p.st->overflow = 1;
      emit = false;
    }
    WF_CYC(2);
    if (emit) {
      WfOut<App> out{&p};
      wfSplitEmit(c, r, w, sink, commit, out, ts, p.tq[nxt] + slot);
    }
    WF_CYC(3);
  }
#ifdef BT_WF_CYCLES
  if (laneId() == 0)
    for (int k = 0; k < 4; ++k) atomicAdd(&g_cyc[4 + k], (unsigned long long)cyc[k]);
#endif
  flushStats(p, kSplit, ts);
}

__global__ void finalizeSoftKernel(Params p) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= p.R) return;
  int pix;
  if (!softPixel(p, k, pix)) return;
  const int s = p.state[k];
  if (s == kRootNoBeam) return;
  if (s == kRootKilled) {
    p.over[pix] = 1;
    p.vis[pix] = 0;
    p.E[pix] = 0;
    return;
  }
  double V = double(p.missFx[k]) / kFix;
  V = V < 0 ? 0 : (V > 1 ? 1 : V);
  p.vis[pix] = float(V);
  if (p.exact) {
    p.E[pix] = float(p.G[k]);
  } else {
    const Vec3 x = p.X[pix], ns = p.N[pix];
    p.E[pix] = float(double(p.light.area() * lightCenterG(p.light, x, ns)) * V);
  }
}

template <class K>
int persistentGrid(K kernel) {
  int dev = 0, sms = 1, perSM = 1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, kernel, kBlock, 0);
  return sms * std::max(1, perSM);
}

// Allocates (or reuses) the engine's buffers for R roots and fills the
// common parameters.
void prepare(GpuRenderer::Impl& I, const WfConfig& cfg, int R, int statWarps, Params& p) {
  if (!I.wf) I.wf = wfCreateBuffers();
  WfBuffers& B = *I.wf;
  if (B.capacity != cfg.capacity) {
    for (WfBeam*& q : B.tq) {
      if (q) cudaFree(q);
      BT_CUDA_CHECK(cudaMalloc(&q, sizeof(WfBeam) * cfg.capacity));
    }
    if (B.sq) cudaFree(B.sq);
    BT_CUDA_CHECK(cudaMalloc(&B.sq, sizeof(WfBeam) * cfg.capacity));
    B.capacity = cfg.capacity;
  }
  if (B.rootCap < size_t(R)) {
    cudaFree(B.roots);
    cudaFree(B.live);
    cudaFree(B.state);
    cudaFree(B.missFx);
    cudaFree(B.G);
    BT_CUDA_CHECK(cudaMalloc(&B.roots, sizeof(WfRoot) * size_t(R)));
    BT_CUDA_CHECK(cudaMalloc(&B.live, sizeof(int) * size_t(R)));
    BT_CUDA_CHECK(cudaMalloc(&B.state, sizeof(int) * size_t(R)));
    BT_CUDA_CHECK(cudaMalloc(&B.missFx, sizeof(unsigned long long) * size_t(R)));
    BT_CUDA_CHECK(cudaMalloc(&B.G, sizeof(double) * size_t(R)));
    B.rootCap = size_t(R);
  }
  if (!B.st) BT_CUDA_CHECK(cudaMalloc(&B.st, sizeof(WfState)));
  const size_t statN = size_t(kNumKernels) * size_t(statWarps) * sNum;
  if (B.statCap < statN) {
    cudaFree(B.wstats);
    BT_CUDA_CHECK(cudaMalloc(&B.wstats, sizeof(unsigned long long) * statN));
    B.statCap = statN;
  }
  BT_CUDA_CHECK(cudaMemset(B.wstats, 0, sizeof(unsigned long long) * statN));
  BT_CUDA_CHECK(cudaMemset(B.st, 0, sizeof(WfState)));
  p = Params();
  p.sv = I.sv;
  p.R = R;
  p.budget = cfg.budget;
  p.capacity = cfg.capacity;
  p.tq[0] = B.tq[0];
  p.tq[1] = B.tq[1];
  p.sq = B.sq;
  p.roots = static_cast<WfRoot*>(B.roots);
  p.live = B.live;
  p.state = B.state;
  p.missFx = B.missFx;
  p.G = B.G;
  p.wstats = B.wstats;
  p.statWarps = statWarps;
  p.st = static_cast<WfState*>(B.st);
}

void ensureOutput(GpuRenderer::Impl& I, size_t cap, Params& p) {
  WfBuffers& B = *I.wf;
  if (B.outCap < cap) {
    cudaFree(B.out);
    cudaFree(B.outRoot);
    BT_CUDA_CHECK(cudaMalloc(&B.out, sizeof(OutBeam) * cap));
    BT_CUDA_CHECK(cudaMalloc(&B.outRoot, sizeof(int) * cap));
    B.outCap = cap;
  }
  p.out = B.out;
  p.outRoot = B.outRoot;
  p.outCap = int(cap);
}

template <int App>
int statWarpsFor() {
  const int gA = persistentGrid(admitKernel<App>), gT = persistentGrid(traceKernel<App>),
            gS = persistentGrid(splitKernel<App>);
  return std::max(gA, std::max(gT, gS)) * kWarpsPerBlock;
}

// Runs rounds until every root is done; then the application's final pass.
template <int App>
void runRounds(const WfConfig& cfg, Params& p, WfRunInfo& info, WfState& hs) {
  const int gA = persistentGrid(admitKernel<App>), gT = persistentGrid(traceKernel<App>),
            gS = persistentGrid(splitKernel<App>);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  int rounds = 0;
  // BT_WF_PROFILE: time each kernel type (events between all launches).
  const bool profile = getenv("BT_WF_PROFILE") != nullptr;
  std::vector<cudaEvent_t> ev;
  auto mark = [&]() {
    if (!profile) return;
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e);
    ev.push_back(e);
  };
  for (;;) {
    for (int k = 0; k < cfg.roundsPerCheck; ++k, ++rounds) {
      const int cur = rounds & 1;
      mark();
      scheduleKernel<<<1, 1>>>(p, cur);
      mark();
      admitKernel<App><<<gA, kBlock>>>(p, cur);
      mark();
      traceKernel<App><<<gT, kBlock>>>(p, cur);
      mark();
      splitKernel<App><<<gS, kBlock>>>(p, cur);
    }
    BT_CUDA_CHECK(cudaGetLastError());
    BT_CUDA_CHECK(cudaMemcpy(&hs, p.st, sizeof(WfState), cudaMemcpyDeviceToHost));
    if (hs.done || hs.overflow || rounds > 1000000) break;
  }
  mark();
  if (App == kAppSoft) finalizeSoftKernel<<<(p.R + 255) / 256, 256>>>(p);
  BT_CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1);
  BT_CUDA_CHECK(cudaEventSynchronize(e1));
  BT_CUDA_CHECK(cudaMemcpy(&hs, p.st, sizeof(WfState), cudaMemcpyDeviceToHost));
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  if (hs.overflow) {
    fprintf(stderr, "wavefront engine: queue overflow (capacity %zu)\n", cfg.capacity);
    std::abort();
  }
#ifdef BT_WF_CYCLES
  {
    unsigned long long h[8];
    cudaMemcpyFromSymbol(h, g_cyc, sizeof(h));
    double tot = double(h[0] + h[1] + h[2] + h[3]) + 1e-30;
    fprintf(stderr, "  trace loop cycles: refill %.1f%%, advance %.1f%%, tests %.1f%%, append %.1f%%\n",
            100 * h[0] / tot, 100 * h[1] / tot, 100 * h[2] / tot, 100 * h[3] / tot);
    tot = double(h[4] + h[5] + h[6] + h[7]) + 1e-30;
    fprintf(stderr, "  split loop cycles: load %.1f%%, clip %.1f%%, reserve %.1f%%, emit %.1f%%\n",
            100 * h[4] / tot, 100 * h[5] / tot, 100 * h[6] / tot, 100 * h[7] / tot);
    unsigned long long z[8] = {};
    cudaMemcpyToSymbol(g_cyc, z, sizeof(z));
  }
#endif
#ifdef BT_WF_SIMT
  {
    unsigned long long h[16];
    cudaMemcpyFromSymbol(h, g_simt, sizeof(h));
    const char* nm[4] = {"outer", "descend", "tri test", "climb"};
    for (int k = 0; k < 4; ++k)
      if (h[2 * k])
        fprintf(stderr, "  SIMT %-8s: %llu warp-steps, %.1f active lanes\n", nm[k], h[2 * k],
                double(h[2 * k + 1]) / double(h[2 * k]));
    unsigned long long z[16] = {};
    cudaMemcpyToSymbol(g_simt, z, sizeof(z));
  }
#endif
  if (profile) {
    double t[4] = {0, 0, 0, 0};
    for (size_t i = 0; i + 1 < ev.size(); ++i) {
      float m = 0;
      cudaEventElapsedTime(&m, ev[i], ev[i + 1]);
      t[i % 4] += m;
    }
    for (cudaEvent_t e : ev) cudaEventDestroy(e);
    fprintf(stderr, "  wavefront kernels: schedule %.2f ms, admit %.2f ms, trace %.2f ms, split %.2f ms "
            "(grids %d/%d/%d blocks)\n", t[0], t[1], t[2], t[3], gA, gT, gS);
  }

  std::vector<unsigned long long> ws(size_t(kNumKernels) * size_t(p.statWarps) * sNum);
  BT_CUDA_CHECK(cudaMemcpy(ws.data(), p.wstats, sizeof(unsigned long long) * ws.size(), cudaMemcpyDeviceToHost));
  unsigned long long tot[sNum] = {};
  for (size_t i = 0; i < ws.size(); ++i) tot[i % sNum] += ws[i];
  TraceStats& ts = info.stats;
  ts = TraceStats();
  ts.kdSteps = tot[sKd];
  ts.leafVisits = tot[sLeaf];
  ts.triTests = tot[sTests];
  ts.hits = tot[sHits];
  ts.splits = tot[sSplits];
  ts.beams = tot[sBeams];
  ts.rootBeams = tot[sRoots];
  ts.presplitBeams = tot[sPresplit];
  ts.fiveSplits = tot[sFive];
  ts.climbs = tot[sClimbs];
  info.ms = ms;
  info.rounds = rounds;
  info.killedRoots = hs.killed;
  info.maxQueue = hs.maxQueue;
  info.traceItems = hs.traceItems;
  info.splitItems = hs.splitItems;
}

void setCamera(const Camera& cam, Params& p) {
  p.W = cam.W;
  p.H = cam.H;
  p.camPlane = cam.plane();
  p.eye = cam.eye;
  p.fwd = cam.fwd;
  p.right = cam.right;
  p.down = cam.down;
  p.halfW = cam.halfW;
  p.halfH = cam.halfH;
}

// Downloads the output polygons of roots that were not abandoned, and the
// list of abandoned roots.
void downloadPolys(GpuRenderer::Impl& I, const Params& p, const WfState& hs, std::vector<OutBeam>& polys,
                   std::vector<int>& killed) {
  const int n = std::min(hs.outCount, p.outCap);
  std::vector<OutBeam> out(static_cast<size_t>(n));
  std::vector<int> root(static_cast<size_t>(n)), state(static_cast<size_t>(p.R));
  if (n > 0) {
    BT_CUDA_CHECK(cudaMemcpy(out.data(), I.wf->out, sizeof(OutBeam) * size_t(n), cudaMemcpyDeviceToHost));
    BT_CUDA_CHECK(cudaMemcpy(root.data(), I.wf->outRoot, sizeof(int) * size_t(n), cudaMemcpyDeviceToHost));
  }
  BT_CUDA_CHECK(cudaMemcpy(state.data(), p.state, sizeof(int) * size_t(p.R), cudaMemcpyDeviceToHost));
  polys.clear();
  polys.reserve(size_t(n));
  for (int i = 0; i < n; ++i)
    if (state[size_t(root[size_t(i)])] != kRootKilled) polys.push_back(out[size_t(i)]);
  killed.clear();
  for (int k = 0; k < p.R; ++k)
    if (state[size_t(k)] == kRootKilled) killed.push_back(k);
}

}  // namespace

WfBuffers* wfCreateBuffers() { return new WfBuffers(); }

void wfDestroyBuffers(WfBuffers* b) {
  if (!b) return;
  cudaFree(b->tq[0]);
  cudaFree(b->tq[1]);
  cudaFree(b->sq);
  cudaFree(b->roots);
  cudaFree(b->live);
  cudaFree(b->state);
  cudaFree(b->missFx);
  cudaFree(b->G);
  cudaFree(b->wstats);
  cudaFree(b->st);
  cudaFree(b->out);
  cudaFree(b->outRoot);
  delete b;
}

void wfSoftShadows(GpuRenderer::Impl& I, const WfConfig& cfg, const AreaLight& L, Real offset, int W, int H,
                   const int* dTri, const Vec3* dX, const Vec3* dN, bool exact, float* dVis, float* dE, float* dEu,
                   unsigned char* dOver, WfRunInfo& info) {
  const int tilesX = (W + 7) / 8, tilesY = (H + 3) / 4;
  Params p;
  prepare(I, cfg, tilesX * tilesY * 32, statWarpsFor<kAppSoft>(), p);
  p.W = W;
  p.H = H;
  p.tilesX = tilesX;
  p.light = L;
  p.offset = offset;
  p.exact = exact;
  p.tri = dTri;
  p.X = dX;
  p.N = dN;
  p.vis = dVis;
  p.E = dE;
  p.Eu = dEu;
  p.over = dOver;
  WfState hs;
  runRounds<kAppSoft>(cfg, p, info, hs);
}

void wfPrimary(GpuRenderer::Impl& I, const WfConfig& cfg, const Camera& cam, bool cull, int tile,
               std::vector<OutBeam>& hits, std::vector<int>& killedTiles, WfRunInfo& info) {
  const int tilesW = (cam.W + tile - 1) / tile, tilesH = (cam.H + tile - 1) / tile;
  size_t outCap = std::max<size_t>(size_t(1) << 16, size_t(cam.W) * size_t(cam.H) / 4);
  for (;;) {
    Params p;
    prepare(I, cfg, tilesW * tilesH, statWarpsFor<kAppPrimary>(), p);
    ensureOutput(I, outCap, p);
    setCamera(cam, p);
    p.tile = tile;
    p.tilesW = tilesW;
    p.cull = cull;
    WfState hs;
    runRounds<kAppPrimary>(cfg, p, info, hs);
    if (hs.outCount > p.outCap) {  // output buffer too small: retry
      outCap = size_t(hs.outCount) + size_t(hs.outCount) / 4;
      continue;
    }
    downloadPolys(I, p, hs, hits, killedTiles);
    return;
  }
}

void wfPointShadows(GpuRenderer::Impl& I, const WfConfig& cfg, const Camera& cam, const Vec3& light,
                    const OutBeam* dPrim, int nPrim, std::vector<OutBeam>& shadowPolys, std::vector<int>& killed,
                    WfRunInfo& info) {
  size_t outCap = std::max<size_t>(size_t(1) << 16, size_t(nPrim) * 2);
  for (;;) {
    Params p;
    prepare(I, cfg, nPrim, statWarpsFor<kAppPoint>(), p);
    ensureOutput(I, outCap, p);
    setCamera(cam, p);
    p.pointLight = light;
    p.prim = dPrim;
    WfState hs;
    runRounds<kAppPoint>(cfg, p, info, hs);
    if (hs.outCount > p.outCap) {
      outCap = size_t(hs.outCount) + size_t(hs.outCount) / 4;
      continue;
    }
    downloadPolys(I, p, hs, shadowPolys, killed);
    return;
  }
}

}  // namespace bt
