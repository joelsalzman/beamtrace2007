// CUDA port: primary rays, the soft-shadow driver, and the first port's
// kernels (one thread per root beam running BeamCore depth-first; still used
// for primary visibility and point shadows, and for soft shadows with
// BT_GPU_ENGINE=v1). Soft shadows use the wavefront engine (wavefront.cu).
#include <algorithm>
#include <cstring>
#include <vector>

#include "beam/beam_core.h"
#include "beam/wavefront_core.h"
#include "cuda/gpu_common.cuh"
#include "ray/ray_core.h"
#include "render/soft_core.h"
#include "util/misc.h"

namespace bt {

namespace {

// Fixed-capacity per-thread storage for BeamCore (lives in local memory).
// Mailboxes and the triangle-setup cache are small direct-mapped tables:
// a miss only costs a redundant (idempotent) test, never a wrong answer.
// Capacities for soft shadows were sized from measured high-water marks
// (plant/sponza/conference/building at 512^2: <= 64 beams, 39 work items, 27
// frames, 59 list entries, 6 polygons); the rare pixel that needs more falls
// back to the CPU. Primary tile beams use larger capacities.
template <int B_, int W_, int F_, int L_>
struct DevStorageT {
  static constexpr int kBeams = B_, kWork = W_, kFrames = F_, kList = L_, kPolys = 8;
  static constexpr int kCache = 1, kMail = 32, kEv = 32;

  CoreBeam pool[kBeams];
  int freeStack[kBeams];
  int nFree, nPool;
  CoreWork work[kWork];
  int workMax[kWork];  // prefix maximum of pending work items' frames
  int nWork;
  CoreFrame frames[kFrames];
  int nFrames;
  int lists[2][kList];
  int nList[2];
  Poly2 polys[2][kPolys];
  int nPolys[2];
  CoreTriInfo cache[kCache];
  int mailTri[kMail];
  uint64_t mailStamp[kMail];
  uint64_t evParent[kEv];
  uint64_t nextEvent;
  double hitArea, missArea, droppedArea, G;
  int outputs;
  bool overflow;
  int why;  // overflow reason bits: 1 beams, 2 work, 4 frames, 8 lists, 16 polys
  int hiBeams, hiWork, hiFrames, hiList, hiPolys;  // high-water marks
  // exact integration of the visible light polygons
  bool exact;
  Vec3 x, ns;
  BeamPlane plane;
  // optional polygon output (hit / occluded pieces only), in global memory
  OutBeam* outBuf;
  int outCap, outN;

  __device__ void beginTrace() {
    nFree = nPool = nWork = nFrames = 0;
    nList[0] = nList[1] = nPolys[0] = nPolys[1] = 0;
    for (int i = 0; i < kCache; ++i) {
      cache[i].stamp = 0;
      cache[i].tri = -1;
    }
    for (int i = 0; i < kMail; ++i) mailTri[i] = -1;
    nextEvent = 1;
    hitArea = missArea = droppedArea = G = 0;
    outputs = 0;
    outN = 0;
    overflow = false;
    why = 0;
    hiBeams = hiWork = hiFrames = hiList = hiPolys = 0;
  }
  __device__ int allocBeam() {
    if (nFree > 0) return freeStack[--nFree];
    if (nPool < kBeams) {
      hiBeams = nPool + 1 > hiBeams ? nPool + 1 : hiBeams;
      return nPool++;
    }
    why |= 1;
    return -1;
  }
  __device__ CoreBeam& beam(int i) { return pool[i]; }
  __device__ void freeBeam(int i) { freeStack[nFree++] = i; }
  __device__ bool pushWork(const CoreWork& w) {
    if (nWork >= kWork) {
      why |= 2;
      return false;
    }
    hiWork = nWork + 1 > hiWork ? nWork + 1 : hiWork;
    workMax[nWork] = nWork == 0 ? w.frame : (w.frame > workMax[nWork - 1] ? w.frame : workMax[nWork - 1]);
    work[nWork++] = w;
    return true;
  }
  __device__ bool popWork(CoreWork& w) {
    if (nWork == 0) return false;
    w = work[--nWork];
    return true;
  }
  __device__ int maxPendingFrame() const { return nWork == 0 ? -1 : workMax[nWork - 1]; }
  __device__ void truncateFrames(int n) {
    if (n < nFrames) nFrames = n;
  }
  __device__ int pushFrame(const CoreFrame& f) {
    if (nFrames >= kFrames) {
      why |= 4;
      return -1;
    }
    frames[nFrames] = f;
    hiFrames = nFrames + 1 > hiFrames ? nFrames + 1 : hiFrames;
    return nFrames++;
  }
  __device__ const CoreFrame& frame(int i) const { return frames[i]; }
  __device__ void listClear(int l) { nList[l] = 0; }
  __device__ bool listPush(int l, int v) {
    if (nList[l] >= kList) {
      why |= 8;
      return false;
    }
    lists[l][nList[l]++] = v;
    hiList = nList[l] > hiList ? nList[l] : hiList;
    return true;
  }
  __device__ int listSize(int l) const { return nList[l]; }
  __device__ int listAt(int l, int i) const { return lists[l][i]; }
  __device__ void polyClear(int l) { nPolys[l] = 0; }
  __device__ bool polyPush(int l, const Poly2& p) {
    if (nPolys[l] >= kPolys) {
      why |= 16;
      return false;
    }
    polys[l][nPolys[l]++] = p;
    hiPolys = nPolys[l] > hiPolys ? nPolys[l] : hiPolys;
    return true;
  }
  __device__ int polySize(int l) const { return nPolys[l]; }
  __device__ const Poly2& polyAt(int l, int i) const { return polys[l][i]; }
  __device__ CoreTriInfo& triSlot(int tri) { return cache[tri & (kCache - 1)]; }
  __device__ uint64_t stamp(int tri) const {
    int i = tri & (kMail - 1);
    return mailTri[i] == tri ? mailStamp[i] : 0;
  }
  __device__ void setStamp(int tri, uint64_t e) {
    int i = tri & (kMail - 1);
    mailTri[i] = tri;
    mailStamp[i] = e;
  }
  __device__ uint64_t newEvent(uint64_t parent) {
    evParent[nextEvent % kEv] = parent;
    return nextEvent++;
  }
  __device__ uint64_t parentOf(uint64_t e) const {
    return (e > 0 && e < nextEvent && e + kEv >= nextEvent) ? evParent[e % kEv] : 0;
  }
  __device__ bool firstHit(int) { return false; }
  __device__ void output(const Poly2& p, int tri, Real area) {
    ++outputs;
    if (tri >= 0 && outBuf) {
      if (outN >= outCap) {
        overflow = true;
        why |= 64;
      } else {
        OutBeam& ob = outBuf[outN++];
        ob.n = p.n;
        for (int i = 0; i < 4; ++i) {
          int k = i < p.n ? i : p.n - 1;
          ob.x[i] = p.x[k];
          ob.y[i] = p.y[k];
        }
        ob.tri = tri;
        ob.area = area;
      }
    }
    if (tri >= 0) {
      hitArea += double(area);
    } else {
      missArea += double(area);
      if (exact) {
        Vec3 P[4];
        for (int i = 0; i < p.n && i < 4; ++i) P[i] = plane.point(p.x[i], p.y[i]);
        G += double(polygonFormFactorG(x, ns, P, p.n < 4 ? p.n : 4));
      }
    }
  }
  __device__ void dropped(Real area) { droppedArea += double(area); }
  __device__ int outputCount() const { return outputs; }
};

using DevStorage = DevStorageT<48, 48, 48, 48>;    // soft shadows (one pixel per thread)
using TileStorage = DevStorageT<96, 96, 96, 96>;   // primary visibility (one tile per thread)
using PointStorage = DevStorageT<48, 48, 48, 48>;  // point-light shadows (one hit beam per thread)

struct GpuCam {
  Vec3 eye, fwd, right, down;
  Real halfW, halfH;
  int W, H;
};

enum StatSlot {
  kKd, kLeaf, kTests, kHits, kSplits, kBeamsOut, kRoots, kPresplit, kMailSkips, kFive, kRays,
  kHiBeams, kHiWork, kHiFrames, kHiList, kHiPolys, kCycDesc, kCycLeaf, kCycAdv, kCycRoot, kCycTotal, kNumStats
};

__device__ void addStats(unsigned long long* g, const TraceStats& s) {
  atomicAdd(&g[kKd], (unsigned long long)s.kdSteps);
  atomicAdd(&g[kLeaf], (unsigned long long)s.leafVisits);
  atomicAdd(&g[kTests], (unsigned long long)s.triTests);
  atomicAdd(&g[kHits], (unsigned long long)s.hits);
  atomicAdd(&g[kSplits], (unsigned long long)s.splits);
  atomicAdd(&g[kBeamsOut], (unsigned long long)s.beams);
  atomicAdd(&g[kRoots], (unsigned long long)s.rootBeams);
  atomicAdd(&g[kPresplit], (unsigned long long)s.presplitBeams);
  atomicAdd(&g[kMailSkips], (unsigned long long)s.mailboxSkips);
  atomicAdd(&g[kFive], (unsigned long long)s.fiveSplits);
  atomicAdd(&g[kRays], (unsigned long long)s.rays);
}

__global__ void primaryKernel(SceneView sv, GpuCam cam, int* tri, Vec3* X, Vec3* N, unsigned long long* gstats) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cam.W || y >= cam.H) return;
  size_t k = size_t(y) * size_t(cam.W) + size_t(x);
  Real qx = (2 * (Real(x) + Real(0.5)) / Real(cam.W) - 1) * cam.halfW;
  Real qy = (2 * (Real(y) + Real(0.5)) / Real(cam.H) - 1) * cam.halfH;
  Ray r;
  r.o = cam.eye;
  r.d = cam.fwd + cam.right * qx + cam.down * qy;
  r.tmin = 0;
  r.tmax = kInf;
  RayHit h;
  TraceStats st;
  tri[k] = -1;
  if (rayTraverse<false>(sv, r, h, st, false, -1)) {
    tri[k] = h.tri;
    X[k] = r.o + r.d * h.t;
    Vec3 n = sv.triN[h.tri];
    if (dot(n, r.d) > 0) n = -n;
    N[k] = n;
  }
  addStats(gstats, st);
}

__device__ void softPixel(const SceneView& sv, const AreaLight& L, Real offset, int k, const int* tri, const Vec3* X,
                          const Vec3* N, bool exact, bool mailbox, float* vis, float* E, float* Eunocc,
                          unsigned char* overflow, TraceStats& stats, int* hi, long long* cyc) {
  overflow[k] = 0;
  int T = tri[k];
  if (T < 0) {
    vis[k] = -1;
    E[k] = 0;
    Eunocc[k] = 0;
    return;
  }
  const Vec3 p = X[k], n = N[k];
  const Real A = L.area();
  const Real G0 = lightCenterG(L, p, n);
  Eunocc[k] = float(A * G0);
  BeamQuery q;
  Poly2 root;
  Real lightArea;
  if (!lightBeamSetup(L, p, n, T, offset, q, root, lightArea)) {
    vis[k] = 0;
    E[k] = 0;
    return;
  }
  DevStorage st;
  st.outBuf = nullptr;
  st.outCap = 0;
  st.exact = exact;
  st.x = p;
  st.ns = n;
  st.plane = q.plane;
  BeamCore<DevStorage> core(sv, st);
  core.useMailbox = mailbox;
  long long t0 = btClock();
  core.trace(q, &root, 1);
  cyc[4] += btClock() - t0;
  for (int i = 0; i < 4; ++i) cyc[i] += core.cyc[i];
  if (st.overflow) {
    overflow[k] = (unsigned char)(st.why ? st.why : 32);
    vis[k] = 0;
    E[k] = 0;
  } else {
    Real V = Real(st.missArea) / lightArea;
    V = V < 0 ? Real(0) : (V > 1 ? Real(1) : V);
    vis[k] = float(V);
    E[k] = exact ? float(st.G) : float(A * G0 * V);
  }
  stats.add(core.stats);
  hi[0] = st.hiBeams > hi[0] ? st.hiBeams : hi[0];
  hi[1] = st.hiWork > hi[1] ? st.hiWork : hi[1];
  hi[2] = st.hiFrames > hi[2] ? st.hiFrames : hi[2];
  hi[3] = st.hiList > hi[3] ? st.hiList : hi[3];
  hi[4] = st.hiPolys > hi[4] ? st.hiPolys : hi[4];
}

// Persistent warps (Aila & Laine 2009): each warp repeatedly fetches an 8x4
// pixel tile, so its 32 lanes trace neighbouring pixels whose shadow beams
// take similar paths through the kd-tree (less divergence).
__global__ void __launch_bounds__(32, 4) softKernel(SceneView sv, AreaLight L, Real offset, int W, int H, const int* tri,
                                                 const Vec3* X, const Vec3* N, bool exact, bool mailbox,
                                                 float* vis, float* E, float* Eunocc, unsigned char* overflow,
                                                 unsigned long long* gstats, int* counter) {
  TraceStats stats;
  int hi[5] = {0, 0, 0, 0, 0};
  long long cyc[5] = {0, 0, 0, 0, 0};
  const int lane = threadIdx.x & 31;
  const int tilesX = (W + 7) / 8, tilesY = (H + 3) / 4, numTiles = tilesX * tilesY;
  for (;;) {
    int tile = 0;
    if (lane == 0) tile = atomicAdd(counter, 1);
    tile = __shfl_sync(0xffffffffu, tile, 0);
    if (tile >= numTiles) break;
    int x = (tile % tilesX) * 8 + (lane & 7), y = (tile / tilesX) * 4 + (lane >> 3);
    if (x < W && y < H)
      softPixel(sv, L, offset, y * W + x, tri, X, N, exact, mailbox, vis, E, Eunocc, overflow, stats, hi, cyc);
    __syncwarp();
  }
  addStats(gstats, stats);
  atomicMax(&gstats[kHiBeams], (unsigned long long)hi[0]);
  atomicMax(&gstats[kHiWork], (unsigned long long)hi[1]);
  atomicMax(&gstats[kHiFrames], (unsigned long long)hi[2]);
  atomicMax(&gstats[kHiList], (unsigned long long)hi[3]);
  atomicMax(&gstats[kHiPolys], (unsigned long long)hi[4]);
  for (int i = 0; i < 5; ++i) atomicAdd(&gstats[kCycDesc + i], (unsigned long long)cyc[i]);
}


// ---- Primary visibility: one image tile (a root beam) per thread ----
__global__ void primaryTileKernel(SceneView sv, BeamPlane camPlane, GpuCam cam, int tile, int tilesX, int numTiles,
                                  bool cull, OutBeam* out, int cap, int* outCount, unsigned char* overflow,
                                  unsigned long long* gstats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numTiles) return;
  int tx = i % tilesX, ty = i / tilesX;
  int x0 = tx * tile, y0 = ty * tile, x1 = min(cam.W, x0 + tile), y1 = min(cam.H, y0 + tile);
  Real qx0 = (2 * Real(x0) / Real(cam.W) - 1) * cam.halfW, qx1 = (2 * Real(x1) / Real(cam.W) - 1) * cam.halfW;
  Real qy0 = (2 * Real(y0) / Real(cam.H) - 1) * cam.halfH, qy1 = (2 * Real(y1) / Real(cam.H) - 1) * cam.halfH;
  Poly2 root;
  root.push(qx0, qy0);
  root.push(qx1, qy0);
  root.push(qx1, qy1);
  root.push(qx0, qy1);
  BeamQuery q;
  q.plane = camPlane;
  q.mode = BeamMode::Nearest;
  q.cullBackfaces = cull;
  TileStorage st;
  st.exact = false;
  st.outBuf = out + size_t(i) * size_t(cap);
  st.outCap = cap;
  BeamCore<TileStorage> core(sv, st);
  core.useMailbox = false;
  core.trace(q, &root, 1);
  outCount[i] = st.outN;
  overflow[i] = st.overflow ? 1 : 0;
  addStats(gstats, core.stats);
}

// ---- Point-light shadows: one primary hit beam per thread ----
__global__ void pointShadowKernel(SceneView sv, BeamPlane camPlane, GpuCam cam, Vec3 light, const OutBeam* prim,
                                  int nPrim, OutBeam* out, int cap, int* outCount, unsigned char* overflow,
                                  unsigned long long* gstats) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nPrim) return;
  const OutBeam pb = prim[j];
  overflow[j] = 0;
  const int T = pb.tri;
  const Vec3 nT = sv.triN[T];
  const Vec3 P0 = sv.pos[sv.tri3[3 * size_t(T)]];
  Real sideCam = dot(nT, cam.eye - P0), sideL = dot(nT, light - P0);
  if (!(sideCam * sideL > 0)) {  // receiver faces away from the light: fully shadowed
    outCount[j] = -1;
    return;
  }
  Poly2 root;
  BeamQuery q;
  Vec3 tu = anyOrthogonal(nT);
  q.plane = BeamPlane::make(light, P0, tu, cross(nT, tu));
  q.mode = BeamMode::AnyHit;
  q.farIsPlane = true;
  q.excludeTri = T;
  q.cullBackfaces = false;
  for (int i = 0; i < pb.n; ++i) {
    Vec3 d = cam.fwd + cam.right * pb.x[i] + cam.down * pb.y[i];
    Vec3 X = cam.eye + d * (dot(nT, P0 - cam.eye) / dot(nT, d));
    Real qx, qy;
    q.plane.coords(X, qx, qy);
    root.push(qx, qy);
  }
  PointStorage st;
  st.exact = false;
  st.outBuf = out + size_t(j) * size_t(cap);
  st.outCap = cap;
  BeamCore<PointStorage> core(sv, st);
  core.useMailbox = false;
  core.trace(q, &root, 1);
  if (st.overflow) {
    overflow[j] = 1;
    outCount[j] = 0;
    addStats(gstats, core.stats);
    return;
  }
  // Map the occluded pieces from the receiver plane back to the image plane.
  int n = 0;
  for (int k = 0; k < st.outN; ++k) {
    OutBeam ob = st.outBuf[k];
    OutBeam ip;
    ip.n = ob.n;
    ip.tri = T;
    ip.area = 0;
    bool ok = true;
    for (int i = 0; i < ob.n && ok; ++i) ok = camPlane.project(q.plane.point(ob.x[i], ob.y[i]), ip.x[i], ip.y[i]);
    if (!ok) continue;
    Poly2 pp;
    for (int i = 0; i < ip.n; ++i) pp.push(ip.x[i], ip.y[i]);
    Real a = pp.area();
    if (!(a > 0 || a < 0)) continue;  // receiver seen exactly edge-on
    if (a < 0) {
      pp.reverse();
      for (int i = 0; i < ip.n; ++i) {
        ip.x[i] = pp.x[i];
        ip.y[i] = pp.y[i];
      }
    }
    for (int i = ip.n; i < 4; ++i) {
      ip.x[i] = ip.x[ip.n - 1];
      ip.y[i] = ip.y[ip.n - 1];
    }
    st.outBuf[n++] = ip;
  }
  outCount[j] = n;
  addStats(gstats, core.stats);
}


// Packs per-item output slices (count[i] entries at i * cap) into a dense array.
__global__ void compactKernel(const OutBeam* in, int cap, const int* count, const int* offset, int n, OutBeam* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = count[i];
  for (int k = 0; k < c; ++k) out[offset[i] + k] = in[size_t(i) * size_t(cap) + size_t(k)];
}

// Downloads the outputs of `n` items (count < 0 or over[i] counts as 0) in one
// transfer; `offsets` receives each item's start in `dense`.
void downloadCompacted(const OutBeam* dIn, int cap, const std::vector<int>& count,
                       const std::vector<unsigned char>& over, std::vector<int>& offsets, std::vector<OutBeam>& dense) {
  const int n = int(count.size());
  offsets.assign(size_t(n), 0);
  std::vector<int> c(static_cast<size_t>(n));
  int total = 0;
  for (int i = 0; i < n; ++i) {
    c[size_t(i)] = (over[size_t(i)] || count[size_t(i)] < 0) ? 0 : count[size_t(i)];
    offsets[size_t(i)] = total;
    total += c[size_t(i)];
  }
  dense.resize(size_t(total));
  if (total == 0) return;
  int *dC = upload(c.data(), c.size()), *dO = upload(offsets.data(), offsets.size());
  OutBeam* dD = nullptr;
  BT_CUDA_CHECK(cudaMalloc(&dD, size_t(total) * sizeof(OutBeam)));
  compactKernel<<<(n + 127) / 128, 128>>>(dIn, cap, dC, dO, n, dD);
  BT_CUDA_CHECK(cudaGetLastError());
  BT_CUDA_CHECK(cudaMemcpy(dense.data(), dD, dense.size() * sizeof(OutBeam), cudaMemcpyDeviceToHost));
  cudaFree(dC);
  cudaFree(dO);
  cudaFree(dD);
}

}  // namespace

bool gpuAvailable(GpuInfo* info) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  if (info) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return false;
    info->name = p.name;
    info->smCount = p.multiProcessorCount;
    info->memGB = double(p.totalGlobalMem) / (1 << 30);
  }
  return true;
}

GpuRenderer::GpuRenderer(const Scene& scene, const KdTree& tree) : impl_(new Impl), scene_(scene), tree_(tree) {
  Impl& I = *impl_;
  I.dPos = upload(scene.pos.data(), scene.pos.size());
  I.dTri = upload(reinterpret_cast<const uint32_t*>(scene.tris.data()), scene.tris.size() * 3);
  I.dTriN = upload(scene.triN.data(), scene.triN.size());
  I.dNodes = upload(tree.nodes.data(), tree.nodes.size());
  I.dIdx = upload(tree.triIndices.data(), tree.triIndices.size());
  I.dParent = upload(tree.parent.data(), tree.parent.size());
  I.dNodeBox = upload(tree.nodeBox.data(), tree.nodeBox.size());
  {
    std::vector<TriRef> refs;
    buildTriRefs(scene, tree, refs);
    I.dRefs = upload(refs.data(), refs.size());
  }
  I.sv.pos = I.dPos;
  I.sv.tri3 = I.dTri;
  I.sv.triN = I.dTriN;
  I.sv.nodes = I.dNodes;
  I.sv.triIndices = I.dIdx;
  I.sv.parent = I.dParent;
  I.sv.nodeBox = I.dNodeBox;
  I.sv.refs = I.dRefs;
  I.sv.bounds = tree.bounds;
  I.sv.numTris = scene.numTris();
  // Local memory for the per-thread beam storage.
  BT_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
  if (const char* e = getenv("BT_GPU_ENGINE")) useWavefront = std::strcmp(e, "v1") != 0;
  // Queues: at most 40% of the free memory. (Capacity / budget = roots in
  // flight; 64K pixels was the measured sweet spot on a 34-SM GPU.)
  size_t freeB = 0, totalB = 0;
  if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess)
    wfCapacity = std::max(size_t(1) << 16, std::min(wfCapacity, size_t(double(freeB) * 0.4) / (3 * sizeof(WfBeam))));
  // Tuning overrides for the wavefront engine.
  if (const char* e = getenv("BT_WF_BUDGET")) wfBudget = std::max(1, atoi(e));
  if (const char* e = getenv("BT_WF_CAPACITY")) wfCapacity = size_t(std::max(1024L, atol(e)));
  if (useWavefront) {
    I.wf = wfCreateBuffers();
    wfReserve(I.wf, wfCapacity);
  }
}

GpuRenderer::~GpuRenderer() {
  cudaFree(impl_->dPos);
  cudaFree(impl_->dTri);
  cudaFree(impl_->dTriN);
  cudaFree(impl_->dNodes);
  cudaFree(impl_->dIdx);
  cudaFree(impl_->dParent);
  cudaFree(impl_->dNodeBox);
  cudaFree(impl_->dRefs);
  wfDestroyBuffers(impl_->wf);
  delete impl_;
}

void GpuRenderer::softShadows(const Camera& cam, const AreaLight& light, const SoftOptions& opt, SoftResult& res) {
  Impl& I = *impl_;
  const int W = cam.W, H = cam.H;
  const size_t npix = size_t(W) * size_t(H);
  res.W = W;
  res.H = H;
  res.stats = RenderStats();
  int* dTriPix;
  Vec3 *dX, *dN;
  float *dVis, *dE, *dEu;
  unsigned char* dOver;
  unsigned long long* dStats;
  BT_CUDA_CHECK(cudaMalloc(&dTriPix, npix * sizeof(int)));
  BT_CUDA_CHECK(cudaMalloc(&dX, npix * sizeof(Vec3)));
  BT_CUDA_CHECK(cudaMalloc(&dN, npix * sizeof(Vec3)));
  BT_CUDA_CHECK(cudaMalloc(&dVis, npix * sizeof(float)));
  BT_CUDA_CHECK(cudaMalloc(&dE, npix * sizeof(float)));
  BT_CUDA_CHECK(cudaMalloc(&dEu, npix * sizeof(float)));
  BT_CUDA_CHECK(cudaMalloc(&dOver, npix));
  BT_CUDA_CHECK(cudaMalloc(&dStats, 2 * kNumStats * sizeof(unsigned long long)));
  BT_CUDA_CHECK(cudaMemset(dStats, 0, 2 * kNumStats * sizeof(unsigned long long)));

  GpuCam gc;
  gc.eye = cam.eye;
  gc.fwd = cam.fwd;
  gc.right = cam.right;
  gc.down = cam.down;
  gc.halfW = cam.halfW;
  gc.halfH = cam.halfH;
  gc.W = W;
  gc.H = H;
  const Real offset = length(scene_.bounds.diag()) * kEpsRel;

  cudaEvent_t e0, e1, e2;
  BT_CUDA_CHECK(cudaEventCreate(&e0));
  BT_CUDA_CHECK(cudaEventCreate(&e1));
  BT_CUDA_CHECK(cudaEventCreate(&e2));
  BT_CUDA_CHECK(cudaEventRecord(e0));
  dim3 pb(16, 8), pg((W + 15) / 16, (H + 7) / 8);
  primaryKernel<<<pg, pb>>>(I.sv, gc, dTriPix, dX, dN, dStats);
  BT_CUDA_CHECK(cudaGetLastError());
  BT_CUDA_CHECK(cudaEventRecord(e1));
  int* dCounter = nullptr;
  WfRunInfo wi;
  if (useWavefront) {
    WfConfig wc;
    wc.budget = wfBudget;
    wc.capacity = wfCapacity;
    wfSoftShadows(I, wc, light, offset, W, H, dTriPix, dX, dN, opt.exact, dVis, dE, dEu, dOver, wi);
    lastRounds = wi.rounds;
    lastMaxQueue = wi.maxQueue;
    if (getenv("BT_GPU_DEBUG"))
      fprintf(stderr, "  wavefront: %d rounds, max queue %d, %d roots abandoned, %llu trace / %llu split entries, %llu splits\n",
              wi.rounds, wi.maxQueue, wi.killedRoots, wi.traceItems, wi.splitItems,
              (unsigned long long)wi.stats.splits);
  } else {
    const int bs = 32;
    int dev = 0, sms = 1, perSM = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    // The per-thread beam storage lives in local memory: give L1 the largest
    // share of the unified L1/shared memory.
    cudaFuncSetAttribute(softKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, softKernel, bs, 0);
    int blocks = std::max(1, sms * std::max(1, perSM));
    if (const char* e = getenv("BT_GPU_BLOCKS")) blocks = std::max(1, atoi(e));
    if (getenv("BT_GPU_DEBUG")) fprintf(stderr, "  launch: %d blocks x %d threads (%d per SM)\n", blocks, bs, perSM);
    BT_CUDA_CHECK(cudaMalloc(&dCounter, sizeof(int)));
    BT_CUDA_CHECK(cudaMemset(dCounter, 0, sizeof(int)));
    softKernel<<<blocks, bs>>>(I.sv, light, offset, W, H, dTriPix, dX, dN, opt.exact, opt.mailbox && useMailbox, dVis,
                               dE, dEu, dOver, dStats + kNumStats, dCounter);
    BT_CUDA_CHECK(cudaGetLastError());
  }
  BT_CUDA_CHECK(cudaEventRecord(e2));
  BT_CUDA_CHECK(cudaEventSynchronize(e2));
  float msPrimary = 0, msSoft = 0;
  cudaEventElapsedTime(&msPrimary, e0, e1);
  cudaEventElapsedTime(&msSoft, e1, e2);
  lastKernelSeconds = msSoft * 1e-3;

  res.tri.resize(npix);
  res.vis.resize(npix);
  res.E.resize(npix);
  res.Eunocc.resize(npix);
  std::vector<unsigned char> over(npix);
  std::vector<Vec3> X(npix), N(npix);
  unsigned long long st[2 * kNumStats];
  BT_CUDA_CHECK(cudaMemcpy(res.tri.data(), dTriPix, npix * sizeof(int), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(res.vis.data(), dVis, npix * sizeof(float), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(res.E.data(), dE, npix * sizeof(float), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(res.Eunocc.data(), dEu, npix * sizeof(float), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(over.data(), dOver, npix, cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(X.data(), dX, npix * sizeof(Vec3), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(N.data(), dN, npix * sizeof(Vec3), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(st, dStats, sizeof(st), cudaMemcpyDeviceToHost));
  cudaFree(dTriPix);
  cudaFree(dX);
  cudaFree(dN);
  cudaFree(dVis);
  cudaFree(dE);
  cudaFree(dEu);
  cudaFree(dOver);
  cudaFree(dStats);
  cudaFree(dCounter);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);

  // CPU fallback for pixels whose per-thread storage overflowed.
  Timer tf;
  std::vector<size_t> todo;
  for (size_t k = 0; k < npix; ++k)
    if (over[k]) todo.push_back(k);
  lastOverflowPixels = int(todo.size());
  if (getenv("BT_GPU_DEBUG") && !useWavefront) {
    double tot = double(st[kNumStats + kCycTotal]) + 1e-30;
    if (tot > 1) fprintf(stderr, "  cycles: descent %.1f%% leaf %.1f%% advance %.1f%% root %.1f%% (of trace)\n",
            100 * st[kNumStats + kCycDesc] / tot, 100 * st[kNumStats + kCycLeaf] / tot,
            100 * st[kNumStats + kCycAdv] / tot, 100 * st[kNumStats + kCycRoot] / tot);
    fprintf(stderr, "  high water: beams %llu work %llu frames %llu lists %llu polys %llu\n", st[kNumStats + kHiBeams],
            st[kNumStats + kHiWork], st[kNumStats + kHiFrames], st[kNumStats + kHiList], st[kNumStats + kHiPolys]);
    int bits[6] = {0, 0, 0, 0, 0, 0};
    for (size_t k : todo)
      for (int b = 0; b < 6; ++b) bits[b] += (over[k] >> b) & 1;
    fprintf(stderr, "  overflow reasons: beams %d work %d frames %d lists %d polys %d other %d\n", bits[0], bits[1],
            bits[2], bits[3], bits[4], bits[5]);
  }
  TraceStats fstats;
  if (!todo.empty()) softShadowPixelsCPU(scene_, tree_, light, opt, todo, X, N, res, fstats);
  lastFallbackSeconds = tf.seconds();

  TraceStats ts;
  if (useWavefront) {
    ts = wi.stats;
  } else {
    ts.kdSteps = st[kNumStats + kKd];
    ts.leafVisits = st[kNumStats + kLeaf];
    ts.triTests = st[kNumStats + kTests];
    ts.hits = st[kNumStats + kHits];
    ts.splits = st[kNumStats + kSplits];
    ts.beams = st[kNumStats + kBeamsOut];
    ts.rootBeams = st[kNumStats + kRoots];
    ts.presplitBeams = st[kNumStats + kPresplit];
    ts.mailboxSkips = st[kNumStats + kMailSkips];
    ts.fiveSplits = st[kNumStats + kFive];
  }
  ts.add(fstats);
  res.stats.trace = ts;
  res.stats.primarySeconds = msPrimary * 1e-3;
  res.stats.traceSeconds = msSoft * 1e-3 + lastFallbackSeconds;
}

namespace {

GpuCam toGpuCam(const Camera& cam) {
  GpuCam gc;
  gc.eye = cam.eye;
  gc.fwd = cam.fwd;
  gc.right = cam.right;
  gc.down = cam.down;
  gc.halfW = cam.halfW;
  gc.halfH = cam.halfH;
  gc.W = cam.W;
  gc.H = cam.H;
  return gc;
}

TraceStats readStats(const unsigned long long* st) {
  TraceStats ts;
  ts.kdSteps = st[kKd];
  ts.leafVisits = st[kLeaf];
  ts.triTests = st[kTests];
  ts.hits = st[kHits];
  ts.splits = st[kSplits];
  ts.beams = st[kBeamsOut];
  ts.rootBeams = st[kRoots];
  ts.presplitBeams = st[kPresplit];
  ts.mailboxSkips = st[kMailSkips];
  ts.fiveSplits = st[kFive];
  ts.rays = st[kRays];
  return ts;
}

}  // namespace

namespace {

Poly2 tilePoly(const Camera& cam, int tile, int tilesX, int i) {
  int tx = i % tilesX, ty = i / tilesX;
  int x0 = tx * tile, y0 = ty * tile, x1 = std::min(cam.W, x0 + tile), y1 = std::min(cam.H, y0 + tile);
  Real a, b, c, d;
  cam.pixelToQ(Real(x0), Real(y0), a, b);
  cam.pixelToQ(Real(x1), Real(y1), c, d);
  Poly2 p;
  p.push(a, b);
  p.push(c, b);
  p.push(c, d);
  p.push(a, d);
  return p;
}

}  // namespace

void GpuRenderer::primary(const Camera& cam, bool cull, std::vector<OutBeam>& hitBeams, RenderStats& rs, int tile) {
  Impl& I = *impl_;
  Timer total;
  if (useWavefront) {
    WfConfig wc;
    wc.budget = wfPrimaryBudget;
    wc.capacity = wfCapacity;
    WfRunInfo wi;
    std::vector<int> killed;
    wfPrimary(I, wc, cam, cull, tile, hitBeams, killed, wi);
    lastKernelSeconds = wi.ms * 1e-3;
    lastRounds = wi.rounds;
    lastMaxQueue = wi.maxQueue;
    const int tilesX = (cam.W + tile - 1) / tile;
    std::vector<Poly2> fallback;
    for (int k : killed) fallback.push_back(tilePoly(cam, tile, tilesX, k));
    Timer tf;
    TraceStats fstats;
    lastOverflowPixels = int(fallback.size());
    if (!fallback.empty()) primaryRootsCPU(scene_, tree_, cam, cull, fallback, hitBeams, fstats);
    lastFallbackSeconds = tf.seconds();
    rs.trace = wi.stats;
    rs.trace.add(fstats);
    rs.trace.hits = hitBeams.size();
    rs.traceSeconds += total.seconds();
    return;
  }
  const int tilesX = (cam.W + tile - 1) / tile, tilesY = (cam.H + tile - 1) / tile, numTiles = tilesX * tilesY;
  const int cap = 1024;
  OutBeam* dOut;
  int* dCount;
  unsigned char* dOver;
  unsigned long long* dStats;
  BT_CUDA_CHECK(cudaMalloc(&dOut, size_t(numTiles) * cap * sizeof(OutBeam)));
  BT_CUDA_CHECK(cudaMalloc(&dCount, size_t(numTiles) * sizeof(int)));
  BT_CUDA_CHECK(cudaMalloc(&dOver, size_t(numTiles)));
  BT_CUDA_CHECK(cudaMalloc(&dStats, kNumStats * sizeof(unsigned long long)));
  BT_CUDA_CHECK(cudaMemset(dStats, 0, kNumStats * sizeof(unsigned long long)));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const int bs = 32;
  primaryTileKernel<<<(numTiles + bs - 1) / bs, bs>>>(I.sv, cam.plane(), toGpuCam(cam), tile, tilesX, numTiles, cull,
                                                      dOut, cap, dCount, dOver, dStats);
  BT_CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1);
  BT_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  lastKernelSeconds = ms * 1e-3;
  std::vector<int> count(static_cast<size_t>(numTiles));
  std::vector<unsigned char> over(static_cast<size_t>(numTiles));
  unsigned long long st[kNumStats];
  BT_CUDA_CHECK(cudaMemcpy(count.data(), dCount, count.size() * sizeof(int), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(over.data(), dOver, over.size(), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(st, dStats, sizeof(st), cudaMemcpyDeviceToHost));
  hitBeams.clear();
  std::vector<int> offsets;
  downloadCompacted(dOut, cap, count, over, offsets, hitBeams);
  std::vector<Poly2> fallback;
  for (int i = 0; i < numTiles; ++i)
    if (over[size_t(i)]) fallback.push_back(tilePoly(cam, tile, tilesX, i));
  cudaFree(dOut);
  cudaFree(dCount);
  cudaFree(dOver);
  cudaFree(dStats);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  Timer tf;
  TraceStats fstats;
  lastOverflowPixels = int(fallback.size());
  if (!fallback.empty()) primaryRootsCPU(scene_, tree_, cam, cull, fallback, hitBeams, fstats);
  lastFallbackSeconds = tf.seconds();
  rs.trace = readStats(st);
  rs.trace.add(fstats);
  rs.trace.hits = hitBeams.size();
  rs.traceSeconds += total.seconds();
}

void GpuRenderer::pointShadows(const Camera& cam, const std::vector<OutBeam>& primHits, const Vec3& light,
                               std::vector<OutBeam>& shadowPolys, RenderStats& rs) {
  Impl& I = *impl_;
  Timer total;
  const int n = int(primHits.size());
  if (n == 0) return;
  if (useWavefront) {
    OutBeam* dPrim = upload(primHits.data(), primHits.size());
    WfConfig wc;
    wc.budget = wfBudget;
    wc.capacity = wfCapacity;
    WfRunInfo wi;
    std::vector<int> killed;
    std::vector<OutBeam> polys;
    wfPointShadows(I, wc, cam, light, dPrim, n, polys, killed, wi);
    cudaFree(dPrim);
    lastKernelSeconds = wi.ms * 1e-3;
    lastRounds = wi.rounds;
    lastMaxQueue = wi.maxQueue;
    shadowPolys.insert(shadowPolys.end(), polys.begin(), polys.end());
    std::vector<OutBeam> fallback;
    for (int k : killed) fallback.push_back(primHits[size_t(k)]);
    Timer tf;
    TraceStats fstats;
    lastOverflowPixels = int(fallback.size());
    if (!fallback.empty()) pointShadowsCPU(scene_, tree_, cam, fallback, light, shadowPolys, fstats);
    lastFallbackSeconds = tf.seconds();
    rs.trace.add(wi.stats);
    rs.trace.add(fstats);
    rs.traceSeconds += total.seconds();
    return;
  }
  const int cap = 64;
  OutBeam *dPrim, *dOut;
  int* dCount;
  unsigned char* dOver;
  unsigned long long* dStats;
  dPrim = upload(primHits.data(), primHits.size());
  BT_CUDA_CHECK(cudaMalloc(&dOut, size_t(n) * cap * sizeof(OutBeam)));
  BT_CUDA_CHECK(cudaMalloc(&dCount, size_t(n) * sizeof(int)));
  BT_CUDA_CHECK(cudaMalloc(&dOver, size_t(n)));
  BT_CUDA_CHECK(cudaMalloc(&dStats, kNumStats * sizeof(unsigned long long)));
  BT_CUDA_CHECK(cudaMemset(dStats, 0, kNumStats * sizeof(unsigned long long)));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  const int bs = 32;
  pointShadowKernel<<<(n + bs - 1) / bs, bs>>>(I.sv, cam.plane(), toGpuCam(cam), light, dPrim, n, dOut, cap, dCount,
                                               dOver, dStats);
  BT_CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1);
  BT_CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  lastKernelSeconds = ms * 1e-3;
  std::vector<int> count(static_cast<size_t>(n));
  std::vector<unsigned char> over(static_cast<size_t>(n));
  unsigned long long st[kNumStats];
  BT_CUDA_CHECK(cudaMemcpy(count.data(), dCount, count.size() * sizeof(int), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(over.data(), dOver, over.size(), cudaMemcpyDeviceToHost));
  BT_CUDA_CHECK(cudaMemcpy(st, dStats, sizeof(st), cudaMemcpyDeviceToHost));
  std::vector<int> offsets;
  std::vector<OutBeam> out;
  downloadCompacted(dOut, cap, count, over, offsets, out);
  cudaFree(dPrim);
  cudaFree(dOut);
  cudaFree(dCount);
  cudaFree(dOver);
  cudaFree(dStats);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  std::vector<OutBeam> fallback;
  for (int j = 0; j < n; ++j) {
    if (over[size_t(j)]) {
      fallback.push_back(primHits[size_t(j)]);
    } else if (count[size_t(j)] < 0) {
      shadowPolys.push_back(primHits[size_t(j)]);
    } else {
      for (int k = 0; k < count[size_t(j)]; ++k) shadowPolys.push_back(out[size_t(offsets[size_t(j)] + k)]);
    }
  }
  Timer tf;
  TraceStats fstats;
  lastOverflowPixels = int(fallback.size());
  if (!fallback.empty()) pointShadowsCPU(scene_, tree_, cam, fallback, light, shadowPolys, fstats);
  lastFallbackSeconds = tf.seconds();
  rs.trace.add(readStats(st));
  rs.trace.add(fstats);
  rs.traceSeconds += total.seconds();
}

}  // namespace bt
