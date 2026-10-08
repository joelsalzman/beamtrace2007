// Wavefront beam tracing: the per-sub-beam steps, written once for the host
// reference engine (beam/wavefront.h) and the CUDA engine (cuda/wavefront.cu).
//
// Instead of one depth-first traversal per beam tree (BeamCore), every
// sub-beam is a self-contained record (WfBeam): its corners, hit status, the
// root (query) it belongs to and a restart-trail continuation (kd node, depth,
// one bit per pending far child, index of the next triangle in the leaf). Any
// thread can therefore take any sub-beam from a queue. Two steps act on a
// record:
//
//   wfTrace  descend to the next leaf, run the cheap triangle tests from the
//            cursor, climb to the next pending far child, ... until the beam
//            is finished (output) or a triangle needs a real clip;
//   wfSplit  clip by that triangle; the pieces continue at the next triangle
//            of the same leaf.
//
// A sub-beam meets the same triangles in the same order as in BeamCore with
// the restart trail and without mailboxing, so the pieces are bitwise the
// same; only the order in which different sub-beams are processed differs.
#pragma once

#include "beam/beam_core.h"

namespace bt {
inline namespace BT_R4_NS {

// Instrumentation (CUDA builds with -DBT_WF_SIMT): counts warp steps and
// active lanes at the marked loop heads (cuda/wavefront.cu prints them).
#if defined(__CUDACC__) && defined(BT_WF_SIMT)
__device__ unsigned long long g_simt[16];
#endif
#if defined(__CUDA_ARCH__) && defined(BT_WF_SIMT)
__device__ __forceinline__ void simtCount(int k) {
  unsigned m = __activemask();
  if ((threadIdx.x & 31) == __ffs(m) - 1) {
    atomicAdd(&g_simt[2 * k], 1ull);
    atomicAdd(&g_simt[2 * k + 1], (unsigned long long)__popc(m));
  }
}
#define BT_SIMT(k) simtCount(k)
#else
#define BT_SIMT(k)
#endif

struct WfBeam {
  Real qx[4], qy[4];  // corners on the root's plane (n = 3 or 4)
  uint64_t trail;     // pending far children: bit d for depth d
  uint32_t node;      // a leaf (cursor >= 0) or the node to descend from (cursor < 0)
  int32_t root;       // the root (query) this sub-beam belongs to
  int32_t hit;        // Nearest: current hit triangle (-1: none)
  int32_t cursor;     // next triangle index within the leaf; -1: descend from `node`
  int16_t depth;      // depth of `node`
  int8_t n;           // number of corners
  int8_t negBits;     // bit a set: rays move toward -a along axis a
};

// Working form of a sub-beam: the 4-wide beam plus its hit triangle's plane.
struct WfWork {
  CoreBeam b;
  Real hitA[3], hitC;
};

// Plane of the current hit, cached in WfWork (BeamCtx::classifyTri).
struct WfHitPlane {
  const WfWork* w;
  BT_HD void operator()(int, Real* A, Real& C) const {
    A[0] = w->hitA[0];
    A[1] = w->hitA[1];
    A[2] = w->hitA[2];
    C = w->hitC;
  }
};

// Triangle id of leaf reference k (packed references if the scene has them).
BT_HD inline int wfTriAt(const SceneView& sv, uint32_t k) {
  return sv.refs ? int(sv.refs[k].tri) : int(sv.triIndices[k]);
}

// Setup of leaf reference k; returns the triangle id.
BT_HD inline int wfTriInfo(const BeamCtx& c, uint32_t k, CoreTriInfo& ti) {
  if (c.sv.refs) {
    const TriRef ref = c.sv.refs[k];
    c.computeTriInfo(ref, ti);
    return ref.tri;
  }
  const int t = int(c.sv.triIndices[k]);
  c.computeTriInfo(t, ti);
  return t;
}

// Expands a record (same arithmetic as BeamCtx::beamGeom / setStatus).
BT_HD inline void wfExpand(const BeamCtx& c, const WfBeam& r, WfWork& w) {
  CoreBeam& b = w.b;
  const int n = r.n, l3 = n > 3 ? 3 : 2;
  b.qx = R4(r.qx[0], r.qx[1], r.qx[2], r.qx[l3]);
  b.qy = R4(r.qy[0], r.qy[1], r.qy[2], r.qy[l3]);
  b.n = n;
  const BeamPlane& P = c.q.plane;
  for (int a = 0; a < 3; ++a) {
    b.sgn[a] = ((r.negBits >> a) & 1) ? int8_t(-1) : int8_t(1);
    R4 wv = b.qx * P.u[a] + b.qy * P.v[a] + P.d[a];
    b.aw[a] = rmax(R4(Real(0)), wv * Real(b.sgn[a]));
  }
  b.mbox = 0;
  if (r.hit >= 0) {
    c.hitPlane(r.hit, w.hitA, w.hitC);
    c.setStatus(b, r.hit, w.hitA, w.hitC);
  } else {
    w.hitA[0] = w.hitA[1] = w.hitA[2] = w.hitC = 0;
    c.setStatus(b, -1, nullptr, 0);
  }
}

BT_HD inline void wfStoreGeom(const CoreBeam& b, WfBeam& r) {
  r.n = int8_t(b.n);
  for (int i = 0; i < 4; ++i) {
    r.qx[i] = b.qx[i];
    r.qy[i] = b.qy[i];
  }
  r.negBits = int8_t((b.sgn[0] < 0 ? 1 : 0) | (b.sgn[1] < 0 ? 2 : 0) | (b.sgn[2] < 0 ? 4 : 0));
}

// A finished piece (BeamCore::output): out.emit(root, polygon, tri, area).
template <class Out>
BT_HD void wfOutput(const BeamCtx& c, const CoreBeam& b, int root, int tri, Out& out, TraceStats& st) {
  PolyN<4> p;
  c.beamToPoly(b, p);
  Real area = p.area();
  if (tri >= 0) st.hits++;
  st.beams++;
  out.emit(root, p, tri, area);
}

enum WfTraceResult : int { kWfFinished = 0, kWfSplit = 1, kWfContinue = 2 };
enum WfPhase : int { kWfDescend = 0, kWfTest = 1, kWfClimb = 2 };

// Where a sub-beam is within wfStep's state machine.
struct WfStepState {
  int phase;
  AABB box;  // box of r.node while descending
};

// Starts the state machine for a record (wfExpand'ed into w).
BT_HD inline void wfBegin(const BeamCtx& c, const WfBeam& r, WfStepState& ss) {
  if (r.cursor < 0) {
    ss.phase = kWfDescend;
    ss.box = c.sv.nodeBox[r.node];
  } else {
    ss.phase = kWfTest;
  }
}

// Cheap steps (kd descent, skipping the excluded triangle, climbing to a
// pending far child) until the sub-beam has a triangle to test: returns
// kWfContinue with ss.phase == kWfTest and triangle `cursor` pending, or
// kWfFinished (no cell left: the beam's output went to `out`).
template <class Out>
BT_HD int wfAdvance(const BeamCtx& c, WfBeam& r, WfWork& w, WfStepState& ss, Out& out, TraceStats& st) {
  const SceneView& sv = c.sv;
  const CoreBeam& B = w.b;
  for (;;) {
    if (ss.phase == kWfDescend) {
      for (;;) {
        const KdNode& kn = sv.nodes[r.node];
        if (kn.isLeaf()) break;
        BT_SIMT(1);
        st.kdSteps++;
        const int d = r.depth++;
        const int a = kn.axis();
        const Real s = kn.split;
        const Real Ns = Real(B.sgn[a]) * (s - c.q.plane.O[a]);
        const uint32_t left = r.node + 1, right = kn.right();
        AABB lbox = ss.box, rbox = ss.box;
        lbox.hi[a] = s;
        rbox.lo[a] = s;
        const bool sigmaRight = B.sgn[a] > 0;
        // cs: the child the rays move into; co: the child containing O.
        const uint32_t cs = sigmaRight ? right : left, co = sigmaRight ? left : right;
        int dec = BeamCtx::kFar;
        if (Ns > 0) dec = c.decide(B, a, Ns, ss.box);  // else the split plane is behind (or through) the apex
        if (dec == BeamCtx::kFar) {
          r.node = cs;
          ss.box = sigmaRight ? rbox : lbox;
        } else {
          if (dec == BeamCtx::kBoth) r.trail |= uint64_t(1) << d;
          r.node = co;
          ss.box = sigmaRight ? lbox : rbox;
        }
      }
      r.cursor = 0;
      st.leafVisits++;
      ss.phase = kWfTest;
    }
    if (ss.phase == kWfTest) {
      const KdNode& leaf = sv.nodes[r.node];
      const uint32_t cnt = leaf.count();
      uint32_t i = uint32_t(r.cursor);
      while (i < cnt && wfTriAt(sv, leaf.offset + i) == c.q.excludeTri) ++i;
      r.cursor = int32_t(i);
      if (i < cnt) return kWfContinue;
      ss.phase = kWfClimb;
    }
    // climb: resume at the deepest pending far child the beam overlaps
    for (;;) {
      if (r.trail == 0) {
        wfOutput(c, B, r.root, c.q.mode == BeamMode::Nearest ? B.hit : -1, out, st);
        return kWfFinished;
      }
      BT_SIMT(3);
      const int d = highestBit(r.trail);
      uint32_t node = r.node;
      for (int depth = r.depth; depth > d; --depth) {
        node = sv.parent[node];
        st.climbs++;
      }
      r.trail &= (uint64_t(1) << d) - 1;
      const KdNode& kn = sv.nodes[node];
      const uint32_t cs = B.sgn[kn.axis()] > 0 ? kn.right() : node + 1;
      r.node = node;
      r.depth = int16_t(d);
      if (c.overlaps(B, sv.nodeBox[cs])) {
        r.node = cs;
        r.depth = int16_t(d + 1);
        r.cursor = -1;
        ss.phase = kWfDescend;
        ss.box = sv.nodeBox[cs];
        break;
      }
    }
  }
}

// Tests the pending triangle (after wfAdvance returned kWfContinue). Returns
// kWfContinue, kWfFinished (AnyHit: occluded, output done) or kWfSplit
// (triangle `cursor` needs a clip).
template <class Out>
BT_HD int wfTestOne(const BeamCtx& c, WfBeam& r, WfWork& w, Out& out, TraceStats& st) {
  BT_SIMT(2);
  CoreBeam& B = w.b;
  const KdNode& leaf = c.sv.nodes[r.node];
  const uint32_t i = uint32_t(r.cursor);
  r.cursor = int32_t(i + 1);
  CoreTriInfo ti;
  const int t = wfTriInfo(c, leaf.offset + i, ti);
  if (!ti.ok) return kWfContinue;
  st.triTests++;
  if (B.hit == t) return kWfContinue;
  Line2 L[5];
  int nl, reason;
  WfHitPlane hp{&w};
  const int res = c.classifyTri(B, ti, hp, L, nl, reason);
  if (res == kTriKeep) return kWfContinue;
  if (res == kTriAllIn) {
    if (c.q.mode == BeamMode::AnyHit) {
      wfOutput(c, B, r.root, t, out, st);  // occluded
      return kWfFinished;
    }
    c.setStatus(B, t, ti.A, ti.C);
    w.hitA[0] = ti.A[0];
    w.hitA[1] = ti.A[1];
    w.hitA[2] = ti.A[2];
    w.hitC = ti.C;
    r.hit = t;
    return kWfContinue;
  }
  r.cursor = int32_t(i);
  return kWfSplit;
}

// True if the current leaf has another triangle to test (skips the excluded
// one); false when the leaf is done.
BT_HD inline bool wfMoreInLeaf(const BeamCtx& c, WfBeam& r) {
  const KdNode& leaf = c.sv.nodes[r.node];
  const uint32_t cnt = leaf.count();
  uint32_t i = uint32_t(r.cursor);
  while (i < cnt && wfTriAt(c.sv, leaf.offset + i) == c.q.excludeTri) ++i;
  r.cursor = int32_t(i);
  return i < cnt;
}

// Advances a sub-beam until it is finished (returns kWfFinished; the output
// went to `out`) or triangle `cursor` of leaf `node` needs a clip (returns
// kWfSplit; the record is updated in place). `w` must hold wfExpand(r).
// This visits exactly the cells and triangles of BeamCore with the restart
// trail.
template <class Out>
BT_HD int wfTrace(const BeamCtx& c, WfBeam& r, WfWork& w, Out& out, TraceStats& st) {
  WfStepState ss;
  wfBegin(c, r, ss);
  for (;;) {
    if (wfAdvance(c, r, w, ss, out, st) == kWfFinished) return kWfFinished;
    const int res = wfTestOne(c, r, w, out, st);
    if (res != kWfContinue) return res;
  }
}

// Clip-piece lists of a split (fixed capacity; overflow abandons the root).
// Clipping a beam (<= 4 corners) by one line gives pieces of <= 5 vertices,
// and pentagons are split before the next line, so stored pieces have <= 5.
// (Measured maxima: 10 pieces per list for primary beams; shadow beams stay
// below 8.)
constexpr int kWfMaxClip = 12;

struct WfPoly5 {
  Real x[5], y[5];
  int n;
};

struct WfClipSink {
  WfPoly5 p[2][kWfMaxClip];
  int n[2] = {0, 0};
  template <int N>
  BT_HD bool push(int l, const PolyN<N>& q) {
    if (n[l] >= kWfMaxClip || q.n > 5) return false;
    WfPoly5& d = p[l][n[l]++];
    d.n = q.n;
    for (int i = 0; i < q.n; ++i) {
      d.x[i] = q.x[i];
      d.y[i] = q.y[i];
    }
    return true;
  }
  template <int N>
  BT_HD void get(int l, int i, PolyN<N>& q) const {
    const WfPoly5& d = p[l][i];
    q.n = d.n;
    for (int k = 0; k < d.n; ++k) {
      q.x[k] = d.x[k];
      q.y[k] = d.y[k];
    }
  }
};

// Split, phase 1: clips the sub-beam by triangle `cursor` of its leaf (after
// wfTrace returned kWfSplit) into `sink`. `commit` is false if no piece
// changes status (nothing measurable is hit): the beam then continues
// unchanged. `slots` receives the number of records phase 2 writes. Returns
// false if a fixed list overflowed.
BT_HD inline bool wfSplitClip(const BeamCtx& c, const WfBeam& r, WfWork& w, bool orderEdges, WfClipSink& sink,
                              TraceStats& st, bool& commit, int& slots) {
  const KdNode& leaf = c.sv.nodes[r.node];
  CoreTriInfo ti;
  wfTriInfo(c, leaf.offset + uint32_t(r.cursor), ti);
  Line2 L[5];
  int nl, reason;
  WfHitPlane hp{&w};
  c.classifyTri(w.b, ti, hp, L, nl, reason);  // kTriClip, as in wfTrace
  sink.n[0] = sink.n[1] = 0;
  if (!c.clipTri<6>(w.b, L, nl, orderEdges, sink, st.fiveSplits)) return false;
  Real newArea = 0;
  PolyN<5> q;
  for (int i = 0; i < sink.n[1]; ++i) {
    sink.get(1, i, q);
    newArea += q.area();
  }
  commit = newArea >= c.minArea;
  slots = 1;
  if (commit) {
    slots = 0;
    const int lists = c.q.mode == BeamMode::AnyHit ? 1 : 2;  // AnyHit: hit pieces are output, not queued
    for (int l = 0; l < lists; ++l)
      for (int i = 0; i < sink.n[l]; ++i) slots += sink.p[l][i].n == 5 ? 2 : 1;
  }
  return true;
}

// Split, phase 2: writes the `slots` continuing records (at the next triangle
// of the leaf) to dst[0..slots) - a record with n = 0 for each piece dropped
// as a sliver - and sends finished pieces (AnyHit: occluded) to `out`.
template <class Out>
BT_HD void wfSplitEmit(const BeamCtx& c, const WfBeam& r, const WfWork& w, const WfClipSink& sink, bool commit,
                       Out& out, TraceStats& st, WfBeam* dst) {
  WfBeam base = r;
  base.cursor = r.cursor + 1;
  if (!commit) {
    dst[0] = base;
    return;
  }
  st.splits++;
  const KdNode& leaf = c.sv.nodes[r.node];
  const int t = wfTriAt(c.sv, leaf.offset + uint32_t(r.cursor));
  const CoreBeam& parent = w.b;
  int k = 0;
  for (int l = 0; l < 2; ++l) {
    const bool hitNow = l == 1;
    const bool queued = !(hitNow && c.q.mode == BeamMode::AnyHit);
    const int hit = hitNow ? t : parent.hit;
    for (int i = 0; i < sink.n[l]; ++i) {
      PolyN<5> p0;
      sink.get(l, i, p0);
      PolyN<5> sub[2];
      int ns = 1;
      if (p0.n == 5) {
        splitFive(p0, sub[0], sub[1]);
        st.fiveSplits++;
        ns = 2;
      } else {
        sub[0] = p0;
      }
      for (int j = 0; j < ns; ++j) {
        CoreBeam nb;
        Real dropped = 0;
        const bool ok = c.beamGeom(sub[j], parent.sgn, nb, dropped);
        if (!ok) {
          st.droppedArea += double(dropped);
          out.dropped(r.root, dropped);
        } else if (!queued) {
          wfOutput(c, nb, r.root, t, out, st);  // occluded piece: done
        }
        if (!queued) continue;
        WfBeam& pr = dst[k++];
        pr = base;
        if (ok) {
          wfStoreGeom(nb, pr);
          pr.hit = hit;
        } else {
          pr.n = 0;  // dropped sliver
        }
      }
    }
  }
}

// Pre-splits a root polygon (BeamCore::addRoot) into records queued at the kd
// root; pieces that miss the scene box are output directly. Calls
// push(record) for each queued piece (false aborts). Returns false if aborted
// or the pre-split overflowed.
template <class Out, class Push>
BT_HD bool wfAdmit(const BeamCtx& c, const Poly2& rootPoly, int root, Out& out, Push& push, TraceStats& st) {
  struct Emit {
    const BeamCtx& c;
    int root;
    Out& out;
    Push& push;
    TraceStats& st;
    BT_HD bool piece(const Poly2& p, const int8_t sgn[3]) {
      CoreBeam b;
      Real dropped = 0;
      if (!c.beamGeom(p, sgn, b, dropped)) {
        st.droppedArea += double(dropped);
        out.dropped(root, dropped);
        return true;
      }
      c.setStatus(b, -1, nullptr, 0);
      st.rootBeams++;
      if (!c.overlaps(b, c.sv.bounds)) {
        wfOutput(c, b, root, -1, out, st);
        return true;
      }
      WfBeam r;
      wfStoreGeom(b, r);
      r.trail = 0;
      r.node = 0;
      r.root = root;
      r.hit = -1;
      r.cursor = -1;
      r.depth = 0;
      return push(r);
    }
    BT_HD void presplit() { st.presplitBeams++; }
    BT_HD void dropped(Real a) {
      st.droppedArea += double(a);
      out.dropped(root, a);
    }
  };
  Poly2 p = rootPoly;
  if (p.n < 3) return true;
  if (p.area() < 0) p.reverse();
  Emit e{c, root, out, push, st};
  return c.presplit(p, e);
}

}  // inline namespace BT_R4_NS
}  // namespace bt
