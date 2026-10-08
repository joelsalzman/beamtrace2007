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
  Poly2 p;
  c.beamToPoly(b, p);
  Real area = p.area();
  if (tri >= 0) st.hits++;
  st.beams++;
  out.emit(root, p, tri, area);
}

enum WfTraceResult : int { kWfFinished = 0, kWfSplit = 1 };

// Advances a sub-beam until it is finished (returns kWfFinished; the output
// went to `out`) or triangle `cursor` of leaf `node` needs a clip (returns
// kWfSplit; the record is updated in place). `w` must hold wfExpand(r).
template <class Out>
BT_HD int wfTrace(const BeamCtx& c, WfBeam& r, WfWork& w, Out& out, TraceStats& st) {
  const SceneView& sv = c.sv;
  const Vec3& O = c.q.plane.O;
  CoreBeam& B = w.b;
  for (;;) {
    if (r.cursor < 0) {  // descend from r.node to a leaf
      uint32_t node = r.node;
      AABB box = sv.nodeBox[node];
      int depth = r.depth;
      uint64_t trail = r.trail;
      for (;;) {
        const KdNode& kn = sv.nodes[node];
        if (kn.isLeaf()) break;
        st.kdSteps++;
        const int d = depth++;
        const int a = kn.axis();
        const Real s = kn.split;
        const Real Ns = Real(B.sgn[a]) * (s - O[a]);
        const uint32_t left = node + 1, right = kn.right();
        AABB lbox = box, rbox = box;
        lbox.hi[a] = s;
        rbox.lo[a] = s;
        const bool sigmaRight = B.sgn[a] > 0;
        // cs: the child the rays move into; co: the child containing O.
        const uint32_t cs = sigmaRight ? right : left, co = sigmaRight ? left : right;
        if (Ns <= 0) {  // split plane behind (or through) the apex
          node = cs;
          box = sigmaRight ? rbox : lbox;
          continue;
        }
        const int dec = c.decide(B, a, Ns, box);
        if (dec == BeamCtx::kFar) {
          node = cs;
          box = sigmaRight ? rbox : lbox;
        } else {
          if (dec == BeamCtx::kBoth) trail |= uint64_t(1) << d;
          node = co;
          box = sigmaRight ? lbox : rbox;
        }
      }
      r.node = node;
      r.depth = int16_t(depth);
      r.trail = trail;
      r.cursor = 0;
      st.leafVisits++;
    }
    // the leaf's triangles, from the cursor on
    const KdNode& leaf = sv.nodes[r.node];
    const uint32_t cnt = leaf.count();
    const uint32_t* idx = sv.triIndices + leaf.offset;
    for (uint32_t i = uint32_t(r.cursor); i < cnt; ++i) {
      const int t = int(idx[i]);
      if (t == c.q.excludeTri) continue;
      CoreTriInfo ti;
      c.computeTriInfo(t, ti);
      if (!ti.ok) continue;
      st.triTests++;
      if (B.hit == t) continue;
      Line2 L[5];
      int nl, reason;
      WfHitPlane hp{&w};
      const int res = c.classifyTri(B, ti, hp, L, nl, reason);
      if (res == kTriKeep) continue;
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
        continue;
      }
      r.cursor = int32_t(i);
      return kWfSplit;
    }
    // leaf done: resume at the deepest pending far child the beam overlaps
    bool resumed = false;
    while (r.trail != 0) {
      const int d = highestBit(r.trail);
      uint32_t node = r.node;
      for (int depth = r.depth; depth > d; --depth) {
        node = sv.parent[node];
        st.climbs++;
      }
      r.node = node;
      r.depth = int16_t(d);
      r.trail &= (uint64_t(1) << d) - 1;
      const KdNode& kn = sv.nodes[node];
      const uint32_t cs = B.sgn[kn.axis()] > 0 ? kn.right() : node + 1;
      if (c.overlaps(B, sv.nodeBox[cs])) {
        r.node = cs;
        r.depth = int16_t(d + 1);
        r.cursor = -1;
        resumed = true;
        break;
      }
    }
    if (!resumed) {
      wfOutput(c, B, r.root, c.q.mode == BeamMode::Nearest ? B.hit : -1, out, st);
      return kWfFinished;
    }
  }
}

// Clip-piece lists of wfSplit (fixed capacity; overflow aborts the split).
// (Measured maxima: 10 polygons and 16 pieces, for primary beams; shadow
// beams stay below 8. A split that needs more abandons its root.)
constexpr int kWfMaxClip = 12;    // polygons per list
constexpr int kWfMaxPieces = 24;  // sub-beams out of one split

struct WfClipSink {
  Poly2 p[2][kWfMaxClip];
  int n[2] = {0, 0};
  BT_HD bool push(int l, const Poly2& q) {
    if (n[l] >= kWfMaxClip) return false;
    p[l][n[l]++] = q;
    return true;
  }
};

// Clips the sub-beam by triangle `cursor` of its leaf (wfTrace returned
// kWfSplit). Writes the pieces that continue (at the next triangle) to
// pieces[0..np) and sends finished ones (AnyHit: occluded) to `out`. If no
// piece changes status (nothing measurable is hit) the beam continues
// unchanged as the single piece. Returns false on overflow of the fixed lists.
template <class Out>
BT_HD bool wfSplit(const BeamCtx& c, const WfBeam& r, WfWork& w, bool orderEdges, WfClipSink& sink, Out& out,
                   TraceStats& st, WfBeam* pieces, int& np) {
  np = 0;
  const KdNode& leaf = c.sv.nodes[r.node];
  const int t = int(c.sv.triIndices[leaf.offset + uint32_t(r.cursor)]);
  CoreTriInfo ti;
  c.computeTriInfo(t, ti);
  Line2 L[5];
  int nl, reason;
  WfHitPlane hp{&w};
  c.classifyTri(w.b, ti, hp, L, nl, reason);  // kTriClip, as in wfTrace
  sink.n[0] = sink.n[1] = 0;
  if (!c.clipTri(w.b, L, nl, orderEdges, sink, st.fiveSplits)) return false;
  Real newArea = 0;
  for (int i = 0; i < sink.n[1]; ++i) newArea += sink.p[1][i].area();
  WfBeam base = r;
  base.cursor = r.cursor + 1;
  if (!(newArea >= c.minArea)) {  // nothing (measurable) hits: continue unchanged
    pieces[np++] = base;
    return true;
  }
  st.splits++;
  const CoreBeam& parent = w.b;
  for (int l = 0; l < 2; ++l) {
    const bool hitNow = l == 1;
    const int hit = hitNow ? t : parent.hit;
    for (int i = 0; i < sink.n[l]; ++i) {
      const Poly2& p0 = sink.p[l][i];
      Poly2 sub[2];
      int ns = 1;
      if (p0.n == 5) {
        splitFive(p0, sub[0], sub[1]);
        st.fiveSplits++;
        ns = 2;
      } else {
        sub[0] = p0;
      }
      for (int k = 0; k < ns; ++k) {
        CoreBeam nb;
        Real dropped = 0;
        if (!c.beamGeom(sub[k], parent.sgn, nb, dropped)) {
          st.droppedArea += double(dropped);
          out.dropped(r.root, dropped);
          continue;
        }
        if (hitNow && c.q.mode == BeamMode::AnyHit) {
          wfOutput(c, nb, r.root, t, out, st);  // occluded piece: done
          continue;
        }
        if (np >= kWfMaxPieces) return false;
        WfBeam& pr = pieces[np++];
        pr = base;
        wfStoreGeom(nb, pr);
        pr.hit = hit;
      }
    }
  }
  return true;
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
