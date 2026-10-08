// The beam tracing algorithm (paper Sec. 3), written once for host and device.
//
//   beam-triangle splitting (Sec. 3.2), beam-kd traversal (Sec. 3.3) and
//   hierarchical "Post Office" mailboxing (Sec. 3.4).
//
// BeamCore<S> is parameterized by a storage policy S that owns the beam pool,
// the work and frame stacks, the triangle-setup cache, the mailboxes and the
// output. The CPU tracer (beam_tracer.h) uses growable std::vector storage;
// the CUDA port uses fixed per-thread arrays and reports overflow.
//
// Everything that depends on R4 lives in the backend's inline namespace (see
// simd4.h), so host (SSE/AVX) and device (generic) builds never mix definitions.
#pragma once

#include "beam/beam_types.h"
#include "core/simd4.h"
#include "util/stats.h"

namespace bt {
inline namespace BT_R4_NS {

#ifdef BT_REAL_DOUBLE
constexpr Real kCoplanarCos = Real(1e-9);
constexpr Real kCoplanarRel = Real(1e-8);
#else
constexpr Real kCoplanarCos = Real(2e-6);
constexpr Real kCoplanarRel = Real(2e-5);
#endif

struct alignas(32) CoreBeam {
  R4 qx, qy;     // corner points on the plane (lanes past n repeat the last corner)
  R4 aw[3];      // |w_axis| at the corners, w = dir(q); sign fixed per beam by the pre-split
  R4 farA;       // |a_far(q)|: far-plane distance t_far = farC / farA
  Real farC;
  int n;
  int hit;
  uint64_t mbox;
  int8_t sgn[3];
  bool hasFar;
};

struct CoreTriInfo {
  uint32_t stamp;
  int tri;
  bool ok;
  Line2 edge[3];
  Real A[3];   // a_T(q) = A0 qx + A1 qy + A2 = n_T . dir(q)
  Real C;      // n_T . (v0 - O); t_T(q) = C / a_T(q)
  Vec3 r[3];   // vertices relative to O
};

struct CoreWork {
  int beam;
  uint32_t node;
  int frame;
  AABB box;
};

struct CoreFrame {
  uint32_t node;
  int parent;
  AABB box;
};

// Bit 0: some vertex strictly inside (L > eps); bit 1: some vertex strictly outside.
BT_HD inline int classifyPoly(const Poly2& p, const Line2& L, Real eps) {
  int c = 0;
  for (int i = 0; i < p.n; ++i) {
    Real d = L.eval(p.x[i], p.y[i]);
    if (d > eps) c |= 1;
    else if (d < -eps) c |= 2;
  }
  return c;
}

// Splits a convex pentagon into a quad and a triangle (paper Fig. 2h).
BT_HD inline void splitFive(const Poly2& p, Poly2& a, Poly2& b) {
  a.clear();
  b.clear();
  for (int i = 0; i < 4; ++i) a.push(p.x[i], p.y[i]);
  b.push(p.x[0], p.y[0]);
  b.push(p.x[3], p.y[3]);
  b.push(p.x[4], p.y[4]);
}

// Optional per-phase cycle counters (device profiling with -DBT_PROFILE_CYCLES;
// no-ops otherwise).
#if defined(__CUDA_ARCH__) && defined(BT_PROFILE_CYCLES)
BT_HD inline long long btClock() { return clock64(); }
#else
BT_HD inline long long btClock() { return 0; }
#endif


// Outcome of the cheap beam-triangle tests (BeamCtx::classifyTri).
enum TriTest : int { kTriKeep = 0, kTriAllIn = 1, kTriClip = 2 };

// Per-query constants and the side-effect-free steps of the algorithm. Shared
// by BeamCore (depth-first, one tree at a time) and the wavefront engines,
// which keep their own beam storage and queues.
struct BeamCtx {
  enum { kNear = 0, kFar = 1, kBoth = 2 };

  SceneView sv;
  BeamQuery q;
  Real eps = 0, minArea = 0, ext = 1;
  Real farA[3] = {0, 0, 0}, farCq = 0;  // far plane (query plane) coefficients

  BT_HD void setup(const BeamQuery& query, Real extent) {
    q = query;
    ext = extent > 0 ? extent : Real(1);
    eps = kEpsRel * ext;
    minArea = eps * eps;
    farCq = dot(q.plane.n, q.plane.d);
    farA[0] = 0;
    farA[1] = 0;
    farA[2] = farCq;
  }

  // kd decision at an inner node for a beam whose rays move toward +sgn[a]
  // and reach the split plane at Dist_a = Ns / aw_a (Ns > 0). Every test is a
  // cross-multiplied comparison of linear-fractional distances; with
  // sign-consistent axes each is a half-plane in q, so the corners decide it
  // for every ray of the beam (conservative: ties fall through to "both").
  BT_HD int decide(const CoreBeam& b, int a, Real Ns, const AABB& box) const {
    const Vec3& O = q.plane.O;
    const R4& awa = b.aw[a];
    for (int k = 1; k <= 2; ++k) {
      int c = a + k >= 3 ? a + k - 3 : a + k;
      Real s = Real(b.sgn[c]);
      Real entry = b.sgn[c] > 0 ? box.lo[c] : box.hi[c];
      Real exit = b.sgn[c] > 0 ? box.hi[c] : box.lo[c];
      Real Nmin = s * (entry - O[c]);
      Real Nmax = s * (exit - O[c]);
      R4 lhs = b.aw[c] * Ns;
      R4 rhs = awa * Nmin;
      R4 tol = (rabs(lhs) + rabs(rhs)) * kKdRel;
      if (lt(lhs, rhs - tol) == kAll4) return kFar;   // crosses the split before entering slab c
      rhs = awa * Nmax;
      tol = (rabs(lhs) + rabs(rhs)) * kKdRel;
      if (gt(lhs, rhs + tol) == kAll4) return kNear;  // leaves slab c before reaching the split
    }
    if (b.hasFar) {
      R4 lhs = b.farA * Ns;
      R4 rhs = awa * b.farC;
      R4 tol = (rabs(lhs) + rabs(rhs)) * kKdRel;
      if (gt(lhs, rhs + tol) == kAll4) return kNear;  // split lies beyond the hit / far plane
    }
    return kBoth;
  }

  BT_HD bool overlaps(const CoreBeam& b, const AABB& box) const {
    const Vec3& O = q.plane.O;
    Real Nmin[3], Nmax[3];
    for (int c = 0; c < 3; ++c) {
      Real s = Real(b.sgn[c]);
      Nmin[c] = s * ((b.sgn[c] > 0 ? box.lo[c] : box.hi[c]) - O[c]);
      Nmax[c] = s * ((b.sgn[c] > 0 ? box.hi[c] : box.lo[c]) - O[c]);
      if (Nmax[c] < 0) return false;  // slab lies entirely behind the apex
    }
    for (int c = 0; c < 3; ++c) {
      for (int e = 0; e < 3; ++e) {
        if (c == e) continue;
        // TMax_c < TMin_e for every corner: leaves slab c before entering slab e.
        R4 lhs = b.aw[e] * Nmax[c];
        R4 rhs = b.aw[c] * Nmin[e];
        R4 tol = (rabs(lhs) + rabs(rhs)) * kKdRel;
        if (lt(lhs, rhs - tol) == kAll4) return false;
      }
    }
    if (b.hasFar) {
      for (int c = 0; c < 3; ++c) {
        // TMin_c beyond the far plane for every corner.
        R4 lhs = b.farA * Nmin[c];
        R4 rhs = b.aw[c] * b.farC;
        R4 tol = (rabs(lhs) + rabs(rhs)) * kKdRel;
        if (gt(lhs, rhs + tol) == kAll4) return false;
      }
    }
    return true;
  }

  // Corners and |w| per axis of a beam with cross-section `p` (convex,
  // <= 4 vertices). Returns false for slivers below minArea; `dropped` then
  // receives the polygon's area.
  BT_HD bool beamGeom(const Poly2& p, const int8_t sgn[3], CoreBeam& b, Real& dropped) const {
    Poly2 c = p;
    cleanPoly(c, eps);
    Real A = c.n >= 3 ? c.area() : Real(0);
    if (c.n < 3 || A < minArea) {
      Real a0 = p.area();
      dropped = a0 < 0 ? -a0 : a0;
      return false;
    }
    int l3 = c.n > 3 ? 3 : 2;
    b.qx = R4(c.x[0], c.x[1], c.x[2], c.x[l3]);
    b.qy = R4(c.y[0], c.y[1], c.y[2], c.y[l3]);
    b.n = c.n;
    const BeamPlane& P = q.plane;
    for (int a = 0; a < 3; ++a) {
      b.sgn[a] = sgn[a];
      R4 w = b.qx * P.u[a] + b.qy * P.v[a] + P.d[a];
      b.aw[a] = rmax(R4(Real(0)), w * Real(sgn[a]));
    }
    return true;
  }

  // Hit status and far plane: the plane (A, C) of the hit triangle, else the
  // query's far plane if it has one.
  BT_HD void setStatus(CoreBeam& b, int hit, const Real* A, Real C) const {
    b.hit = hit;
    if (hit >= 0) {
      b.farA = rabs(b.qx * A[0] + b.qy * A[1] + A[2]);
      b.farC = C < 0 ? -C : C;
      b.hasFar = true;
    } else if (q.farIsPlane) {
      Real c0 = farCq < 0 ? -farCq : farCq;
      b.farA = R4(c0);
      b.farC = c0;
      b.hasFar = true;
    } else {
      b.farC = 0;
      b.hasFar = false;
    }
  }

  BT_HD void computeTriInfo(int t, CoreTriInfo& ti) const {
    ti.tri = t;
    ti.ok = false;
    const uint32_t* T = sv.tri3 + 3 * size_t(t);
    const Vec3& O = q.plane.O;
    for (int k = 0; k < 3; ++k) ti.r[k] = sv.pos[T[k]] - O;
    // Edge planes through O, computed in canonical vertex order so the two
    // triangles sharing an edge get exactly negated planes (no cracks).
    Vec3 m[3];
    const int ei[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (int k = 0; k < 3; ++k) {
      int i = ei[k][0], j = ei[k][1];
      m[k] = T[i] < T[j] ? cross(ti.r[i], ti.r[j]) : -cross(ti.r[j], ti.r[i]);
    }
    // D = det(v0-O, v1-O, v2-O) = (v0 - O) . N; its sign orients the edges so
    // that "inside all three" means "hits the triangle at t > 0".
    Real D = dot(ti.r[2], m[0]);
    Real scale = length(m[0]) * length(ti.r[2]);
    if (!((D < 0 ? -D : D) > kDegRel * scale)) return;  // O in the triangle's plane: zero solid angle
    if (q.cullBackfaces && D > 0) return;               // back-facing as seen from O
    Real s = D > 0 ? Real(1) : Real(-1);
    const BeamPlane& P = q.plane;
    for (int k = 0; k < 3; ++k)
      ti.edge[k] = makeLine(s * dot(m[k], P.u), s * dot(m[k], P.v), s * dot(m[k], P.d), ext);
    hitPlane(t, ti.A, ti.C);
    ti.ok = true;
  }

  // a_T(q) = A0 qx + A1 qy + A2 = n_T . dir(q) and C = n_T . (v0 - O) for
  // triangle t; t_T(q) = C / a_T(q).
  BT_HD void hitPlane(int t, Real A[3], Real& C) const {
    const BeamPlane& P = q.plane;
    const Vec3& nT = sv.triN[t];
    A[0] = dot(nT, P.u);
    A[1] = dot(nT, P.v);
    A[2] = dot(nT, P.d);
    C = dot(nT, sv.pos[sv.tri3[3 * size_t(t)]] - q.plane.O);
  }

  // Line whose positive side is where triangle T is strictly closer to O than
  // plane F (t_T < t_F). With t_k(q) = C_k / a_k(q) and sign(a_k) = sign(C_k)
  // wherever the rays hit, t_T < t_F <=> sign(C_T C_F) (C_F a_T(q) - C_T a_F(q)) > 0.
  BT_HD Line2 closerLine(const CoreTriInfo& T, const Real* AF, Real CF, const Vec3& nF, int fId) const {
    const Vec3& nT = sv.triN[T.tri];
    Real cosang = dot(nT, nF);
    Real aC = T.C < 0 ? -T.C : T.C, aF = CF < 0 ? -CF : CF;
    Real scaleC = bmax(aC, aF);
    Real ca = cosang < 0 ? -cosang : cosang;
    Real diff = T.C - cosang * CF;
    if (ca > 1 - kCoplanarCos && (diff < 0 ? -diff : diff) <= kCoplanarRel * scaleC) {
      Line2 L;  // coplanar: deterministic tie-break (lower id wins); never closer than the far plane
      L.c = (fId >= 0 && T.tri < fId) ? Real(1) : Real(-1);
      return L;
    }
    Real sg = T.C * CF > 0 ? Real(1) : Real(-1);
    return makeLine(sg * (CF * T.A[0] - T.C * AF[0]), sg * (CF * T.A[1] - T.C * AF[1]),
                    sg * (CF * T.A[2] - T.C * AF[2]), ext);
  }

  // Trivial case (b) of Fig. 2: some beam side plane has all three triangle
  // vertices strictly outside it.
  BT_HD bool separated(const CoreBeam& b, const CoreTriInfo& ti) const {
    const BeamPlane& P = q.plane;
    R4 wx = b.qx * P.u.x + b.qy * P.v.x + P.d.x;
    R4 wy = b.qx * P.u.y + b.qy * P.v.y + P.d.y;
    R4 wz = b.qx * P.u.z + b.qy * P.v.z + P.d.z;
    R4 wx1 = rot1(wx), wy1 = rot1(wy), wz1 = rot1(wz);
    R4 nx = wy * wz1 - wz * wy1;
    R4 ny = wz * wx1 - wx * wz1;
    R4 nz = wx * wy1 - wy * wx1;
    R4 nlen = rsqrt_(nx * nx + ny * ny + nz * nz);
    // Lane k holds side k -> k+1. A triangular beam repeats its last corner in
    // lane 3, so lane 2 is a degenerate side and must not vote (with FMA its
    // cross product is a tiny nonzero, not exactly 0).
    int outAll = b.n == 4 ? kAll4 : 0xB;
    for (int k = 0; k < 3 && outAll; ++k) {
      const Vec3& r = ti.r[k];
      R4 s = nx * r.x + ny * r.y + nz * r.z;
      R4 tol = nlen * (length(r) * kEpsRel);
      outAll &= lt(s, -tol);
    }
    return outAll != 0;
  }

  BT_HD void beamToPoly(const CoreBeam& b, Poly2& p) const {
    p.n = b.n;
    for (int i = 0; i < b.n; ++i) {
      p.x[i] = b.qx[i];
      p.y[i] = b.qy[i];
    }
  }

  // Fuzzy test of the 4 corners against a line (paper App. A): only corners
  // farther than eps from the line vote. Returns false if no corner is
  // definitely inside, i.e. the beam lies outside the half-plane.
  BT_HD bool testLine(const CoreBeam& B, const Line2& line, Line2* L, int& nl, bool& allIn) const {
    R4 dv = B.qx * line.a + B.qy * line.b + line.c;
    if (gt(dv, R4(eps)) == 0) return false;  // no corner definitely inside
    if (lt(dv, R4(-eps)) != 0) allIn = false;
    L[nl++] = line;
    return true;
  }

  // The cheap part of a beam-triangle test (B.hit != ti.tri): the trivial
  // cases of Fig. 2 plus the depth line (Nearest: is T in front of the current
  // hit?) and the far plane. Returns
  //   kTriKeep  - B is unaffected by T,
  //   kTriAllIn - B lies inside T and T is in front everywhere (case c),
  //   kTriClip  - B must be clipped by the lines L[0..nl).
  // `hitPlane(A, C)` supplies the plane of B.hit and is called only when
  // needed (Nearest mode, B has a hit, the edges did not decide). `reason`
  // receives the deciding step (1-3 an edge, 4 depth, 5 far, 6 inside,
  // 7 beam side).
  template <class HitPlane>
  BT_HD int classifyTri(const CoreBeam& B, const CoreTriInfo& ti, HitPlane& hp, Line2* L, int& nl,
                        int& reason) const {
    nl = 0;
    bool allIn = true;
    reason = 0;
    for (int k = 0; k < 3; ++k) {
      if (!testLine(B, ti.edge[k], L, nl, allIn)) {  // trivial case (a): a triangle edge separates
        reason = 1 + k;
        return kTriKeep;
      }
    }
    if (q.mode == BeamMode::Nearest && B.hit >= 0) {
      Real AF[3], CF;
      hp(B.hit, AF, CF);
      if (!testLine(B, closerLine(ti, AF, CF, sv.triN[B.hit], B.hit), L, nl, allIn)) {
        reason = 4;  // T is behind the current hit everywhere in the beam
        return kTriKeep;
      }
    }
    if (q.farIsPlane) {
      if (!testLine(B, closerLine(ti, farA, farCq, q.plane.n, -1), L, nl, allIn)) {
        reason = 5;  // T lies beyond the far plane
        return kTriKeep;
      }
    }
    if (allIn) {  // trivial case (c): the beam lies inside the triangle
      reason = 6;
      return kTriAllIn;
    }
    if (separated(B, ti)) {  // trivial case (b): a beam side separates
      reason = 7;
      return kTriKeep;
    }
    return kTriClip;
  }

  // Clips B by the lines L[0..nl) one at a time (paper Fig. 2e-h). Pieces
  // outside some line keep the beam's old status and go to sink.push(0, p);
  // pieces inside every line take the new status and go to sink.push(1, p).
  // Clips first by the edge that leaves the smallest remainder: later cuts
  // then act on smaller pieces, which shortens their extension lines across
  // the miss region (less fragmentation of later beams). Returns false if a
  // push fails or the local stack overflows.
  template <class Sink>
  BT_HD bool clipTri(const CoreBeam& B, Line2* L, int nl, bool orderEdges, Sink& sink, uint64_t& fiveSplits) const {
    if (orderEdges) {
      Real score[3];
      for (int k = 0; k < 3; ++k) {
        R4 dv = B.qx * L[k].a + B.qy * L[k].b + L[k].c;
        R4 pos = rmax(dv, R4(Real(0)));
        score[k] = pos[0] + pos[1] + pos[2] + (B.n == 4 ? pos[3] : Real(0));
      }
      for (int i = 1; i < 3; ++i)
        for (int j = i; j > 0 && score[j] < score[j - 1]; --j) {
          bswap(score[j], score[j - 1]);
          bswap(L[j], L[j - 1]);
        }
    }
    bool ok = true;
    Poly2 stack[8];
    int stackLine[8];
    int sp = 0;
    beamToPoly(B, stack[0]);
    stackLine[0] = 0;
    sp = 1;
    while (sp > 0) {
      --sp;
      Poly2 p = stack[sp];
      int li = stackLine[sp];
      bool inside = true;
      for (; li < nl; ++li) {
        int c = classifyPoly(p, L[li], eps);
        if (!(c & 2)) continue;  // nothing strictly outside
        if (!(c & 1)) {          // nothing strictly inside
          if (!sink.push(0, p)) ok = false;
          inside = false;
          break;
        }
        Poly2 in, out;
        splitPoly(p, L[li], eps, in, out);
        cleanPoly(out, eps);
        if (out.n >= 3 && !sink.push(0, out)) ok = false;
        cleanPoly(in, eps);
        if (in.n < 3) {
          inside = false;
          break;
        }
        if (in.n == 5) {
          Poly2 a, b;
          splitFive(in, a, b);
          fiveSplits++;
          if (sp >= 8) return false;
          stack[sp] = b;
          stackLine[sp] = li + 1;
          ++sp;
          p = a;
        } else {
          p = in;
        }
      }
      if (inside && !sink.push(1, p)) ok = false;
    }
    return ok;
  }

  // Pre-split by the lines where an axis component of the ray direction is
  // zero, so each beam has a single direction sign per axis; then fan into
  // convex pieces of <= 4 vertices. Calls emit.piece(poly, sgn) for each
  // piece (returning false aborts), emit.presplit() for each extra split and
  // emit.dropped(area) for slivers along a sign line. Returns false if aborted
  // or the local stack overflows.
  template <class Emit>
  BT_HD bool presplit(const Poly2& root, Emit& emit) const {
    struct Item {
      Poly2 p;
      int axis;
      int8_t sgn[3];
    };
    Item stack[16];
    int sp = 0;
    stack[sp].p = root;
    stack[sp].axis = 0;
    stack[sp].sgn[0] = stack[sp].sgn[1] = stack[sp].sgn[2] = 1;
    ++sp;
    const BeamPlane& P = q.plane;
    while (sp > 0) {
      Item it = stack[--sp];
      const Poly2& p = it.p;
      if (p.n < 3) continue;
      if (it.axis == 3) {
        for (int k = 1; k + 1 < p.n; k += 2) {
          Poly2 qd;
          qd.push(p.x[0], p.y[0]);
          qd.push(p.x[k], p.y[k]);
          qd.push(p.x[k + 1], p.y[k + 1]);
          if (k + 2 < p.n) qd.push(p.x[k + 2], p.y[k + 2]);
          if (!emit.piece(qd, it.sgn)) return false;
        }
        continue;
      }
      const int axis = it.axis;
      Line2 L = makeLine(P.u[axis], P.v[axis], P.d[axis], ext);
      int c = classifyPoly(p, L, eps);
      if (c == 3) {
        if (sp + 2 > 16) return false;
        Poly2 in, out;
        splitPoly(p, L, eps, in, out);
        cleanPoly(in, eps);
        cleanPoly(out, eps);
        Item a = it, b = it;
        a.p = in;
        a.axis = axis + 1;
        a.sgn[axis] = 1;
        b.p = out;
        b.axis = axis + 1;
        b.sgn[axis] = -1;
        stack[sp++] = b;
        stack[sp++] = a;
        emit.presplit();
      } else if (c != 0) {
        it.axis = axis + 1;
        it.sgn[axis] = (c & 1) ? 1 : -1;
        stack[sp++] = it;
      } else {
        Real a0 = p.area();
        a0 = a0 < 0 ? -a0 : a0;
        emit.dropped(a0);  // sliver along a direction-sign line
      }
    }
    return true;
  }
};

template <class S>
class BeamCore {
 public:
  enum { kNear = BeamCtx::kNear, kFar = BeamCtx::kFar, kBoth = BeamCtx::kBoth };
  enum { kCur = 0, kNxt = 1 };

  BT_HD BeamCore(const SceneView& sv, S& st) : st_(st) { c_.sv = sv; }

  TraceStats stats;
  long long cyc[4] = {0, 0, 0, 0};  // descent, leaf, advance, root setup (device only)
  bool useMailbox = true;
  bool orderEdges = true;  // clip by the most-cutting triangle edge first
  int lastReason = 0;      // deciding step of the last intersect() (tests)

  // Traces convex root polygons (plane coordinates) through the scene.
  BT_HD void trace(const BeamQuery& query, const Poly2* roots, int numRoots) {
    long long c0 = btClock();
    st_.beginTrace();
    Real ext = 0;
    for (int i = 0; i < numRoots; ++i) ext = bmax(ext, roots[i].extent());
    setupQuery(query, ext);
    for (int i = 0; i < numRoots && !st_.overflow; ++i) {
      Poly2 p = roots[i];
      if (p.n < 3) continue;
      if (p.area() < 0) p.reverse();
      addRoot(p);
    }
    cyc[3] += btClock() - c0;
    const Vec3& O = c_.q.plane.O;
    CoreWork w;
    while (!st_.overflow && st_.popWork(w)) {
      // Frames form parent chains toward smaller indices, and work is LIFO:
      // frames above every pending item's frame are dead and can be reused.
      st_.truncateFrames(bmax(w.frame, st_.maxPendingFrame()) + 1);
      uint32_t node = w.node;
      AABB box = w.box;
      int frame = w.frame;
      long long c1 = btClock();
      for (;;) {
        const KdNode& kn = c_.sv.nodes[node];
        if (kn.isLeaf()) break;
        stats.kdSteps++;
        const int a = kn.axis();
        const Real s = kn.split;
        const CoreBeam& B = st_.beam(w.beam);
        const Real Ns = Real(B.sgn[a]) * (s - O[a]);
        const uint32_t left = node + 1, right = kn.right();
        AABB lbox = box, rbox = box;
        lbox.hi[a] = s;
        rbox.lo[a] = s;
        const bool sigmaRight = B.sgn[a] > 0;
        // cs: the child the rays move into; co: the child containing O.
        const uint32_t cs = sigmaRight ? right : left, co = sigmaRight ? left : right;
        const AABB& bs = sigmaRight ? rbox : lbox;
        const AABB& bo = sigmaRight ? lbox : rbox;
        if (Ns <= 0) {  // split plane behind (or through) the apex
          node = cs;
          box = bs;
          continue;
        }
        int dec = c_.decide(B, a, Ns, box);
        if (dec == kNear) {
          node = co;
          box = bo;
        } else if (dec == kFar) {
          node = cs;
          box = bs;
        } else {
          CoreFrame f;
          f.node = cs;
          f.parent = frame;
          f.box = bs;
          int fi = st_.pushFrame(f);
          if (fi < 0) {
            st_.overflow = true;
            return;
          }
          frame = fi;
          node = co;
          box = bo;
        }
      }
      long long c2 = btClock();
      cyc[0] += c2 - c1;
      const KdNode& leaf = c_.sv.nodes[node];
      if (leaf.count() == 0) {  // empty leaf: just move on
        stats.leafVisits++;
        advance(w.beam, frame);
        cyc[2] += btClock() - c2;
        continue;
      }
      processLeaf(w.beam, leaf);
      long long c3 = btClock();
      cyc[1] += c3 - c2;
      const int n = st_.listSize(cur_);
      for (int i = 0; i < n && !st_.overflow; ++i) advance(st_.listAt(cur_, i), frame);
      cyc[2] += btClock() - c3;
    }
  }

  // ---- pieces exposed for tests ----
  BT_HD void setupQuery(const BeamQuery& q, Real ext) {
    c_.setup(q, ext);
    if (++cacheStamp_ == 0) cacheStamp_ = 1;  // slots stamped 0 are never valid
  }
  BT_HD int decide(const CoreBeam& b, int a, Real Ns, const AABB& box) const { return c_.decide(b, a, Ns, box); }
  BT_HD bool overlaps(const CoreBeam& b, const AABB& box) const { return c_.overlaps(b, box); }
  BT_HD const BeamCtx& ctx() const { return c_; }

  // Builds a beam from a convex polygon (<= 4 vertices) for the current query.
  BT_HD bool makeBeam(const Poly2& p, const int8_t sgn[3], int hit, CoreBeam& b) {
    Real dropped = 0;
    if (!c_.beamGeom(p, sgn, b, dropped)) {
      stats.droppedArea += double(dropped);
      st_.dropped(dropped);
      return false;
    }
    b.mbox = 0;
    if (hit >= 0) {
      const CoreTriInfo& ti = triInfo(hit);
      c_.setStatus(b, hit, ti.A, ti.C);
    } else {
      c_.setStatus(b, -1, nullptr, 0);
    }
    return true;
  }

  // Re-tests a final beam against one triangle (idempotency check). Returns
  // 0 if nothing would change, 1 if the beam's status would change as a whole,
  // 2 if it would be split.
  BT_HD int retest(const BeamQuery& q, Real ext, const OutBeam& ob, int tri, int* reason, Real* newArea = nullptr) {
    st_.beginTrace();
    setupQuery(q, ext);
    Poly2 p;
    for (int i = 0; i < ob.n; ++i) p.push(ob.x[i], ob.y[i]);
    int8_t sgn[3];
    for (int a = 0; a < 3; ++a) {
      Real best = 0;
      for (int i = 0; i < p.n; ++i) {
        Real w = q.plane.d[a] + p.x[i] * q.plane.u[a] + p.y[i] * q.plane.v[a];
        if ((w < 0 ? -w : w) > (best < 0 ? -best : best)) best = w;
      }
      sgn[a] = best >= 0 ? 1 : -1;
    }
    int idx = st_.allocBeam();
    if (idx < 0 || !makeBeam(p, sgn, q.mode == BeamMode::Nearest ? ob.tri : -1, st_.beam(idx))) return 0;
    const int hitBefore = st_.beam(idx).hit;
    int res = 0;
    lastReason = 0;
    if (tri != q.excludeTri && tri != hitBefore) {
      CoreTriInfo ti = triInfo(tri);
      if (ti.ok) {
        st_.listClear(nxt_);
        int outBefore = st_.outputCount();
        intersect(idx, tri, ti);
        bool unchanged = st_.listSize(nxt_) == 1 && st_.listAt(nxt_, 0) == idx && st_.beam(idx).hit == hitBefore &&
                         st_.outputCount() == outBefore;
        if (!unchanged) res = st_.listSize(nxt_) <= 1 ? 1 : 2;
        if (newArea) {
          *newArea = 0;
          for (int i = 0; i < st_.polySize(1); ++i) *newArea += st_.polyAt(1, i).area();
        }
      }
    }
    if (reason) *reason = lastReason;
    return res;
  }

 private:
  // Plane of the current hit triangle, from the triangle-setup cache.
  struct CachedHitPlane {
    BeamCore* core;
    BT_HD void operator()(int tri, Real* A, Real& C) {
      const CoreTriInfo& F = core->triInfo(tri);
      A[0] = F.A[0];
      A[1] = F.A[1];
      A[2] = F.A[2];
      C = F.C;
    }
  };
  // Clip pieces go to the storage's polygon lists.
  struct PolySink {
    S* st;
    BT_HD bool push(int l, const Poly2& p) {
      if (st->polyPush(l, p)) return true;
      st->overflow = true;
      return false;
    }
  };
  // Root pieces become beams queued at the kd root.
  struct RootEmit {
    BeamCore* core;
    BT_HD bool piece(const Poly2& p, const int8_t sgn[3]) { return core->addRootPiece(p, sgn); }
    BT_HD void presplit() { core->stats.presplitBeams++; }
    BT_HD void dropped(Real a) {
      core->stats.droppedArea += double(a);
      core->st_.dropped(a);
    }
  };

  BT_HD const CoreTriInfo& triInfo(int tri) {
    CoreTriInfo& e = st_.triSlot(tri);
    if (e.stamp != cacheStamp_ || e.tri != tri) {
      c_.computeTriInfo(tri, e);
      e.stamp = cacheStamp_;
    }
    return e;
  }

  BT_HD bool seen(int tri, uint64_t mbox) const {
    uint64_t s = st_.stamp(tri);
    if (s == 0) return false;
    uint64_t e = mbox;
    for (int k = 0; k < 4 && e != 0; ++k) {  // the beam and its 3 nearest ancestors
      if (s == e) return true;
      e = st_.parentOf(e);
    }
    return false;
  }

  BT_HD void output(const CoreBeam& b, int tri) {
    Poly2 p;
    c_.beamToPoly(b, p);
    Real area = p.area();
    if (tri >= 0) {
      stats.hits++;
      if (st_.firstHit(tri)) stats.visibleTris++;
    }
    stats.beams++;
    st_.output(p, tri, area);
  }

  BT_HD void finalize(int beamIdx) {
    const CoreBeam& b = st_.beam(beamIdx);
    output(b, c_.q.mode == BeamMode::Nearest ? b.hit : -1);
    st_.freeBeam(beamIdx);
  }

  BT_HD void emit(const Poly2& p0, const CoreBeam& parent, int newHit, bool hitNow, int tri) {
    Poly2 pieces[2];
    int np = 1;
    if (p0.n == 5) {
      splitFive(p0, pieces[0], pieces[1]);
      stats.fiveSplits++;
      np = 2;
    } else {
      pieces[0] = p0;
    }
    for (int i = 0; i < np; ++i) {
      CoreBeam nb;
      if (!makeBeam(pieces[i], parent.sgn, hitNow ? newHit : parent.hit, nb)) continue;
      nb.mbox = parent.mbox;
      if (hitNow && c_.q.mode == BeamMode::AnyHit) {
        output(nb, tri);  // occluded piece: done
        continue;
      }
      int idx = st_.allocBeam();
      if (idx < 0 || !st_.listPush(nxt_, idx)) {
        st_.overflow = true;
        return;
      }
      st_.beam(idx) = nb;
    }
  }

  BT_HD void keep(int beamIdx) {
    if (!st_.listPush(nxt_, beamIdx)) st_.overflow = true;
  }

  BT_HD void intersect(int beamIdx, int t, const CoreTriInfo& ti) {
    stats.triTests++;
    // Reference for the cheap early-out tests; copied only before splitting
    // (the slot is recycled while the pieces are emitted).
    const CoreBeam& B = st_.beam(beamIdx);
    if (B.hit == t) {
      keep(beamIdx);
      return;
    }
    Line2 L[5];
    int nl = 0;
    CachedHitPlane hp{this};
    int r = c_.classifyTri(B, ti, hp, L, nl, lastReason);
    if (r == kTriKeep) {
      keep(beamIdx);
      return;
    }
    if (r == kTriAllIn) {
      if (c_.q.mode == BeamMode::AnyHit) {
        output(B, t);
        st_.freeBeam(beamIdx);
        return;
      }
      CoreBeam& b = st_.beam(beamIdx);  // (same object as B)
      c_.setStatus(b, t, ti.A, ti.C);
      keep(beamIdx);
      return;
    }
    // The pieces are committed only if some piece really changes status, so a
    // triangle whose overlap with the beam is empty or hidden never fragments it.
    st_.polyClear(0);
    st_.polyClear(1);
    PolySink sink{&st_};
    if (!c_.clipTri(B, L, nl, orderEdges, sink, stats.fiveSplits)) {
      st_.overflow = true;
      return;
    }
    Real newArea = 0;
    for (int i = 0; i < st_.polySize(1); ++i) newArea += st_.polyAt(1, i).area();
    if (!(newArea >= c_.minArea)) {
      lastReason = 9;  // nothing (measurable) hits: leave the beam untouched
      keep(beamIdx);
      return;
    }
    stats.splits++;
    lastReason = 8;
    const CoreBeam parent = B;  // copy: the slot is reused by the pieces
    st_.freeBeam(beamIdx);
    for (int i = 0; i < st_.polySize(0); ++i) emit(st_.polyAt(0, i), parent, -1, false, t);
    for (int i = 0; i < st_.polySize(1); ++i) emit(st_.polyAt(1, i), parent, t, true, t);
  }

  BT_HD void processLeaf(int beamIdx, const KdNode& leaf) {
    stats.leafVisits++;
    const uint64_t lineage = st_.beam(beamIdx).mbox;
    uint64_t E = 0;
    if (useMailbox) E = st_.newEvent(lineage);
    st_.listClear(cur_);
    st_.listPush(cur_, beamIdx);
    const uint32_t* idx = c_.sv.triIndices + leaf.offset;
    const uint32_t cnt = leaf.count();
    for (uint32_t i = 0; i < cnt && !st_.overflow; ++i) {
      int t = int(idx[i]);
      if (t == c_.q.excludeTri) continue;
      if (useMailbox && seen(t, lineage)) {
        stats.mailboxSkips++;
        continue;
      }
      CoreTriInfo ti = triInfo(t);
      if (!ti.ok) continue;
      st_.listClear(nxt_);
      const int n = st_.listSize(cur_);
      for (int k = 0; k < n && !st_.overflow; ++k) intersect(st_.listAt(cur_, k), t, ti);
      bswap(cur_, nxt_);
      if (st_.listSize(cur_) == 0) break;
    }
    if (useMailbox)
      for (uint32_t i = 0; i < cnt; ++i) st_.setStamp(int(idx[i]), E);
    const int n = st_.listSize(cur_);
    for (int k = 0; k < n; ++k) st_.beam(st_.listAt(cur_, k)).mbox = E;
  }

  BT_HD void advance(int beamIdx, int frame) {
    int f = frame;
    while (f >= 0) {
      const CoreFrame& fr = st_.frame(f);
      if (c_.overlaps(st_.beam(beamIdx), fr.box)) {
        CoreWork w;
        w.beam = beamIdx;
        w.node = fr.node;
        w.frame = fr.parent;
        w.box = fr.box;
        if (!st_.pushWork(w)) st_.overflow = true;
        return;
      }
      f = fr.parent;
    }
    finalize(beamIdx);
  }

  BT_HD void addRoot(const Poly2& root) {
    RootEmit e{this};
    if (!c_.presplit(root, e)) st_.overflow = true;
  }

  // A pre-split root piece: becomes a beam queued at the kd root.
  BT_HD bool addRootPiece(const Poly2& p, const int8_t sgn[3]) {
    CoreBeam b;
    if (!makeBeam(p, sgn, -1, b)) return true;
    int idx = st_.allocBeam();
    if (idx < 0) {
      st_.overflow = true;
      return false;
    }
    st_.beam(idx) = b;
    stats.rootBeams++;
    if (c_.overlaps(b, c_.sv.bounds)) {
      CoreWork w;
      w.beam = idx;
      w.node = 0;
      w.frame = -1;
      w.box = c_.sv.bounds;
      if (!st_.pushWork(w)) {
        st_.overflow = true;
        return false;
      }
    } else {
      finalize(idx);
    }
    return true;
  }

  BeamCtx c_;
  S& st_;
  uint32_t cacheStamp_ = 0;
  int cur_ = kCur, nxt_ = kNxt;
};

}  // inline namespace BT_R4_NS
}  // namespace bt
