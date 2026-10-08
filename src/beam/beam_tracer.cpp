#include "beam/beam_tracer.h"

#include <algorithm>
#include <cassert>

namespace bt {

namespace {

#ifdef BT_REAL_DOUBLE
constexpr Real kCoplanarCos = Real(1e-9);
constexpr Real kCoplanarRel = Real(1e-8);
#else
constexpr Real kCoplanarCos = Real(2e-6);
constexpr Real kCoplanarRel = Real(2e-5);
#endif

constexpr uint32_t kCacheBits = 14;

// Bit 0: some vertex strictly inside (L > eps); bit 1: some vertex strictly outside.
inline int classifyPoly(const Poly2& p, const Line2& L, Real eps) {
  int c = 0;
  for (int i = 0; i < p.n; ++i) {
    Real d = L.eval(p.x[i], p.y[i]);
    if (d > eps) c |= 1;
    else if (d < -eps) c |= 2;
  }
  return c;
}

// Splits a convex pentagon into a quad and a triangle (paper Fig. 2h).
inline void splitFive(const Poly2& p, Poly2& a, Poly2& b) {
  a.clear();
  b.clear();
  for (int i = 0; i < 4; ++i) a.push(p.x[i], p.y[i]);
  b.push(p.x[0], p.y[0]);
  b.push(p.x[3], p.y[3]);
  b.push(p.x[4], p.y[4]);
}

}  // namespace

BeamPlane BeamPlane::make(const Vec3& O, const Vec3& p0, const Vec3& u_, const Vec3& v_) {
  BeamPlane P;
  P.O = O;
  P.p0 = p0;
  Vec3 n = normalize(cross(u_, v_));
  if (dot(n, p0 - O) < 0) n = -n;
  P.u = normalize(u_);
  P.v = cross(n, P.u);
  P.n = n;
  P.d = p0 - O;
  return P;
}

BeamTracer::BeamTracer(const Scene& scene, const KdTree& tree)
    : scene_(scene), tree_(tree), cache_(size_t(1) << kCacheBits), mail_(size_t(scene.numTris()), 0) {}

void BeamTracer::setupQuery(const BeamQuery& q, Real ext) {
  q_ = q;
  ext_ = ext > 0 ? ext : Real(1);
  eps_ = kEpsRel * ext_;
  minArea_ = eps_ * eps_;
  if (++stamp_ == 0) {
    for (auto& e : cache_) e.stamp = 0;
    stamp_ = 1;
  }
  farCq_ = dot(q.plane.n, q.plane.d);
  farA_[0] = 0;
  farA_[1] = 0;
  farA_[2] = farCq_;
}

void BeamTracer::setQueryForTest(const BeamQuery& q, Real ext) { setupQuery(q, ext); }

int BeamTracer::retest(const BeamQuery& q, Real ext, const OutBeam& ob, int tri, int* reason) {
  setupQuery(q, ext);
  BeamOutput dummy;
  out_ = &dummy;
  pool_.clear();
  nxt_.clear();
  Poly2 p;
  for (int i = 0; i < ob.n; ++i) p.push(ob.x[i], ob.y[i]);
  int8_t sgn[3];
  for (int a = 0; a < 3; ++a) {
    Real best = 0;
    for (int i = 0; i < p.n; ++i) {
      Real w = q.plane.d[a] + p.x[i] * q.plane.u[a] + p.y[i] * q.plane.v[a];
      if (std::fabs(w) > std::fabs(best)) best = w;
    }
    sgn[a] = best >= 0 ? 1 : -1;
  }
  Beam b;
  if (!makeBeam(p, sgn, q.mode == BeamMode::Nearest ? ob.tri : -1, b)) {
    out_ = nullptr;
    return 0;
  }
  pool_.push_back(b);
  int res = 0;
  if (tri != q.excludeTri && tri != b.hit) {
    TriInfo ti = triInfo(tri);
    if (ti.ok) {
      size_t outBefore = dummy.beams.size();
      intersect(0, tri, ti);
      bool unchanged = nxt_.size() == 1 && nxt_[0] == 0 && pool_[0].hit == b.hit && dummy.beams.size() == outBefore;
      if (!unchanged) res = (nxt_.size() <= 1 && pool_.size() == 1) ? 1 : 2;
    }
  }
  if (reason) *reason = lastReason;
  out_ = nullptr;
  return res;
}

void BeamTracer::computeTriInfo(int t, TriInfo& ti) const {
  ti.tri = t;
  ti.ok = false;
  const auto& T = scene_.tris[size_t(t)];
  const Vec3& O = q_.plane.O;
  for (int k = 0; k < 3; ++k) ti.r[k] = scene_.pos[T[size_t(k)]] - O;
  // Edge planes through O, computed in canonical vertex order so the two
  // triangles sharing an edge get exactly negated planes (no cracks).
  auto canon = [&](int i, int j) -> Vec3 {
    return T[size_t(i)] < T[size_t(j)] ? cross(ti.r[i], ti.r[j]) : -cross(ti.r[j], ti.r[i]);
  };
  Vec3 m[3] = {canon(0, 1), canon(1, 2), canon(2, 0)};
  // D = det(v0-O, v1-O, v2-O) = (v0 - O) . N; its sign orients the edges so
  // that "inside all three" means "hits the triangle at t > 0".
  Real D = dot(ti.r[2], m[0]);
  Real scale = length(m[0]) * length(ti.r[2]);
  if (!(std::fabs(D) > kDegRel * scale)) return;  // O in the triangle's plane: zero solid angle
  if (q_.cullBackfaces && D > 0) return;           // back-facing as seen from O
  Real s = D > 0 ? Real(1) : Real(-1);
  const BeamPlane& P = q_.plane;
  for (int k = 0; k < 3; ++k)
    ti.edge[k] = makeLine(s * dot(m[k], P.u), s * dot(m[k], P.v), s * dot(m[k], P.d), ext_);
  const Vec3& nT = scene_.triN[size_t(t)];
  ti.A[0] = dot(nT, P.u);
  ti.A[1] = dot(nT, P.v);
  ti.A[2] = dot(nT, P.d);
  ti.C = dot(nT, ti.r[0]);
  ti.ok = true;
}

const BeamTracer::TriInfo& BeamTracer::triInfo(int tri) {
  TriInfo& e = cache_[size_t(tri) & ((size_t(1) << kCacheBits) - 1)];
  if (e.stamp != stamp_ || e.tri != tri) {
    computeTriInfo(tri, e);
    e.stamp = stamp_;
  }
  return e;
}

// Line whose positive side is where triangle T is strictly closer to O than
// plane F (t_T < t_F). With t_k(q) = C_k / a_k(q) and sign(a_k) = sign(C_k)
// wherever the rays hit, t_T < t_F <=> sign(C_T C_F) (C_F a_T(q) - C_T a_F(q)) > 0.
Line2 BeamTracer::closerLine(const TriInfo& T, const Real* AF, Real CF, const Vec3& nF, int fId) const {
  const Vec3& nT = scene_.triN[size_t(T.tri)];
  Real cosang = dot(nT, nF);
  Real scaleC = std::max(std::fabs(T.C), std::fabs(CF));
  if (std::fabs(cosang) > 1 - kCoplanarCos && std::fabs(T.C - cosang * CF) <= kCoplanarRel * scaleC) {
    Line2 L;  // coplanar: deterministic tie-break (lower id wins); never closer than the far plane
    L.c = (fId >= 0 && T.tri < fId) ? Real(1) : Real(-1);
    return L;
  }
  Real sg = T.C * CF > 0 ? Real(1) : Real(-1);
  return makeLine(sg * (CF * T.A[0] - T.C * AF[0]), sg * (CF * T.A[1] - T.C * AF[1]),
                  sg * (CF * T.A[2] - T.C * AF[2]), ext_);
}

// Trivial case (b) of Fig. 2: some beam side plane has all three triangle
// vertices strictly outside it.
bool BeamTracer::separated(const Beam& b, const TriInfo& ti) const {
  const BeamPlane& P = q_.plane;
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

bool BeamTracer::makeBeam(const Poly2& p, const int8_t sgn[3], int hit, Beam& b) {
  Poly2 c = p;
  cleanPoly(c, eps_);
  Real A = c.n >= 3 ? c.area() : Real(0);
  if (c.n < 3 || A < minArea_) {
    Real a0 = std::fabs(p.area());
    stats.droppedArea += double(a0);
    if (out_) out_->droppedArea += double(a0);
    return false;
  }
  assert(c.n <= 4);
  int l3 = c.n > 3 ? 3 : 2;
  b.qx = R4(c.x[0], c.x[1], c.x[2], c.x[l3]);
  b.qy = R4(c.y[0], c.y[1], c.y[2], c.y[l3]);
  b.n = c.n;
  const BeamPlane& P = q_.plane;
  for (int a = 0; a < 3; ++a) {
    b.sgn[a] = sgn[a];
    R4 w = b.qx * P.u[a] + b.qy * P.v[a] + P.d[a];
    b.aw[a] = rmax(R4(Real(0)), w * Real(sgn[a]));
  }
  b.hit = hit;
  b.mbox = 0;
  if (hit >= 0) {
    const TriInfo& ti = triInfo(hit);
    b.farA = rabs(b.qx * ti.A[0] + b.qy * ti.A[1] + ti.A[2]);
    b.farC = std::fabs(ti.C);
    b.hasFar = true;
  } else if (q_.farIsPlane) {
    b.farA = R4(std::fabs(farCq_));
    b.farC = std::fabs(farCq_);
    b.hasFar = true;
  } else {
    b.hasFar = false;
  }
  return true;
}

void BeamTracer::beamToPoly(const Beam& b, Poly2& p) const {
  p.n = b.n;
  for (int i = 0; i < b.n; ++i) {
    p.x[i] = b.qx[i];
    p.y[i] = b.qy[i];
  }
}

// kd decision at an inner node for a beam whose rays move toward +sgn[a]
// and reach the split plane at Dist_a = Ns / aw_a (Ns > 0). Every test is a
// cross-multiplied comparison of linear-fractional distances; with
// sign-consistent axes each is a half-plane in q, so the corners decide it
// for every ray of the beam (conservative: ties fall through to "both").
int BeamTracer::decide(const Beam& b, int a, Real Ns, const AABB& box) const {
  const Vec3& O = q_.plane.O;
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

bool BeamTracer::overlaps(const Beam& b, const AABB& box) const {
  const Vec3& O = q_.plane.O;
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

bool BeamTracer::seen(int tri, uint64_t mbox) const {
  uint64_t s = mail_[size_t(tri)];
  if (s == 0) return false;
  uint64_t e = mbox;
  for (int k = 0; k < 4 && e != 0; ++k) {  // the beam and its 3 nearest ancestors
    if (s == e) return true;
    e = parentOf(e);
  }
  return false;
}

void BeamTracer::output(const Beam& b, int tri) {
  Poly2 p;
  beamToPoly(b, p);
  Real area = p.area();
  if (tri >= 0) {
    out_->hitArea += double(area);
    stats.hits++;
  } else {
    out_->missArea += double(area);
  }
  stats.beams++;
  if (keepOutput) {
    OutBeam ob;
    ob.n = b.n;
    for (int i = 0; i < 4; ++i) {
      ob.x[i] = b.qx[i];
      ob.y[i] = b.qy[i];
    }
    ob.tri = tri;
    ob.area = area;
    out_->beams.push_back(ob);
  }
}

void BeamTracer::finalize(int beamIdx) {
  const Beam& b = pool_[size_t(beamIdx)];
  output(b, q_.mode == BeamMode::Nearest ? b.hit : -1);
}

void BeamTracer::emit(const Poly2& p0, const Beam& parent, int newHit, bool hitNow, int tri) {
  Poly2 pieces[2];
  int np = 1;
  if (p0.n == 5) {
    splitFive(p0, pieces[0], pieces[1]);
    np = 2;
  } else {
    pieces[0] = p0;
  }
  for (int i = 0; i < np; ++i) {
    Beam nb;
    if (!makeBeam(pieces[i], parent.sgn, hitNow ? newHit : parent.hit, nb)) continue;
    nb.mbox = parent.mbox;
    if (hitNow && q_.mode == BeamMode::AnyHit) {
      output(nb, tri);  // occluded piece: done
      continue;
    }
    pool_.push_back(nb);
    nxt_.push_back(int(pool_.size()) - 1);
  }
}

void BeamTracer::intersect(int beamIdx, int t, const TriInfo& ti) {
  stats.triTests++;
  const Beam B = pool_[size_t(beamIdx)];
  if (B.hit == t) {
    nxt_.push_back(beamIdx);
    return;
  }
  Line2 L[5];
  int nl = 0;
  bool allIn = true;
  // Fuzzy test of the 4 corners against a line (paper App. A): only corners
  // farther than eps from the line vote.
  auto test = [&](const Line2& line) -> bool {
    R4 dv = B.qx * line.a + B.qy * line.b + line.c;
    if (gt(dv, R4(eps_)) == 0) return false;  // no corner definitely inside
    if (lt(dv, R4(-eps_)) != 0) allIn = false;
    L[nl++] = line;
    return true;
  };
  lastReason = 0;
  for (int k = 0; k < 3; ++k) {
    if (!test(ti.edge[k])) {  // trivial case (a): a triangle edge separates
      lastReason = 1 + k;
      nxt_.push_back(beamIdx);
      return;
    }
  }
  if (q_.mode == BeamMode::Nearest && B.hit >= 0) {
    const TriInfo& F = triInfo(B.hit);
    Real AF[3] = {F.A[0], F.A[1], F.A[2]};
    Real CF = F.C;
    if (!test(closerLine(ti, AF, CF, scene_.triN[size_t(B.hit)], B.hit))) {
      lastReason = 4;
      nxt_.push_back(beamIdx);  // T is behind the current hit everywhere in the beam
      return;
    }
  }
  if (q_.farIsPlane) {
    if (!test(closerLine(ti, farA_, farCq_, q_.plane.n, -1))) {
      lastReason = 5;
      nxt_.push_back(beamIdx);  // T lies beyond the far plane
      return;
    }
  }
  if (allIn) {  // trivial case (c): the beam lies inside the triangle
    lastReason = 6;
    if (q_.mode == BeamMode::AnyHit) {
      output(B, t);
      return;
    }
    Beam& b = pool_[size_t(beamIdx)];
    b.hit = t;
    b.farA = rabs(b.qx * ti.A[0] + b.qy * ti.A[1] + ti.A[2]);
    b.farC = std::fabs(ti.C);
    b.hasFar = true;
    nxt_.push_back(beamIdx);
    return;
  }
  if (separated(B, ti)) {  // trivial case (b): a beam side separates
    lastReason = 7;
    nxt_.push_back(beamIdx);
    return;
  }
  // Clip one line at a time (paper Fig. 2e-h). Pieces outside some line keep
  // the beam's old status; pieces inside every line take the new status. The
  // pieces are committed only if some piece really changes status, so a
  // triangle whose overlap with the beam is empty or hidden never fragments it.
  tmpKeep_.clear();
  tmpNew_.clear();
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
      int c = classifyPoly(p, L[li], eps_);
      if (!(c & 2)) continue;  // nothing strictly outside
      if (!(c & 1)) {          // nothing strictly inside
        tmpKeep_.push_back(p);
        inside = false;
        break;
      }
      Poly2 in, out;
      splitPoly(p, L[li], eps_, in, out);
      cleanPoly(out, eps_);
      if (out.n >= 3) tmpKeep_.push_back(out);
      cleanPoly(in, eps_);
      if (in.n < 3) {
        inside = false;
        break;
      }
      if (in.n == 5) {
        Poly2 a, b;
        splitFive(in, a, b);
        stack[sp] = b;
        stackLine[sp] = li + 1;
        ++sp;
        p = a;
      } else {
        p = in;
      }
    }
    if (inside) tmpNew_.push_back(p);
  }
  Real newArea = 0;
  for (const Poly2& p : tmpNew_) newArea += p.area();
  if (!(newArea >= minArea_)) {
    lastReason = 9;  // nothing (measurable) hits: leave the beam untouched
    nxt_.push_back(beamIdx);
    return;
  }
  stats.splits++;
  lastReason = 8;
  for (const Poly2& p : tmpKeep_) emit(p, B, -1, false, t);
  for (const Poly2& p : tmpNew_) emit(p, B, t, true, t);
}

void BeamTracer::processLeaf(int beamIdx, const KdNode& leaf) {
  stats.leafVisits++;
  const uint64_t lineage = pool_[size_t(beamIdx)].mbox;
  uint64_t E = 0;
  if (useMailbox) {
    E = nextEvent_++;
    evParent_.push_back(lineage);
  }
  cur_.clear();
  cur_.push_back(beamIdx);
  const uint32_t* idx = tree_.triIndices.data() + leaf.offset;
  const uint32_t cnt = leaf.count();
  for (uint32_t i = 0; i < cnt; ++i) {
    int t = int(idx[i]);
    if (t == q_.excludeTri) continue;
    if (useMailbox && seen(t, lineage)) {
      stats.mailboxSkips++;
      continue;
    }
    TriInfo ti = triInfo(t);
    if (!ti.ok) continue;
    nxt_.clear();
    for (int b : cur_) intersect(b, t, ti);
    std::swap(cur_, nxt_);
    if (cur_.empty()) break;
  }
  if (useMailbox)
    for (uint32_t i = 0; i < cnt; ++i) mail_[idx[i]] = E;
  for (int b : cur_) pool_[size_t(b)].mbox = E;
}

void BeamTracer::advance(int beamIdx, int frame) {
  int f = frame;
  while (f >= 0) {
    const Frame& fr = frames_[size_t(f)];
    if (overlaps(pool_[size_t(beamIdx)], fr.box)) {
      work_.push_back({beamIdx, fr.node, fr.parent, fr.box});
      return;
    }
    f = fr.parent;
  }
  finalize(beamIdx);
}

void BeamTracer::addRoot(const Poly2& p, int axis, int8_t sgn[3]) {
  if (p.n < 3) return;
  if (axis == 3) {
    // fan into convex pieces of <= 4 vertices
    for (int k = 1; k + 1 < p.n; k += 2) {
      Poly2 q;
      q.push(p.x[0], p.y[0]);
      q.push(p.x[k], p.y[k]);
      q.push(p.x[k + 1], p.y[k + 1]);
      if (k + 2 < p.n) q.push(p.x[k + 2], p.y[k + 2]);
      Beam b;
      if (!makeBeam(q, sgn, -1, b)) continue;
      pool_.push_back(b);
      int idx = int(pool_.size()) - 1;
      stats.rootBeams++;
      if (overlaps(b, tree_.bounds))
        work_.push_back({idx, 0u, -1, tree_.bounds});
      else
        finalize(idx);
    }
    return;
  }
  // Pre-split by the line where the axis component of the ray direction is
  // zero, so each beam has a single direction sign per axis.
  const BeamPlane& P = q_.plane;
  Line2 L = makeLine(P.u[axis], P.v[axis], P.d[axis], ext_);
  int c = classifyPoly(p, L, eps_);
  if (c == 3) {
    Poly2 in, out;
    splitPoly(p, L, eps_, in, out);
    cleanPoly(in, eps_);
    cleanPoly(out, eps_);
    int8_t s2[3] = {sgn[0], sgn[1], sgn[2]};
    s2[axis] = 1;
    addRoot(in, axis + 1, s2);
    s2[axis] = -1;
    addRoot(out, axis + 1, s2);
    stats.presplitBeams++;
  } else if (c != 0) {
    int8_t s2[3] = {sgn[0], sgn[1], sgn[2]};
    s2[axis] = (c & 1) ? 1 : -1;
    addRoot(p, axis + 1, s2);
  } else {
    stats.droppedArea += double(std::fabs(p.area()));  // sliver along a direction-sign line
    out_->droppedArea += double(std::fabs(p.area()));
  }
}

void BeamTracer::trace(const BeamQuery& query, const Poly2* roots, int numRoots, BeamOutput& out) {
  out_ = &out;
  out.clear();
  Real ext = 0;
  for (int i = 0; i < numRoots; ++i) ext = std::max(ext, roots[i].extent());
  setupQuery(query, ext);
  evBase_ = nextEvent_;
  evParent_.clear();
  pool_.clear();
  work_.clear();
  frames_.clear();

  for (int i = 0; i < numRoots; ++i) {
    Poly2 p = roots[i];
    if (p.n < 3) continue;
    if (p.area() < 0) p.reverse();
    int8_t sgn[3] = {1, 1, 1};
    addRoot(p, 0, sgn);
  }

  const Vec3& O = q_.plane.O;
  while (!work_.empty()) {
    Work w = work_.back();
    work_.pop_back();
    uint32_t node = w.node;
    AABB box = w.box;
    int frame = w.frame;
    for (;;) {
      const KdNode& kn = tree_.nodes[node];
      if (kn.isLeaf()) break;
      stats.kdSteps++;
      const int a = kn.axis();
      const Real s = kn.split;
      const Beam& B = pool_[size_t(w.beam)];
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
      int dec = decide(B, a, Ns, box);
      if (dec == kNear) {
        node = co;
        box = bo;
      } else if (dec == kFar) {
        node = cs;
        box = bs;
      } else {
        frames_.push_back({cs, frame, bs});
        frame = int(frames_.size()) - 1;
        node = co;
        box = bo;
      }
    }
    processLeaf(w.beam, tree_.nodes[node]);
    // processLeaf leaves the live pieces in cur_; advance() only touches work_.
    std::vector<int> live;
    live.swap(cur_);
    for (int p : live) advance(p, frame);
    live.clear();
    cur_.swap(live);
  }
  out_ = nullptr;
}

}  // namespace bt
