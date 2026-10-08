// Unit tests: SIMD lanes, polygon splitting, kd decisions.
#include "tests/helpers.h"

using namespace bttest;

namespace {

#ifdef BT_REAL_DOUBLE
constexpr double kAreaTol = 1e-9;
#else
constexpr double kAreaTol = 2e-4;
#endif

Poly2 randomConvex(Rng& rng, int n) {
  // n points on a randomly squashed circle -> convex, CCW.
  Poly2 p;
  double cx = rng.uniform(-1, 1), cy = rng.uniform(-1, 1), rx = rng.uniform(0.1, 1), ry = rng.uniform(0.1, 1);
  double a0 = rng.uniform(0, 2 * M_PI);
  double angs[8];
  for (int i = 0; i < n; ++i) angs[i] = rng.uniform(0, 2 * M_PI);
  std::sort(angs, angs + n);
  for (int i = 0; i < n; ++i)
    p.push(Real(cx + rx * std::cos(angs[i] + a0)), Real(cy + ry * std::sin(angs[i] + a0)));
  if (p.area() < 0) p.reverse();
  return p;
}

bool isConvexCCW(const Poly2& p, Real tol) {
  for (int i = 0; i < p.n; ++i) {
    int j = (i + 1) % p.n, k = (i + 2) % p.n;
    Real cr = (p.x[j] - p.x[i]) * (p.y[k] - p.y[j]) - (p.y[j] - p.y[i]) * (p.x[k] - p.x[j]);
    if (cr < -tol) return false;
  }
  return true;
}

}  // namespace

TEST(r4_lanes_and_masks) {
  R4 a(1, 2, 3, 4), b(4, 3, 2, 1);
  CHECK(lt(a, b) == 0x3);
  CHECK(gt(a, b) == 0xC);
  CHECK(le(a, a) == kAll4);
  R4 r = rot1(a);
  CHECK(r[0] == 2 && r[1] == 3 && r[2] == 4 && r[3] == 1);
  R4 m = rmin(a, b);
  CHECK(m[0] == 1 && m[3] == 1);
  R4 ab = rabs(-a);
  CHECK(ab[2] == 3);
  R4 s = a * b + Real(1);
  CHECK(s[1] == 7);
}

TEST(split_conserves_area_and_convexity) {
  Rng rng(42);
  int bad = 0;
  for (int it = 0; it < 20000; ++it) {
    Poly2 p = randomConvex(rng, 3 + int(rng.next() % 2));
    double ang = rng.uniform(0, 2 * M_PI);
    Line2 L;
    L.a = Real(std::cos(ang));
    L.b = Real(std::sin(ang));
    // Sometimes put the line through a vertex to exercise "on" handling.
    if (it % 5 == 0) {
      int v = int(rng.next() % unsigned(p.n));
      L.c = -(L.a * p.x[v] + L.b * p.y[v]);
    } else {
      L.c = Real(rng.uniform(-1.5, 1.5));
    }
    Real eps = Real(1e-6);
    Poly2 in, out;
    splitPoly(p, L, eps, in, out);
    double a = double(p.area());
    double ai = in.n >= 3 ? double(in.area()) : 0, ao = out.n >= 3 ? double(out.area()) : 0;
    bool ok = std::fabs(ai + ao - a) <= kAreaTol * std::max(1.0, a) + 1e-5 * a;
    ok = ok && (in.n < 3 || isConvexCCW(in, Real(1e-5))) && (out.n < 3 || isConvexCCW(out, Real(1e-5)));
    ok = ok && in.n <= 5 && out.n <= 5;
    for (int i = 0; i < in.n; ++i) ok = ok && L.eval(in.x[i], in.y[i]) >= -eps * 2;
    for (int i = 0; i < out.n; ++i) ok = ok && L.eval(out.x[i], out.y[i]) <= eps * 2;
    if (!ok && bad++ < 3) fprintf(stderr, "    split failure: areas %g %g %g n %d %d\n", a, ai, ao, in.n, out.n);
  }
  CHECK(bad == 0);
}

TEST(split_neighbours_agree_bitwise) {
  // Two triangles sharing an edge produce exactly negated lines; the crossing
  // points they compute on any polygon edge must then be identical.
  Rng rng(7);
  int bad = 0;
  for (int it = 0; it < 5000; ++it) {
    Poly2 p = randomConvex(rng, 4);
    double ang = rng.uniform(0, 2 * M_PI);
    Line2 L;
    L.a = Real(std::cos(ang));
    L.b = Real(std::sin(ang));
    L.c = Real(rng.uniform(-0.5, 0.5));
    Line2 M;
    M.a = -L.a;
    M.b = -L.b;
    M.c = -L.c;
    Poly2 i1, o1, i2, o2;
    splitPoly(p, L, Real(1e-7), i1, o1);
    splitPoly(p, M, Real(1e-7), i2, o2);
    // in-piece of L == out-piece of M, vertex for vertex.
    bool same = i1.n == o2.n;
    for (int k = 0; same && k < i1.n; ++k) same = i1.x[k] == o2.x[k] && i1.y[k] == o2.y[k];
    if (!same) ++bad;
  }
  CHECK(bad == 0);
}

// For random beams and boxes, every interior ray's needed children must be
// covered by the beam's kd decision, and overlaps() must never cull a box an
// interior ray passes through.
TEST(kd_decision_is_conservative) {
  Rng rng(99);
  Scene dummy;
  uint16_t m = dummy.addMaterial(Vec3(1, 1, 1));
  addQuad(dummy, Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0), m);
  dummy.finalize();
  KdTree tree;
  tree.build(dummy);
  BeamTracer btr(dummy, tree);
  int checked = 0, violations = 0, cullViolations = 0;
  for (int it = 0; it < 4000; ++it) {
    AABB box;
    box.lo = Vec3(Real(rng.uniform(-1, 0)), Real(rng.uniform(-1, 0)), Real(rng.uniform(-1, 0)));
    box.hi = box.lo + Vec3(Real(rng.uniform(0.2, 2)), Real(rng.uniform(0.2, 2)), Real(rng.uniform(0.2, 2)));
    int a = int(rng.next() % 3);
    Real s = box.lo[a] + Real(rng.uniform(0.1, 0.9)) * (box.hi[a] - box.lo[a]);
    // Apex: outside, inside, on the split plane, or on a box face.
    Vec3 O(Real(rng.uniform(-3, 3)), Real(rng.uniform(-3, 3)), Real(rng.uniform(-3, 3)));
    int mode = it % 4;
    if (mode == 1) O = box.center() + Vec3(Real(rng.uniform(-0.1, 0.1)), Real(rng.uniform(-0.1, 0.1)), 0);
    if (mode == 2) O[a] = s;
    if (mode == 3) O[(a + 1) % 3] = box.lo[(a + 1) % 3];
    // Beam towards a random direction; small cross-section inside one octant.
    Vec3 dir = normalize(Vec3(Real(rng.uniform(-1, 1)), Real(rng.uniform(-1, 1)), Real(rng.uniform(-1, 1))));
    if (it % 7 == 0) dir[(a + 2) % 3] = 0;  // zero direction component
    dir = normalize(dir);
    Vec3 u = anyOrthogonal(dir), v = cross(dir, u);
    BeamQuery q;
    q.plane = BeamPlane::make(O, O + dir, u, v);
    q.mode = BeamMode::Nearest;
    btr.setQueryForTest(q, 1);
    Poly2 p = randomConvex(rng, 4);
    Real sc = Real(rng.uniform(0.01, 0.6));
    for (int i = 0; i < p.n; ++i) {
      p.x[i] *= sc;
      p.y[i] *= sc;
    }
    int8_t sgn[3];
    bool consistent = true;
    for (int c = 0; c < 3 && consistent; ++c) {
      int pos = 0, neg = 0;
      for (int i = 0; i < p.n; ++i) {
        Real w = q.plane.dir(p.x[i], p.y[i])[c];
        pos += w > 0;
        neg += w < 0;
      }
      if (pos && neg) consistent = false;
      sgn[c] = neg ? -1 : 1;
    }
    if (!consistent) continue;
    BeamTracer::Beam b;
    if (!btr.makeBeam(p, sgn, -1, b)) continue;
    // Optional far plane at a random distance (as for a hit beam).
    Real tfar = kInf;
    if (it % 3 == 0) {
      tfar = Real(rng.uniform(0.2, 4));
      b.hasFar = true;
      b.farC = tfar;
      b.farA = R4(1);  // t_far = farC / farA (same for every ray)
    }
    Real Ns = Real(sgn[a]) * (s - O[a]);
    int dec = Ns <= 0 ? -1 : btr.decide(b, a, Ns, box);
    bool ov = btr.overlaps(b, box);
    bool sigmaRight = sgn[a] > 0;
    for (int k = 0; k < 64; ++k) {
      // random interior ray (convex combination of the corners)
      Real wts[4], sum = 0;
      for (int i = 0; i < p.n; ++i) sum += (wts[i] = Real(rng.uniform(0.01, 1)));
      Real qx = 0, qy = 0;
      for (int i = 0; i < p.n; ++i) {
        qx += p.x[i] * wts[i] / sum;
        qy += p.y[i] * wts[i] / sum;
      }
      Vec3 w = q.plane.dir(qx, qy);
      // Ray segment inside the box (in double).
      double t0 = 0, t1 = std::isfinite(double(tfar)) ? double(tfar) : 1e30;
      for (int c = 0; c < 3; ++c) {
        if (w[c] == 0) {
          if (O[c] < box.lo[c] || O[c] > box.hi[c]) t0 = 1e31;
          continue;
        }
        double ta = (double(box.lo[c]) - O[c]) / w[c], tb = (double(box.hi[c]) - O[c]) / w[c];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
      }
      if (!(t0 < t1)) continue;  // ray misses the box: nothing needed
      ++checked;
      if (!ov) {
        ++cullViolations;
        continue;
      }
      // Which sides does the segment visit (beyond a tiny margin)?
      double x0 = O[a] + t0 * w[a], x1 = O[a] + t1 * w[a];
      double lo = std::min(x0, x1), hi = std::max(x0, x1);
      double margin = 1e-6 * (box.hi[a] - box.lo[a]);
      bool needLeft = lo < s - margin, needRight = hi > s + margin;
      bool needSigma = sigmaRight ? needRight : needLeft;
      bool needOther = sigmaRight ? needLeft : needRight;
      bool okDec = true;
      if (dec == -1) okDec = !needOther;
      else if (dec == BeamTracer::kNear) okDec = !needSigma;
      else if (dec == BeamTracer::kFar) okDec = !needOther;
      if (!okDec) ++violations;
    }
  }
  if (violations || cullViolations)
    fprintf(stderr, "    checked %d rays: %d decision violations, %d cull violations\n", checked, violations,
            cullViolations);
  CHECK(checked > 10000);
  CHECK(violations == 0);
  CHECK(cullViolations == 0);
}

TEST(lambert_formula_matches_quadrature) {
  Vec3 x(Real(0.2), 0, Real(-0.1)), n(0, 1, 0);
  Vec3 poly[4] = {Vec3(-0.5f, 2, -0.5f), Vec3(0.5f, 2, -0.5f), Vec3(0.7f, 2.3f, 0.5f), Vec3(-0.5f, 2.2f, 0.5f)};
  Real G = polygonFormFactorG(x, n, poly, 4);
  // midpoint quadrature over two triangles (0,1,2) and (0,2,3)
  double sum = 0;
  const int N = 300;
  for (int tri = 0; tri < 2; ++tri) {
    Vec3 A = poly[0], B = poly[tri == 0 ? 1 : 2], C = poly[tri == 0 ? 2 : 3];
    Vec3 nn = cross(B - A, C - A);
    double area = double(length(nn)) / 2;
    Vec3 nl = normalize(nn);
    int cnt = 0;
    double acc = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N - i; ++j) {
        double u = (i + 1.0 / 3) / N, v = (j + 1.0 / 3) / N;
        if (u + v > 1) continue;
        Vec3 p = A + (B - A) * Real(u) + (C - A) * Real(v);
        Vec3 w = p - x;
        double r2 = double(dot(w, w));
        Vec3 wn = w / std::sqrt(Real(r2));
        acc += double(dot(n, wn)) * std::fabs(double(dot(nl, wn))) / r2;
        ++cnt;
      }
    sum += acc / cnt * area;
  }
  CHECK_MSG(std::fabs(double(G) - sum) < 2e-3 * sum, "G %g quad %g", double(G), sum);
}
