// 2D convex-polygon geometry for beam cross-sections (host and device).
//
// A beam's corner-ray directions are points q = (qx, qy) on a plane
// (paper Sec. 3.1). Triangle edges, depth comparisons and the pre-split by
// direction sign all become 2D lines L(q) = a*qx + b*qy + c on that plane.
#pragma once

#include "core/vec.h"

namespace bt {

constexpr int kMaxPoly = 8;

// Convex polygon with room for N vertices (push ignores overflow; splitPoly
// reports it).
template <int N>
struct PolyN {
  Real x[N], y[N];
  int n = 0;
  BT_HD void clear() { n = 0; }
  BT_HD void push(Real px, Real py) {
    if (n < N) {
      x[n] = px;
      y[n] = py;
      ++n;
    }
  }
  // Signed area (positive = counter-clockwise in (qx, qy)).
  BT_HD Real area() const {
    Real a = 0;
    for (int i = 0; i < n; ++i) {
      int j = i + 1 == n ? 0 : i + 1;
      a += x[i] * y[j] - x[j] * y[i];
    }
    return a * Real(0.5);
  }
  BT_HD void reverse() {
    for (int i = 0, j = n - 1; i < j; ++i, --j) {
      Real tx = x[i]; x[i] = x[j]; x[j] = tx;
      Real ty = y[i]; y[i] = y[j]; y[j] = ty;
    }
  }
  BT_HD Real extent() const {
    Real lx = kInf, hx = -kInf, ly = kInf, hy = -kInf;
    for (int i = 0; i < n; ++i) {
      lx = x[i] < lx ? x[i] : lx; hx = x[i] > hx ? x[i] : hx;
      ly = y[i] < ly ? y[i] : ly; hy = y[i] > hy ? y[i] : hy;
    }
    Real ex = hx - lx, ey = hy - ly;
    return ex > ey ? ex : ey;
  }
};

using Poly2 = PolyN<kMaxPoly>;

// Oriented line; L(q) > 0 is the "inside" half-plane. Normalized so that
// (a, b) is a unit vector unless the line is degenerate (a = b = 0, constant c).
struct Line2 {
  Real a = 0, b = 0, c = 0;
  BT_HD Real eval(Real x, Real y) const { return a * x + b * y + c; }
};

// Normalizes (a, b, c). `ext` is the beam extent; a line farther than
// ~1e9 extents away (or exactly parallel) becomes the constant sign(c).
BT_HD inline Line2 makeLine(Real a, Real b, Real c, Real ext) {
  Line2 L;
  Real nrm = std::sqrt(a * a + b * b);
  if (!(nrm > 0) || nrm * ext * Real(1e9) < std::fabs(c)) {
    L.c = c > 0 ? Real(1) : (c < 0 ? Real(-1) : Real(0));
    return L;
  }
  L.a = a / nrm;
  L.b = b / nrm;
  L.c = c / nrm;
  return L;
}

// Splits convex `p` by line `L` with fuzzy tolerance `eps`: vertices with
// |L| <= eps are "on" and go to both pieces; crossing points are computed
// from the lexicographically smaller endpoint so neighbours agree bitwise.
// Returns false if a piece needed more than N vertices.
template <int N>
BT_HD inline bool splitPoly(const PolyN<N>& p, const Line2& L, Real eps, PolyN<N>& in, PolyN<N>& out) {
  in.clear();
  out.clear();
  Real d[N];
  int cls[N];
  int nin = 0, nout = 0;
  for (int i = 0; i < p.n; ++i) {
    d[i] = L.eval(p.x[i], p.y[i]);
    cls[i] = d[i] > eps ? 1 : (d[i] < -eps ? -1 : 0);
  }
  for (int i = 0; i < p.n; ++i) {
    int j = i + 1 == p.n ? 0 : i + 1;
    nin += (cls[i] >= 0) + (cls[i] * cls[j] < 0);
    nout += (cls[i] <= 0) + (cls[i] * cls[j] < 0);
  }
  if (nin > N || nout > N) return false;
  for (int i = 0; i < p.n; ++i) {
    int j = i + 1 == p.n ? 0 : i + 1;
    if (cls[i] >= 0) in.push(p.x[i], p.y[i]);
    if (cls[i] <= 0) out.push(p.x[i], p.y[i]);
    if (cls[i] * cls[j] < 0) {
      bool iFirst = p.x[i] < p.x[j] || (p.x[i] == p.x[j] && p.y[i] < p.y[j]);
      int a = iFirst ? i : j, b = iFirst ? j : i;
      Real t = d[a] / (d[a] - d[b]);
      Real X = p.x[a] + t * (p.x[b] - p.x[a]);
      Real Y = p.y[a] + t * (p.y[b] - p.y[a]);
      in.push(X, Y);
      out.push(X, Y);
    }
  }
  return true;
}

// Welds vertices that are (numerically) duplicate or collinear, within a
// tolerance far below the fuzzy epsilon so that welding never opens visible
// gaps between neighbouring beams. (Splitting never creates near-duplicates:
// crossing points are only computed between vertices strictly farther than
// eps from the line.) Leaves n = 0 if the polygon collapses.
template <int N>
BT_HD inline void cleanPoly(PolyN<N>& p, Real epsSplit) {
  const Real eps = epsSplit * Real(1e-3);
  bool changed = true;
  while (changed && p.n >= 3) {
    changed = false;
    for (int i = 0; i < p.n && p.n >= 3; ++i) {
      int h = i == 0 ? p.n - 1 : i - 1;
      int j = i + 1 == p.n ? 0 : i + 1;
      Real ex = p.x[j] - p.x[h], ey = p.y[j] - p.y[h];
      Real dx = p.x[i] - p.x[h], dy = p.y[i] - p.y[h];
      Real len = std::sqrt(ex * ex + ey * ey);
      Real dist2 = dx * dx + dy * dy;
      // distance of vertex i from the chord h->j
      Real cr = std::fabs(ex * dy - ey * dx);
      bool dup = dist2 <= eps * eps;
      bool collinear = cr <= eps * len;
      if (dup || collinear) {
        for (int k = i; k + 1 < p.n; ++k) {
          p.x[k] = p.x[k + 1];
          p.y[k] = p.y[k + 1];
        }
        --p.n;
        changed = true;
        --i;
      }
    }
  }
  if (p.n < 3) p.n = 0;
}

}  // namespace bt
