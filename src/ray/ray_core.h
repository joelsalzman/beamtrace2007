// One-ray-at-a-time kd-tree traversal and ray-triangle test (host and device).
#pragma once

#include <limits>

#include "beam/beam_types.h"
#include "util/stats.h"

namespace bt {

struct Ray {
  Vec3 o, d;
  Real tmin = 0, tmax = kInf;
};

struct RayHit {
  int tri = -1;
  Real t = kInf;
};

// Moller-Trumbore; returns t or +inf.
BT_HD inline Real rayHitTri(const SceneView& sv, const Ray& ray, int tri, bool cull) {
  const uint32_t* T = sv.tri3 + 3 * size_t(tri);
  const Vec3& v0 = sv.pos[T[0]];
  Vec3 e1 = sv.pos[T[1]] - v0, e2 = sv.pos[T[2]] - v0;
  Vec3 pv = cross(ray.d, e2);
  Real det = dot(e1, pv);
  if (cull ? !(det > 0) : det == 0) return kInf;
  Real inv = 1 / det;
  Vec3 tv = ray.o - v0;
  Real u = dot(tv, pv) * inv;
  if (u < 0 || u > 1) return kInf;
  Vec3 qv = cross(tv, e1);
  Real v = dot(ray.d, qv) * inv;
  if (v < 0 || u + v > 1) return kInf;
  return dot(e2, qv) * inv;
}

// Front-to-back kd traversal. kAny: stop at the first hit in (tmin, tmax).
// Nearest hits break coplanar ties toward the lower triangle id (as the beam
// tracer does).
template <bool kAny>
BT_HD inline bool rayTraverse(const SceneView& sv, const Ray& ray, RayHit& hit, TraceStats& stats, bool cull,
                              int excludeTri) {
  stats.rays++;
  Real t0 = ray.tmin, t1 = ray.tmax;
  Real inv[3];
  for (int a = 0; a < 3; ++a) {
    inv[a] = 1 / ray.d[a];
    if (ray.d[a] == 0) {
      if (ray.o[a] < sv.bounds.lo[a] || ray.o[a] > sv.bounds.hi[a]) return false;
      continue;
    }
    Real ta = (sv.bounds.lo[a] - ray.o[a]) * inv[a];
    Real tb = (sv.bounds.hi[a] - ray.o[a]) * inv[a];
    if (ta > tb) bswap(ta, tb);
    t0 = bmax(t0, ta);
    t1 = bmin(t1, tb);
  }
  if (t0 > t1) return false;

  struct Entry {
    uint32_t node;
    Real t0, t1;
  };
  Entry stack[64];
  int sp = 0;
  uint32_t node = 0;
  Real best = ray.tmax;
  int bestTri = -1;
  for (;;) {
    const KdNode& n = sv.nodes[node];
    if (!n.isLeaf()) {
      stats.kdSteps++;
      int a = n.axis();
      Real o = ray.o[a], d = ray.d[a];
      bool belowFirst = o < n.split || (o == n.split && d <= 0);
      uint32_t first = belowFirst ? node + 1 : n.right();
      uint32_t second = belowFirst ? n.right() : node + 1;
      if (d == 0) {
        node = first;
        continue;
      }
      Real tp = (n.split - o) * inv[a];
      if (tp > t1 || tp <= 0) {
        node = first;
      } else if (tp < t0) {
        node = second;
      } else {
        if (sp < 64) stack[sp++] = {second, tp, t1};
        node = first;
        t1 = tp;
      }
      continue;
    }
    stats.leafVisits++;
    const uint32_t* idx = sv.triIndices + n.offset;
    for (uint32_t i = 0; i < n.count(); ++i) {
      int t = int(idx[i]);
      if (t == excludeTri) continue;
      stats.triTests++;
      Real th = rayHitTri(sv, ray, t, cull && !kAny);
      if (!(th > ray.tmin && th < ray.tmax)) continue;
      if (kAny) {
        stats.hits++;
        hit.tri = t;
        hit.t = th;
        return true;
      }
      // Ties (coplanar overlaps) only within a few ulps: anything larger is a
      // genuinely closer surface.
      Real tol = bestTri < 0 ? Real(0) : best * (4 * std::numeric_limits<Real>::epsilon());
      if (bestTri < 0 || th < best - tol || (th <= best + tol && t < bestTri)) {
        best = th;
        bestTri = t;
      }
    }
    // Stop only if the hit lies inside this leaf: a hit slightly beyond it can
    // still be beaten by a closer surface in the next leaf.
    if (bestTri >= 0 && best <= t1) break;
    if (sp == 0) break;
    --sp;
    node = stack[sp].node;
    t0 = stack[sp].t0;
    t1 = stack[sp].t1;
  }
  if (bestTri < 0) return false;
  stats.hits++;
  hit.tri = bestTri;
  hit.t = best;
  return true;
}

}  // namespace bt
