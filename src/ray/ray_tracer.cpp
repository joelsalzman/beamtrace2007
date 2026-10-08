#include "ray/ray_tracer.h"

#include <algorithm>

namespace bt {

Real RayTracer::hitTri(const Ray& ray, int tri, bool cull) const {
  const auto& T = scene_.tris[size_t(tri)];
  const Vec3& v0 = scene_.pos[T[0]];
  Vec3 e1 = scene_.pos[T[1]] - v0, e2 = scene_.pos[T[2]] - v0;
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
  Real t = dot(e2, qv) * inv;
  return t;
}

template <bool kAny>
bool RayTracer::traverse(const Ray& ray, RayHit& hit, TraceStats& stats, bool cull, int excludeTri) const {
  stats.rays++;
  // Clip to the scene box.
  Real t0 = ray.tmin, t1 = ray.tmax;
  Real inv[3];
  for (int a = 0; a < 3; ++a) {
    inv[a] = 1 / ray.d[a];
    Real ta = (tree_.bounds.lo[a] - ray.o[a]) * inv[a];
    Real tb = (tree_.bounds.hi[a] - ray.o[a]) * inv[a];
    if (ta > tb) std::swap(ta, tb);
    if (ray.d[a] == 0) {
      if (ray.o[a] < tree_.bounds.lo[a] || ray.o[a] > tree_.bounds.hi[a]) return false;
      continue;
    }
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
  }
  if (t0 > t1) return false;

  struct Entry {
    uint32_t node;
    Real t0, t1;
  };
  Entry stack[96];
  int sp = 0;
  uint32_t node = 0;
  Real best = ray.tmax;
  int bestTri = -1;
  for (;;) {
    const KdNode& n = tree_.nodes[node];
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
        stack[sp++] = {second, tp, t1};
        node = first;
        t1 = tp;
      }
      continue;
    }
    stats.leafVisits++;
    const uint32_t* idx = tree_.triIndices.data() + n.offset;
    for (uint32_t i = 0; i < n.count(); ++i) {
      int t = int(idx[i]);
      if (t == excludeTri) continue;
      stats.triTests++;
      Real th = hitTri(ray, t, cull && !kAny);
      if (!(th > ray.tmin && th < ray.tmax)) continue;
      if (kAny) {
        stats.hits++;
        hit.tri = t;
        hit.t = th;
        return true;
      }
      // Coplanar tie-break by lower id, matching the beam tracer.
      Real tol = bestTri < 0 ? Real(0) : best * kEpsRel;
      if (bestTri < 0 || th < best - tol || (th <= best + tol && t < bestTri)) {
        best = th;
        bestTri = t;
      }
    }
    if (bestTri >= 0 && best <= t1 * (1 + kEpsRel)) break;
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

bool RayTracer::intersect(const Ray& ray, RayHit& hit, TraceStats& stats, bool cull, int excludeTri) const {
  return traverse<false>(ray, hit, stats, cull, excludeTri);
}

bool RayTracer::occluded(const Ray& ray, TraceStats& stats, int excludeTri) const {
  RayHit h;
  return traverse<true>(ray, h, stats, false, excludeTri);
}

}  // namespace bt
