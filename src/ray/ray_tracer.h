// One-ray-at-a-time kd-tree ray tracer: the paper's comparison baseline
// ("our own optimized one-at-a-time ray tracer", Sec. 4.1) and our
// correctness oracle.
#pragma once

#include "accel/kdtree.h"
#include "scene/scene.h"
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

class RayTracer {
 public:
  RayTracer(const Scene& scene, const KdTree& tree) : scene_(scene), tree_(tree) {}

  // Nearest hit in (ray.tmin, ray.tmax). Coplanar ties go to the lower id.
  bool intersect(const Ray& ray, RayHit& hit, TraceStats& stats, bool cullBackfaces = false,
                 int excludeTri = -1) const;
  // Any hit in (ray.tmin, ray.tmax), ignoring `excludeTri`.
  bool occluded(const Ray& ray, TraceStats& stats, int excludeTri = -1) const;
  // Ray-triangle test (Moller-Trumbore); returns t or +inf.
  Real hitTri(const Ray& ray, int tri, bool cullBackfaces) const;

 private:
  template <bool kAny>
  bool traverse(const Ray& ray, RayHit& hit, TraceStats& stats, bool cull, int excludeTri) const;

  const Scene& scene_;
  const KdTree& tree_;
};

}  // namespace bt
