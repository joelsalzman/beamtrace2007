// One-ray-at-a-time kd-tree ray tracer: the paper's comparison baseline
// ("our own optimized one-at-a-time ray tracer", Sec. 4.1) and our
// correctness oracle. The traversal itself is in ray_core.h (host + device).
#pragma once

#include "accel/kdtree.h"
#include "ray/ray_core.h"
#include "scene/scene.h"
#include "util/stats.h"

namespace bt {

class RayTracer {
 public:
  RayTracer(const Scene& scene, const KdTree& tree) : sv_(makeSceneView(scene, tree)) {}

  // Nearest hit in (ray.tmin, ray.tmax). Coplanar ties go to the lower id.
  bool intersect(const Ray& ray, RayHit& hit, TraceStats& stats, bool cullBackfaces = false,
                 int excludeTri = -1) const {
    return rayTraverse<false>(sv_, ray, hit, stats, cullBackfaces, excludeTri);
  }
  // Any hit in (ray.tmin, ray.tmax), ignoring `excludeTri`.
  bool occluded(const Ray& ray, TraceStats& stats, int excludeTri = -1) const {
    RayHit h;
    return rayTraverse<true>(sv_, ray, h, stats, false, excludeTri);
  }
  // Ray-triangle test (Moller-Trumbore); returns t or +inf.
  Real hitTri(const Ray& ray, int tri, bool cullBackfaces) const { return rayHitTri(sv_, ray, tri, cullBackfaces); }

 private:
  SceneView sv_;
};

}  // namespace bt
