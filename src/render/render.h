// Camera, CPU rasterization of beam cross-sections (the paper used the GPU
// rasterizer for this step) and the three rendering pipelines:
//   primary visibility, point-light hard shadows, area-light soft shadows,
// each with a beam-traced and a ray-traced implementation.
#pragma once

#include <string>
#include <vector>

#include "beam/beam_tracer.h"
#include "ray/ray_tracer.h"
#include "render/render_types.h"
#include "util/image.h"

namespace bt {

// Per-sample visibility buffer.
struct SampleBuffer {
  int W = 0, H = 0, S = 1;
  std::vector<Vec2> offs;      // sample offsets within a pixel
  std::vector<int32_t> tri;    // visible triangle per sample (-1 background)
  std::vector<uint8_t> flag;   // 1 = in shadow
  void init(int W, int H, int aa);
  size_t idx(int x, int y, int s) const { return (size_t(y) * size_t(W) + size_t(x)) * size_t(S) + size_t(s); }
};

// Rasterizes convex polygons given in image-plane coordinates. With
// `shadowPass`, sets flag = 1 for covered samples; otherwise writes tri ids.
void rasterize(const std::vector<OutBeam>& polys, const Camera& cam, SampleBuffer& sb, bool shadowPass);
// Draws polygon outlines (wireframe beams, paper Fig. 9 right).
void drawOutlines(const std::vector<OutBeam>& polys, const Camera& cam, Image& img, float r, float g, float b);

// Diffuse shading of a sample buffer with one point light (paper Sec. 4:
// "diffuse shading and a single point light at the camera").
Image shadeSamples(const Scene& scene, const Camera& cam, const SampleBuffer& sb, const Vec3& light,
                   const Vec3& background);

// ---- Primary visibility ----
void beamPrimary(BeamTracer& bt, const Camera& cam, bool cull, BeamOutput& out, RenderStats& rs);
void rayPrimary(const RayTracer& rt, const Camera& cam, bool cull, SampleBuffer& sb, RenderStats& rs);

// ---- Point-light hard shadows ----
// Converts primary hit beams into shadowed polygons (image-plane coordinates).
void beamPointShadows(BeamTracer& bt, const Scene& scene, const Camera& cam, const BeamOutput& primary,
                      const Vec3& light, std::vector<OutBeam>& shadowPolys, RenderStats& rs);
void rayPointShadows(const RayTracer& rt, const Scene& scene, const Camera& cam, const Vec3& light,
                     SampleBuffer& sb, RenderStats& rs);

// ---- Area-light soft shadows (paper Sec. 5) ----


void softShadows(const Scene& scene, const KdTree& tree, const Camera& cam, const AreaLight& light,
                 const SoftOptions& opt, SoftResult& res);
Image shadeSoft(const Scene& scene, const SoftResult& res, Real exposure, const Vec3& background);
// 99th percentile of the unoccluded term, used as a shared exposure.
Real softExposure(const SoftResult& res);

// Exact visibility of the light from a point (beam tracer), used by tests.
// Returns the visible fraction; optionally the exact integral of
// cos(theta_i) cos(theta_l) / r^2 over the visible part.
Real beamLightVisibility(BeamTracer& bt, const Vec3& x, const Vec3& ns, int tri, const AreaLight& light,
                         Real offset, Real* exactG = nullptr);
Real rayLightVisibility(const RayTracer& rt, const Vec3& x, const Vec3& ns, int tri, const AreaLight& light,
                        Real offset, int n, bool jitter, uint64_t seed, TraceStats& stats,
                        Real* exactG = nullptr);

}  // namespace bt
