// Camera, CPU rasterization of beam cross-sections (the paper used the GPU
// rasterizer for this step) and the three rendering pipelines:
//   primary visibility, point-light hard shadows, area-light soft shadows,
// each with a beam-traced and a ray-traced implementation.
#pragma once

#include <string>
#include <vector>

#include "beam/beam_tracer.h"
#include "ray/ray_tracer.h"
#include "util/image.h"

namespace bt {

struct Camera {
  Vec3 eye, fwd, right, down;
  Real halfW = 1, halfH = 1;
  int W = 1, H = 1;
  static Camera make(const View& v, const Vec3& up, int W, int H);
  // Image plane at distance 1 along fwd; q in [-halfW, halfW] x [-halfH, halfH].
  BeamPlane plane() const { return BeamPlane::make(eye, eye + fwd, right, down); }
  Vec3 dir(Real qx, Real qy) const { return fwd + right * qx + down * qy; }
  void pixelToQ(Real px, Real py, Real& qx, Real& qy) const {
    qx = (2 * px / Real(W) - 1) * halfW;
    qy = (2 * py / Real(H) - 1) * halfH;
  }
  void qToPixel(Real qx, Real qy, Real& px, Real& py) const {
    px = (qx / halfW + 1) * Real(0.5) * Real(W);
    py = (qy / halfH + 1) * Real(0.5) * Real(H);
  }
  Poly2 rootPoly() const {
    Poly2 p;
    p.push(-halfW, -halfH);
    p.push(halfW, -halfH);
    p.push(halfW, halfH);
    p.push(-halfW, halfH);
    return p;
  }
};

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

struct RenderStats {
  TraceStats trace;
  double traceSeconds = 0;    // visibility computation (what the paper times)
  double primarySeconds = 0;  // soft shadows: ray-traced primary visibility
  double rasterSeconds = 0;   // CPU raster + shading
  int visibleTris = 0;
  int hitBeams = 0;
};

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
struct AreaLight {
  Vec3 c, U, V;  // center and full edge vectors; emits along normalize(U x V)
  Vec3 n() const { return normalize(cross(U, V)); }
  Real area() const { return length(cross(U, V)); }
};

struct SoftOptions {
  bool useBeams = true;
  int samples = 256;        // rays: shadow rays per pixel (rounded to a square)
  bool exact = false;       // integrate cos/r^2 over the visible light exactly instead of eq. 2
  bool jitter = true;
  int threads = 1;
  bool mailbox = true;
};

struct SoftResult {
  int W = 0, H = 0;
  std::vector<float> vis;     // visible fraction of the light, -1 for background
  std::vector<float> E;       // unscaled irradiance-like term
  std::vector<float> Eunocc;  // same with V = 1 (for exposure)
  std::vector<int> tri;       // primary hit
  RenderStats stats;
};

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
// Integral of cos(theta_i) cos(theta_l) / r^2 over a planar polygon (Lambert's formula).
Real polygonFormFactorG(const Vec3& x, const Vec3& n, const Vec3* poly, int count);

}  // namespace bt
