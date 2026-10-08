// Render-side plain types shared by the CPU pipelines and the CUDA port.
// (No dependency on the beam tracer's SIMD types.)
#pragma once

#include <vector>

#include "accel/kdtree.h"
#include "beam/beam_types.h"
#include "render/soft_core.h"
#include "scene/scene.h"
#include "util/stats.h"

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

struct RenderStats {
  TraceStats trace;
  double traceSeconds = 0;    // visibility computation (what the paper times)
  double primarySeconds = 0;  // soft shadows: ray-traced primary visibility
  double rasterSeconds = 0;   // CPU raster + shading
  int visibleTris = 0;
  int hitBeams = 0;
};

struct SoftOptions {
  bool useBeams = true;
  int samples = 256;        // rays: shadow rays per pixel (rounded to a square)
  bool exact = false;       // integrate cos/r^2 over the visible light exactly instead of eq. 2
  bool jitter = true;
  int threads = 1;
  bool mailbox = true;
  bool trail = false;       // beams: restart-trail continuation instead of the frame stack
};

struct SoftResult {
  int W = 0, H = 0;
  std::vector<float> vis;     // visible fraction of the light, -1 for background
  std::vector<float> E;       // unscaled irradiance-like term
  std::vector<float> Eunocc;  // same with V = 1 (for exposure)
  std::vector<int> tri;       // primary hit
  RenderStats stats;
};

// CPU soft-shadow beams for a list of pixels (used for the CUDA port's
// overflow fallback): fills res.vis / res.E at those pixels.
void softShadowPixelsCPU(const Scene& scene, const KdTree& tree, const AreaLight& light, const SoftOptions& opt,
                         const std::vector<size_t>& pixels, const std::vector<Vec3>& X, const std::vector<Vec3>& N,
                         SoftResult& res, TraceStats& stats);

// CPU primary visibility for a list of root polygons (image plane); appends
// the hit beams. Used for the CUDA port's overflow fallback.
void primaryRootsCPU(const Scene& scene, const KdTree& tree, const Camera& cam, bool cull,
                     const std::vector<Poly2>& roots, std::vector<OutBeam>& hitBeams, TraceStats& stats);
// CPU point-light shadows for a list of primary hit beams (overflow fallback).
void pointShadowsCPU(const Scene& scene, const KdTree& tree, const Camera& cam, const std::vector<OutBeam>& primaryHits,
                     const Vec3& light, std::vector<OutBeam>& shadowPolys, TraceStats& stats);

}  // namespace bt
