// CUDA port of the beam tracer (host interface; no CUDA types here).
//
// Soft shadows run one thread per pixel: a ray-traced primary hit, then one
// exact shadow beam traced by the same BeamCore as the CPU tracer, with
// fixed-size per-thread storage. Pixels whose storage overflows are flagged
// and recomputed on the CPU, so the result is always the exact one.
#pragma once

#include <string>

#include "accel/kdtree.h"
#include "render/render_types.h"

namespace bt {

struct GpuInfo {
  std::string name;
  int smCount = 0;
  double memGB = 0;
};

// True if a CUDA device is usable; fills `info` if given.
bool gpuAvailable(GpuInfo* info = nullptr);

class GpuRenderer {
 public:
  GpuRenderer(const Scene& scene, const KdTree& tree);
  ~GpuRenderer();
  GpuRenderer(const GpuRenderer&) = delete;
  GpuRenderer& operator=(const GpuRenderer&) = delete;

  // Same contract as softShadows() in render.h (opt.useBeams must be true).
  // res.stats.primarySeconds = GPU primary rays; traceSeconds = GPU shadow
  // beams + CPU fallback.
  void softShadows(const Camera& cam, const AreaLight& light, const SoftOptions& opt, SoftResult& res);

  // Primary visibility: one root beam per `tile` x `tile` pixel tile, one tile
  // per thread. Returns the hit beams (image-plane polygons). Tiles whose
  // storage overflows are traced on the CPU.
  void primary(const Camera& cam, bool cull, std::vector<OutBeam>& hitBeams, RenderStats& rs, int tile = 16);

  // Point-light shadows for the given primary hit beams, one per thread.
  // Appends shadowed polygons (image plane) like beamPointShadows().
  void pointShadows(const Camera& cam, const std::vector<OutBeam>& primaryHits, const Vec3& light,
                    std::vector<OutBeam>& shadowPolys, RenderStats& rs);

  // Post Office mailboxing on the GPU (off by default: with per-thread
  // direct-mapped mailboxes in local memory it costs more than it saves).
  bool useMailbox = false;

  // Soft shadows: the wavefront engine (sub-beams in queues, cuda/wavefront.cu)
  // or, with useWavefront = false (environment BT_GPU_ENGINE=v1), the first
  // port's one thread per pixel.
  bool useWavefront = true;
  int wfBudget = 256;                     // live sub-beams per pixel / shadow beam before it goes to the CPU
  int wfPrimaryBudget = 4096;             // live sub-beams per primary tile
  size_t wfCapacity = size_t(1) << 22;    // sub-beams per queue (3 queues of 64 B records)
  int lastRounds = 0, lastMaxQueue = 0;

  int lastOverflowPixels = 0;     // pixels recomputed on the CPU
  double lastFallbackSeconds = 0;
  double lastKernelSeconds = 0;   // shadow-beam kernel only

 public:
  struct Impl;  // device state (cuda/gpu_common.cuh)

 private:
  Impl* impl_;
  const Scene& scene_;
  const KdTree& tree_;
};

}  // namespace bt
