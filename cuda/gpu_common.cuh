// Internal to the CUDA port: error checking, uploads and the renderer's
// device state, shared by gpu_render.cu (kernels of the first port, primary
// rays) and wavefront.cu (the wavefront engine).
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "beam/beam_types.h"
#include "cuda/gpu_render.h"

#define BT_CUDA_CHECK(x)                                                                   \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)

namespace bt {

template <class T>
T* upload(const T* data, size_t n) {
  T* d = nullptr;
  BT_CUDA_CHECK(cudaMalloc(&d, sizeof(T) * (n ? n : 1)));
  if (n) BT_CUDA_CHECK(cudaMemcpy(d, data, sizeof(T) * n, cudaMemcpyHostToDevice));
  return d;
}

struct WfBuffers;  // wavefront engine allocations (wavefront.cu)
WfBuffers* wfCreateBuffers();
void wfDestroyBuffers(WfBuffers* b);

struct GpuRenderer::Impl {
  SceneView sv;  // device pointers
  Vec3* dPos = nullptr;
  uint32_t* dTri = nullptr;
  Vec3* dTriN = nullptr;
  KdNode* dNodes = nullptr;
  uint32_t* dIdx = nullptr;
  uint32_t* dParent = nullptr;
  AABB* dNodeBox = nullptr;
  TriRef* dRefs = nullptr;
  WfBuffers* wf = nullptr;
};

// The wavefront engine (wavefront.cu).
struct WfConfig {
  int budget = 128;            // live sub-beams per root
  size_t capacity = 1u << 23;  // sub-beam records per queue
  int roundsPerCheck = 8;      // rounds launched between host checks for completion
};

struct WfRunInfo {
  double ms = 0;      // GPU time of the rounds and the final pass
  int rounds = 0;
  int killedRoots = 0;
  int maxQueue = 0;   // largest trace queue
  unsigned long long traceItems = 0, splitItems = 0;  // queue entries processed
  TraceStats stats;
};

// Exact soft shadows for the pixels of a W x H image whose primary hits
// (triangle, point, normal) are in device memory. Writes vis / E / Eunocc and
// sets over[pixel] for pixels the engine abandoned (to be recomputed on the
// CPU).
void wfSoftShadows(GpuRenderer::Impl& I, const WfConfig& cfg, const AreaLight& L, Real offset, int W, int H,
                   const int* dTri, const Vec3* dX, const Vec3* dN, bool exact, float* dVis, float* dE, float* dEu,
                   unsigned char* dOver, WfRunInfo& info);

// Primary visibility, one root per tile x tile pixels: the visible polygons
// (image plane) of all tiles except the abandoned ones (`killedTiles`, row
// order), which the caller recomputes on the CPU.
void wfPrimary(GpuRenderer::Impl& I, const WfConfig& cfg, const Camera& cam, bool cull, int tile,
               std::vector<OutBeam>& hits, std::vector<int>& killedTiles, WfRunInfo& info);

// Point-light shadows, one root per primary hit beam (in device memory):
// the shadowed polygons (image plane), except for the abandoned beams
// (`killed`).
void wfPointShadows(GpuRenderer::Impl& I, const WfConfig& cfg, const Camera& cam, const Vec3& light,
                    const OutBeam* dPrim, int nPrim, std::vector<OutBeam>& shadowPolys, std::vector<int>& killed,
                    WfRunInfo& info);

}  // namespace bt
