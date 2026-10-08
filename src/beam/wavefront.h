// Host reference for the wavefront beam tracer (the CUDA engine's schedule,
// run sequentially): sub-beams live in queues, and each round runs
//   trace  on every queued sub-beam (wfTrace), collecting those that need a clip,
//   split  on those (wfSplit), whose pieces form the next round's queue.
// Each root (an independent beam tree, as one BeamCore::trace call) may hold
// at most `budget` live sub-beams; a root that would exceed it, or overflow a
// fixed-size list, is abandoned ("killed") and reported, so the caller can
// recompute it with BeamCore. The GPU engine uses the same rule to bound its
// memory.
#pragma once

#include <vector>

#include "accel/kdtree.h"
#include "beam/beam_tracer.h"
#include "beam/wavefront_core.h"
#include "scene/scene.h"

namespace bt {

struct WfRootSpec {
  BeamQuery q;
  Poly2 poly;
};

struct WfRootResult {
  double hitArea = 0, missArea = 0, droppedArea = 0;
  bool killed = false;
  std::vector<OutBeam> beams;  // all output pieces (if keepOutput)
};

class WavefrontTracer {
 public:
  WavefrontTracer(const Scene& scene, const KdTree& tree);

  void trace(const std::vector<WfRootSpec>& roots, std::vector<WfRootResult>& results);

  int budget = 1 << 30;     // live sub-beams per root
  bool keepOutput = true;
  bool orderEdges = true;
  TraceStats stats;
  int rounds = 0;           // of the last trace()
  size_t maxQueue = 0;      // largest trace queue of the last trace()
  size_t maxLive = 0;       // largest number of live sub-beams of one root
  int maxClip = 0, maxPieces = 0;  // largest clip list / piece count of one split

 private:
  SceneView sv_;
};

}  // namespace bt
