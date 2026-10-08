#include "beam/beam_tracer.h"

#include <algorithm>

namespace bt {

namespace {
constexpr size_t kCacheSize = size_t(1) << 14;
}

HostBeamStorage::HostBeamStorage(int numTris)
    : cache(kCacheSize), mail(size_t(numTris), 0), visStamp(size_t(numTris), 0) {
  for (auto& c : cache) {
    c.stamp = 0;
    c.tri = -1;
  }
}

void HostBeamStorage::beginTrace() {
  pool.clear();
  freeList.clear();
  work.clear();
  workMaxFrame.clear();
  frames.clear();
  lists[0].clear();
  lists[1].clear();
  evBase = nextEvent;
  evParent.clear();
  if (++traceId == 0) {
    std::fill(visStamp.begin(), visStamp.end(), 0u);
    traceId = 1;
  }
  overflow = false;
  outputs = 0;
}

void HostBeamStorage::output(const Poly2& p, int tri, Real area) {
  ++outputs;
  if (!out) return;
  if (tri >= 0)
    out->hitArea += double(area);
  else
    out->missArea += double(area);
  if (keepOutput) {
    OutBeam ob;
    ob.n = p.n;
    for (int i = 0; i < 4; ++i) {
      int k = i < p.n ? i : p.n - 1;
      ob.x[i] = p.x[k];
      ob.y[i] = p.y[k];
    }
    ob.tri = tri;
    ob.area = area;
    out->beams.push_back(ob);
  }
}

BeamTracer::BeamTracer(const Scene& scene, const KdTree& tree)
    : store_(scene.numTris()),
      core_(makeSceneView(scene, tree), store_),
      stats(core_.stats),
      useMailbox(core_.useMailbox),
      keepOutput(store_.keepOutput),
      orderEdges(core_.orderEdges),
      lastReason(core_.lastReason) {}

void BeamTracer::trace(const BeamQuery& query, const Poly2* roots, int numRoots, BeamOutput& out) {
  out.clear();
  store_.out = &out;
  core_.trace(query, roots, numRoots);
  store_.out = nullptr;
}

int BeamTracer::retest(const BeamQuery& q, Real ext, const OutBeam& ob, int tri, int* reason, Real* newArea) {
  BeamOutput dummy;
  store_.out = &dummy;
  int r = core_.retest(q, ext, ob, tri, reason, newArea);
  store_.out = nullptr;
  return r;
}

}  // namespace bt
