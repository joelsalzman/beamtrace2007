// CPU beam tracer: BeamCore (beam_core.h) with growable host storage.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "accel/kdtree.h"
#include "beam/beam_core.h"
#include "scene/scene.h"
#include "util/stats.h"

namespace bt {

struct BeamOutput {
  std::vector<OutBeam> beams;
  double hitArea = 0, missArea = 0, droppedArea = 0;
  void clear() {
    beams.clear();
    hitArea = missArea = droppedArea = 0;
  }
};

// Storage policy for BeamCore on the host: std::vector everything, never
// overflows. Beam slots are recycled through a free list.
struct HostBeamStorage {
  explicit HostBeamStorage(int numTris);

  std::vector<CoreBeam> pool;
  std::vector<int> freeList;
  std::vector<CoreWork> work;
  std::vector<int> workMaxFrame;  // prefix maximum of pending work items' frames
  std::vector<CoreFrame> frames;
  std::vector<int> lists[2];
  std::vector<Poly2> polys[2];
  std::vector<CoreTriInfo> cache;
  std::vector<uint64_t> mail;
  std::vector<uint64_t> evParent;
  uint64_t evBase = 1, nextEvent = 1;
  std::vector<uint32_t> visStamp;
  uint32_t traceId = 0;
  BeamOutput* out = nullptr;
  bool keepOutput = true;
  bool overflow = false;
  int outputs = 0;

  void beginTrace();
  int allocBeam() {
    if (!freeList.empty()) {
      int i = freeList.back();
      freeList.pop_back();
      return i;
    }
    pool.emplace_back();
    return int(pool.size()) - 1;
  }
  CoreBeam& beam(int i) { return pool[size_t(i)]; }
  void freeBeam(int i) { freeList.push_back(i); }
  bool pushWork(const CoreWork& w) {
    workMaxFrame.push_back(work.empty() ? w.frame : std::max(w.frame, workMaxFrame.back()));
    work.push_back(w);
    return true;
  }
  bool popWork(CoreWork& w) {
    if (work.empty()) return false;
    w = work.back();
    work.pop_back();
    workMaxFrame.pop_back();
    return true;
  }
  int maxPendingFrame() const { return workMaxFrame.empty() ? -1 : workMaxFrame.back(); }
  void truncateFrames(int n) {
    if (n < int(frames.size())) frames.resize(size_t(n));
  }
  int pushFrame(const CoreFrame& f) {
    frames.push_back(f);
    return int(frames.size()) - 1;
  }
  const CoreFrame& frame(int i) const { return frames[size_t(i)]; }
  void listClear(int l) { lists[l].clear(); }
  bool listPush(int l, int v) {
    lists[l].push_back(v);
    return true;
  }
  int listSize(int l) const { return int(lists[l].size()); }
  int listAt(int l, int i) const { return lists[l][size_t(i)]; }
  void polyClear(int l) { polys[l].clear(); }
  bool polyPush(int l, const Poly2& p) {
    polys[l].push_back(p);
    return true;
  }
  int polySize(int l) const { return int(polys[l].size()); }
  const Poly2& polyAt(int l, int i) const { return polys[l][size_t(i)]; }
  CoreTriInfo& triSlot(int tri) { return cache[size_t(tri) & (cache.size() - 1)]; }
  uint64_t stamp(int tri) const { return mail[size_t(tri)]; }
  void setStamp(int tri, uint64_t e) { mail[size_t(tri)] = e; }
  uint64_t newEvent(uint64_t parent) {
    evParent.push_back(parent);
    return nextEvent++;
  }
  uint64_t parentOf(uint64_t e) const {
    return (e >= evBase && e < nextEvent) ? evParent[size_t(e - evBase)] : 0;
  }
  bool firstHit(int tri) {
    if (visStamp[size_t(tri)] == traceId) return false;
    visStamp[size_t(tri)] = traceId;
    return true;
  }
  void output(const Poly2& p, int tri, Real area);
  void dropped(Real area) {
    if (out) out->droppedArea += double(area);
  }
  int outputCount() const { return outputs; }
};

class BeamTracer {
 public:
  using Beam = CoreBeam;
  enum {
    kNear = BeamCore<HostBeamStorage>::kNear,
    kFar = BeamCore<HostBeamStorage>::kFar,
    kBoth = BeamCore<HostBeamStorage>::kBoth
  };

  BeamTracer(const Scene& scene, const KdTree& tree);

  // Traces the convex root polygons (in plane coordinates) through the scene.
  void trace(const BeamQuery& query, const Poly2* roots, int numRoots, BeamOutput& out);

 private:
  HostBeamStorage store_;
  BeamCore<HostBeamStorage> core_;

 public:
  TraceStats& stats;
  bool& useMailbox;
  bool& keepOutput;  // if false only areas are accumulated (soft shadows)
  bool& orderEdges;  // clip by the most-cutting triangle edge first
  int& lastReason;

  // ---- internals exposed for unit tests ----
  int decide(const Beam& b, int axis, Real Ns, const AABB& box) const { return core_.decide(b, axis, Ns, box); }
  bool overlaps(const Beam& b, const AABB& box) const { return core_.overlaps(b, box); }
  bool makeBeam(const Poly2& p, const int8_t sgn[3], int hit, Beam& b) { return core_.makeBeam(p, sgn, hit, b); }
  void setQueryForTest(const BeamQuery& q, Real ext) { core_.setupQuery(q, ext); }
  // Re-tests a final beam against one triangle (idempotency check). Returns
  // 0 if nothing would change, 1 if the beam's status would change as a
  // whole, 2 if it would be split. `reason` receives the deciding step.
  int retest(const BeamQuery& q, Real ext, const OutBeam& ob, int tri, int* reason = nullptr,
             Real* newArea = nullptr);
};

}  // namespace bt
