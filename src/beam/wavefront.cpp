#include "beam/wavefront.h"

#include <algorithm>

namespace bt {

namespace {

struct HostOut {
  std::vector<WfRootResult>* res;
  bool keep;
  void emit(int root, const Poly2& p, int tri, Real area) {
    WfRootResult& r = (*res)[size_t(root)];
    if (tri >= 0)
      r.hitArea += double(area);
    else
      r.missArea += double(area);
    if (!keep) return;
    OutBeam ob;
    ob.n = p.n;
    for (int i = 0; i < 4; ++i) {
      int k = i < p.n ? i : p.n - 1;
      ob.x[i] = p.x[k];
      ob.y[i] = p.y[k];
    }
    ob.tri = tri;
    ob.area = area;
    r.beams.push_back(ob);
  }
  void dropped(int root, Real area) { (*res)[size_t(root)].droppedArea += double(area); }
};

struct HostPush {
  std::vector<WfBeam>* q;
  int* count;
  bool operator()(const WfBeam& r) {
    q->push_back(r);
    ++*count;
    return true;
  }
};

}  // namespace

WavefrontTracer::WavefrontTracer(const Scene& scene, const KdTree& tree) : sv_(makeSceneView(scene, tree)) {}

void WavefrontTracer::trace(const std::vector<WfRootSpec>& roots, std::vector<WfRootResult>& results) {
  const size_t R = roots.size();
  results.assign(R, WfRootResult());
  std::vector<BeamCtx> ctx(R);
  std::vector<int> live(R, 0);
  std::vector<char> killed(R, 0);
  HostOut out{&results, keepOutput};
  rounds = 0;
  maxQueue = 0;
  maxLive = 0;
  maxClip = maxPieces = 0;
  auto kill = [&](int root) {
    killed[size_t(root)] = 1;
    results[size_t(root)].killed = true;
  };

  std::vector<WfBeam> traceQ, splitQ, next;
  for (size_t i = 0; i < R; ++i) {
    BeamCtx& c = ctx[i];
    c.sv = sv_;
    c.setup(roots[i].q, roots[i].poly.extent());
    int pushed = 0;
    size_t before = traceQ.size();
    HostPush push{&traceQ, &pushed};
    bool ok = wfAdmit(c, roots[i].poly, int(i), out, push, stats);
    if (!ok || pushed > budget) {
      traceQ.resize(before);
      kill(int(i));
      continue;
    }
    live[i] = pushed;
  }

  WfClipSink sink;
  WfBeam pieces[kWfMaxPieces];
  while (!traceQ.empty()) {
    ++rounds;
    maxQueue = std::max(maxQueue, traceQ.size());
    splitQ.clear();
    for (WfBeam& r : traceQ) {
      const size_t root = size_t(r.root);
      if (killed[root]) {
        --live[root];
        continue;
      }
      WfWork w;
      wfExpand(ctx[root], r, w);
      if (wfTrace(ctx[root], r, w, out, stats) == kWfSplit)
        splitQ.push_back(r);
      else
        --live[root];
    }
    next.clear();
    for (const WfBeam& r : splitQ) {
      const size_t root = size_t(r.root);
      if (killed[root]) {
        --live[root];
        continue;
      }
      WfWork w;
      wfExpand(ctx[root], r, w);
      int np = 0;
      bool ok = wfSplit(ctx[root], r, w, orderEdges, sink, out, stats, pieces, np);
      maxClip = std::max(maxClip, std::max(sink.n[0], sink.n[1]));
      maxPieces = std::max(maxPieces, np);
      if (!ok || live[root] - 1 + np > budget) {
        --live[root];
        kill(int(root));
        continue;
      }
      live[root] += np - 1;
      maxLive = std::max(maxLive, size_t(live[root]));
      for (int k = 0; k < np; ++k) next.push_back(pieces[k]);
    }
    traceQ.swap(next);
  }
}

}  // namespace bt
