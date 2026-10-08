// Prints hashes of the CPU beam tracer's exact outputs and statistics for a
// set of scenes. Used to check that refactorings of the tracer change nothing:
// run before and after, then diff. (Hashes depend on the compiler and flags.)
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "render/render.h"
#include "scene/scene.h"

using namespace bt;

namespace {

struct Hash {
  uint64_t h = 1469598103934665603ull;
  void bytes(const void* p, size_t n) {
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= c[i];
      h *= 1099511628211ull;
    }
  }
  template <class T>
  void add(const T& v) {
    bytes(&v, sizeof(T));
  }
};

void addStats(Hash& h, const TraceStats& s) {
  h.add(s.kdSteps);
  h.add(s.leafVisits);
  h.add(s.triTests);
  h.add(s.hits);
  h.add(s.splits);
  h.add(s.beams);
  h.add(s.rootBeams);
  h.add(s.presplitBeams);
  h.add(s.mailboxSkips);
  h.add(s.fiveSplits);
  h.add(s.droppedArea);
}

// Order-independent hash of a polygon list (sorted by their own hashes).
uint64_t polysHash(const std::vector<OutBeam>& v) {
  std::vector<uint64_t> hs;
  hs.reserve(v.size());
  for (const OutBeam& ob : v) {
    Hash h;
    h.add(ob.n);
    h.add(ob.tri);
    for (int i = 0; i < ob.n; ++i) {
      h.add(ob.x[i]);
      h.add(ob.y[i]);
    }
    hs.push_back(h.h);
  }
  std::sort(hs.begin(), hs.end());
  Hash h;
  for (uint64_t x : hs) h.add(x);
  return h.h;
}

void report(const char* scene, const char* what, uint64_t out, const TraceStats& s) {
  Hash hs;
  addStats(hs, s);
  printf("%-12s %-22s out %016llx stats %016llx  kd %llu tests %llu splits %llu beams %llu\n", scene, what,
         (unsigned long long)out, (unsigned long long)hs.h, (unsigned long long)s.kdSteps,
         (unsigned long long)s.triTests, (unsigned long long)s.splits, (unsigned long long)s.beams);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: bt_golden CONFIG...\n");
    return 2;
  }
  for (int i = 1; i < argc; ++i) {
    SceneConfig cfg;
    std::string err;
    Scene scene;
    if (!loadConfig(argv[i], cfg, err) || !buildScene(cfg, scene, err)) {
      fprintf(stderr, "%s: %s\n", argv[i], err.c_str());
      return 1;
    }
    KdTree tree;
    KdBuildParams kp;
    kp.costIntersect = Real(0.4);
    tree.build(scene, kp);
    const char* name = cfg.name.c_str();
    Camera cam = Camera::make(cfg.views[0], cfg.up, 256, 256);
    BeamTracer bt(scene, tree);
    for (int mb = 1; mb >= 0; --mb) {
      bt.useMailbox = mb != 0;
      BeamOutput prim;
      RenderStats rs;
      beamPrimary(bt, cam, cfg.cullBackfaces, prim, rs);
      report(name, mb ? "primary" : "primary nomail", polysHash(prim.beams), rs.trace);
      if (cfg.hasPointLight) {
        std::vector<OutBeam> sp;
        RenderStats rp;
        beamPointShadows(bt, scene, cam, prim, cfg.pointLight, sp, rp);
        report(name, mb ? "point" : "point nomail", polysHash(sp), rp.trace);
      }
    }
    if (cfg.hasAreaLight) {
      Camera sc = Camera::make(cfg.views[0], cfg.up, 96, 96);
      AreaLight L;
      L.c = cfg.lightC;
      L.U = cfg.lightU;
      L.V = cfg.lightV;
      for (int c = 0; c < 3; ++c) {
        SoftOptions so;
        so.threads = 1;
        so.mailbox = c != 1;
        so.exact = c == 2;
        SoftResult res;
        softShadows(scene, tree, sc, L, so, res);
        Hash h;
        h.bytes(res.vis.data(), res.vis.size() * sizeof(float));
        h.bytes(res.E.data(), res.E.size() * sizeof(float));
        report(name, c == 0 ? "soft" : (c == 1 ? "soft nomail" : "soft exact"), h.h, res.stats.trace);
      }
    }
  }
  return 0;
}
