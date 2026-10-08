// The wavefront engine (sub-beams in queues, processed in rounds) must give
// exactly the pieces of the depth-first BeamCore: per root, the same multiset
// of output polygons bit for bit and the same counters. Roots that exceed
// their sub-beam budget must be reported, never silently wrong.
#include "beam/wavefront.h"
#include "render/point_core.h"
#include "tests/helpers.h"

using namespace bttest;

namespace {

bool beamLess(const OutBeam& a, const OutBeam& b) {
  if (a.tri != b.tri) return a.tri < b.tri;
  if (a.n != b.n) return a.n < b.n;
  for (int k = 0; k < a.n; ++k) {
    if (a.x[k] != b.x[k]) return a.x[k] < b.x[k];
    if (a.y[k] != b.y[k]) return a.y[k] < b.y[k];
  }
  return a.area < b.area;
}

bool sameMultiset(std::vector<OutBeam> a, std::vector<OutBeam> b) {
  if (a.size() != b.size()) return false;
  std::sort(a.begin(), a.end(), beamLess);
  std::sort(b.begin(), b.end(), beamLess);
  for (size_t i = 0; i < a.size(); ++i) {
    const OutBeam &x = a[i], &y = b[i];
    if (x.n != y.n || x.tri != y.tri || x.area != y.area) return false;
    for (int k = 0; k < x.n; ++k)
      if (x.x[k] != y.x[k] || x.y[k] != y.y[k]) return false;
  }
  return true;
}

bool close(double a, double b) { return std::fabs(a - b) <= 1e-9 * std::max(1.0, std::max(std::fabs(a), std::fabs(b))); }

// Compares the wavefront engine with BeamCore (restart trail, no mailbox),
// tracing every root on its own.
void compareEngines(const char* name, const Scene& s, const KdTree& tree, const std::vector<WfRootSpec>& roots) {
  BeamTracer bt(s, tree);
  bt.useMailbox = false;
  bt.useTrail = true;
  std::vector<BeamOutput> ref(roots.size());
  for (size_t i = 0; i < roots.size(); ++i) bt.trace(roots[i].q, &roots[i].poly, 1, ref[i]);
  WavefrontTracer wf(s, tree);
  std::vector<WfRootResult> res;
  wf.trace(roots, res);
  int bad = 0, killed = 0;
  for (size_t i = 0; i < roots.size(); ++i) {
    killed += res[i].killed;
    bool ok = sameMultiset(ref[i].beams, res[i].beams) && close(ref[i].hitArea, res[i].hitArea) &&
              close(ref[i].missArea, res[i].missArea) && close(ref[i].droppedArea, res[i].droppedArea);
    if (!ok && bad++ < 3)
      fprintf(stderr, "    %s root %zu: %zu vs %zu pieces, miss %.17g vs %.17g\n", name, i, ref[i].beams.size(),
              res[i].beams.size(), ref[i].missArea, res[i].missArea);
  }
  const TraceStats &a = bt.stats, &b = wf.stats;
  CHECK_MSG(killed == 0, "%s: %d roots killed", name, killed);
  CHECK_MSG(bad == 0, "%s: %d of %zu roots differ", name, bad, roots.size());
  CHECK_MSG(a.kdSteps == b.kdSteps && a.leafVisits == b.leafVisits && a.triTests == b.triTests,
            "%s: kd %llu/%llu leaves %llu/%llu tests %llu/%llu", name, (unsigned long long)a.kdSteps,
            (unsigned long long)b.kdSteps, (unsigned long long)a.leafVisits, (unsigned long long)b.leafVisits,
            (unsigned long long)a.triTests, (unsigned long long)b.triTests);
  CHECK_MSG(a.hits == b.hits && a.splits == b.splits && a.beams == b.beams && a.rootBeams == b.rootBeams &&
                a.presplitBeams == b.presplitBeams && a.fiveSplits == b.fiveSplits && a.climbs == b.climbs,
            "%s: hits %llu/%llu splits %llu/%llu beams %llu/%llu five %llu/%llu climbs %llu/%llu", name,
            (unsigned long long)a.hits, (unsigned long long)b.hits, (unsigned long long)a.splits,
            (unsigned long long)b.splits, (unsigned long long)a.beams, (unsigned long long)b.beams,
            (unsigned long long)a.fiveSplits, (unsigned long long)b.fiveSplits, (unsigned long long)a.climbs,
            (unsigned long long)b.climbs);
  CHECK_MSG(close(a.droppedArea, b.droppedArea), "%s: dropped area %g vs %g", name, a.droppedArea, b.droppedArea);
  CHECK(wf.rounds > 2);
}

struct Case {
  const char* name;
  Scene scene;
  View view;
  Vec3 light;
  AreaLight area;
};

std::vector<Case> cases() {
  std::vector<Case> cs(3);
  cs[0].name = "plant";
  cs[0].scene = makeScene("mesh procedural plant\n");
  cs[0].view.eye = Vec3(1.5f, 1.4f, 2);
  cs[0].view.target = Vec3(0, 0.8f, 0);
  cs[0].view.fovY = 50;
  cs[0].light = Vec3(1, 3, 1);
  cs[0].area.c = Vec3(0.5f, 3.5f, 0.5f);
  cs[0].area.U = Vec3(0.6f, 0, 0);
  cs[0].area.V = Vec3(0, 0, 0.6f);
  cs[1].name = "room";
  cs[1].scene = makeScene("mesh procedural room\n");
  cs[1].view.eye = Vec3(6.2f, 2.2f, 6.5f);
  cs[1].view.target = Vec3(4, 0.8f, 3.5f);
  cs[1].view.fovY = 70;
  cs[1].light = Vec3(5.5f, 2.6f, 2.5f);
  cs[1].area.c = Vec3(5, 2.9f, 3);
  cs[1].area.U = Vec3(0.8f, 0, 0);
  cs[1].area.V = Vec3(0, 0, 0.8f);
  cs[2].name = "random_tris";
  cs[2].scene = makeScene("mesh procedural random_tris n=150 seed=11\n");
  cs[2].view.eye = Vec3(1.8f, 1.6f, 2.2f);
  cs[2].view.target = Vec3(0.5f, 0.5f, 0.5f);
  cs[2].view.fovY = 60;
  cs[2].light = Vec3(0.4f, 2, 0.6f);
  cs[2].area.c = Vec3(0.5f, 2.5f, 0.5f);
  cs[2].area.U = Vec3(0.5f, 0, 0);
  cs[2].area.V = Vec3(0, 0, 0.5f);
  return cs;
}

// Primary hit beams of the whole image (BeamCore), for building shadow roots.
std::vector<OutBeam> primaryHits(const Scene& s, const KdTree& tree, const Camera& cam) {
  BeamTracer bt(s, tree);
  BeamOutput out;
  RenderStats rs;
  beamPrimary(bt, cam, false, out, rs);
  std::vector<OutBeam> hits;
  for (const OutBeam& ob : out.beams)
    if (ob.tri >= 0) hits.push_back(ob);
  return hits;
}

std::vector<WfRootSpec> softRoots(const Scene& s, const Camera& cam, const std::vector<OutBeam>& hits,
                                  const AreaLight& L, int stride) {
  std::vector<WfRootSpec> roots;
  const BeamPlane P = cam.plane();
  for (size_t i = 0; i < hits.size(); i += size_t(stride)) {
    const OutBeam& ob = hits[i];
    Real cx = 0, cy = 0;
    for (int k = 0; k < ob.n; ++k) {
      cx += ob.x[k];
      cy += ob.y[k];
    }
    cx /= Real(ob.n);
    cy /= Real(ob.n);
    Vec3 d = P.dir(cx, cy);
    Vec3 nT = s.triN[size_t(ob.tri)];
    Vec3 x = cam.eye + d * (dot(nT, s.v(ob.tri, 0) - cam.eye) / dot(nT, d));
    Vec3 ns = dot(nT, d) > 0 ? -nT : nT;
    WfRootSpec r;
    Real lightArea;
    if (lightBeamSetup(L, x, ns, ob.tri, Real(1e-4), r.q, r.poly, lightArea)) roots.push_back(r);
  }
  return roots;
}

std::vector<WfRootSpec> pointRoots(const Scene& s, const Camera& cam, const std::vector<OutBeam>& hits,
                                   const Vec3& light) {
  std::vector<WfRootSpec> roots;
  for (const OutBeam& pb : hits) {
    WfRootSpec r;
    if (pointShadowSetup(cam.eye, cam.fwd, cam.right, cam.down, s.triN[size_t(pb.tri)], s.v(pb.tri, 0), pb.tri,
                         light, pb.x, pb.y, pb.n, r.q, r.poly))
      roots.push_back(r);
  }
  return roots;
}

std::vector<WfRootSpec> primaryRoots(const Camera& cam, int tiles) {
  std::vector<WfRootSpec> roots;
  WfRootSpec r;
  r.q.plane = cam.plane();
  r.q.mode = BeamMode::Nearest;
  for (int ty = 0; ty < tiles; ++ty)
    for (int tx = 0; tx < tiles; ++tx) {
      Real x0 = (Real(2 * tx) / Real(tiles) - 1) * cam.halfW, x1 = (Real(2 * tx + 2) / Real(tiles) - 1) * cam.halfW;
      Real y0 = (Real(2 * ty) / Real(tiles) - 1) * cam.halfH, y1 = (Real(2 * ty + 2) / Real(tiles) - 1) * cam.halfH;
      r.poly.clear();
      r.poly.push(x0, y0);
      r.poly.push(x1, y0);
      r.poly.push(x1, y1);
      r.poly.push(x0, y1);
      roots.push_back(r);
    }
  return roots;
}

}  // namespace

TEST(wavefront_matches_beamcore_bitwise) {
  for (Case& c : cases()) {
    KdTree tree;
    KdBuildParams kp;
    kp.costIntersect = Real(0.4);
    tree.build(c.scene, kp);
    Camera cam = Camera::make(c.view, Vec3(0, 1, 0), 128, 128);
    std::string n = c.name;
    compareEngines((n + " primary 1 root").c_str(), c.scene, tree, primaryRoots(cam, 1));
    compareEngines((n + " primary 4x4 tiles").c_str(), c.scene, tree, primaryRoots(cam, 4));
    std::vector<OutBeam> hits = primaryHits(c.scene, tree, cam);
    compareEngines((n + " point shadows").c_str(), c.scene, tree, pointRoots(c.scene, cam, hits, c.light));
    compareEngines((n + " soft shadows").c_str(), c.scene, tree, softRoots(c.scene, cam, hits, c.area, 3));
  }
}

TEST(wavefront_budget_kills_roots_and_keeps_the_rest_exact) {
  Case c = cases()[0];  // plant: many sub-beams per soft-shadow root
  KdTree tree;
  KdBuildParams kp;
  kp.costIntersect = Real(0.4);
  tree.build(c.scene, kp);
  Camera cam = Camera::make(c.view, Vec3(0, 1, 0), 128, 128);
  std::vector<WfRootSpec> roots = softRoots(c.scene, cam, primaryHits(c.scene, tree, cam), c.area, 2);
  BeamTracer bt(c.scene, tree);
  bt.useMailbox = false;
  WavefrontTracer wf(c.scene, tree);
  wf.budget = 4;
  std::vector<WfRootResult> res;
  wf.trace(roots, res);
  int killed = 0, bad = 0;
  for (size_t i = 0; i < roots.size(); ++i) {
    if (res[i].killed) {
      ++killed;
      continue;
    }
    BeamOutput ref;
    bt.trace(roots[i].q, &roots[i].poly, 1, ref);
    bad += !sameMultiset(ref.beams, res[i].beams);
  }
  CHECK_MSG(killed > 0 && killed < int(roots.size()), "%d of %zu roots killed", killed, roots.size());
  CHECK_MSG(bad == 0, "%d surviving roots differ", bad);
  CHECK(wf.maxLive <= 4);
}
