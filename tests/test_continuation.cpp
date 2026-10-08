// Continuations: the restart trail (one bit per pending far child plus kd
// parent links) must reproduce the paper's frame stack exactly: the same
// cells in the same order, hence bitwise-identical output and statistics.
#include "tests/helpers.h"

using namespace bttest;

namespace {

bool sameStats(const TraceStats& a, const TraceStats& b) {
  return a.kdSteps == b.kdSteps && a.leafVisits == b.leafVisits && a.triTests == b.triTests && a.hits == b.hits &&
         a.splits == b.splits && a.beams == b.beams && a.rootBeams == b.rootBeams &&
         a.presplitBeams == b.presplitBeams && a.mailboxSkips == b.mailboxSkips && a.fiveSplits == b.fiveSplits &&
         a.droppedArea == b.droppedArea;
}

bool sameBeams(const std::vector<OutBeam>& a, const std::vector<OutBeam>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    const OutBeam &x = a[i], &y = b[i];
    if (x.n != y.n || x.tri != y.tri || x.area != y.area) return false;
    for (int k = 0; k < x.n; ++k)
      if (x.x[k] != y.x[k] || x.y[k] != y.y[k]) return false;
  }
  return true;
}

struct Run {
  BeamOutput prim;
  std::vector<OutBeam> shadow;
  std::vector<Real> vis;
  TraceStats stats;
};

Run runAll(const Scene& s, const KdTree& tree, const Camera& cam, const Vec3& light, const AreaLight& L,
           bool trail, bool mailbox) {
  Run r;
  BeamTracer bt(s, tree);
  bt.useTrail = trail;
  bt.useMailbox = mailbox;
  RenderStats rs;
  beamPrimary(bt, cam, false, r.prim, rs);
  beamPointShadows(bt, s, cam, r.prim, light, r.shadow, rs);
  // soft shadows at the centres of the hit beams
  const BeamPlane P = cam.plane();
  for (size_t i = 0; i < r.prim.beams.size(); i += 7) {
    const OutBeam& ob = r.prim.beams[i];
    if (ob.tri < 0) continue;  // background
    Real cx = 0, cy = 0;
    for (int k = 0; k < ob.n; ++k) {
      cx += ob.x[k];
      cy += ob.y[k];
    }
    cx /= Real(ob.n);
    cy /= Real(ob.n);
    Vec3 d = P.dir(cx, cy);
    Vec3 nT = s.triN[size_t(ob.tri)];
    Real t = dot(nT, s.v(ob.tri, 0) - cam.eye) / dot(nT, d);
    Vec3 x = cam.eye + d * t;
    Vec3 ns = dot(nT, d) > 0 ? -nT : nT;
    r.vis.push_back(beamLightVisibility(bt, x, ns, ob.tri, L, Real(1e-4)));
  }
  r.stats = bt.stats;
  return r;
}

void compareModes(const char* name, const Scene& s, const View& v, const Vec3& light, const AreaLight& L) {
  KdTree tree;
  KdBuildParams kp;
  kp.costIntersect = Real(0.4);  // the beam tracer's tree
  tree.build(s, kp);
  Camera cam = Camera::make(v, Vec3(0, 1, 0), 128, 128);
  for (int mb = 0; mb < 2; ++mb) {
    Run a = runAll(s, tree, cam, light, L, false, mb != 0);
    Run b = runAll(s, tree, cam, light, L, true, mb != 0);
    CHECK_MSG(sameBeams(a.prim.beams, b.prim.beams), "%s mailbox %d: primary beams differ", name, mb);
    CHECK_MSG(sameBeams(a.shadow, b.shadow), "%s mailbox %d: shadow polygons differ", name, mb);
    CHECK_MSG(a.vis == b.vis, "%s mailbox %d: soft visibility differs", name, mb);
    CHECK_MSG(sameStats(a.stats, b.stats), "%s mailbox %d: statistics differ (kd %llu vs %llu)", name, mb,
              (unsigned long long)a.stats.kdSteps, (unsigned long long)b.stats.kdSteps);
    // (Pieces split in one leaf each climb its parent chain, so climbs can
    // exceed the descent steps; each climb is a single load.)
    CHECK_MSG(b.stats.climbs > 0 && a.stats.climbs == 0, "%s: climbs %llu / %llu", name,
              (unsigned long long)a.stats.climbs, (unsigned long long)b.stats.climbs);
    CHECK(a.prim.beams.size() > 50 && !a.vis.empty());
  }
}

}  // namespace

TEST(trail_matches_frame_stack_bitwise) {
  AreaLight L;
  {
    Scene s = makeScene("mesh procedural plant\n");
    View v;
    v.eye = Vec3(1.5f, 1.4f, 2);
    v.target = Vec3(0, 0.8f, 0);
    v.fovY = 50;
    L.c = Vec3(0.5f, 3.5f, 0.5f);
    L.U = Vec3(0.6f, 0, 0);
    L.V = Vec3(0, 0, 0.6f);
    compareModes("plant", s, v, Vec3(1, 3, 1), L);
  }
  {
    Scene s = makeScene("mesh procedural room\n");
    View v;
    v.eye = Vec3(6.2f, 2.2f, 6.5f);
    v.target = Vec3(4, 0.8f, 3.5f);
    v.fovY = 70;
    L.c = Vec3(5, 2.9f, 3);
    L.U = Vec3(0.8f, 0, 0);
    L.V = Vec3(0, 0, 0.8f);
    compareModes("room", s, v, Vec3(5.5f, 2.6f, 2.5f), L);
  }
  {
    Scene s = makeScene("mesh procedural random_tris n=150 seed=11\n");
    View v;
    v.eye = Vec3(1.8f, 1.6f, 2.2f);
    v.target = Vec3(0.5f, 0.5f, 0.5f);
    v.fovY = 60;
    L.c = Vec3(0.5f, 2.5f, 0.5f);
    L.U = Vec3(0.5f, 0, 0);
    L.V = Vec3(0, 0, 0.5f);
    compareModes("random_tris", s, v, Vec3(0.4f, 2, 0.6f), L);
  }
}
