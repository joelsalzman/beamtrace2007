// Integration tests: beam tracing vs ray tracing, kd vs brute force,
// soft shadows vs closed form, point shadows vs shadow rays, mailboxing.
#include "tests/helpers.h"

using namespace bttest;

namespace {

#ifdef BT_REAL_DOUBLE
constexpr Real kDelta = Real(1e-6);
#else
constexpr Real kDelta = Real(2e-4);
#endif

void checkPrimary(const char* name, const Scene& scene, const View& view, const Vec3& up, bool cull,
                  int samples, bool singleLeaf = false) {
  KdTree tree;
  KdBuildParams kp;
  kp.singleLeaf = singleLeaf;
  tree.build(scene, kp);
  Camera cam = Camera::make(view, up, 256, 256);
  PrimaryCompare pc = comparePrimary(scene, tree, cam, cull, samples, 1234, kDelta);
  bool ok = pc.mismatches == 0 && pc.holes == 0 && pc.overlaps == 0 && pc.areaError < 1e-3;
  if (!ok) reportCompare(name, pc);
  CHECK_MSG(pc.mismatches == 0, "%s", name);
  CHECK_MSG(pc.holes == 0, "%s", name);
  CHECK_MSG(pc.overlaps == 0, "%s", name);
  CHECK_MSG(pc.areaError < 1e-3, "%s area error %g", name, pc.areaError);
}

}  // namespace

TEST(primary_two_triangles_single_leaf) {
  Scene s;
  uint16_t m = s.addMaterial(Vec3(1, 1, 1));
  addQuad(s, Vec3(-5, 0, -5), Vec3(-5, 0, 5), Vec3(5, 0, 5), Vec3(5, 0, -5), m);
  uint32_t a = s.addVertex(Vec3(-1, 1, 0)), b = s.addVertex(Vec3(1, 1, 0)), c = s.addVertex(Vec3(0, 1, 1));
  s.addTri(a, b, c, m);
  s.finalize();
  View v;
  v.eye = Vec3(0, 4, 3);
  v.target = Vec3(0, 0, 0);
  v.fovY = 60;
  checkPrimary("two_tris_single", s, v, Vec3(0, 1, 0), false, 4000, true);
  checkPrimary("two_tris_kd", s, v, Vec3(0, 1, 0), false, 4000, false);
}

TEST(primary_room_single_leaf) {
  Scene s = makeScene("mesh procedural room\n");
  View v;
  v.eye = Vec3(4 + 2.8f * 0.7071f, 1.9f, 4 + 2.8f * 0.7071f);
  v.target = Vec3(4, 1.2f, 4);
  v.fovY = 70;
  checkPrimary("room_single", s, v, Vec3(0, 1, 0), false, 20000, true);
}

TEST(primary_room_kd) {
  Scene s = makeScene("mesh procedural room\n");
  View v;
  v.eye = Vec3(4 + 2.8f * 0.7071f, 1.9f, 4 + 2.8f * 0.7071f);
  v.target = Vec3(4, 1.2f, 4);
  v.fovY = 70;
  checkPrimary("room_kd", s, v, Vec3(0, 1, 0), false, 20000, false);
}

TEST(primary_random_interpenetrating) {
  // Random triangle soups (lots of interpenetration), cameras outside and
  // inside the cloud (triangles behind and straddling the apex), cull on/off.
  Rng rng(5);
  for (int k = 0; k < 6; ++k) {
    std::string cfg = "mesh procedural random_tris n=" + std::to_string(40 + 40 * k) + " seed=" + std::to_string(k + 1) + "\n";
    Scene s = makeScene(cfg);
    View v;
    if (k % 2 == 0) {
      v.eye = Vec3(Real(rng.uniform(-2, 3)), Real(rng.uniform(1.5, 3)), Real(rng.uniform(-2, 3)));
    } else {
      v.eye = Vec3(Real(rng.uniform(0.3, 0.7)), Real(rng.uniform(0.3, 0.7)), Real(rng.uniform(0.3, 0.7)));
    }
    v.target = Vec3(0.5f, 0.5f, 0.5f) + Vec3(Real(rng.uniform(-0.2, 0.2)), 0, Real(rng.uniform(-0.2, 0.2)));
    v.fovY = 75;
    char name[64];
    snprintf(name, sizeof name, "random_%d_kd", k);
    checkPrimary(name, s, v, Vec3(0, 1, 0), k >= 4, 6000, false);
    snprintf(name, sizeof name, "random_%d_single", k);
    checkPrimary(name, s, v, Vec3(0, 1, 0), k >= 4, 3000, true);
  }
}

TEST(primary_coplanar_and_axis_aligned) {
  // Overlapping coplanar quads + boxes on a floor; camera exactly on kd split planes.
  Scene s;
  uint16_t m = s.addMaterial(Vec3(1, 1, 1));
  addQuad(s, Vec3(-3, 0, -3), Vec3(-3, 0, 3), Vec3(3, 0, 3), Vec3(3, 0, -3), m);
  addQuad(s, Vec3(-1, 0, -1), Vec3(-1, 0, 2), Vec3(2, 0, 2), Vec3(2, 0, -1), m);  // coplanar with floor
  addBox(s, Vec3(-1, 0, -1), Vec3(0, 1, 0), m);
  addBox(s, Vec3(0.5f, 0, 0.5f), Vec3(1.5f, 0.5f, 1.5f), m);
  addBox(s, Vec3(-2, 0, 1), Vec3(-1.5f, 2, 1.5f), m);
  s.finalize();
  KdTree tree;
  tree.build(s);
  View v;
  v.eye = Vec3(2.5f, 2, 3);
  v.target = Vec3(0, 0.3f, 0);
  v.fovY = 60;
  checkPrimary("coplanar", s, v, Vec3(0, 1, 0), false, 8000);
  // Camera on the root split plane and on box faces.
  const KdNode& root = tree.nodes[0];
  if (!root.isLeaf()) {
    View v2 = v;
    v2.eye[root.axis()] = root.split;
    checkPrimary("camera_on_split", s, v2, Vec3(0, 1, 0), false, 8000);
  }
  View v3 = v;
  v3.eye = Vec3(0.5f, 0.5f, 3);  // on planes x = 0.5 and y = 0.5 (box faces)
  v3.target = Vec3(0.5f, 0.5f, 0);
  checkPrimary("camera_on_faces", s, v3, Vec3(0, 1, 0), false, 8000);
}

namespace {

// Pixel is "interior" if it and its 8 neighbours carry the same id.
bool uniform3x3(const SampleBuffer& sb, int x, int y) {
  int id = sb.tri[sb.idx(x, y, 0)];
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= sb.W || yy >= sb.H) return false;
      if (sb.tri[sb.idx(xx, yy, 0)] != id) return false;
    }
  return true;
}

}  // namespace

TEST(primary_no_cracks_on_closed_mesh) {
  // A closed sphere over an empty background: no pixel inside the silhouette
  // may see the background (a crack between adjacent triangles).
  Scene s = makeScene("mesh procedural sphere stacks=60\n");
  View v;
  v.eye = Vec3(0.3f, 0.4f, 3);
  v.target = Vec3(0, 0, 0);
  v.fovY = 50;
  KdTree tree;
  tree.build(s);
  Camera cam = Camera::make(v, Vec3(0, 1, 0), 512, 512);
  BeamTracer btr(s, tree);
  BeamOutput out;
  RenderStats rs;
  beamPrimary(btr, cam, false, out, rs);
  SampleBuffer beamBuf, rayBuf;
  beamBuf.init(512, 512, 1);
  rayBuf.init(512, 512, 1);
  rasterize(out.beams, cam, beamBuf, false);
  RayTracer rt(s, tree);
  rayPrimary(rt, cam, false, rayBuf, rs);
  int inside = 0, cracks = 0, differ = 0;
  for (int y = 0; y < 512; ++y)
    for (int x = 0; x < 512; ++x) {
      int r = rayBuf.tri[rayBuf.idx(x, y, 0)];
      if (r < 0) continue;
      bool hitAll = true;
      for (int dy = -1; dy <= 1 && hitAll; ++dy)
        for (int dx = -1; dx <= 1 && hitAll; ++dx) {
          int xx = std::min(511, std::max(0, x + dx)), yy = std::min(511, std::max(0, y + dy));
          hitAll = rayBuf.tri[rayBuf.idx(xx, yy, 0)] >= 0;
        }
      if (!hitAll) continue;
      ++inside;
      int b = beamBuf.tri[beamBuf.idx(x, y, 0)];
      if (b < 0) ++cracks;
      else if (b != r && uniform3x3(beamBuf, x, y) && uniform3x3(rayBuf, x, y)) ++differ;
    }
  CHECK(inside > 50000);
  CHECK_MSG(cracks == 0, "%d crack pixels", cracks);
  CHECK_MSG(differ == 0, "%d interior pixels differ", differ);
}

TEST(primary_idempotent_and_mailbox_invariant) {
  Scene s = makeScene("mesh procedural random_tris n=150 seed=11\n");
  KdTree tree;
  tree.build(s);
  View v;
  v.eye = Vec3(1.8f, 1.6f, 2.2f);
  v.target = Vec3(0.5f, 0.5f, 0.5f);
  v.fovY = 60;
  Camera cam = Camera::make(v, Vec3(0, 1, 0), 256, 256);
  BeamTracer btr(s, tree);
  BeamOutput a, b;
  RenderStats rs;
  beamPrimary(btr, cam, false, a, rs);
  btr.useMailbox = false;
  beamPrimary(btr, cam, false, b, rs);
  bool same = a.beams.size() == b.beams.size();
  for (size_t i = 0; same && i < a.beams.size(); ++i) {
    const OutBeam &x = a.beams[i], &y = b.beams[i];
    same = x.n == y.n && x.tri == y.tri;
    for (int k = 0; same && k < x.n; ++k) same = x.x[k] == y.x[k] && x.y[k] == y.y[k];
  }
  CHECK_MSG(same, "mailbox changed the output (%zu vs %zu beams)", a.beams.size(), b.beams.size());
  // Re-test every final beam against every triangle: nothing may change.
  BeamQuery q;
  q.plane = cam.plane();
  q.mode = BeamMode::Nearest;
  int changed = 0;
  for (const OutBeam& ob : a.beams)
    for (int t = 0; t < s.numTris(); ++t) {
      int reason;
      Real newArea = 0;
      // Changes within the fuzzy band (new area <= 16 eps^2) are rounding noise.
      const Real eps = kEpsRel * cam.rootPoly().extent();
      if (btr.retest(q, cam.rootPoly().extent(), ob, t, &reason, &newArea) != 0 && newArea > 16 * eps * eps) {
        if (changed++ < 3)
          fprintf(stderr, "    beam (tri %d, area %g) changes against tri %d (reason %d, new area %g)\n", ob.tri,
                  double(ob.area), t, reason, double(newArea));
      }
    }
  CHECK(changed == 0);
}

TEST(primary_kd_matches_brute_force) {
  Scene s = makeScene("mesh procedural plant\n");
  KdTree kd, single;
  kd.build(s);
  KdBuildParams kp;
  kp.singleLeaf = true;
  single.build(s, kp);
  View v;
  v.eye = Vec3(1.5f, 1.4f, 2);
  v.target = Vec3(0, 0.8f, 0);
  v.fovY = 50;
  Camera cam = Camera::make(v, Vec3(0, 1, 0), 512, 512);
  BeamOutput a, b;
  BeamTracer t1(s, kd), t2(s, single);
  RenderStats rs;
  beamPrimary(t1, cam, false, a, rs);
  beamPrimary(t2, cam, false, b, rs);
  SampleBuffer ba, bb;
  ba.init(512, 512, 1);
  bb.init(512, 512, 1);
  rasterize(a.beams, cam, ba, false);
  rasterize(b.beams, cam, bb, false);
  int compared = 0, diff = 0;
  for (int y = 0; y < 512; ++y)
    for (int x = 0; x < 512; ++x) {
      if (!uniform3x3(ba, x, y) || !uniform3x3(bb, x, y)) continue;
      ++compared;
      diff += ba.tri[ba.idx(x, y, 0)] != bb.tri[bb.idx(x, y, 0)];
    }
  CHECK(compared > 100000);
  CHECK_MSG(diff == 0, "%d of %d pixels differ", diff, compared);
}

namespace {

// Visible fraction of a unit square light at height H from floor point (px, pz)
// with a square occluder (side occ, height h, centered at the origin).
double analyticVisibility(double px, double pz, double H, double h, double occ) {
  double k = H / h;
  double cx = px + (0 - px) * k, cz = pz + (0 - pz) * k, half = occ * k / 2;
  double ox = std::max(0.0, std::min(0.5, cx + half) - std::max(-0.5, cx - half));
  double oz = std::max(0.0, std::min(0.5, cz + half) - std::max(-0.5, cz - half));
  return 1 - ox * oz;
}

}  // namespace

TEST(soft_shadows_match_closed_form) {
  Scene s = makeScene("mesh procedural analytic h=1 occ=1\n");
  KdTree tree;
  tree.build(s);
  BeamTracer btr(s, tree);
  RayTracer rt(s, tree);
  AreaLight L;
  L.c = Vec3(0, 3, 0);
  L.U = Vec3(1, 0, 0);
  L.V = Vec3(0, 0, 1);
  Rng rng(17);
  double maxErr = 0, maxRayErr = 0;
  TraceStats st;
#ifdef BT_REAL_DOUBLE
  const Real offset = Real(1e-9);
  const double tol = 1e-7;
#else
  const Real offset = Real(1e-4);
  const double tol = 2e-3;
#endif
  for (int i = 0; i < 400; ++i) {
    double px = rng.uniform(-1.5, 1.5), pz = rng.uniform(-1.5, 1.5);
    Vec3 x(Real(px), 0, Real(pz));
    int floorTri = -1;
    for (int t = 0; t < s.numTris(); ++t)
      if (s.v(t, 0).y == 0 && s.v(t, 1).y == 0 && s.v(t, 2).y == 0) floorTri = t;  // either floor tri is fine
    Real V = beamLightVisibility(btr, x, Vec3(0, 1, 0), floorTri, L, offset);
    double ref = analyticVisibility(px, pz, 3 - double(offset), 1 - double(offset), 1);
    maxErr = std::max(maxErr, std::fabs(double(V) - ref));
    if (i % 8 == 0) {
      Real Vr = rayLightVisibility(rt, x, Vec3(0, 1, 0), floorTri, L, offset, 64 * 64, true, uint64_t(i), st);
      maxRayErr = std::max(maxRayErr, std::fabs(double(Vr) - ref));
    }
  }
  CHECK_MSG(maxErr < tol, "beam max error %g", maxErr);
  CHECK_MSG(maxRayErr < 0.01, "ray (4096 spp) max error %g", maxRayErr);
  // Exact integration of the unoccluded light equals Lambert's formula.
  Vec3 x(Real(1.4), 0, Real(1.4));
  Real G = 0;
  beamLightVisibility(btr, x, Vec3(0, 1, 0), -1, L, offset, &G);
  Vec3 quad[4] = {L.c - L.U * 0.5f - L.V * 0.5f, L.c + L.U * 0.5f - L.V * 0.5f, L.c + L.U * 0.5f + L.V * 0.5f,
                  L.c - L.U * 0.5f + L.V * 0.5f};
  Real G0 = polygonFormFactorG(x, Vec3(0, 1, 0), quad, 4);
  double V = analyticVisibility(1.4, 1.4, 3, 1, 1);
  CHECK_MSG(V == 1 ? std::fabs(double(G - G0)) < 1e-3 * double(G0) : true, "G %g vs %g", double(G), double(G0));
}

TEST(soft_shadows_match_dense_ray_sampling) {
  Scene s = makeScene("mesh procedural plant\n");
  KdTree tree;
  tree.build(s);
  BeamTracer btr(s, tree);
  RayTracer rt(s, tree);
  AreaLight L;
  L.c = Vec3(0.3f, 2.5f, 0.2f);
  L.U = Vec3(0.6f, 0, 0);
  L.V = Vec3(0, 0, 0.6f);
  Rng rng(23);
  double sumErr = 0, maxErr = 0;
  int n = 0;
  TraceStats st;
  const Real offset = length(s.bounds.diag()) * kEpsRel;
  for (int i = 0; i < 150; ++i) {
    // points on the floor and the back wall
    Vec3 x = i % 3 ? Vec3(Real(rng.uniform(-1.2, 1.2)), 0, Real(rng.uniform(-1.4, 1.4)))
                   : Vec3(Real(rng.uniform(-1.5, 1.5)), Real(rng.uniform(0.05, 2)), -1.5f);
    Vec3 nrm = i % 3 ? Vec3(0, 1, 0) : Vec3(0, 0, 1);
    Ray r;
    r.o = x + nrm * Real(0.5);
    r.d = -nrm;
    RayHit h;
    if (!rt.intersect(r, h, st)) continue;
    int tri = h.tri;
    x = r.o + r.d * h.t;
    Real Vb = beamLightVisibility(btr, x, nrm, tri, L, offset);
    Real Vr = rayLightVisibility(rt, x, nrm, tri, L, offset, 96 * 96, true, uint64_t(i), st);
    double e = std::fabs(double(Vb - Vr));
    sumErr += e;
    maxErr = std::max(maxErr, e);
    ++n;
  }
  CHECK(n > 100);
  CHECK_MSG(maxErr < 0.02, "max |beam - ray| %g", maxErr);
  CHECK_MSG(sumErr / n < 0.003, "mean |beam - ray| %g", sumErr / n);
}

TEST(point_shadows_match_shadow_rays) {
  Scene s = makeScene("mesh procedural room\n");
  KdTree tree;
  tree.build(s);
  View v;
  v.eye = Vec3(6.2f, 2.2f, 6.5f);
  v.target = Vec3(4, 0.8f, 3.5f);
  v.fovY = 70;
  Camera cam = Camera::make(v, Vec3(0, 1, 0), 256, 256);
  Vec3 light(5.5f, 2.6f, 2.5f);
  BeamTracer btr(s, tree);
  RayTracer rt(s, tree);
  BeamOutput prim;
  RenderStats rs;
  beamPrimary(btr, cam, false, prim, rs);
  std::vector<OutBeam> shadow;
  beamPointShadows(btr, s, cam, prim, light, shadow, rs);
  Rng rng(8);
  int compared = 0, bad = 0;
  TraceStats st;
  auto rayShadow = [&](Real qx, Real qy, int& tri) -> int {
    Ray r;
    r.o = cam.eye;
    r.d = cam.dir(qx, qy);
    RayHit h;
    if (!rt.intersect(r, h, st)) {
      tri = -1;
      return 0;
    }
    tri = h.tri;
    Vec3 X = r.o + r.d * h.t;
    Vec3 nT = s.triN[size_t(h.tri)];
    Vec3 P0 = s.v(h.tri, 0);
    if (!(dot(nT, cam.eye - P0) * dot(nT, light - P0) > 0)) return 1;
    Ray sr;
    sr.o = X;
    sr.d = light - X;
    sr.tmax = 1 - kEpsRel;
    return rt.occluded(sr, st, h.tri) ? 1 : 0;
  };
  for (int i = 0; i < 20000; ++i) {
    Real qx = Real(rng.uniform(-1, 1)) * cam.halfW, qy = Real(rng.uniform(-1, 1)) * cam.halfH;
    bool nearEdge = false;
    int inShadow = 0;
    for (const OutBeam& ob : shadow) {
      Real d = insideDist(ob, qx, qy);
      if (std::fabs(d) <= kDelta * 10) nearEdge = true;
      if (d > 0) inShadow = 1;
    }
    for (const OutBeam& ob : prim.beams)
      if (std::fabs(insideDist(ob, qx, qy)) <= kDelta * 10) nearEdge = true;
    if (nearEdge) continue;
    int tri;
    int rs0 = rayShadow(qx, qy, tri);
    if (tri < 0) continue;
    ++compared;
    if (rs0 != inShadow) {
      // allow disagreement only if a nearby ray agrees with the beams
      bool explained = false;
      for (int k = 0; k < 8 && !explained; ++k) {
        double ang = k * M_PI / 4;
        int t2;
        explained = rayShadow(qx + Real(std::cos(ang)) * kDelta * 20, qy + Real(std::sin(ang)) * kDelta * 20, t2) == inShadow;
      }
      if (!explained && bad++ < 3) fprintf(stderr, "    mismatch at %g %g beam %d ray %d\n", double(qx), double(qy), inShadow, rs0);
    }
  }
  CHECK(compared > 15000);
  CHECK_MSG(bad == 0, "%d of %d samples disagree", bad, compared);
}

TEST(kd_ray_traversal_matches_brute_force) {
  // Regression (Sponza): overlapping, nearly planar floor triangles whose
  // vertex heights differ by a few ulps. SAH then splits inside their tiny
  // z-extent, and clipping a near-planar triangle against such a z-plane is
  // ill-conditioned; every kd cell a triangle crosses must still list it.
  Scene s;
  uint16_t m = s.addMaterial(Vec3(1, 1, 1));
  Rng rng(31);
  const Real base = Real(2.521487);
  auto zjit = [&]() {
    Real z = base;
    int k = int(rng.next() % 5);
    for (int i = 0; i < k; ++i) z = std::nextafter(z, Real(10));
    return z;
  };
  for (int i = 0; i < 1500; ++i) {
    Real x = Real(rng.uniform(-3, 3)), y = Real(rng.uniform(-3, 3)), sz = Real(rng.uniform(0.05, 2.0));
    uint32_t a = s.addVertex(Vec3(x, y, zjit())), b = s.addVertex(Vec3(x + sz, y + Real(rng.uniform(-0.3, 0.3)) * sz, zjit())),
             c = s.addVertex(Vec3(x + Real(rng.uniform(-0.3, 0.3)) * sz, y + sz, zjit()));
    s.addTri(a, b, c, m);
  }
  s.finalize();
  KdTree tree;
  tree.build(s);
  RayTracer rt(s, tree);
  TraceStats st;
  int bad = 0, hits = 0;
  for (int i = 0; i < 15000; ++i) {
    Ray r;
    r.o = Vec3(Real(rng.uniform(-6, 6)), Real(rng.uniform(-6, 6)), Real(rng.uniform(3, 6)));
    Vec3 target(Real(rng.uniform(-3, 3)), Real(rng.uniform(-3, 3)), base);
    r.d = target - r.o;
    RayHit h;
    rt.intersect(r, h, st);
    Real best = kInf;
    for (int t = 0; t < s.numTris(); ++t) {
      Real th = rt.hitTri(r, t, false);
      if (th > 0 && th < best) best = th;
    }
    hits += std::isfinite(best);
    bool ok = std::isfinite(best) ? (h.tri >= 0 && std::fabs(h.t - best) <= Real(1e-5) * best) : h.tri < 0;
    if (!ok && bad++ < 3) fprintf(stderr, "    ray %d: kd t=%g brute t=%g\n", i, double(h.t), double(best));
  }
  CHECK(hits > 8000);
  CHECK_MSG(bad == 0, "%d rays missed their nearest triangle", bad);
}
