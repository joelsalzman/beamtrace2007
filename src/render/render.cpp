#include "render/render.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <unordered_set>

#include "util/misc.h"

namespace bt {

Camera Camera::make(const View& v, const Vec3& up, int W, int H) {
  Camera c;
  c.eye = v.eye;
  c.fwd = normalize(v.target - v.eye);
  c.right = normalize(cross(c.fwd, up));
  c.down = cross(c.fwd, c.right);
  c.W = W;
  c.H = H;
  c.halfH = Real(std::tan(double(v.fovY) * M_PI / 360.0));
  c.halfW = c.halfH * Real(W) / Real(H);
  return c;
}

void SampleBuffer::init(int W_, int H_, int aa) {
  W = W_;
  H = H_;
  offs.clear();
  if (aa <= 1) {
    offs.push_back(Vec2(0.5f, 0.5f));
  } else if (aa == 4) {
    const Real p[4][2] = {{0.375f, 0.125f}, {0.875f, 0.375f}, {0.125f, 0.625f}, {0.625f, 0.875f}};
    for (auto& q : p) offs.push_back(Vec2(q[0], q[1]));
  } else if (aa == 6) {
    const int perm[6] = {3, 0, 4, 1, 5, 2};  // 6-rooks pattern
    for (int i = 0; i < 6; ++i) offs.push_back(Vec2((Real(i) + 0.5f) / 6, (Real(perm[i]) + 0.5f) / 6));
  } else {
    int k = std::max(2, int(std::lround(std::sqrt(double(aa)))));
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < k; ++i) offs.push_back(Vec2((Real(i) + 0.5f) / Real(k), (Real(j) + 0.5f) / Real(k)));
  }
  S = int(offs.size());
  tri.assign(size_t(W) * size_t(H) * size_t(S), -1);
  flag.assign(tri.size(), 0);
}

void rasterize(const std::vector<OutBeam>& polys, const Camera& cam, SampleBuffer& sb, bool shadowPass) {
  for (const OutBeam& ob : polys) {
    if (!shadowPass && ob.tri < 0) continue;
    // Edge half-planes A x + B y + C >= 0 in pixel coordinates (polygons are
    // counter-clockwise in q, and q -> pixel scales both axes positively).
    double px[4], py[4], A[4], B[4], C[4];
    double ly = 1e300, hy = -1e300;
    for (int i = 0; i < ob.n; ++i) {
      Real x, y;
      cam.qToPixel(ob.x[i], ob.y[i], x, y);
      px[i] = double(x);
      py[i] = double(y);
      ly = std::min(ly, py[i]);
      hy = std::max(hy, py[i]);
    }
    for (int i = 0; i < ob.n; ++i) {
      int j = i + 1 == ob.n ? 0 : i + 1;
      A[i] = -(py[j] - py[i]);
      B[i] = px[j] - px[i];
      C[i] = -(A[i] * px[i] + B[i] * py[i]);
    }
    for (int s = 0; s < sb.S; ++s) {
      const double ox = double(sb.offs[size_t(s)].x), oy = double(sb.offs[size_t(s)].y);
      int y0 = std::max(0, int(std::ceil(ly - oy))), y1 = std::min(sb.H - 1, int(std::floor(hy - oy)));
      for (int y = y0; y <= y1; ++y) {
        const double sy = y + oy;
        double xl = -1e300, xr = 1e300;
        bool empty = false;
        for (int i = 0; i < ob.n && !empty; ++i) {
          double r = B[i] * sy + C[i];  // need A x + r >= 0
          if (A[i] > 0) xl = std::max(xl, -r / A[i]);
          else if (A[i] < 0) xr = std::min(xr, -r / A[i]);
          else if (r < 0) empty = true;
        }
        if (empty || xl > xr) continue;
        int x0 = std::max(0, int(std::ceil(xl - ox))), x1 = std::min(sb.W - 1, int(std::floor(xr - ox)));
        for (int x = x0; x <= x1; ++x) {
          size_t k = sb.idx(x, y, s);
          if (shadowPass)
            sb.flag[k] = 1;
          else
            sb.tri[k] = ob.tri;
        }
      }
    }
  }
}

void drawOutlines(const std::vector<OutBeam>& polys, const Camera& cam, Image& img, float r, float g, float b) {
  for (const OutBeam& ob : polys) {
    for (int i = 0; i < ob.n; ++i) {
      int j = i + 1 == ob.n ? 0 : i + 1;
      Real ax, ay, bx, by;
      cam.qToPixel(ob.x[i], ob.y[i], ax, ay);
      cam.qToPixel(ob.x[j], ob.y[j], bx, by);
      int steps = int(std::ceil(std::max(std::fabs(bx - ax), std::fabs(by - ay)))) + 1;
      for (int k = 0; k <= steps; ++k) {
        Real t = Real(k) / Real(steps);
        int x = int(std::floor(ax + (bx - ax) * t)), y = int(std::floor(ay + (by - ay) * t));
        if (x >= 0 && y >= 0 && x < img.w && y < img.h) img.set(x, y, r, g, b);
      }
    }
  }
}

Image shadeSamples(const Scene& scene, const Camera& cam, const SampleBuffer& sb, const Vec3& light,
                   const Vec3& bg) {
  Image img(sb.W, sb.H);
  for (int y = 0; y < sb.H; ++y) {
    for (int x = 0; x < sb.W; ++x) {
      float acc[3] = {0, 0, 0};
      for (int s = 0; s < sb.S; ++s) {
        size_t k = sb.idx(x, y, s);
        int t = sb.tri[k];
        if (t < 0) {
          acc[0] += float(bg.x);
          acc[1] += float(bg.y);
          acc[2] += float(bg.z);
          continue;
        }
        Real qx, qy;
        cam.pixelToQ(Real(x) + sb.offs[size_t(s)].x, Real(y) + sb.offs[size_t(s)].y, qx, qy);
        Vec3 d = cam.dir(qx, qy);
        Vec3 n = scene.triN[size_t(t)];
        Real den = dot(n, d);
        Vec3 X = den != 0 ? cam.eye + d * (dot(n, scene.v(t, 0) - cam.eye) / den) : cam.eye;
        if (den > 0) n = -n;
        Real diff = std::max(Real(0), dot(n, normalize(light - X)));
        if (sb.flag[k]) diff = 0;
        const Vec3& alb = scene.matColor[scene.triMat[size_t(t)]];
        float sh = float(0.15 + 0.85 * diff);
        acc[0] += float(alb.x) * sh;
        acc[1] += float(alb.y) * sh;
        acc[2] += float(alb.z) * sh;
      }
      img.set(x, y, acc[0] / float(sb.S), acc[1] / float(sb.S), acc[2] / float(sb.S));
    }
  }
  return img;
}

void beamPrimary(BeamTracer& bt, const Camera& cam, bool cull, BeamOutput& out, RenderStats& rs) {
  BeamQuery q;
  q.plane = cam.plane();
  q.mode = BeamMode::Nearest;
  q.cullBackfaces = cull;
  Poly2 root = cam.rootPoly();
  TraceStats before = bt.stats;
  Timer timer;
  bt.trace(q, &root, 1, out);
  rs.traceSeconds += timer.seconds();
  TraceStats d = bt.stats;
  d.kdSteps -= before.kdSteps;
  d.leafVisits -= before.leafVisits;
  d.triTests -= before.triTests;
  d.hits -= before.hits;
  d.splits -= before.splits;
  d.beams -= before.beams;
  d.rootBeams -= before.rootBeams;
  d.presplitBeams -= before.presplitBeams;
  d.mailboxSkips -= before.mailboxSkips;
  d.fiveSplits -= before.fiveSplits;
  d.visibleTris -= before.visibleTris;
  d.droppedArea -= before.droppedArea;
  rs.trace.add(d);
  std::unordered_set<int> vis;
  int hb = 0;
  for (const OutBeam& ob : out.beams)
    if (ob.tri >= 0) {
      vis.insert(ob.tri);
      ++hb;
    }
  rs.visibleTris = int(vis.size());
  rs.hitBeams = hb;
}

void rayPrimary(const RayTracer& rt, const Camera& cam, bool cull, SampleBuffer& sb, RenderStats& rs) {
  Timer timer;
  std::unordered_set<int> vis;
  for (int y = 0; y < sb.H; ++y) {
    for (int x = 0; x < sb.W; ++x) {
      for (int s = 0; s < sb.S; ++s) {
        Real qx, qy;
        cam.pixelToQ(Real(x) + sb.offs[size_t(s)].x, Real(y) + sb.offs[size_t(s)].y, qx, qy);
        Ray r;
        r.o = cam.eye;
        r.d = cam.dir(qx, qy);
        r.tmin = 0;
        RayHit h;
        rt.intersect(r, h, rs.trace, cull);
        sb.tri[sb.idx(x, y, s)] = h.tri;
      }
    }
  }
  rs.traceSeconds += timer.seconds();
  for (int t : sb.tri)
    if (t >= 0) vis.insert(t);
  rs.visibleTris = int(vis.size());
}

void beamPointShadows(BeamTracer& bt, const Scene& scene, const Camera& cam, const BeamOutput& primary,
                      const Vec3& light, std::vector<OutBeam>& shadowPolys, RenderStats& rs) {
  Timer timer;
  BeamPlane camPlane = cam.plane();
  BeamOutput out;
  TraceStats before = bt.stats;
  for (const OutBeam& pb : primary.beams) {
    if (pb.tri < 0) continue;
    const int T = pb.tri;
    const Vec3 nT = scene.triN[size_t(T)];
    const Vec3 P0 = scene.v(T, 0);
    Real sideCam = dot(nT, cam.eye - P0), sideL = dot(nT, light - P0);
    if (!(sideCam * sideL > 0)) {  // receiver faces away from the light: fully shadowed
      shadowPolys.push_back(pb);
      continue;
    }
    // Back-project the beam's corners onto the receiver plane (a homography).
    Vec3 X[4];
    for (int i = 0; i < pb.n; ++i) {
      Vec3 d = cam.dir(pb.x[i], pb.y[i]);
      X[i] = cam.eye + d * (dot(nT, P0 - cam.eye) / dot(nT, d));
    }
    Vec3 tu = anyOrthogonal(nT);
    BeamQuery q;
    q.plane = BeamPlane::make(light, P0, tu, cross(nT, tu));
    q.mode = BeamMode::AnyHit;
    q.farIsPlane = true;
    q.excludeTri = T;
    Poly2 root;
    for (int i = 0; i < pb.n; ++i) {
      Real qx, qy;
      q.plane.coords(X[i], qx, qy);
      root.push(qx, qy);
    }
    bt.trace(q, &root, 1, out);
    for (const OutBeam& sb : out.beams) {
      if (sb.tri < 0) continue;
      OutBeam ip;
      ip.n = sb.n;
      ip.tri = T;
      ip.area = 0;
      bool ok = true;
      for (int i = 0; i < sb.n; ++i) ok = ok && camPlane.project(q.plane.point(sb.x[i], sb.y[i]), ip.x[i], ip.y[i]);
      if (!ok) continue;
      // Keep counter-clockwise orientation in image coordinates.
      Poly2 p;
      for (int i = 0; i < ip.n; ++i) p.push(ip.x[i], ip.y[i]);
      if (!(std::fabs(p.area()) > 0)) continue;  // receiver seen exactly edge-on
      if (p.area() < 0) {
        p.reverse();
        for (int i = 0; i < ip.n; ++i) {
          ip.x[i] = p.x[i];
          ip.y[i] = p.y[i];
        }
      }
      shadowPolys.push_back(ip);
    }
  }
  rs.traceSeconds += timer.seconds();
  TraceStats d = bt.stats;
  d.kdSteps -= before.kdSteps;
  d.leafVisits -= before.leafVisits;
  d.triTests -= before.triTests;
  d.hits -= before.hits;
  d.splits -= before.splits;
  d.beams -= before.beams;
  d.rootBeams -= before.rootBeams;
  d.presplitBeams -= before.presplitBeams;
  d.mailboxSkips -= before.mailboxSkips;
  d.fiveSplits -= before.fiveSplits;
  d.visibleTris -= before.visibleTris;
  d.droppedArea -= before.droppedArea;
  rs.trace.add(d);
}

void rayPointShadows(const RayTracer& rt, const Scene& scene, const Camera& cam, const Vec3& light,
                     SampleBuffer& sb, RenderStats& rs) {
  Timer timer;
  for (int y = 0; y < sb.H; ++y)
    for (int x = 0; x < sb.W; ++x)
      for (int s = 0; s < sb.S; ++s) {
        size_t k = sb.idx(x, y, s);
        int T = sb.tri[k];
        if (T < 0) continue;
        Real qx, qy;
        cam.pixelToQ(Real(x) + sb.offs[size_t(s)].x, Real(y) + sb.offs[size_t(s)].y, qx, qy);
        Vec3 d = cam.dir(qx, qy);
        Vec3 nT = scene.triN[size_t(T)];
        Vec3 P0 = scene.v(T, 0);
        Vec3 X = cam.eye + d * (dot(nT, P0 - cam.eye) / dot(nT, d));
        Real sideCam = dot(nT, cam.eye - P0), sideL = dot(nT, light - P0);
        if (!(sideCam * sideL > 0)) {
          sb.flag[k] = 1;
          continue;
        }
        Ray r;
        r.o = X;
        r.d = light - X;
        r.tmin = 0;
        r.tmax = 1 - kEpsRel;
        sb.flag[k] = rt.occluded(r, rs.trace, T) ? 1 : 0;
      }
  rs.traceSeconds += timer.seconds();
}

Real beamLightVisibility(BeamTracer& bt, const Vec3& x, const Vec3& ns, int tri, const AreaLight& L,
                         Real offset, Real* exactG) {
  if (exactG) *exactG = 0;
  BeamQuery q;
  Poly2 root;
  Real lightArea;
  if (!lightBeamSetup(L, x, ns, tri, offset, q, root, lightArea)) return 0;
  BeamOutput out2;
  bool keep = bt.keepOutput;
  bt.keepOutput = exactG != nullptr;
  bt.trace(q, &root, 1, out2);
  bt.keepOutput = keep;
  Real V = Real(out2.missArea) / lightArea;
  if (exactG) {
    Real G = 0;
    for (const OutBeam& ob : out2.beams) {
      if (ob.tri >= 0) continue;
      Vec3 P[4];
      for (int i = 0; i < ob.n; ++i) P[i] = q.plane.point(ob.x[i], ob.y[i]);
      G += polygonFormFactorG(x, ns, P, ob.n);
    }
    *exactG = G;
  }
  return std::max(Real(0), std::min(Real(1), V));
}

Real rayLightVisibility(const RayTracer& rt, const Vec3& x, const Vec3& ns, int tri, const AreaLight& L,
                        Real offset, int n, bool jitter, uint64_t seed, TraceStats& stats, Real* exactG) {
  if (exactG) *exactG = 0;
  Vec3 nL = L.n();
  if (!(dot(x - L.c, nL) > 0)) return 0;
  int k = std::max(1, int(std::lround(std::sqrt(double(n)))));
  Rng rng(seed);
  Vec3 apex = x + ns * offset;
  int vis = 0;
  double G = 0;
  Real A = L.area();
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < k; ++i) {
      Real su = jitter ? Real(rng.uniform()) : Real(0.5), sv = jitter ? Real(rng.uniform()) : Real(0.5);
      Vec3 p = L.c + L.U * ((Real(i) + su) / Real(k) - Real(0.5)) + L.V * ((Real(j) + sv) / Real(k) - Real(0.5));
      if (!(dot(ns, p - x) > 0)) continue;  // below the horizon
      Ray r;
      r.o = apex;
      r.d = p - apex;
      r.tmin = 0;
      r.tmax = 1 - kEpsRel;
      if (rt.occluded(r, stats, tri)) continue;
      ++vis;
      if (exactG) {
        Vec3 w = p - x;
        double r2 = double(dot(w, w));
        Vec3 wn = w / std::sqrt(Real(r2));
        G += double(std::max(Real(0), dot(ns, wn))) * double(std::max(Real(0), -dot(nL, wn))) / r2;
      }
    }
  if (exactG) *exactG = Real(G * double(A) / double(k * k));
  return Real(vis) / Real(k * k);
}

void softShadows(const Scene& scene, const KdTree& tree, const Camera& cam, const AreaLight& light,
                 const SoftOptions& opt, SoftResult& res) {
  const int W = cam.W, H = cam.H;
  res.W = W;
  res.H = H;
  res.vis.assign(size_t(W) * size_t(H), -1.f);
  res.E.assign(res.vis.size(), 0.f);
  res.Eunocc.assign(res.vis.size(), 0.f);
  res.tri.assign(res.vis.size(), -1);
  const Real offset = length(scene.bounds.diag()) * kEpsRel;
  const Real A = light.area();

  // Primary visibility with the ray tracer (as in the paper, Sec. 5.1).
  RayTracer rt(scene, tree);
  std::vector<Vec3> X(res.vis.size()), N(res.vis.size());
  Timer tp;
  {
    std::atomic<int> nextRow(0);
    std::vector<TraceStats> st(size_t(std::max(1, opt.threads)));
    auto worker = [&](int id) {
      for (int y = nextRow++; y < H; y = nextRow++)
        for (int x = 0; x < W; ++x) {
          size_t k = size_t(y) * size_t(W) + size_t(x);
          Real qx, qy;
          cam.pixelToQ(Real(x) + Real(0.5), Real(y) + Real(0.5), qx, qy);
          Ray r;
          r.o = cam.eye;
          r.d = cam.dir(qx, qy);
          RayHit h;
          if (!rt.intersect(r, h, st[size_t(id)])) continue;
          res.tri[k] = h.tri;
          X[k] = r.o + r.d * h.t;
          Vec3 n = scene.triN[size_t(h.tri)];
          if (dot(n, r.d) > 0) n = -n;
          N[k] = n;
        }
    };
    std::vector<std::thread> th;
    for (int i = 1; i < opt.threads; ++i) th.emplace_back(worker, i);
    worker(0);
    for (auto& t : th) t.join();
    for (auto& s : st) res.stats.trace.add(s);
  }
  res.stats.primarySeconds = tp.seconds();
  TraceStats primaryStats = res.stats.trace;
  res.stats.trace = TraceStats();

  Timer ts;
  std::atomic<int> nextRow(0);
  std::vector<TraceStats> st(size_t(std::max(1, opt.threads)));
  auto worker = [&](int id) {
    BeamTracer btr(scene, tree);
    btr.useMailbox = opt.mailbox;
    btr.keepOutput = false;
    for (int y = nextRow++; y < H; y = nextRow++) {
      for (int x = 0; x < W; ++x) {
        size_t k = size_t(y) * size_t(W) + size_t(x);
        int T = res.tri[k];
        if (T < 0) continue;
        const Vec3& p = X[k];
        const Vec3& n = N[k];
        Real G0 = lightCenterG(light, p, n);
        Real V, Gx = 0;
        if (opt.useBeams) {
          V = beamLightVisibility(btr, p, n, T, light, offset, opt.exact ? &Gx : nullptr);
        } else {
          uint64_t seed = (uint64_t(y) << 32) ^ uint64_t(x) ^ 0x9E3779B97F4A7C15ULL;
          V = rayLightVisibility(rt, p, n, T, light, offset, opt.samples, opt.jitter, seed, st[size_t(id)],
                                 opt.exact ? &Gx : nullptr);
        }
        res.vis[k] = float(V);
        res.Eunocc[k] = float(A * G0);
        res.E[k] = opt.exact ? float(Gx) : float(A * G0 * V);  // eq. 2 uses the light-center geometry
      }
    }
    if (opt.useBeams) st[size_t(id)].add(btr.stats);
  };
  std::vector<std::thread> th;
  for (int i = 1; i < opt.threads; ++i) th.emplace_back(worker, i);
  worker(0);
  for (auto& t : th) t.join();
  for (auto& s : st) res.stats.trace.add(s);
  res.stats.traceSeconds = ts.seconds();
  (void)primaryStats;
}

Real softExposure(const SoftResult& res) {
  std::vector<float> v;
  for (float e : res.Eunocc)
    if (e > 0) v.push_back(e);
  if (v.empty()) return 1;
  size_t k = size_t(double(v.size() - 1) * 0.99);
  std::nth_element(v.begin(), v.begin() + long(k), v.end());
  return v[k] > 0 ? Real(1) / Real(v[k]) : Real(1);
}

Image shadeSoft(const Scene& scene, const SoftResult& res, Real exposure, const Vec3& bg) {
  Image img(res.W, res.H);
  for (int y = 0; y < res.H; ++y)
    for (int x = 0; x < res.W; ++x) {
      size_t k = size_t(y) * size_t(res.W) + size_t(x);
      if (res.tri[k] < 0) {
        img.set(x, y, float(bg.x), float(bg.y), float(bg.z));
        continue;
      }
      const Vec3& alb = scene.matColor[scene.triMat[size_t(res.tri[k])]];
      float e = float(res.E[k] * exposure) * 0.9f + 0.03f;
      img.set(x, y, float(alb.x) * e, float(alb.y) * e, float(alb.z) * e);
    }
  return img;
}

}  // namespace bt

namespace bt {

void softShadowPixelsCPU(const Scene& scene, const KdTree& tree, const AreaLight& light, const SoftOptions& opt,
                         const std::vector<size_t>& pixels, const std::vector<Vec3>& X, const std::vector<Vec3>& N,
                         SoftResult& res, TraceStats& stats) {
  const Real offset = length(scene.bounds.diag()) * kEpsRel;
  std::atomic<size_t> next(0);
  std::vector<TraceStats> st(size_t(std::max(1, opt.threads)));
  auto worker = [&](int id) {
    BeamTracer btr(scene, tree);
    btr.useMailbox = opt.mailbox;
    btr.keepOutput = false;
    for (size_t i = next++; i < pixels.size(); i = next++) {
      size_t k = pixels[i];
      Real Gx = 0;
      Real V = beamLightVisibility(btr, X[k], N[k], res.tri[k], light, offset, opt.exact ? &Gx : nullptr);
      res.vis[k] = float(V);
      res.E[k] = opt.exact ? float(Gx) : float(light.area() * lightCenterG(light, X[k], N[k]) * V);
    }
    st[size_t(id)].add(btr.stats);
  };
  std::vector<std::thread> th;
  for (int i = 1; i < opt.threads; ++i) th.emplace_back(worker, i);
  worker(0);
  for (auto& t : th) t.join();
  for (auto& s : st) stats.add(s);
}

}  // namespace bt

namespace bt {

void primaryRootsCPU(const Scene& scene, const KdTree& tree, const Camera& cam, bool cull,
                     const std::vector<Poly2>& roots, std::vector<OutBeam>& hitBeams, TraceStats& stats) {
  BeamTracer btr(scene, tree);
  BeamQuery q;
  q.plane = cam.plane();
  q.mode = BeamMode::Nearest;
  q.cullBackfaces = cull;
  BeamOutput out;
  for (const Poly2& r : roots) {
    btr.trace(q, &r, 1, out);
    for (const OutBeam& ob : out.beams)
      if (ob.tri >= 0) hitBeams.push_back(ob);
  }
  stats.add(btr.stats);
}

void pointShadowsCPU(const Scene& scene, const KdTree& tree, const Camera& cam, const std::vector<OutBeam>& primaryHits,
                     const Vec3& light, std::vector<OutBeam>& shadowPolys, TraceStats& stats) {
  BeamTracer btr(scene, tree);
  BeamOutput prim;
  prim.beams = primaryHits;
  RenderStats rs;
  beamPointShadows(btr, scene, cam, prim, light, shadowPolys, rs);
  stats.add(rs.trace);
}

}  // namespace bt
