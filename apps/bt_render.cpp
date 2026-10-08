// bt_render: render a scene with the beam tracer or the baseline ray tracer
// and report the paper's statistics.
//
//   bt_render --config configs/sponza.cfg --mode softshadow --method beam --res 512x512
//
// Modes: primary | pointshadow | softshadow. Methods: beam | ray.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "render/render.h"
#include "util/misc.h"

using namespace bt;

namespace {

struct Args {
  std::string config, mode = "primary", method = "beam", out, csv, tag;
  int view = 0;
  bool allViews = false;
  int W = 512, H = 512;
  int aa = 6;          // raster samples per pixel for beams
  int rayAA = 1;       // primary rays per pixel for the ray tracer
  int samples = 256;   // shadow rays per pixel (soft shadows)
  int threads = 1;
  bool exact = false, jitter = true, mailbox = true, wire = false, singleLeaf = false, quiet = false;
  bool writePfm = false;
  double lightScale = 1;
  int cull = -1;  // -1: from config
};

void usage() {
  fprintf(stderr,
          "usage: bt_render --config FILE [--view N | --views all] [--mode primary|pointshadow|softshadow]\n"
          "                 [--method beam|ray] [--res WxH] [--aa N] [--ray-aa N] [--samples N] [--threads N]\n"
          "                 [--exact] [--no-jitter] [--no-mailbox] [--light-scale S] [--cull 0|1]\n"
          "                 [--out PREFIX] [--wire] [--pfm] [--csv FILE] [--tag STR] [--kd-single-leaf] [--quiet]\n");
}

bool parseArgs(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", s.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (s == "--config") a.config = next();
    else if (s == "--mode") a.mode = next();
    else if (s == "--method") a.method = next();
    else if (s == "--view") a.view = std::atoi(next().c_str());
    else if (s == "--views") a.allViews = next() == "all";
    else if (s == "--res") {
      std::string r = next();
      if (sscanf(r.c_str(), "%dx%d", &a.W, &a.H) != 2) return false;
    } else if (s == "--aa") a.aa = std::atoi(next().c_str());
    else if (s == "--ray-aa") a.rayAA = std::atoi(next().c_str());
    else if (s == "--samples") a.samples = std::atoi(next().c_str());
    else if (s == "--threads") a.threads = std::max(1, std::atoi(next().c_str()));
    else if (s == "--exact") a.exact = true;
    else if (s == "--no-jitter") a.jitter = false;
    else if (s == "--no-mailbox") a.mailbox = false;
    else if (s == "--light-scale") a.lightScale = std::atof(next().c_str());
    else if (s == "--cull") a.cull = std::atoi(next().c_str());
    else if (s == "--out") a.out = next();
    else if (s == "--wire") a.wire = true;
    else if (s == "--pfm") a.writePfm = true;
    else if (s == "--csv") a.csv = next();
    else if (s == "--tag") a.tag = next();
    else if (s == "--kd-single-leaf") a.singleLeaf = true;
    else if (s == "--quiet") a.quiet = true;
    else if (s == "--help" || s == "-h") return false;
    else {
      fprintf(stderr, "unknown option %s\n", s.c_str());
      return false;
    }
  }
  return !a.config.empty();
}

void appendCsv(const Args& a, const SceneConfig& cfg, const Scene& scene, const KdTree& tree, int view,
               const RenderStats& rs, double meanVis) {
  if (a.csv.empty()) return;
  bool exists = std::ifstream(a.csv).good();
  FILE* f = fopen(a.csv.c_str(), "a");
  if (!f) return;
  if (!exists)
    fprintf(f,
            "scene,mode,method,precision,view,W,H,aa,samples,threads,tris,visible_tris,hit_beams,beams,kd_steps,"
            "leaf_visits,tri_tests,hits,splits,presplit,mailbox_skips,rays,dropped_area,trace_s,primary_s,"
            "raster_s,build_s,light_scale,exact,mailbox,mean_vis,tag\n");
  const TraceStats& t = rs.trace;
  fprintf(f, "%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.6g,%.6f,%.6f,%.6f,%.4f,%g,%d,%d,%.6f,%s\n",
          cfg.name.c_str(), a.mode.c_str(), a.method.c_str(), sizeof(Real) == 8 ? "double" : "float", view, a.W,
          a.H, a.method == "beam" ? a.aa : a.rayAA, a.samples, a.threads, scene.numTris(), rs.visibleTris,
          rs.hitBeams, (unsigned long long)t.beams, (unsigned long long)t.kdSteps,
          (unsigned long long)t.leafVisits, (unsigned long long)t.triTests, (unsigned long long)t.hits,
          (unsigned long long)t.splits, (unsigned long long)t.presplitBeams, (unsigned long long)t.mailboxSkips,
          (unsigned long long)t.rays, t.droppedArea, rs.traceSeconds, rs.primarySeconds, rs.rasterSeconds,
          tree.buildSeconds, a.lightScale, a.exact ? 1 : 0, a.mailbox ? 1 : 0, meanVis, a.tag.c_str());
  fclose(f);
}

std::string outName(const Args& a, int view, const char* suffix) {
  char buf[64];
  if (a.allViews)
    snprintf(buf, sizeof buf, "_v%02d%s", view, suffix);
  else
    snprintf(buf, sizeof buf, "%s", suffix);
  return a.out + buf;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parseArgs(argc, argv, a)) {
    usage();
    return 2;
  }
  SceneConfig cfg;
  std::string err;
  if (!loadConfig(a.config, cfg, err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  Scene scene;
  Timer tload;
  if (!buildScene(cfg, scene, err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  double loadS = tload.seconds();
  KdTree tree;
  KdBuildParams kp;
  kp.singleLeaf = a.singleLeaf;
  tree.build(scene, kp);
  bool cull = a.cull >= 0 ? a.cull != 0 : cfg.cullBackfaces;
  if (!a.quiet)
    fprintf(stderr, "[%s] %d tris, load %.2fs, kd build %.2fs (%zu nodes, %d leaves, depth %d), %s\n",
            cfg.name.c_str(), scene.numTris(), loadS, tree.buildSeconds, tree.nodes.size(), tree.numLeaves,
            tree.maxDepthReached, sizeof(Real) == 8 ? "double" : "float");
  if (cfg.views.empty()) {
    fprintf(stderr, "config has no views\n");
    return 1;
  }
  int v0 = a.allViews ? 0 : a.view, v1 = a.allViews ? int(cfg.views.size()) - 1 : a.view;
  if (v0 < 0 || v1 >= int(cfg.views.size())) {
    fprintf(stderr, "view out of range\n");
    return 1;
  }
  BeamTracer btr(scene, tree);
  btr.useMailbox = a.mailbox;
  RayTracer rtr(scene, tree);
  for (int view = v0; view <= v1; ++view) {
    Camera cam = Camera::make(cfg.views[size_t(view)], cfg.up, a.W, a.H);
    RenderStats rs;
    double meanVis = 0;
    Image img;
    if (a.mode == "primary" || a.mode == "pointshadow") {
      bool shadows = a.mode == "pointshadow";
      if (shadows && !cfg.hasPointLight) {
        fprintf(stderr, "config has no point_light\n");
        return 1;
      }
      Vec3 light = shadows ? cfg.pointLight : cam.eye;
      SampleBuffer sb;
      if (a.method == "beam") {
        BeamOutput prim;
        beamPrimary(btr, cam, cull, prim, rs);
        std::vector<OutBeam> shadowPolys;
        if (shadows) beamPointShadows(btr, scene, cam, prim, light, shadowPolys, rs);
        Timer tr;
        sb.init(a.W, a.H, a.aa);
        rasterize(prim.beams, cam, sb, false);
        if (shadows) rasterize(shadowPolys, cam, sb, true);
        if (!a.out.empty()) img = shadeSamples(scene, cam, sb, light, cfg.background);
        rs.rasterSeconds = tr.seconds();
        if (a.wire && !a.out.empty()) {
          Image w = img;
          for (auto& c : w.rgb) c *= 0.5f;
          drawOutlines(prim.beams, cam, w, 1.f, 1.f, 0.3f);
          if (shadows) drawOutlines(shadowPolys, cam, w, 0.3f, 0.8f, 1.f);
          writePNG(outName(a, view, "_wire.png"), w);
        }
      } else {
        sb.init(a.W, a.H, a.rayAA);
        rayPrimary(rtr, cam, cull, sb, rs);
        if (shadows) rayPointShadows(rtr, scene, cam, light, sb, rs);
        Timer tr;
        if (!a.out.empty()) img = shadeSamples(scene, cam, sb, light, cfg.background);
        rs.rasterSeconds = tr.seconds();
      }
      if (rs.visibleTris == 0) {
        std::vector<char> seen(size_t(scene.numTris()), 0);
        for (int t : sb.tri)
          if (t >= 0 && !seen[size_t(t)]) {
            seen[size_t(t)] = 1;
            rs.visibleTris++;
          }
      }
    } else if (a.mode == "softshadow") {
      if (!cfg.hasAreaLight) {
        fprintf(stderr, "config has no area_light\n");
        return 1;
      }
      AreaLight L;
      L.c = cfg.lightC;
      L.U = cfg.lightU * Real(a.lightScale);
      L.V = cfg.lightV * Real(a.lightScale);
      SoftOptions so;
      so.useBeams = a.method == "beam";
      so.samples = a.samples;
      so.exact = a.exact;
      so.jitter = a.jitter;
      so.threads = a.threads;
      so.mailbox = a.mailbox;
      SoftResult res;
      softShadows(scene, tree, cam, L, so, res);
      rs = res.stats;
      double sum = 0;
      int cnt = 0;
      for (float v : res.vis)
        if (v >= 0) {
          sum += v;
          ++cnt;
        }
      meanVis = cnt ? sum / cnt : 0;
      if (!a.out.empty()) {
        img = shadeSoft(scene, res, softExposure(res), cfg.background);
        if (a.writePfm) {
          Image vis(res.W, res.H);
          for (int y = 0; y < res.H; ++y)
            for (int x = 0; x < res.W; ++x) {
              float v = res.vis[size_t(y) * size_t(res.W) + size_t(x)];
              vis.set(x, y, v, v, v);
            }
          writePFM(outName(a, view, "_vis.pfm"), vis);
        }
      }
    } else {
      fprintf(stderr, "unknown mode %s\n", a.mode.c_str());
      return 2;
    }
    if (!a.out.empty()) writePNG(outName(a, view, ".png"), img);
    const TraceStats& t = rs.trace;
    double px = double(a.W) * a.H;
    if (!a.quiet)
      printf("%s view %d %s/%s: trace %.4fs%s | visible tris %d, hit beams %d, beams %llu | kd steps/px %.4g, "
             "isect/px %.4g, hits/px %.4g%s\n",
             cfg.name.c_str(), view, a.mode.c_str(), a.method.c_str(), rs.traceSeconds,
             a.mode == "softshadow" ? (" (+primary " + std::to_string(rs.primarySeconds) + "s)").c_str() : "",
             rs.visibleTris, rs.hitBeams, (unsigned long long)t.beams, double(t.kdSteps) / px,
             double(t.triTests) / px, double(t.hits) / px,
             a.mode == "softshadow" ? (", mean V " + std::to_string(meanVis)).c_str() : "");
    appendCsv(a, cfg, scene, tree, view, rs, meanVis);
  }
  return 0;
}
