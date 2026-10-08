// GPU vs CPU: the CUDA soft-shadow beams must reproduce the CPU beams.
#include "cuda/gpu_render.h"
#include "tests/helpers.h"

using namespace bttest;

namespace {

void compareSoft(const char* cfgText, const View& v, const AreaLight& L, int res, double maxTol, double meanTol) {
  Scene s = makeScene(cfgText);
  KdTree tree;
  KdBuildParams kp;
  kp.costIntersect = Real(0.4);
  tree.build(s, kp);
  Camera cam = Camera::make(v, Vec3(0, 1, 0), res, res);
  SoftOptions opt;
  opt.useBeams = true;
  SoftResult cpu, gpu;
  softShadows(s, tree, cam, L, opt, cpu);
  GpuRenderer g(s, tree);
  g.softShadows(cam, L, opt, gpu);
  double maxd = 0, sum = 0;
  int n = 0, triDiff = 0;
  for (size_t k = 0; k < cpu.vis.size(); ++k) {
    if (cpu.tri[k] != gpu.tri[k]) {
      ++triDiff;
      continue;
    }
    if (cpu.vis[k] < 0) continue;
    double d = std::fabs(double(cpu.vis[k]) - double(gpu.vis[k]));
    maxd = std::max(maxd, d);
    sum += d;
    ++n;
  }
  fprintf(stderr, "    %d px: max |cpu-gpu| %.3g, mean %.3g, primary tri differs at %d px, %d overflow px\n", n, maxd,
          n ? sum / n : 0.0, triDiff, g.lastOverflowPixels);
  CHECK(n > res * res / 4);
  CHECK(triDiff <= res * res / 1000);
  CHECK_MSG(maxd <= maxTol, "max %g", maxd);
  CHECK_MSG(sum / std::max(1, n) <= meanTol, "mean %g", sum / std::max(1, n));
}

}  // namespace

TEST(cuda_soft_shadows_match_cpu_analytic) {
  if (!gpuAvailable()) {
    fprintf(stderr, "    no CUDA device: skipped\n");
    return;
  }
  View v;
  v.eye = Vec3(0, 5, 5);
  v.target = Vec3(0, 0, 0);
  v.fovY = 60;
  AreaLight L;
  L.c = Vec3(0, 3, 0);
  L.U = Vec3(1, 0, 0);
  L.V = Vec3(0, 0, 1);
  compareSoft("mesh procedural analytic h=1 occ=1\n", v, L, 128, 1e-3, 1e-5);
}

TEST(cuda_soft_shadows_match_cpu_plant) {
  if (!gpuAvailable()) return;
  View v;
  v.eye = Vec3(2.6f, 1.7f, 3.4f);
  v.target = Vec3(0, 0.55f, 0);
  v.fovY = 45;
  AreaLight L;
  L.c = Vec3(0.9f, 2.8f, 0.7f);
  L.U = Vec3(0.4f, 0, 0);
  L.V = Vec3(0, 0, 0.4f);
  compareSoft("mesh procedural plant\n", v, L, 160, 0.02, 1e-4);
}

TEST(cuda_soft_shadows_match_cpu_room) {
  if (!gpuAvailable()) return;
  View v;
  v.eye = Vec3(6.2f, 2.2f, 6.5f);
  v.target = Vec3(4, 0.8f, 3.5f);
  v.fovY = 70;
  AreaLight L;
  L.c = Vec3(4, 2.95f, 4);
  L.U = Vec3(1, 0, 0);
  L.V = Vec3(0, 0, 1);
  compareSoft("mesh procedural room\n", v, L, 160, 0.02, 1e-4);
}

namespace {

bool uniform3x3(const SampleBuffer& sb, int x, int y, bool flags) {
  int id = flags ? sb.flag[sb.idx(x, y, 0)] : sb.tri[sb.idx(x, y, 0)];
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= sb.W || yy >= sb.H) return false;
      int v = flags ? sb.flag[sb.idx(xx, yy, 0)] : sb.tri[sb.idx(xx, yy, 0)];
      if (v != id) return false;
    }
  return true;
}

}  // namespace

TEST(cuda_primary_and_point_shadows_match_cpu) {
  if (!gpuAvailable()) return;
  for (const char* cfg : {"mesh procedural room\n", "mesh procedural plant\n"}) {
    Scene s = makeScene(cfg);
    KdTree tree;
    KdBuildParams kp;
    kp.costIntersect = Real(0.4);
    tree.build(s, kp);
    View v;
    bool room = std::string(cfg).find("room") != std::string::npos;
    v.eye = room ? Vec3(6.2f, 2.2f, 6.5f) : Vec3(2.6f, 1.7f, 3.4f);
    v.target = room ? Vec3(4, 0.8f, 3.5f) : Vec3(0, 0.55f, 0);
    v.fovY = room ? 70 : 45;
    Vec3 light = room ? Vec3(5.5f, 2.6f, 2.5f) : Vec3(1.0f, 3.0f, 0.8f);
    Camera cam = Camera::make(v, Vec3(0, 1, 0), 384, 384);
    BeamTracer btr(s, tree);
    BeamOutput cpuPrim;
    RenderStats rs;
    beamPrimary(btr, cam, false, cpuPrim, rs);
    std::vector<OutBeam> cpuShadow;
    beamPointShadows(btr, s, cam, cpuPrim, light, cpuShadow, rs);
    GpuRenderer g(s, tree);
    std::vector<OutBeam> gpuPrim, gpuShadow;
    g.primary(cam, false, gpuPrim, rs);
    g.pointShadows(cam, gpuPrim, light, gpuShadow, rs);
    SampleBuffer a, b;
    a.init(384, 384, 1);
    b.init(384, 384, 1);
    rasterize(cpuPrim.beams, cam, a, false);
    rasterize(cpuShadow, cam, a, true);
    rasterize(gpuPrim, cam, b, false);
    rasterize(gpuShadow, cam, b, true);
    int compared = 0, triDiff = 0, shadowCompared = 0, shadowDiff = 0;
    for (int y = 0; y < 384; ++y)
      for (int x = 0; x < 384; ++x) {
        if (uniform3x3(a, x, y, false) && uniform3x3(b, x, y, false)) {
          ++compared;
          triDiff += a.tri[a.idx(x, y, 0)] != b.tri[b.idx(x, y, 0)];
        }
        if (uniform3x3(a, x, y, true) && uniform3x3(b, x, y, true)) {
          ++shadowCompared;
          shadowDiff += a.flag[a.idx(x, y, 0)] != b.flag[b.idx(x, y, 0)];
        }
      }
    fprintf(stderr, "    %s: %d/%d interior pixels differ (tri), %d/%d (shadow)\n", room ? "room" : "plant", triDiff,
            compared, shadowDiff, shadowCompared);
    CHECK(compared > 384 * 384 / 2);
    CHECK(triDiff == 0);
    CHECK(shadowDiff == 0);
  }
}

TEST(cuda_wavefront_deterministic_and_exact_under_tiny_budgets) {
  if (!gpuAvailable()) return;
  Scene s = makeScene("mesh procedural plant\n");
  KdTree tree;
  KdBuildParams kp;
  kp.costIntersect = Real(0.4);
  tree.build(s, kp);
  View v;
  v.eye = Vec3(2.6f, 1.7f, 3.4f);
  v.target = Vec3(0, 0.55f, 0);
  v.fovY = 45;
  AreaLight L;
  L.c = Vec3(0.9f, 2.8f, 0.7f);
  L.U = Vec3(0.4f, 0, 0);
  L.V = Vec3(0, 0, 0.4f);
  Camera cam = Camera::make(v, Vec3(0, 1, 0), 128, 128);
  SoftOptions opt;
  GpuRenderer g(s, tree);
  SoftResult a, b, small, cpu;
  g.softShadows(cam, L, opt, a);
  g.softShadows(cam, L, opt, b);
  // Fixed-point accumulation: the result does not depend on scheduling.
  CHECK_MSG(a.vis == b.vis && a.E == b.E, "two runs differ");
  // Budget 4 and small queues: many pixels go back to the CPU, and the
  // admission rule must still never overflow a queue.
  g.wfBudget = 4;
  g.wfCapacity = 4096;
  g.softShadows(cam, L, opt, small);
  softShadows(s, tree, cam, L, opt, cpu);
  int fallback = g.lastOverflowPixels, bad = 0, n = 0;
  for (size_t k = 0; k < cpu.vis.size(); ++k) {
    if (cpu.vis[k] < 0) continue;
    ++n;
    bad += std::fabs(double(cpu.vis[k]) - double(small.vis[k])) > 0.02;
  }
  fprintf(stderr, "    %d of %d pixels recomputed on the CPU, %d rounds\n", fallback, n, g.lastRounds);
  CHECK(fallback > 100 && fallback < n);
  CHECK_MSG(bad == 0, "%d pixels differ", bad);
}
