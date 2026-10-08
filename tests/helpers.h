// Shared helpers for the correctness tests.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "render/render.h"
#include "tests/test.h"
#include "util/misc.h"

namespace bttest {

using namespace bt;

inline Scene makeScene(const std::string& cfgText) {
  SceneConfig cfg;
  std::string err;
  if (!parseConfig(cfgText, cfg, err)) fprintf(stderr, "config error: %s\n", err.c_str());
  Scene s;
  if (!buildScene(cfg, s, err)) fprintf(stderr, "scene error: %s\n", err.c_str());
  return s;
}

// Signed distance-like inside measure of point (x,y) for a CCW convex polygon:
// min over edges of the signed distance to the edge line (positive inside).
inline Real insideDist(const OutBeam& b, Real x, Real y) {
  Real m = kInf;
  Real area = 0;
  for (int i = 0; i < b.n; ++i) {
    int j = i + 1 == b.n ? 0 : i + 1;
    area += b.x[i] * b.y[j] - b.x[j] * b.y[i];
  }
  if (!(area > 0)) return -kInf;  // degenerate (zero-area) polygon covers nothing
  for (int i = 0; i < b.n; ++i) {
    int j = i + 1 == b.n ? 0 : i + 1;
    Real ex = b.x[j] - b.x[i], ey = b.y[j] - b.y[i];
    Real len = std::sqrt(ex * ex + ey * ey);
    if (len == 0) continue;
    Real d = (ex * (y - b.y[i]) - ey * (x - b.x[i])) / len;
    m = std::min(m, d);
  }
  return m;
}

struct PrimaryCompare {
  int samples = 0;
  int mismatches = 0;      // unexplained tri id mismatches
  int nearEdge = 0;        // mismatches within delta of a beam edge (allowed)
  int holes = 0;           // points covered by no beam (beyond delta of an edge)
  int overlaps = 0;        // points covered by 2+ beams (beyond delta)
  double areaError = 0;    // |sum of beam areas - root area| / root area
  int firstBadBeam = -2, firstBadRay = -2;
  Real firstX = 0, firstY = 0;
};

// Compares beam-traced primary visibility with the ray tracer at random
// image-plane points. `delta` is the tolerance band around beam edges, in
// image-plane units.
inline PrimaryCompare comparePrimary(const Scene& scene, const KdTree& tree, const Camera& cam, bool cull,
                                     int nSamples, uint64_t seed, Real delta, bool mailbox = true,
                                     BeamOutput* outCopy = nullptr) {
  PrimaryCompare pc;
  BeamTracer btr(scene, tree);
  btr.useMailbox = mailbox;
  BeamOutput out;
  RenderStats rs;
  beamPrimary(btr, cam, cull, out, rs);
  if (outCopy) *outCopy = out;
  RayTracer rt(scene, tree);
  Rng rng(seed);
  double rootArea = double(4 * cam.halfW * cam.halfH), sum = 0;
  for (const OutBeam& b : out.beams) sum += double(b.area);
  pc.areaError = std::fabs(sum + out.droppedArea - rootArea) / rootArea;
  TraceStats st;
  for (int i = 0; i < nSamples; ++i) {
    Real qx = Real(rng.uniform(-1, 1)) * cam.halfW, qy = Real(rng.uniform(-1, 1)) * cam.halfH;
    int cover = 0, beamTri = -3;
    Real bestInside = -kInf;
    for (const OutBeam& b : out.beams) {
      Real d = insideDist(b, qx, qy);
      if (d > bestInside) bestInside = d;
      if (d > delta) {
        ++cover;
        beamTri = b.tri;
      }
    }
    Ray r;
    r.o = cam.eye;
    r.d = cam.dir(qx, qy);
    RayHit h;
    rt.intersect(r, h, st, cull);
    ++pc.samples;
    if (cover == 0) {
      if (bestInside < -delta) {
        ++pc.holes;
        if (pc.firstBadBeam == -2) {
          pc.firstBadBeam = -3;
          pc.firstBadRay = h.tri;
          pc.firstX = qx;
          pc.firstY = qy;
        }
      }
      continue;  // on an edge
    }
    if (cover > 1) {
      ++pc.overlaps;
      continue;
    }
    if (beamTri == h.tri) continue;
    // Coplanar ambiguity: both triangles hit at the same distance.
    if (beamTri >= 0 && h.tri >= 0) {
      Real tb = rt.hitTri(r, beamTri, false);
      if (std::isfinite(tb) && std::fabs(tb - h.t) <= Real(1e-4) * h.t) continue;
    }
    // Near-edge: does a nearby ray agree with the beam?
    bool explained = false;
    for (int k = 0; k < 8 && !explained; ++k) {
      double ang = k * M_PI / 4;
      Ray r2 = r;
      r2.d = cam.dir(qx + Real(std::cos(ang)) * delta * 2, qy + Real(std::sin(ang)) * delta * 2);
      RayHit h2;
      rt.intersect(r2, h2, st, cull);
      explained = h2.tri == beamTri;
    }
    if (explained) {
      ++pc.nearEdge;
      continue;
    }
    ++pc.mismatches;
    if (pc.firstBadBeam == -2) {
      pc.firstBadBeam = beamTri;
      pc.firstBadRay = h.tri;
      pc.firstX = qx;
      pc.firstY = qy;
    }
  }
  return pc;
}

inline void reportCompare(const char* what, const PrimaryCompare& pc) {
  fprintf(stderr,
          "    %s: samples %d mismatches %d nearEdge %d holes %d overlaps %d areaErr %.3g first(beam %d ray %d at "
          "%.5f,%.5f)\n",
          what, pc.samples, pc.mismatches, pc.nearEdge, pc.holes, pc.overlaps, pc.areaError, pc.firstBadBeam,
          pc.firstBadRay, double(pc.firstX), double(pc.firstY));
}

}  // namespace bttest
