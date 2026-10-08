// The beam tracer: beam-triangle splitting (paper Sec. 3.2), beam-kd-tree
// traversal (Sec. 3.3) and hierarchical "Post Office" mailboxing (Sec. 3.4).
#pragma once

#include <cstdint>
#include <vector>

#include "accel/kdtree.h"
#include "beam/beam_geom.h"
#include "core/simd4.h"
#include "scene/scene.h"
#include "util/stats.h"

namespace bt {

// The plane on which a beam tree's corner-ray directions are represented.
// Corner ray k is O + t * dir(q_k); t = 1 on the plane.
struct BeamPlane {
  Vec3 O;      // apex (common origin of all rays)
  Vec3 p0;     // plane origin
  Vec3 u, v;   // orthonormal in-plane basis
  Vec3 n;      // unit normal = u x v, oriented so that n.(p0 - O) > 0
  Vec3 d;      // p0 - O
  Vec3 point(Real qx, Real qy) const { return p0 + u * qx + v * qy; }
  Vec3 dir(Real qx, Real qy) const { return d + u * qx + v * qy; }
  // Plane coordinates of a point lying on the plane.
  void coords(const Vec3& p, Real& qx, Real& qy) const {
    Vec3 r = p - p0;
    qx = dot(r, u);
    qy = dot(r, v);
  }
  // Central projection through O onto the plane. Returns false if p is not
  // strictly in front of O.
  bool project(const Vec3& p, Real& qx, Real& qy) const {
    Vec3 r = p - O;
    Real den = dot(n, r);
    if (!(den > 0)) return false;
    Vec3 x = O + r * (dot(n, d) / den);
    coords(x, qx, qy);
    return true;
  }
  // Builds a plane through p0 spanned by (u, v); flips v if needed so the
  // plane faces away from O.
  static BeamPlane make(const Vec3& O, const Vec3& p0, const Vec3& u, const Vec3& v);
};

enum class BeamMode {
  Nearest,  // primary visibility: find the nearest surface for every ray
  AnyHit    // shadows: remove every part of the beam that hits anything
};

struct BeamQuery {
  BeamPlane plane;
  BeamMode mode = BeamMode::Nearest;
  bool farIsPlane = false;   // only geometry strictly in front of the plane (t < 1) counts
  int excludeTri = -1;       // triangle ignored (the receiver / shading triangle)
  bool cullBackfaces = false;
};

// A final beam cross-section on the query plane.
struct OutBeam {
  Real x[4], y[4];
  int n;
  int tri;  // Nearest: visible triangle (-1 = background); AnyHit: occluder (-1 = unoccluded)
  Real area;
};

struct BeamOutput {
  std::vector<OutBeam> beams;
  double hitArea = 0, missArea = 0, droppedArea = 0;
  void clear() {
    beams.clear();
    hitArea = missArea = droppedArea = 0;
  }
};

class BeamTracer {
 public:
  BeamTracer(const Scene& scene, const KdTree& tree);

  // Traces the convex root polygons (in plane coordinates) through the scene.
  void trace(const BeamQuery& query, const Poly2* roots, int numRoots, BeamOutput& out);

  TraceStats stats;
  bool useMailbox = true;
  bool keepOutput = true;  // if false only areas are accumulated (soft shadows)
  bool orderEdges = true;  // clip by the most-cutting triangle edge first

  // ---- internals exposed for unit tests ----
  struct alignas(32) Beam {
    R4 qx, qy;     // corner points on the plane (lanes past n repeat the last corner)
    R4 aw[3];      // |w_axis| at the corners, w = dir(q); sign fixed per beam by the pre-split
    R4 farA;       // |a_far(q)|: far-plane distance t_far = farC / farA
    Real farC = 0;
    int n = 0;
    int hit = -1;
    uint64_t mbox = 0;
    int8_t sgn[3] = {1, 1, 1};
    bool hasFar = false;
  };
  enum { kNear = 0, kFar = 1, kBoth = 2 };
  // kd decision for a beam at an inner node (axis a, plane numerator Ns > 0).
  int decide(const Beam& b, int axis, Real Ns, const AABB& box) const;
  bool overlaps(const Beam& b, const AABB& box) const;
  // Builds a beam from a convex polygon (<= 4 vertices) for the current query.
  bool makeBeam(const Poly2& p, const int8_t sgn[3], int hit, Beam& b);
  void setQueryForTest(const BeamQuery& q, Real ext);
  // Re-tests a final beam against one triangle (idempotency check). Returns
  // 0 if nothing would change, 1 if the beam's status would change as a
  // whole, 2 if it would be split. `reason` receives the deciding step.
  int retest(const BeamQuery& q, Real ext, const OutBeam& ob, int tri, int* reason = nullptr);
  int lastReason = 0;

 private:
  struct TriInfo {
    uint32_t stamp = 0;
    int tri = -1;
    bool ok = false;
    Line2 edge[3];
    Real A[3] = {0, 0, 0};  // a_T(q) = A0 qx + A1 qy + A2 = n_T . dir(q)
    Real C = 0;             // n_T . (v0 - O); t_T(q) = C / a_T(q)
    Vec3 r[3];              // vertices relative to O
  };
  struct Work {
    int beam;
    uint32_t node;
    int frame;
    AABB box;
  };
  struct Frame {
    uint32_t node;
    int parent;
    AABB box;
  };

  void setupQuery(const BeamQuery& q, Real ext);
  const TriInfo& triInfo(int tri);
  void computeTriInfo(int tri, TriInfo& ti) const;
  Line2 closerLine(const TriInfo& T, const Real* AF, Real CF, const Vec3& nF, int fId) const;
  bool separated(const Beam& b, const TriInfo& ti) const;
  void processLeaf(int beamIdx, const KdNode& leaf);
  void intersect(int beamIdx, int tri, const TriInfo& ti);
  void emit(const Poly2& p, const Beam& parent, int newHit, bool hitNow, int tri);
  void advance(int beamIdx, int frame);
  void finalize(int beamIdx);
  void output(const Beam& b, int tri);
  void beamToPoly(const Beam& b, Poly2& p) const;
  void addRoot(const Poly2& p, int depth, int8_t sgn[3]);
  bool seen(int tri, uint64_t mbox) const;
  uint64_t parentOf(uint64_t e) const {
    return (e >= evBase_ && e < nextEvent_) ? evParent_[size_t(e - evBase_)] : 0;
  }

  const Scene& scene_;
  const KdTree& tree_;
  BeamQuery q_;
  Real eps_ = 0, minArea_ = 0, ext_ = 1;
  Real farA_[3] = {0, 0, 0}, farCq_ = 0;  // far plane (query plane) coefficients
  uint32_t stamp_ = 0;
  std::vector<TriInfo> cache_;
  std::vector<uint64_t> mail_;
  std::vector<uint32_t> visStamp_;  // distinct-hit counting per trace
  uint32_t traceId_ = 0;
  std::vector<uint64_t> evParent_;
  uint64_t evBase_ = 1, nextEvent_ = 1;
  std::vector<Beam> pool_;
  std::vector<Work> work_;
  std::vector<Frame> frames_;
  std::vector<int> cur_, nxt_;
  std::vector<Poly2> tmpKeep_, tmpNew_;
  BeamOutput* out_ = nullptr;
};

}  // namespace bt
