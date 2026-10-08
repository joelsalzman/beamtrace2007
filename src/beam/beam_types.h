// Plain data types shared by the CPU beam tracer and the CUDA port.
#pragma once

#include <cstdint>
#include <vector>

#include "accel/kdtree.h"
#include "beam/beam_geom.h"

namespace bt {

// The plane on which a beam tree's corner-ray directions are represented
// (paper Sec. 3.1). Corner ray k is O + t * dir(q_k); t = 1 on the plane.
struct BeamPlane {
  Vec3 O;      // apex (common origin of all rays)
  Vec3 p0;     // plane origin
  Vec3 u, v;   // orthonormal in-plane basis
  Vec3 n;      // unit normal = u x v, oriented so that n.(p0 - O) > 0
  Vec3 d;      // p0 - O
  BT_HD Vec3 point(Real qx, Real qy) const { return p0 + u * qx + v * qy; }
  BT_HD Vec3 dir(Real qx, Real qy) const { return d + u * qx + v * qy; }
  // Plane coordinates of a point lying on the plane.
  BT_HD void coords(const Vec3& p, Real& qx, Real& qy) const {
    Vec3 r = p - p0;
    qx = dot(r, u);
    qy = dot(r, v);
  }
  // Central projection through O onto the plane; false if p is not strictly
  // in front of O.
  BT_HD bool project(const Vec3& p, Real& qx, Real& qy) const {
    Vec3 r = p - O;
    Real den = dot(n, r);
    if (!(den > 0)) return false;
    Vec3 x = O + r * (dot(n, d) / den);
    coords(x, qx, qy);
    return true;
  }
  // Plane through p0 spanned by (u, v); v is flipped if needed so that the
  // plane faces away from O.
  BT_HD static BeamPlane make(const Vec3& O, const Vec3& p0, const Vec3& u_, const Vec3& v_) {
    BeamPlane P;
    P.O = O;
    P.p0 = p0;
    Vec3 n = normalize(cross(u_, v_));
    if (dot(n, p0 - O) < 0) n = -n;
    P.u = normalize(u_);
    P.v = cross(n, P.u);
    P.n = n;
    P.d = p0 - O;
    return P;
  }
};

enum class BeamMode : int {
  Nearest = 0,  // primary visibility: find the nearest surface for every ray
  AnyHit = 1    // shadows: remove every part of the beam that hits anything
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

// A triangle as referenced from a kd leaf, packed for one wide load (the
// wavefront engines): vertices, unit normal, global vertex indices (which
// fix the canonical orientation of shared edges) and the triangle id.
struct alignas(16) TriRef {
  Vec3 v[3];
  Vec3 n;
  uint32_t vi[3];
  int32_t tri;
};

// Raw pointers to the scene and kd-tree (host or device memory).
struct SceneView {
  const Vec3* pos = nullptr;
  const uint32_t* tri3 = nullptr;  // 3 vertex indices per triangle
  const Vec3* triN = nullptr;
  const KdNode* nodes = nullptr;
  const uint32_t* triIndices = nullptr;
  const uint32_t* parent = nullptr;  // kd node parents (restart-trail traversal)
  const AABB* nodeBox = nullptr;     // kd node boxes
  const TriRef* refs = nullptr;      // optional: packed triangles in leaf order (parallel to triIndices)
  AABB bounds;
  int numTris = 0;
};

// Packed triangles for every leaf reference (in triIndices order).
inline void buildTriRefs(const Scene& s, const KdTree& t, std::vector<TriRef>& refs) {
  refs.resize(t.triIndices.size());
  for (size_t i = 0; i < refs.size(); ++i) {
    const uint32_t tri = t.triIndices[i];
    TriRef& r = refs[i];
    for (int k = 0; k < 3; ++k) {
      r.vi[k] = s.tris[tri][size_t(k)];
      r.v[k] = s.pos[r.vi[k]];
    }
    r.n = s.triN[tri];
    r.tri = int32_t(tri);
  }
}

inline SceneView makeSceneView(const Scene& s, const KdTree& t) {
  SceneView v;
  v.pos = s.pos.data();
  v.tri3 = reinterpret_cast<const uint32_t*>(s.tris.data());
  v.triN = s.triN.data();
  v.nodes = t.nodes.data();
  v.triIndices = t.triIndices.data();
  v.parent = t.parent.data();
  v.nodeBox = t.nodeBox.data();
  v.bounds = t.bounds;
  v.numTris = s.numTris();
  return v;
}

}  // namespace bt
