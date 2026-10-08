// Per-pixel setup for exact soft shadows (paper Sec. 5), host and device:
// the shadow beam's plane is the light's plane, its apex the shading point,
// and its cross-section the light quad clipped by the shading point's horizon.
#pragma once

#include "beam/beam_types.h"

namespace bt {

struct AreaLight {
  Vec3 c, U, V;  // center and full edge vectors; emits along normalize(U x V)
  BT_HD Vec3 n() const { return normalize(cross(U, V)); }
  BT_HD Real area() const { return length(cross(U, V)); }
};

// Integral of cos(theta_i) cos(theta_l) / r^2 over a planar polygon seen from
// x with normal n (Lambert's formula); the polygon must lie above x's horizon.
BT_HD inline Real polygonFormFactorG(const Vec3& x, const Vec3& n, const Vec3* poly, int count) {
  double sum = 0;
  for (int i = 0; i < count; ++i) {
    Vec3 a = poly[i] - x, b = poly[i + 1 == count ? 0 : i + 1] - x;
    double la = double(length(a)), lb = double(length(b));
    if (la == 0 || lb == 0) continue;
    double c = double(dot(a, b)) / (la * lb);
    c = c < -1 ? -1 : (c > 1 ? 1 : c);
    double theta = ::acos(c);
    Vec3 g = cross(a, b);
    double lg = double(length(g));
    if (lg == 0) continue;
    sum += theta * double(dot(g, n)) / lg;
  }
  return Real((sum < 0 ? -sum : sum) * 0.5);
}

// Builds the shadow-beam query and root polygon for shading point x (normal
// ns, triangle tri). Returns false if no part of the light is visible from
// the front of the surface (x behind the emitter or the light below the horizon).
BT_HD inline bool lightBeamSetup(const AreaLight& L, const Vec3& x, const Vec3& ns, int tri, Real offset,
                                 BeamQuery& q, Poly2& root, Real& lightArea) {
  Vec3 nL = L.n();
  if (!(dot(x - L.c, nL) > 0)) return false;  // behind the emitter
  Vec3 apex = x + ns * offset;
  q.plane = BeamPlane::make(apex, L.c, L.U, L.V);
  q.mode = BeamMode::AnyHit;
  q.farIsPlane = true;
  q.excludeTri = tri;
  q.cullBackfaces = false;
  Vec3 corners[4] = {L.c - L.U * Real(0.5) - L.V * Real(0.5), L.c + L.U * Real(0.5) - L.V * Real(0.5),
                     L.c + L.U * Real(0.5) + L.V * Real(0.5), L.c - L.U * Real(0.5) + L.V * Real(0.5)};
  Poly2 quad;
  for (int i = 0; i < 4; ++i) {
    Real qx, qy;
    q.plane.coords(corners[i], qx, qy);
    quad.push(qx, qy);
  }
  if (quad.area() < 0) quad.reverse();
  lightArea = quad.area();
  // horizon: ns . (p - x) >= 0, p = p0 + qx u + qy v
  Line2 hz;
  hz.a = dot(ns, q.plane.u);
  hz.b = dot(ns, q.plane.v);
  hz.c = dot(ns, L.c - x);
  int above = 0;
  for (int i = 0; i < quad.n; ++i) above += hz.eval(quad.x[i], quad.y[i]) > 0;
  if (above == 0) return false;
  if (above == quad.n) {
    root = quad;
  } else {
    Poly2 out;
    splitPoly(quad, hz, 0, root, out);
  }
  return root.n >= 3 && root.area() > 0;
}

// Light-center geometry term of eq. 2: cos(theta_i) cos(theta_l) / r^2.
BT_HD inline Real lightCenterG(const AreaLight& L, const Vec3& x, const Vec3& ns) {
  Vec3 nL = L.n();
  if (!(dot(x - L.c, nL) > 0)) return 0;
  Vec3 w = L.c - x;
  Real r2 = dot(w, w);
  Vec3 wn = w / std::sqrt(r2);
  Real ci = dot(ns, wn), cl = -dot(nL, wn);
  return (ci > 0 ? ci : 0) * (cl > 0 ? cl : 0) / r2;
}

}  // namespace bt
