// Point-light hard shadows from primary hit beams (paper Sec. 4), host and
// device: each primary hit beam is back-projected onto its triangle's plane
// and traced from the light; the occluded pieces are projected back into the
// image.
#pragma once

#include "beam/beam_types.h"

namespace bt {

// Shadow beam for the primary hit beam with image-plane corners (px, py)[n]
// on triangle T (normal nT, first vertex P0), seen from a camera at `eye`
// with ray directions fwd + right * qx + down * qy. Returns false if the
// receiver faces away from the light: the whole beam is in shadow.
BT_HD inline bool pointShadowSetup(const Vec3& eye, const Vec3& fwd, const Vec3& right, const Vec3& down,
                                   const Vec3& nT, const Vec3& P0, int T, const Vec3& light, const Real* px,
                                   const Real* py, int n, BeamQuery& q, Poly2& root) {
  Real sideCam = dot(nT, eye - P0), sideL = dot(nT, light - P0);
  if (!(sideCam * sideL > 0)) return false;
  Vec3 tu = anyOrthogonal(nT);
  q.plane = BeamPlane::make(light, P0, tu, cross(nT, tu));
  q.mode = BeamMode::AnyHit;
  q.farIsPlane = true;
  q.excludeTri = T;
  q.cullBackfaces = false;
  root.clear();
  // Back-project the beam's corners onto the receiver plane (a homography).
  for (int i = 0; i < n; ++i) {
    Vec3 d = fwd + right * px[i] + down * py[i];
    Vec3 X = eye + d * (dot(nT, P0 - eye) / dot(nT, d));
    Real qx, qy;
    q.plane.coords(X, qx, qy);
    root.push(qx, qy);
  }
  return true;
}

// Projects an occluded piece (corners on the shadow beam's plane) back onto
// the image plane, counter-clockwise; false if it cannot be (receiver seen
// exactly edge-on).
BT_HD inline bool pointShadowToImage(const BeamPlane& camPlane, const BeamPlane& shadowPlane, const Real* sx,
                                     const Real* sy, int n, int T, OutBeam& ip) {
  ip.n = n;
  ip.tri = T;
  ip.area = 0;
  for (int i = 0; i < n; ++i)
    if (!camPlane.project(shadowPlane.point(sx[i], sy[i]), ip.x[i], ip.y[i])) return false;
  Poly2 p;
  for (int i = 0; i < n; ++i) p.push(ip.x[i], ip.y[i]);
  Real a = p.area();
  if (!(a > 0 || a < 0)) return false;
  if (a < 0) {
    p.reverse();
    for (int i = 0; i < n; ++i) {
      ip.x[i] = p.x[i];
      ip.y[i] = p.y[i];
    }
  }
  for (int i = n; i < 4; ++i) {
    ip.x[i] = ip.x[n - 1];
    ip.y[i] = ip.y[n - 1];
  }
  return true;
}

}  // namespace bt
