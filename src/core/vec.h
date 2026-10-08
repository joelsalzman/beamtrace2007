// Minimal 2D/3D vector math.
#pragma once

#include "core/real.h"

namespace bt {

struct Vec3 {
  Real x = 0, y = 0, z = 0;
  BT_HD Vec3() {}
  BT_HD Vec3(Real x_, Real y_, Real z_) : x(x_), y(y_), z(z_) {}
  BT_HD Real operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  BT_HD Real& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};

BT_HD inline Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
BT_HD inline Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
BT_HD inline Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
BT_HD inline Vec3 operator*(const Vec3& a, Real s) { return {a.x * s, a.y * s, a.z * s}; }
BT_HD inline Vec3 operator*(Real s, const Vec3& a) { return {a.x * s, a.y * s, a.z * s}; }
BT_HD inline Vec3 operator/(const Vec3& a, Real s) { return {a.x / s, a.y / s, a.z / s}; }
BT_HD inline Vec3& operator+=(Vec3& a, const Vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
BT_HD inline Vec3& operator-=(Vec3& a, const Vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
BT_HD inline Vec3& operator*=(Vec3& a, Real s) { a.x *= s; a.y *= s; a.z *= s; return a; }
BT_HD inline Real dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
BT_HD inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
BT_HD inline Real length(const Vec3& a) { return std::sqrt(dot(a, a)); }
BT_HD inline Vec3 normalize(const Vec3& a) {
  Real l = length(a);
  return l > 0 ? a / l : a;
}
BT_HD inline Vec3 vmin(const Vec3& a, const Vec3& b) {
  return {a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z};
}
BT_HD inline Vec3 vmax(const Vec3& a, const Vec3& b) {
  return {a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z};
}

// Any unit vector orthogonal to n (n must be unit length).
BT_HD inline Vec3 anyOrthogonal(const Vec3& n) {
  Vec3 a = std::fabs(n.x) < Real(0.6) ? Vec3(1, 0, 0) : (std::fabs(n.y) < Real(0.6) ? Vec3(0, 1, 0) : Vec3(0, 0, 1));
  return normalize(cross(n, a));
}

struct Vec2 {
  Real x = 0, y = 0;
  BT_HD Vec2() {}
  BT_HD Vec2(Real x_, Real y_) : x(x_), y(y_) {}
};

struct AABB {
  Vec3 lo{kInf, kInf, kInf};
  Vec3 hi{-kInf, -kInf, -kInf};
  BT_HD void expand(const Vec3& p) { lo = vmin(lo, p); hi = vmax(hi, p); }
  BT_HD void expand(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
  BT_HD bool empty() const { return !(lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z); }
  BT_HD Vec3 diag() const { return hi - lo; }
  BT_HD Vec3 center() const { return (lo + hi) * Real(0.5); }
  BT_HD Real surfaceArea() const {
    Vec3 d = diag();
    return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
  }
};

}  // namespace bt
