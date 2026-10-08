// Scalar type selection and small numeric helpers.
//
// The whole library is compiled twice: once with Real = float (the paper's
// 32-bit SSE configuration) and once with Real = double (used as the
// reference for correctness work).
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define BT_HD __host__ __device__
#else
#define BT_HD
#endif

namespace bt {

#ifdef BT_REAL_DOUBLE
using Real = double;
#else
using Real = float;
#endif

constexpr Real kInf = std::numeric_limits<Real>::infinity();
constexpr Real kPi = Real(3.14159265358979323846);

// Relative tolerances. Float tolerances are set a few orders of magnitude
// above float epsilon; double ones a few above double epsilon.
#ifdef BT_REAL_DOUBLE
constexpr Real kEpsRel = Real(1e-9);    // fuzzy split epsilon, relative to beam extent
constexpr Real kKdRel = Real(1e-12);    // kd decision margin (relative)
constexpr Real kDegRel = Real(1e-10);   // edge-on / degenerate detection
#else
constexpr Real kEpsRel = Real(2e-5);
constexpr Real kKdRel = Real(1e-5);
constexpr Real kDegRel = Real(2e-6);
#endif

template <class T>
BT_HD inline T bmin(T a, T b) { return a < b ? a : b; }
template <class T>
BT_HD inline T bmax(T a, T b) { return a > b ? a : b; }
template <class T>
BT_HD inline void bswap(T& a, T& b) {
  T t = a;
  a = b;
  b = t;
}

BT_HD inline Real sgn(Real x) { return x > 0 ? Real(1) : (x < 0 ? Real(-1) : Real(0)); }
template <class T>
BT_HD inline T clampv(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

}  // namespace bt
