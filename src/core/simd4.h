// R4: four lanes of Real, one per beam corner ray (SoA layout, paper App. A).
//
// Backends:
//   float  + SSE2  -> __m128   (the paper's configuration)
//   double + AVX2  -> __m256d
//   otherwise      -> plain arrays (also used on the GPU)
//
// Comparisons return a 4-bit lane mask (bit i = lane i), like _mm_movemask_ps.
#pragma once

#include "core/real.h"

#if !defined(BT_NO_SIMD) && !defined(__CUDACC__) && !defined(BT_REAL_DOUBLE) && defined(__SSE2__)
#define BT_R4_SSE 1
#include <immintrin.h>
#elif !defined(BT_NO_SIMD) && !defined(__CUDACC__) && defined(BT_REAL_DOUBLE) && defined(__AVX2__)
#define BT_R4_AVX 1
#include <immintrin.h>
#endif

namespace bt {

constexpr int kAll4 = 0xF;

#if defined(BT_R4_SSE)

struct alignas(16) R4 {
  __m128 v;
  R4() {}
  R4(__m128 x) : v(x) {}
  explicit R4(Real s) : v(_mm_set1_ps(s)) {}
  R4(Real a, Real b, Real c, Real d) : v(_mm_setr_ps(a, b, c, d)) {}
  Real operator[](int i) const {
    alignas(16) float t[4];
    _mm_store_ps(t, v);
    return t[i];
  }
  void store(Real* out) const { _mm_storeu_ps(out, v); }
};
inline R4 operator+(R4 a, R4 b) { return _mm_add_ps(a.v, b.v); }
inline R4 operator-(R4 a, R4 b) { return _mm_sub_ps(a.v, b.v); }
inline R4 operator*(R4 a, R4 b) { return _mm_mul_ps(a.v, b.v); }
inline R4 operator/(R4 a, R4 b) { return _mm_div_ps(a.v, b.v); }
inline R4 operator*(R4 a, Real s) { return _mm_mul_ps(a.v, _mm_set1_ps(s)); }
inline R4 operator*(Real s, R4 a) { return _mm_mul_ps(a.v, _mm_set1_ps(s)); }
inline R4 operator+(R4 a, Real s) { return _mm_add_ps(a.v, _mm_set1_ps(s)); }
inline R4 operator-(R4 a, Real s) { return _mm_sub_ps(a.v, _mm_set1_ps(s)); }
inline R4 operator-(R4 a) { return _mm_xor_ps(a.v, _mm_set1_ps(-0.0f)); }
inline R4 rmin(R4 a, R4 b) { return _mm_min_ps(a.v, b.v); }
inline R4 rmax(R4 a, R4 b) { return _mm_max_ps(a.v, b.v); }
inline R4 rabs(R4 a) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a.v); }
inline R4 rsqrt_(R4 a) { return _mm_sqrt_ps(a.v); }
inline int lt(R4 a, R4 b) { return _mm_movemask_ps(_mm_cmplt_ps(a.v, b.v)); }
inline int le(R4 a, R4 b) { return _mm_movemask_ps(_mm_cmple_ps(a.v, b.v)); }
inline int gt(R4 a, R4 b) { return _mm_movemask_ps(_mm_cmpgt_ps(a.v, b.v)); }
inline int ge(R4 a, R4 b) { return _mm_movemask_ps(_mm_cmpge_ps(a.v, b.v)); }
// Rotate lanes: result[i] = a[(i+1)&3].
inline R4 rot1(R4 a) { return _mm_shuffle_ps(a.v, a.v, _MM_SHUFFLE(0, 3, 2, 1)); }

#elif defined(BT_R4_AVX)

struct alignas(32) R4 {
  __m256d v;
  R4() {}
  R4(__m256d x) : v(x) {}
  explicit R4(Real s) : v(_mm256_set1_pd(s)) {}
  R4(Real a, Real b, Real c, Real d) : v(_mm256_setr_pd(a, b, c, d)) {}
  Real operator[](int i) const {
    alignas(32) double t[4];
    _mm256_store_pd(t, v);
    return t[i];
  }
  void store(Real* out) const { _mm256_storeu_pd(out, v); }
};
inline R4 operator+(R4 a, R4 b) { return _mm256_add_pd(a.v, b.v); }
inline R4 operator-(R4 a, R4 b) { return _mm256_sub_pd(a.v, b.v); }
inline R4 operator*(R4 a, R4 b) { return _mm256_mul_pd(a.v, b.v); }
inline R4 operator/(R4 a, R4 b) { return _mm256_div_pd(a.v, b.v); }
inline R4 operator*(R4 a, Real s) { return _mm256_mul_pd(a.v, _mm256_set1_pd(s)); }
inline R4 operator*(Real s, R4 a) { return _mm256_mul_pd(a.v, _mm256_set1_pd(s)); }
inline R4 operator+(R4 a, Real s) { return _mm256_add_pd(a.v, _mm256_set1_pd(s)); }
inline R4 operator-(R4 a, Real s) { return _mm256_sub_pd(a.v, _mm256_set1_pd(s)); }
inline R4 operator-(R4 a) { return _mm256_xor_pd(a.v, _mm256_set1_pd(-0.0)); }
inline R4 rmin(R4 a, R4 b) { return _mm256_min_pd(a.v, b.v); }
inline R4 rmax(R4 a, R4 b) { return _mm256_max_pd(a.v, b.v); }
inline R4 rabs(R4 a) { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a.v); }
inline R4 rsqrt_(R4 a) { return _mm256_sqrt_pd(a.v); }
inline int lt(R4 a, R4 b) { return _mm256_movemask_pd(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)); }
inline int le(R4 a, R4 b) { return _mm256_movemask_pd(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)); }
inline int gt(R4 a, R4 b) { return _mm256_movemask_pd(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)); }
inline int ge(R4 a, R4 b) { return _mm256_movemask_pd(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ)); }
inline R4 rot1(R4 a) { return _mm256_permute4x64_pd(a.v, _MM_SHUFFLE(0, 3, 2, 1)); }

#else

struct R4 {
  Real v[4];
  BT_HD R4() {}
  BT_HD explicit R4(Real s) { v[0] = v[1] = v[2] = v[3] = s; }
  BT_HD R4(Real a, Real b, Real c, Real d) { v[0] = a; v[1] = b; v[2] = c; v[3] = d; }
  BT_HD Real operator[](int i) const { return v[i]; }
  BT_HD void store(Real* out) const { for (int i = 0; i < 4; ++i) out[i] = v[i]; }
};
#define BT_R4_BINOP(OP)                                                            \
  BT_HD inline R4 operator OP(R4 a, R4 b) {                                        \
    return R4(a.v[0] OP b.v[0], a.v[1] OP b.v[1], a.v[2] OP b.v[2], a.v[3] OP b.v[3]); \
  }                                                                                \
  BT_HD inline R4 operator OP(R4 a, Real s) { return a OP R4(s); }
BT_R4_BINOP(+)
BT_R4_BINOP(-)
BT_R4_BINOP(*)
BT_R4_BINOP(/)
#undef BT_R4_BINOP
BT_HD inline R4 operator*(Real s, R4 a) { return a * R4(s); }
BT_HD inline R4 operator-(R4 a) { return R4(-a.v[0], -a.v[1], -a.v[2], -a.v[3]); }
BT_HD inline R4 rmin(R4 a, R4 b) {
  R4 r;
  for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i];
  return r;
}
BT_HD inline R4 rmax(R4 a, R4 b) {
  R4 r;
  for (int i = 0; i < 4; ++i) r.v[i] = a.v[i] > b.v[i] ? a.v[i] : b.v[i];
  return r;
}
BT_HD inline R4 rabs(R4 a) {
  R4 r;
  for (int i = 0; i < 4; ++i) r.v[i] = std::fabs(a.v[i]);
  return r;
}
BT_HD inline R4 rsqrt_(R4 a) {
  R4 r;
  for (int i = 0; i < 4; ++i) r.v[i] = std::sqrt(a.v[i]);
  return r;
}
#define BT_R4_CMP(NAME, OP)                                  \
  BT_HD inline int NAME(R4 a, R4 b) {                        \
    return (a.v[0] OP b.v[0] ? 1 : 0) | (a.v[1] OP b.v[1] ? 2 : 0) | \
           (a.v[2] OP b.v[2] ? 4 : 0) | (a.v[3] OP b.v[3] ? 8 : 0);  \
  }
BT_R4_CMP(lt, <)
BT_R4_CMP(le, <=)
BT_R4_CMP(gt, >)
BT_R4_CMP(ge, >=)
#undef BT_R4_CMP
BT_HD inline R4 rot1(R4 a) { return R4(a.v[1], a.v[2], a.v[3], a.v[0]); }

#endif

BT_HD inline R4 fmadd(R4 a, R4 b, R4 c) { return a * b + c; }

}  // namespace bt
