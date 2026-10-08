// Timer, RNG and string helpers.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bt {

class Timer {
 public:
  Timer() { reset(); }
  void reset() { t0_ = std::chrono::steady_clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

 private:
  std::chrono::steady_clock::time_point t0_;
};

// PCG32 (O'Neill). Deterministic, small, good enough for jitter and scenes.
class Rng {
 public:
  explicit Rng(uint64_t seed = 0x853c49e6748fea9bULL, uint64_t seq = 0xda3e39cb94b95bdbULL) {
    state_ = 0;
    inc_ = (seq << 1u) | 1u;
    next();
    state_ += seed;
    next();
  }
  uint32_t next() {
    uint64_t old = state_;
    state_ = old * 6364136223846793005ULL + inc_;
    uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = uint32_t(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }
  double uniform() { return (next() >> 5) * (1.0 / 134217728.0); }  // [0,1)
  double uniform(double a, double b) { return a + (b - a) * uniform(); }

 private:
  uint64_t state_, inc_;
};

std::vector<std::string> splitWS(const std::string& s);
std::string trim(const std::string& s);

}  // namespace bt
