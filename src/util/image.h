// Tiny RGB image + PNG/PFM writers (no external dependencies).
#pragma once

#include <string>
#include <vector>

namespace bt {

struct Image {
  int w = 0, h = 0;
  std::vector<float> rgb;  // linear, 3 floats per pixel, row 0 = top
  Image() {}
  Image(int w_, int h_) : w(w_), h(h_), rgb(size_t(w_) * h_ * 3, 0.f) {}
  float* px(int x, int y) { return &rgb[(size_t(y) * w + x) * 3]; }
  const float* px(int x, int y) const { return &rgb[(size_t(y) * w + x) * 3]; }
  void set(int x, int y, float r, float g, float b) {
    float* p = px(x, y);
    p[0] = r; p[1] = g; p[2] = b;
  }
};

// Writes 8-bit sRGB PNG (values clamped to [0,1], gamma 2.2 if `gamma`).
bool writePNG(const std::string& path, const Image& img, bool gamma = true);
// Writes a little-endian color PFM (exact float values; bottom-to-top rows).
bool writePFM(const std::string& path, const Image& img);

}  // namespace bt
