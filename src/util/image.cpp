#include "util/image.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace bt {

namespace {

uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc = 0) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
  v.push_back(uint8_t(x >> 24));
  v.push_back(uint8_t(x >> 16));
  v.push_back(uint8_t(x >> 8));
  v.push_back(uint8_t(x));
}

void chunk(FILE* f, const char* type, const std::vector<uint8_t>& data) {
  std::vector<uint8_t> buf;
  put32(buf, uint32_t(data.size()));
  size_t start = buf.size();
  buf.insert(buf.end(), type, type + 4);
  buf.insert(buf.end(), data.begin(), data.end());
  uint32_t c = crc32(buf.data() + start, buf.size() - start);
  put32(buf, c);
  fwrite(buf.data(), 1, buf.size(), f);
}

}  // namespace

bool writePNG(const std::string& path, const Image& img, bool gamma) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  fwrite(sig, 1, 8, f);
  std::vector<uint8_t> ihdr;
  put32(ihdr, uint32_t(img.w));
  put32(ihdr, uint32_t(img.h));
  ihdr.push_back(8);  // bit depth
  ihdr.push_back(2);  // RGB
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  chunk(f, "IHDR", ihdr);

  // Raw scanlines with filter byte 0.
  std::vector<uint8_t> raw;
  raw.reserve(size_t(img.h) * (img.w * 3 + 1));
  for (int y = 0; y < img.h; ++y) {
    raw.push_back(0);
    for (int x = 0; x < img.w; ++x) {
      const float* p = img.px(x, y);
      for (int c = 0; c < 3; ++c) {
        float v = p[c];
        if (!(v > 0)) v = 0;
        if (v > 1) v = 1;
        if (gamma) v = std::pow(v, 1.f / 2.2f);
        raw.push_back(uint8_t(std::lround(v * 255.f)));
      }
    }
  }
  // zlib stream with stored (uncompressed) deflate blocks.
  std::vector<uint8_t> z;
  z.push_back(0x78);
  z.push_back(0x01);
  size_t pos = 0;
  uint32_t a = 1, b = 0;
  for (uint8_t c : raw) {
    a = (a + c) % 65521;
    b = (b + a) % 65521;
  }
  do {
    size_t n = raw.size() - pos;
    if (n > 65535) n = 65535;
    bool last = pos + n == raw.size();
    z.push_back(last ? 1 : 0);
    z.push_back(uint8_t(n & 0xFF));
    z.push_back(uint8_t(n >> 8));
    z.push_back(uint8_t(~n & 0xFF));
    z.push_back(uint8_t((~n >> 8) & 0xFF));
    z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
    pos += n;
  } while (pos < raw.size());
  put32(z, (b << 16) | a);
  chunk(f, "IDAT", z);
  chunk(f, "IEND", {});
  fclose(f);
  return true;
}

bool writePFM(const std::string& path, const Image& img) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  fprintf(f, "PF\n%d %d\n-1.0\n", img.w, img.h);
  for (int y = img.h - 1; y >= 0; --y) fwrite(img.px(0, y), sizeof(float), size_t(img.w) * 3, f);
  fclose(f);
  return true;
}

}  // namespace bt
