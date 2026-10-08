// bt_compare: error metrics between two PFM images (e.g. ray-traced vs
// beam-traced soft-shadow visibility). Pixels where the reference is
// negative (background) are ignored.
//
//   bt_compare reference.pfm test.pfm [diff.pfm]
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool readPFM(const char* path, int& w, int& h, std::vector<float>& rgb) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  char magic[3] = {0, 0, 0};
  float scale;
  if (fscanf(f, "%2s %d %d %f", magic, &w, &h, &scale) != 4 || std::string(magic) != "PF") {
    fclose(f);
    return false;
  }
  fgetc(f);
  rgb.resize(size_t(w) * size_t(h) * 3);
  size_t got = fread(rgb.data(), sizeof(float), rgb.size(), f);
  fclose(f);
  return got == rgb.size();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: bt_compare reference.pfm test.pfm [diff.pfm]\n");
    return 2;
  }
  int w0, h0, w1, h1;
  std::vector<float> a, b;
  if (!readPFM(argv[1], w0, h0, a) || !readPFM(argv[2], w1, h1, b) || w0 != w1 || h0 != h1) {
    fprintf(stderr, "cannot read images or size mismatch\n");
    return 1;
  }
  double se = 0, ae = 0, maxe = 0;
  size_t n = 0, over1 = 0, over5 = 0;
  std::vector<float> diff(a.size(), 0.f);
  for (size_t i = 0; i < a.size(); i += 3) {
    if (a[i] < 0 || b[i] < 0) continue;
    double e = std::fabs(double(a[i]) - double(b[i]));
    se += e * e;
    ae += e;
    if (e > maxe) maxe = e;
    if (e > 0.01) ++over1;
    if (e > 0.05) ++over5;
    diff[i] = diff[i + 1] = diff[i + 2] = float(e);
    ++n;
  }
  if (n == 0) n = 1;
  printf("rmse %.6g mae %.6g max %.6g frac>0.01 %.6g frac>0.05 %.6g pixels %zu\n", std::sqrt(se / double(n)),
         ae / double(n), maxe, double(over1) / double(n), double(over5) / double(n), n);
  if (argc > 3) {
    FILE* f = fopen(argv[3], "wb");
    if (f) {
      fprintf(f, "PF\n%d %d\n-1.0\n", w0, h0);
      fwrite(diff.data(), sizeof(float), diff.size(), f);
      fclose(f);
    }
  }
  return 0;
}
