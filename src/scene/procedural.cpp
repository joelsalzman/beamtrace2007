// Procedural scenes: stand-ins for the paper's non-public scenes, plus test scenes.
//
//   room        ~0.9K tris, an Erw6-like furnished room
//   plant       ~5.2K tris, a potted plant whose leaves are almost all silhouette edges
//   building    ~2.2M tris (default), a multi-floor office building (Soda Hall scale test)
//   tess_sweep  floor + 5x5 spheres, `stacks=` controls tessellation
//   analytic    floor + square occluder + (config) square light: closed-form soft shadows
//   random_tris `n=` random, interpenetrating triangles in the unit cube (tests)
//   sphere      one tessellated sphere (crack tests)
#include <cmath>

#include "scene/scene.h"
#include "util/misc.h"

namespace bt {

namespace {

double param(const std::map<std::string, std::string>& p, const std::string& k, double def) {
  auto it = p.find(k);
  return it == p.end() ? def : std::stod(it->second);
}

void addCylinder(Scene& s, const Vec3& base, Real r, Real h, int seg, uint16_t mat) {
  std::vector<uint32_t> lo(static_cast<size_t>(seg)), hi(static_cast<size_t>(seg));
  for (int i = 0; i < seg; ++i) {
    double a = 2 * M_PI * i / seg;
    Vec3 d(Real(std::cos(a)) * r, 0, Real(std::sin(a)) * r);
    lo[size_t(i)] = s.addVertex(base + d);
    hi[size_t(i)] = s.addVertex(base + d + Vec3(0, h, 0));
  }
  uint32_t cb = s.addVertex(base);
  for (int i = 0; i < seg; ++i) {
    int j = (i + 1) % seg;
    s.addTri(lo[size_t(i)], hi[size_t(i)], hi[size_t(j)], mat);
    s.addTri(lo[size_t(i)], hi[size_t(j)], lo[size_t(j)], mat);
    s.addTri(cb, lo[size_t(i)], lo[size_t(j)], mat);
  }
}

void addChair(Scene& s, const Vec3& p, Real sz, uint16_t mat) {
  Real leg = Real(0.05) * sz, seatH = Real(0.45) * sz, w = Real(0.45) * sz;
  for (int i = 0; i < 4; ++i) {
    Real x = (i & 1) ? w - leg : 0, z = (i & 2) ? w - leg : 0;
    addBox(s, p + Vec3(x, 0, z), p + Vec3(x + leg, seatH, z + leg), mat);
  }
  addBox(s, p + Vec3(0, seatH, 0), p + Vec3(w, seatH + Real(0.05) * sz, w), mat);
  addBox(s, p + Vec3(0, seatH, w - leg), p + Vec3(w, seatH + Real(0.5) * sz, w), mat);
}

void addTable(Scene& s, const Vec3& p, Real wx, Real wz, Real h, uint16_t mat) {
  Real leg = Real(0.06);
  addBox(s, p + Vec3(0, h - Real(0.05), 0), p + Vec3(wx, h, wz), mat);
  for (int i = 0; i < 4; ++i) {
    Real x = (i & 1) ? wx - leg : 0, z = (i & 2) ? wz - leg : 0;
    addBox(s, p + Vec3(x, 0, z), p + Vec3(x + leg, h - Real(0.05), z + leg), mat);
  }
}

// A wall slab along x (axis=0) or z (axis=2) with an optional door gap.
void addWall(Scene& s, const Vec3& o, int axis, Real len, Real height, Real thick, bool door, uint16_t mat) {
  auto box = [&](Real a0, Real a1, Real y0, Real y1) {
    if (axis == 0)
      addBox(s, o + Vec3(a0, y0, 0), o + Vec3(a1, y1, thick), mat);
    else
      addBox(s, o + Vec3(0, y0, a0), o + Vec3(thick, y1, a1), mat);
  };
  if (!door) {
    box(0, len, 0, height);
    return;
  }
  Real dw = Real(1.2), dh = Real(2.2);
  Real a = (len - dw) / 2;
  box(0, a, 0, height);
  box(a + dw, len, 0, height);
  box(a, a + dw, dh, height);
}

void buildRoom(Scene& s) {
  uint16_t wall = s.addMaterial(Vec3(0.8f, 0.78f, 0.72f));
  uint16_t wood = s.addMaterial(Vec3(0.6f, 0.42f, 0.28f));
  uint16_t blue = s.addMaterial(Vec3(0.35f, 0.45f, 0.75f));
  uint16_t red = s.addMaterial(Vec3(0.75f, 0.3f, 0.25f));
  addBox(s, Vec3(0, 0, 0), Vec3(8, 3, 8), wall, true);
  addTable(s, Vec3(3, 0, 3), 2, Real(1.2), Real(0.75), wood);
  addTable(s, Vec3(0.5f, 0, 6), Real(1.5), Real(1.5), Real(0.75), wood);
  for (int i = 0; i < 4; ++i) addChair(s, Vec3(Real(2.6 + 0.8 * i), 0, Real(4.4)), 1, blue);
  for (int i = 0; i < 2; ++i) addChair(s, Vec3(Real(2.8 + 1.0 * i), 0, Real(2.2)), 1, blue);
  // shelf with a few books
  addBox(s, Vec3(7.3f, 0, 1), Vec3(7.9f, 2, 3.5f), wood);
  for (int i = 0; i < 8; ++i) addBox(s, Vec3(7.2f, 1.2f, Real(1.1 + 0.28 * i)), Vec3(7.3f, Real(1.45 + 0.04 * (i % 3)), Real(1.3 + 0.28 * i)), red);
  addSphere(s, Vec3(4, Real(0.95), Real(3.6)), Real(0.2), 6, 10, red);
  addSphere(s, Vec3(1.2f, Real(1.0), Real(6.7)), Real(0.25), 6, 10, blue);
}

void buildPlant(Scene& s, uint64_t seed) {
  uint16_t floorM = s.addMaterial(Vec3(0.75f, 0.72f, 0.68f));
  uint16_t potM = s.addMaterial(Vec3(0.7f, 0.4f, 0.25f));
  uint16_t stemM = s.addMaterial(Vec3(0.35f, 0.5f, 0.2f));
  uint16_t leafM = s.addMaterial(Vec3(0.3f, 0.65f, 0.25f));
  addQuad(s, Vec3(-3, 0, -3), Vec3(-3, 0, 3), Vec3(3, 0, 3), Vec3(3, 0, -3), floorM);
  addQuad(s, Vec3(-3, 0, -1.5f), Vec3(3, 0, -1.5f), Vec3(3, 3, -1.5f), Vec3(-3, 3, -1.5f), floorM);
  addCylinder(s, Vec3(0, 0, 0), Real(0.3), Real(0.45), 24, potM);
  Rng rng(seed);
  const int branches = 9;
  const int leavesPerBranch = 250;  // 9 * 250 * 2 = 4500 leaf triangles (+ stems, pot, floor ~ 5.2K)
  for (int b = 0; b < branches; ++b) {
    double az = 2 * M_PI * (b + rng.uniform() * 0.5) / branches;
    double el = 0.9 + 0.5 * rng.uniform();
    Vec3 dir(Real(std::cos(az) * std::cos(el)), Real(std::sin(el)), Real(std::sin(az) * std::cos(el)));
    Vec3 start(0, Real(0.4), 0);
    Real len = Real(0.9 + 0.5 * rng.uniform());
    Vec3 endp = start + dir * len;
    for (int k = 0; k < 6; ++k) {  // stem as a chain of small cubes
      Vec3 c = start + (endp - start) * (Real(k) / 5);
      addBox(s, c - Vec3(0.015f, 0.015f, 0.015f), c + Vec3(0.015f, 0.015f, 0.015f), stemM);
    }
    for (int l = 0; l < leavesPerBranch; ++l) {
      Real t = Real(0.2 + 0.8 * rng.uniform());
      Vec3 c = start + dir * (t * len) +
               Vec3(Real(rng.uniform(-0.25, 0.25)), Real(rng.uniform(-0.15, 0.25)), Real(rng.uniform(-0.25, 0.25)));
      Vec3 a = normalize(Vec3(Real(rng.uniform(-1, 1)), Real(rng.uniform(-0.3, 1)), Real(rng.uniform(-1, 1))));
      Vec3 n = anyOrthogonal(a);
      Vec3 bvec = normalize(cross(a, n) + n * Real(rng.uniform(-0.5, 0.5)));
      Real L = Real(0.06 + 0.05 * rng.uniform()), W = L * Real(0.35);
      uint32_t i0 = s.addVertex(c - a * L), i1 = s.addVertex(c + bvec * W), i2 = s.addVertex(c + a * L),
               i3 = s.addVertex(c - bvec * W);
      s.addTri(i0, i1, i2, leafM);
      s.addTri(i0, i2, i3, leafM);
    }
  }
}

void buildBuilding(Scene& s, int floors, int rooms, int detail) {
  uint16_t wallM = s.addMaterial(Vec3(0.78f, 0.76f, 0.7f));
  uint16_t floorM = s.addMaterial(Vec3(0.55f, 0.5f, 0.45f));
  uint16_t woodM = s.addMaterial(Vec3(0.6f, 0.42f, 0.28f));
  uint16_t chairM = s.addMaterial(Vec3(0.35f, 0.45f, 0.75f));
  uint16_t ballM = s.addMaterial(Vec3(0.75f, 0.35f, 0.3f));
  uint16_t bookM = s.addMaterial(Vec3(0.4f, 0.6f, 0.4f));
  const Real W = 5, H = 3, T = Real(0.15);
  for (int f = 0; f < floors; ++f) {
    for (int i = 0; i < rooms; ++i) {
      for (int j = 0; j < rooms; ++j) {
        Vec3 o(Real(i) * W, Real(f) * H, Real(j) * W);
        addQuad(s, o, o + Vec3(0, 0, W), o + Vec3(W, 0, W), o + Vec3(W, 0, 0), floorM);
        addQuad(s, o + Vec3(0, H - T, 0), o + Vec3(W, H - T, 0), o + Vec3(W, H - T, W), o + Vec3(0, H - T, W), wallM);
        addWall(s, o, 2, W, H - T, T, true, wallM);   // west wall (x = x0), door
        addWall(s, o, 0, W, H - T, T, true, wallM);   // south wall (z = z0), door
        if (i == rooms - 1) addWall(s, o + Vec3(W - T, 0, 0), 2, W, H - T, T, false, wallM);
        if (j == rooms - 1) addWall(s, o + Vec3(0, 0, W - T), 0, W, H - T, T, false, wallM);
        // furniture, offset so doors (centered on walls) stay clear
        addTable(s, o + Vec3(Real(0.6), 0, Real(3.2)), Real(1.6), Real(0.8), Real(0.75), woodM);
        addChair(s, o + Vec3(Real(1.1), 0, Real(2.5)), 1, chairM);
        addBox(s, o + Vec3(Real(3.9), 0, Real(3.3)), o + Vec3(Real(4.7), Real(1.8), Real(4.7)), woodM);
        for (int b = 0; b < 16; ++b)
          addBox(s, o + Vec3(Real(3.8), Real(0.3 + 0.35 * (b / 8)), Real(3.4 + 0.16 * (b % 8))),
                 o + Vec3(Real(3.9), Real(0.58 + 0.35 * (b / 8)), Real(3.52 + 0.16 * (b % 8))), bookM);
        addSphere(s, o + Vec3(Real(1.4), Real(1.05), Real(3.6)), Real(0.28), detail, 2 * detail, ballM);
      }
    }
  }
}

void buildTessSweep(Scene& s, int stacks) {
  uint16_t floorM = s.addMaterial(Vec3(0.75f, 0.75f, 0.75f));
  uint16_t ballM = s.addMaterial(Vec3(0.8f, 0.5f, 0.35f));
  addQuad(s, Vec3(-6, 0, -6), Vec3(-6, 0, 6), Vec3(6, 0, 6), Vec3(6, 0, -6), floorM);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      addSphere(s, Vec3(Real(-4 + 2 * i), Real(0.8), Real(-4 + 2 * j)), Real(0.75), stacks, 2 * stacks, ballM);
}

}  // namespace

bool buildProcedural(const std::string& name, const std::map<std::string, std::string>& p, Scene& s,
                     std::string& err) {
  if (name == "room") {
    buildRoom(s);
  } else if (name == "plant") {
    buildPlant(s, uint64_t(param(p, "seed", 7)));
  } else if (name == "building") {
    buildBuilding(s, int(param(p, "floors", 7)), int(param(p, "rooms", 8)), int(param(p, "detail", 33)));
  } else if (name == "tess_sweep") {
    buildTessSweep(s, int(param(p, "stacks", 16)));
  } else if (name == "analytic") {
    Real h = Real(param(p, "h", 1)), occ = Real(param(p, "occ", 1)), F = Real(param(p, "floor", 10));
    Real ox = Real(param(p, "ox", 0)), oz = Real(param(p, "oz", 0));
    uint16_t floorM = s.addMaterial(Vec3(0.8f, 0.8f, 0.8f));
    uint16_t occM = s.addMaterial(Vec3(0.8f, 0.4f, 0.3f));
    addQuad(s, Vec3(-F, 0, -F), Vec3(-F, 0, F), Vec3(F, 0, F), Vec3(F, 0, -F), floorM);
    Real a = occ / 2;
    addQuad(s, Vec3(ox - a, h, oz - a), Vec3(ox - a, h, oz + a), Vec3(ox + a, h, oz + a), Vec3(ox + a, h, oz - a), occM);
  } else if (name == "random_tris") {
    int n = int(param(p, "n", 50));
    Rng rng(uint64_t(param(p, "seed", 1)));
    Real size = Real(param(p, "size", 0.3));
    uint16_t m = s.addMaterial(Vec3(0.7f, 0.7f, 0.7f));
    for (int i = 0; i < n; ++i) {
      Vec3 c(Real(rng.uniform()), Real(rng.uniform()), Real(rng.uniform()));
      uint32_t idx[3];
      for (int k = 0; k < 3; ++k)
        idx[k] = s.addVertex(c + Vec3(Real(rng.uniform(-size, size)), Real(rng.uniform(-size, size)),
                                      Real(rng.uniform(-size, size))));
      s.addTri(idx[0], idx[1], idx[2], m);
    }
  } else if (name == "sphere") {
    uint16_t m = s.addMaterial(Vec3(0.7f, 0.7f, 0.7f));
    int st = int(param(p, "stacks", 24));
    addSphere(s, Vec3(0, 0, 0), Real(param(p, "r", 1)), st, 2 * st, m);
  } else {
    err = "unknown procedural scene '" + name + "'";
    return false;
  }
  return true;
}

}  // namespace bt
