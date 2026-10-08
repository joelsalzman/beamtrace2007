// Triangle scene, loaders, procedural generators and scene configs.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/vec.h"

namespace bt {

struct Scene {
  std::vector<Vec3> pos;
  std::vector<std::array<uint32_t, 3>> tris;
  std::vector<Vec3> triN;      // unit geometric normal, from winding (v1-v0)x(v2-v0)
  std::vector<Real> triArea;
  std::vector<uint16_t> triMat;
  std::vector<Vec3> matColor;  // diffuse albedo per material
  AABB bounds;

  uint32_t addVertex(const Vec3& p) {
    pos.push_back(p);
    return uint32_t(pos.size() - 1);
  }
  void addTri(uint32_t a, uint32_t b, uint32_t c, uint16_t mat = 0) {
    tris.push_back({a, b, c});
    triMat.push_back(mat);
  }
  uint16_t addMaterial(const Vec3& albedo) {
    matColor.push_back(albedo);
    return uint16_t(matColor.size() - 1);
  }
  int numTris() const { return int(tris.size()); }
  Vec3 v(int tri, int k) const { return pos[tris[size_t(tri)][size_t(k)]]; }
  // Removes zero-area triangles, computes normals, areas and bounds.
  void finalize();
};

struct View {
  Vec3 eye, target;
  Real fovY = 60;  // degrees
};

struct SceneConfig {
  std::string name;
  std::string path;  // config file path
  std::vector<std::string> meshLines;
  bool cullBackfaces = false;
  Vec3 up{0, 1, 0};
  std::vector<View> views;
  bool hasPointLight = false;
  Vec3 pointLight;
  bool hasAreaLight = false;
  Vec3 lightC, lightU, lightV;  // center and full edge vectors; emits along normalize(U x V)
  Real lightIntensity = 1;
  Vec3 background{Real(0.0), Real(0.0), Real(0.0)};
};

bool loadConfig(const std::string& path, SceneConfig& cfg, std::string& err);
// Parses config text directly (used by tests and built-in scenes).
bool parseConfig(const std::string& text, SceneConfig& cfg, std::string& err);
bool buildScene(const SceneConfig& cfg, Scene& scene, std::string& err);

// Loaders append to `scene` using material `mat`.
bool loadOBJ(const std::string& path, Scene& scene, uint16_t mat, std::string& err);
bool loadPLY(const std::string& path, Scene& scene, uint16_t mat, std::string& err);

// Procedural generators. `params` are key=value pairs from the config line.
bool buildProcedural(const std::string& name, const std::map<std::string, std::string>& params,
                     Scene& scene, std::string& err);

// Geometry helpers used by generators and tests.
void addQuad(Scene& s, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, uint16_t mat);
void addBox(Scene& s, const Vec3& lo, const Vec3& hi, uint16_t mat, bool inward = false);
void addSphere(Scene& s, const Vec3& c, Real r, int stacks, int slices, uint16_t mat);

}  // namespace bt
