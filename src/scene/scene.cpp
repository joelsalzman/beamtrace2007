#include <cmath>
#include <fstream>
#include <sstream>

#include "scene/scene.h"
#include "util/misc.h"

namespace bt {

void Scene::finalize() {
  std::vector<std::array<uint32_t, 3>> keep;
  std::vector<uint16_t> keepMat;
  keep.reserve(tris.size());
  keepMat.reserve(tris.size());
  triN.clear();
  triArea.clear();
  bounds = AABB();
  for (size_t i = 0; i < tris.size(); ++i) {
    const auto& t = tris[i];
    // Compute in double so the degeneracy test is the same for both builds.
    const Vec3 &a = pos[t[0]], &b = pos[t[1]], &c = pos[t[2]];
    double e1x = double(b.x) - a.x, e1y = double(b.y) - a.y, e1z = double(b.z) - a.z;
    double e2x = double(c.x) - a.x, e2y = double(c.y) - a.y, e2z = double(c.z) - a.z;
    double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    double l = std::sqrt(nx * nx + ny * ny + nz * nz);
    double scale = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z) * std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);
    if (!(l > 1e-7 * scale) || t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) continue;
    keep.push_back(t);
    keepMat.push_back(triMat[i]);
    triN.push_back(Vec3(Real(nx / l), Real(ny / l), Real(nz / l)));
    triArea.push_back(Real(0.5 * l));
    bounds.expand(a);
    bounds.expand(b);
    bounds.expand(c);
  }
  tris.swap(keep);
  triMat.swap(keepMat);
  if (matColor.empty()) matColor.push_back(Vec3(Real(0.75), Real(0.75), Real(0.75)));
}

void addQuad(Scene& s, const Vec3& a, const Vec3& b, const Vec3& c, const Vec3& d, uint16_t mat) {
  uint32_t i0 = s.addVertex(a), i1 = s.addVertex(b), i2 = s.addVertex(c), i3 = s.addVertex(d);
  s.addTri(i0, i1, i2, mat);
  s.addTri(i0, i2, i3, mat);
}

void addBox(Scene& s, const Vec3& lo, const Vec3& hi, uint16_t mat, bool inward) {
  // 8 shared corners so adjacent faces share edges.
  uint32_t v[8];
  for (int i = 0; i < 8; ++i)
    v[i] = s.addVertex(Vec3((i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z));
  // Faces listed counter-clockwise when seen from outside.
  const int f[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
  for (auto& q : f) {
    if (!inward) {
      s.addTri(v[q[0]], v[q[1]], v[q[2]], mat);
      s.addTri(v[q[0]], v[q[2]], v[q[3]], mat);
    } else {
      s.addTri(v[q[0]], v[q[2]], v[q[1]], mat);
      s.addTri(v[q[0]], v[q[3]], v[q[2]], mat);
    }
  }
}

void addSphere(Scene& s, const Vec3& c, Real r, int stacks, int slices, uint16_t mat) {
  if (stacks < 2) stacks = 2;
  if (slices < 3) slices = 3;
  uint32_t top = s.addVertex(c + Vec3(0, r, 0));
  uint32_t bot = s.addVertex(c - Vec3(0, r, 0));
  std::vector<uint32_t> ring(size_t(stacks - 1) * slices);
  for (int i = 1; i < stacks; ++i) {
    double th = M_PI * i / stacks;
    for (int j = 0; j < slices; ++j) {
      double ph = 2 * M_PI * j / slices;
      Vec3 p(Real(std::sin(th) * std::cos(ph)), Real(std::cos(th)), Real(std::sin(th) * std::sin(ph)));
      ring[size_t(i - 1) * slices + j] = s.addVertex(c + p * r);
    }
  }
  auto at = [&](int i, int j) { return ring[size_t(i - 1) * slices + (j % slices)]; };
  for (int j = 0; j < slices; ++j) {
    s.addTri(top, at(1, j + 1), at(1, j), mat);
    s.addTri(bot, at(stacks - 1, j), at(stacks - 1, j + 1), mat);
  }
  for (int i = 1; i < stacks - 1; ++i)
    for (int j = 0; j < slices; ++j) {
      s.addTri(at(i, j), at(i, j + 1), at(i + 1, j + 1), mat);
      s.addTri(at(i, j), at(i + 1, j + 1), at(i + 1, j), mat);
    }
}

namespace {

bool parseReals(const std::vector<std::string>& tok, size_t from, size_t count, Real* out, std::string& err) {
  if (tok.size() < from + count) {
    err = "expected " + std::to_string(count) + " numbers after '" + tok[0] + "'";
    return false;
  }
  for (size_t i = 0; i < count; ++i) out[i] = Real(std::stod(tok[from + i]));
  return true;
}

}  // namespace

bool parseConfig(const std::string& text, SceneConfig& cfg, std::string& err) {
  std::istringstream is(text);
  std::string line;
  int lineNo = 0;
  try {
    while (std::getline(is, line)) {
      ++lineNo;
      size_t hash = line.find('#');
      if (hash != std::string::npos) line = line.substr(0, hash);
      auto tok = splitWS(line);
      if (tok.empty()) continue;
      const std::string& k = tok[0];
      Real v[9];
      if (k == "name") {
        cfg.name = tok.size() > 1 ? tok[1] : "";
      } else if (k == "mesh") {
        cfg.meshLines.push_back(trim(line.substr(line.find("mesh") + 4)));
      } else if (k == "cull_backfaces") {
        cfg.cullBackfaces = tok.size() > 1 && tok[1] != "0";
      } else if (k == "up") {
        if (!parseReals(tok, 1, 3, v, err)) return false;
        cfg.up = Vec3(v[0], v[1], v[2]);
      } else if (k == "view") {
        if (!parseReals(tok, 1, 7, v, err)) return false;
        View w;
        w.eye = Vec3(v[0], v[1], v[2]);
        w.target = Vec3(v[3], v[4], v[5]);
        w.fovY = v[6];
        cfg.views.push_back(w);
      } else if (k == "orbit") {
        // orbit cx cy cz radius height count fov : views on a circle around (cx,cy,cz)
        if (!parseReals(tok, 1, 7, v, err)) return false;
        int n = int(v[5]);
        Vec3 c(v[0], v[1], v[2]);
        Vec3 upv = normalize(cfg.up);
        Vec3 a = anyOrthogonal(upv), b = cross(upv, a);
        for (int i = 0; i < n; ++i) {
          double ang = 2 * M_PI * (i + 0.125) / n;
          View w;
          w.eye = c + a * Real(v[3] * std::cos(ang)) + b * Real(v[3] * std::sin(ang)) + upv * v[4];
          w.target = c;
          w.fovY = v[6];
          cfg.views.push_back(w);
        }
      } else if (k == "walk") {
        // walk x0 y0 z0 x1 y1 z1 count fov : views along a segment looking forward
        if (!parseReals(tok, 1, 8, v, err)) return false;
        Vec3 p0(v[0], v[1], v[2]), p1(v[3], v[4], v[5]);
        int n = int(v[6]);
        Vec3 dir = normalize(p1 - p0);
        for (int i = 0; i < n; ++i) {
          Real t = n > 1 ? Real(i) / Real(n - 1) : 0;
          View w;
          w.eye = p0 + (p1 - p0) * (t * Real(0.85));
          // look forward, swinging left/right a little for variety
          Vec3 side = normalize(cross(dir, normalize(cfg.up)));
          Real swing = Real(0.6 * std::sin(2.3 * i));
          w.target = w.eye + dir + side * swing;
          w.fovY = v[7];
          cfg.views.push_back(w);
        }
      } else if (k == "point_light") {
        if (!parseReals(tok, 1, 3, v, err)) return false;
        cfg.hasPointLight = true;
        cfg.pointLight = Vec3(v[0], v[1], v[2]);
      } else if (k == "area_light") {
        if (!parseReals(tok, 1, 9, v, err)) return false;
        cfg.hasAreaLight = true;
        cfg.lightC = Vec3(v[0], v[1], v[2]);
        cfg.lightU = Vec3(v[3], v[4], v[5]);
        cfg.lightV = Vec3(v[6], v[7], v[8]);
      } else if (k == "light_intensity") {
        if (!parseReals(tok, 1, 1, v, err)) return false;
        cfg.lightIntensity = v[0];
      } else if (k == "background") {
        if (!parseReals(tok, 1, 3, v, err)) return false;
        cfg.background = Vec3(v[0], v[1], v[2]);
      } else {
        err = "unknown key '" + k + "'";
        return false;
      }
    }
  } catch (const std::exception& e) {
    err = "line " + std::to_string(lineNo) + ": " + e.what();
    return false;
  }
  return true;
}

bool loadConfig(const std::string& path, SceneConfig& cfg, std::string& err) {
  std::ifstream f(path);
  if (!f) {
    err = "cannot open config " + path;
    return false;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  cfg.path = path;
  if (!parseConfig(ss.str(), cfg, err)) {
    err = path + ": " + err;
    return false;
  }
  return true;
}

bool buildScene(const SceneConfig& cfg, Scene& scene, std::string& err) {
  static const Vec3 palette[] = {{0.75f, 0.75f, 0.75f}, {0.8f, 0.55f, 0.4f}, {0.45f, 0.65f, 0.8f},
                                 {0.6f, 0.8f, 0.45f},   {0.85f, 0.8f, 0.5f}, {0.7f, 0.5f, 0.75f}};
  int meshIndex = 0;
  for (const std::string& ml : cfg.meshLines) {
    auto tok = splitWS(ml);
    if (tok.size() < 2) {
      err = "bad mesh line: " + ml;
      return false;
    }
    // Optional modifiers: scale s | translate x y z | color r g b ; procedural takes key=value.
    Real scale = 1;
    Vec3 translate;
    Vec3 color = palette[meshIndex % 6];
    std::map<std::string, std::string> params;
    for (size_t i = 2; i < tok.size(); ++i) {
      if (tok[i] == "scale" && i + 1 < tok.size()) {
        scale = Real(std::stod(tok[++i]));
      } else if (tok[i] == "translate" && i + 3 < tok.size()) {
        translate = Vec3(Real(std::stod(tok[i + 1])), Real(std::stod(tok[i + 2])), Real(std::stod(tok[i + 3])));
        i += 3;
      } else if (tok[i] == "color" && i + 3 < tok.size()) {
        color = Vec3(Real(std::stod(tok[i + 1])), Real(std::stod(tok[i + 2])), Real(std::stod(tok[i + 3])));
        i += 3;
      } else if (tok[i].find('=') != std::string::npos) {
        auto p = tok[i].find('=');
        params[tok[i].substr(0, p)] = tok[i].substr(p + 1);
      }
    }
    size_t firstVertex = scene.pos.size();
    bool ok;
    if (tok[0] == "obj") {
      ok = loadOBJ(tok[1], scene, scene.addMaterial(color), err);
    } else if (tok[0] == "ply") {
      ok = loadPLY(tok[1], scene, scene.addMaterial(color), err);
    } else if (tok[0] == "procedural") {
      ok = buildProcedural(tok[1], params, scene, err);
    } else {
      err = "unknown mesh type " + tok[0];
      return false;
    }
    if (!ok) return false;
    for (size_t i = firstVertex; i < scene.pos.size(); ++i) scene.pos[i] = scene.pos[i] * scale + translate;
    ++meshIndex;
  }
  scene.finalize();
  if (scene.numTris() == 0) {
    err = "scene has no triangles";
    return false;
  }
  return true;
}

}  // namespace bt
