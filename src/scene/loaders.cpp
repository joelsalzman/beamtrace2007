// OBJ and PLY loaders (positions and faces only; polygons are fan-triangulated).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "scene/scene.h"
#include "util/misc.h"

namespace bt {

namespace {

bool readFile(const std::string& path, std::string& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize(size_t(n));
  size_t got = n > 0 ? fread(&out[0], 1, size_t(n), f) : 0;
  fclose(f);
  return got == size_t(n);
}

}  // namespace

bool loadOBJ(const std::string& path, Scene& scene, uint16_t mat, std::string& err) {
  std::string data;
  if (!readFile(path, data)) {
    err = "cannot read " + path;
    return false;
  }
  const uint32_t base = uint32_t(scene.pos.size());
  uint32_t nv = 0;
  std::vector<long> face;
  const char* p = data.c_str();
  const char* end = p + data.size();
  while (p < end) {
    const char* eol = static_cast<const char*>(memchr(p, '\n', size_t(end - p)));
    if (!eol) eol = end;
    while (p < eol && (*p == ' ' || *p == '\t')) ++p;
    if (p + 1 < eol && p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
      char* q;
      double x = strtod(p + 1, &q);
      double y = strtod(q, &q);
      double z = strtod(q, &q);
      scene.addVertex(Vec3(Real(x), Real(y), Real(z)));
      ++nv;
    } else if (p + 1 < eol && p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
      face.clear();
      const char* q = p + 1;
      while (q < eol) {
        while (q < eol && (*q == ' ' || *q == '\t' || *q == '\r')) ++q;
        if (q >= eol) break;
        char* r;
        long idx = strtol(q, &r, 10);
        if (r == q) break;
        face.push_back(idx);
        q = r;
        while (q < eol && *q != ' ' && *q != '\t') ++q;  // skip /vt/vn
      }
      auto resolve = [&](long i) -> long { return i > 0 ? i - 1 : long(nv) + i; };
      for (size_t k = 2; k < face.size(); ++k) {
        long a = resolve(face[0]), b = resolve(face[k - 1]), c = resolve(face[k]);
        if (a < 0 || b < 0 || c < 0 || a >= long(nv) || b >= long(nv) || c >= long(nv)) continue;
        scene.addTri(base + uint32_t(a), base + uint32_t(b), base + uint32_t(c), mat);
      }
    }
    p = eol + 1;
  }
  if (nv == 0) {
    err = "no vertices in " + path;
    return false;
  }
  return true;
}

namespace {

enum class PlyType { I8, U8, I16, U16, I32, U32, F32, F64, Invalid };

PlyType plyType(const std::string& s) {
  if (s == "char" || s == "int8") return PlyType::I8;
  if (s == "uchar" || s == "uint8") return PlyType::U8;
  if (s == "short" || s == "int16") return PlyType::I16;
  if (s == "ushort" || s == "uint16") return PlyType::U16;
  if (s == "int" || s == "int32") return PlyType::I32;
  if (s == "uint" || s == "uint32") return PlyType::U32;
  if (s == "float" || s == "float32") return PlyType::F32;
  if (s == "double" || s == "float64") return PlyType::F64;
  return PlyType::Invalid;
}

int plySize(PlyType t) {
  switch (t) {
    case PlyType::I8: case PlyType::U8: return 1;
    case PlyType::I16: case PlyType::U16: return 2;
    case PlyType::I32: case PlyType::U32: case PlyType::F32: return 4;
    case PlyType::F64: return 8;
    default: return 0;
  }
}

struct PlyProp {
  std::string name;
  PlyType type = PlyType::Invalid;
  bool isList = false;
  PlyType countType = PlyType::Invalid;
};

struct PlyElement {
  std::string name;
  size_t count = 0;
  std::vector<PlyProp> props;
};

double readBin(const char*& p, PlyType t, bool bigEndian) {
  unsigned char b[8];
  int n = plySize(t);
  for (int i = 0; i < n; ++i) b[i] = static_cast<unsigned char>(p[bigEndian ? n - 1 - i : i]);
  p += n;
  switch (t) {
    case PlyType::I8: return double(int8_t(b[0]));
    case PlyType::U8: return double(b[0]);
    case PlyType::I16: { int16_t v; memcpy(&v, b, 2); return v; }
    case PlyType::U16: { uint16_t v; memcpy(&v, b, 2); return v; }
    case PlyType::I32: { int32_t v; memcpy(&v, b, 4); return v; }
    case PlyType::U32: { uint32_t v; memcpy(&v, b, 4); return v; }
    case PlyType::F32: { float v; memcpy(&v, b, 4); return v; }
    case PlyType::F64: { double v; memcpy(&v, b, 8); return v; }
    default: return 0;
  }
}

}  // namespace

bool loadPLY(const std::string& path, Scene& scene, uint16_t mat, std::string& err) {
  std::string data;
  if (!readFile(path, data)) {
    err = "cannot read " + path;
    return false;
  }
  size_t hdrEnd = data.find("end_header");
  if (data.compare(0, 3, "ply") != 0 || hdrEnd == std::string::npos) {
    err = path + ": not a PLY file";
    return false;
  }
  size_t bodyStart = data.find('\n', hdrEnd);
  if (bodyStart == std::string::npos) {
    err = path + ": truncated header";
    return false;
  }
  ++bodyStart;
  std::istringstream hs(data.substr(0, hdrEnd));
  std::string line;
  std::string format;
  std::vector<PlyElement> elems;
  while (std::getline(hs, line)) {
    auto tok = splitWS(line);
    if (tok.empty()) continue;
    if (tok[0] == "format" && tok.size() > 1) {
      format = tok[1];
    } else if (tok[0] == "element" && tok.size() > 2) {
      PlyElement e;
      e.name = tok[1];
      e.count = size_t(std::stoull(tok[2]));
      elems.push_back(e);
    } else if (tok[0] == "property" && !elems.empty()) {
      PlyProp pr;
      if (tok.size() > 4 && tok[1] == "list") {
        pr.isList = true;
        pr.countType = plyType(tok[2]);
        pr.type = plyType(tok[3]);
        pr.name = tok[4];
      } else if (tok.size() > 2) {
        pr.type = plyType(tok[1]);
        pr.name = tok[2];
      }
      elems.back().props.push_back(pr);
    }
  }
  const bool ascii = format == "ascii";
  const bool bigEndian = format == "binary_big_endian";
  if (!ascii && format != "binary_little_endian" && !bigEndian) {
    err = path + ": unsupported PLY format " + format;
    return false;
  }
  const uint32_t base = uint32_t(scene.pos.size());
  uint32_t nv = 0;
  const char* p = data.c_str() + bodyStart;
  const char* end = data.c_str() + data.size();
  auto readVal = [&](PlyType t) -> double {
    if (ascii) {
      char* q;
      double v = strtod(p, &q);
      p = q;
      return v;
    }
    if (p + plySize(t) > end) return 0;
    return readBin(p, t, bigEndian);
  };
  std::vector<long> idx;
  for (const PlyElement& e : elems) {
    for (size_t i = 0; i < e.count; ++i) {
      double xyz[3] = {0, 0, 0};
      idx.clear();
      for (const PlyProp& pr : e.props) {
        if (pr.isList) {
          long n = long(readVal(pr.countType));
          for (long k = 0; k < n; ++k) {
            double v = readVal(pr.type);
            if (pr.name == "vertex_indices" || pr.name == "vertex_index") idx.push_back(long(v));
          }
        } else {
          double v = readVal(pr.type);
          if (pr.name == "x") xyz[0] = v;
          else if (pr.name == "y") xyz[1] = v;
          else if (pr.name == "z") xyz[2] = v;
        }
      }
      if (e.name == "vertex") {
        scene.addVertex(Vec3(Real(xyz[0]), Real(xyz[1]), Real(xyz[2])));
        ++nv;
      } else if (e.name == "face") {
        for (size_t k = 2; k < idx.size(); ++k) {
          long a = idx[0], b = idx[k - 1], c = idx[k];
          if (a < 0 || b < 0 || c < 0 || a >= long(nv) || b >= long(nv) || c >= long(nv)) continue;
          scene.addTri(base + uint32_t(a), base + uint32_t(b), base + uint32_t(c), mat);
        }
      }
    }
  }
  if (nv == 0) {
    err = path + ": no vertices";
    return false;
  }
  return true;
}

}  // namespace bt
