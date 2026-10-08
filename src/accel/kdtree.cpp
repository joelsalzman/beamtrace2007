#include "accel/kdtree.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "util/misc.h"

namespace bt {

namespace {

struct Ref {
  uint32_t tri;
  AABB b;
};

}  // namespace

// Bounds of triangle `t` clipped to `box` (computed in double).
AABB clippedTriangleBounds(const Scene& s, uint32_t t, const AABB& box, const AABB& fallback) {
  double poly[2][12][3];
  int n = 3;
  for (int k = 0; k < 3; ++k) {
    Vec3 p = s.v(int(t), k);
    poly[0][k][0] = p.x;
    poly[0][k][1] = p.y;
    poly[0][k][2] = p.z;
  }
  int cur = 0;
  for (int axis = 0; axis < 3 && n > 0; ++axis) {
    for (int side = 0; side < 2 && n > 0; ++side) {
      double plane = side == 0 ? double(box.lo[axis]) : double(box.hi[axis]);
      double sign = side == 0 ? 1.0 : -1.0;  // keep sign*(x - plane) >= 0
      int m = 0;
      for (int i = 0; i < n; ++i) {
        const double* a = poly[cur][i];
        const double* b = poly[cur][(i + 1) % n];
        double da = sign * (a[axis] - plane), db = sign * (b[axis] - plane);
        if (da >= 0) {
          for (int c = 0; c < 3; ++c) poly[1 - cur][m][c] = a[c];
          ++m;
        }
        if ((da >= 0) != (db >= 0)) {
          double tt = da / (da - db);
          for (int c = 0; c < 3; ++c) poly[1 - cur][m][c] = a[c] + tt * (b[c] - a[c]);
          poly[1 - cur][m][axis] = plane;
          ++m;
        }
      }
      n = m;
      cur = 1 - cur;
    }
  }
  if (n == 0) return fallback;
  // Bounds in double, then rounded outward: a bound that rounds inward could
  // drop the triangle from a thin cell it actually crosses.
  double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
      lo[c] = std::min(lo[c], poly[cur][i][c]);
      hi[c] = std::max(hi[c], poly[cur][i][c]);
    }
  AABB r;
  const double ulp = double(std::numeric_limits<Real>::epsilon());
  for (int c = 0; c < 3; ++c) {
    double pad = 4 * ulp * (std::fabs(lo[c]) + std::fabs(hi[c]) + double(box.hi[c] - box.lo[c]));
    r.lo[c] = Real(lo[c] - pad);
    r.hi[c] = Real(hi[c] + pad);
  }
  // Never grow beyond the box or the original bounds.
  r.lo = vmax(r.lo, vmax(box.lo, fallback.lo));
  r.hi = vmin(r.hi, vmin(box.hi, fallback.hi));
  for (int a = 0; a < 3; ++a)
    if (r.lo[a] > r.hi[a]) r.lo[a] = r.hi[a];
  return r;
}

namespace {

class Builder {
 public:
  Builder(const Scene& s, const KdBuildParams& p, KdTree& t) : scene_(s), p_(p), tree_(t) {}

  void build(std::vector<Ref>& refs, const AABB& box, int depth) {
    const size_t nodeIndex = tree_.nodes.size();
    tree_.nodes.push_back(KdNode());
    tree_.maxDepthReached = std::max(tree_.maxDepthReached, depth);
    const size_t N = refs.size();

    int bestAxis = -1;
    Real bestSplit = 0, bestCost = kInf;
    if (!p_.singleLeaf && N > 1 && depth < maxDepth_) {
      findSplit(refs, box, bestAxis, bestSplit, bestCost);
    }
    const Real leafCost = p_.costIntersect * Real(N);
    if (bestAxis < 0 || !(bestCost < leafCost)) {
      makeLeaf(nodeIndex, refs);
      return;
    }

    AABB lbox = box, rbox = box;
    lbox.hi[bestAxis] = bestSplit;
    rbox.lo[bestAxis] = bestSplit;
    std::vector<Ref> left, right;
    left.reserve(N / 2 + 1);
    right.reserve(N / 2 + 1);
    for (const Ref& r : refs) {
      Real lo = r.b.lo[bestAxis], hi = r.b.hi[bestAxis];
      bool planar = lo == hi && lo == bestSplit;
      bool goLeft = planar || lo < bestSplit;
      bool goRight = planar || hi > bestSplit;
      if (!goLeft && !goRight) goLeft = goRight = true;  // cannot happen, but be safe
      if (goLeft && goRight && !planar && p_.clipBounds) {
        left.push_back({r.tri, clippedTriangleBounds(scene_, r.tri, lbox, r.b)});
        right.push_back({r.tri, clippedTriangleBounds(scene_, r.tri, rbox, r.b)});
      } else {
        if (goLeft) left.push_back(r);
        if (goRight) right.push_back(r);
      }
    }
    std::vector<Ref>().swap(refs);  // free memory before recursing
    build(left, lbox, depth + 1);
    std::vector<Ref>().swap(left);
    const size_t rightIndex = tree_.nodes.size();
    build(right, rbox, depth + 1);
    KdNode& n = tree_.nodes[nodeIndex];
    n.split = bestSplit;
    n.data = (uint32_t(rightIndex) << 2) | uint32_t(bestAxis);
  }

  int maxDepth_ = 32;

 private:
  void makeLeaf(size_t nodeIndex, std::vector<Ref>& refs) {
    std::sort(refs.begin(), refs.end(), [&](const Ref& a, const Ref& b) {
      Real aa = scene_.triArea[a.tri], ab = scene_.triArea[b.tri];
      return aa != ab ? aa > ab : a.tri < b.tri;
    });
    KdNode& n = tree_.nodes[nodeIndex];
    n.offset = uint32_t(tree_.triIndices.size());
    n.data = (uint32_t(refs.size()) << 2) | 3u;
    for (const Ref& r : refs) tree_.triIndices.push_back(r.tri);
    tree_.numLeaves++;
    tree_.numRefs += refs.size();
  }

  Real sahCost(const AABB& box, int axis, Real split, size_t nl, size_t nr) const {
    AABB l = box, r = box;
    l.hi[axis] = split;
    r.lo[axis] = split;
    Real inv = 1 / box.surfaceArea();
    Real cost = p_.costTraverse + p_.costIntersect * (l.surfaceArea() * inv * Real(nl) + r.surfaceArea() * inv * Real(nr));
    if (nl == 0 || nr == 0) cost *= p_.emptyBonus;
    return cost;
  }

  void findSplit(const std::vector<Ref>& refs, const AABB& box, int& bestAxis, Real& bestSplit, Real& bestCost) {
    const size_t N = refs.size();
    if (!(box.surfaceArea() > 0)) return;
    for (int axis = 0; axis < 3; ++axis) {
      Real lo = box.lo[axis], hi = box.hi[axis];
      if (!(hi > lo)) continue;
      if (N > size_t(p_.binThreshold)) {
        const int B = p_.bins;
        std::vector<size_t> starts(size_t(B), 0), ends(size_t(B), 0);
        Real scale = Real(B) / (hi - lo);
        for (const Ref& r : refs) {
          int bs = clampv(int((r.b.lo[axis] - lo) * scale), 0, B - 1);
          int be = clampv(int((r.b.hi[axis] - lo) * scale), 0, B - 1);
          starts[size_t(bs)]++;
          ends[size_t(be)]++;
        }
        // Plane k sits at the left edge of bin k. Refs starting in bins < k go left
        // (approximately); refs ending in bins >= k go right.
        size_t nl = 0, nr = N;
        for (int k = 1; k < B; ++k) {
          nl += starts[size_t(k - 1)];
          nr -= ends[size_t(k - 1)];
          Real split = lo + (hi - lo) * Real(k) / Real(B);
          Real c = sahCost(box, axis, split, nl, nr);
          if (c < bestCost) {
            bestCost = c;
            bestAxis = axis;
            bestSplit = split;
          }
        }
      } else {
        // Exact sweep over sorted events. Planar refs are counted on both sides.
        struct Ev {
          Real pos;
          int type;  // 0 = end, 1 = planar, 2 = start
        };
        std::vector<Ev> ev;
        ev.reserve(2 * N);
        for (const Ref& r : refs) {
          Real a = r.b.lo[axis], b = r.b.hi[axis];
          if (a == b) {
            ev.push_back({a, 1});
          } else {
            ev.push_back({a, 2});
            ev.push_back({b, 0});
          }
        }
        std::sort(ev.begin(), ev.end(), [](const Ev& x, const Ev& y) {
          return x.pos != y.pos ? x.pos < y.pos : x.type < y.type;
        });
        size_t nl = 0, nr = N;
        for (size_t i = 0; i < ev.size();) {
          Real p = ev[i].pos;
          size_t pe = 0, pp = 0, ps = 0;
          while (i < ev.size() && ev[i].pos == p && ev[i].type == 0) { ++pe; ++i; }
          while (i < ev.size() && ev[i].pos == p && ev[i].type == 1) { ++pp; ++i; }
          while (i < ev.size() && ev[i].pos == p && ev[i].type == 2) { ++ps; ++i; }
          nr -= pe + pp;
          if (p > lo && p < hi) {
            Real c = sahCost(box, axis, p, nl + pp, nr + pp);
            if (c < bestCost) {
              bestCost = c;
              bestAxis = axis;
              bestSplit = p;
            }
          }
          nl += ps + pp;
        }
      }
    }
  }

  const Scene& scene_;
  const KdBuildParams& p_;
  KdTree& tree_;
};

}  // namespace

void KdTree::build(const Scene& scene, const KdBuildParams& params) {
  Timer timer;
  nodes.clear();
  triIndices.clear();
  numLeaves = 0;
  maxDepthReached = 0;
  numRefs = 0;
  bounds = scene.bounds;
  // Pad the root box slightly so no geometry lies exactly on its faces.
  Vec3 pad = bounds.diag() * Real(1e-4) + Vec3(Real(1e-6), Real(1e-6), Real(1e-6));
  bounds.lo -= pad;
  bounds.hi += pad;
  std::vector<Ref> refs(size_t(scene.numTris()));
  for (int t = 0; t < scene.numTris(); ++t) {
    AABB b;
    for (int k = 0; k < 3; ++k) b.expand(scene.v(t, k));
    refs[size_t(t)] = {uint32_t(t), b};
  }
  Builder builder(scene, params, *this);
  int N = std::max(1, scene.numTris());
  builder.maxDepth_ = params.maxDepth >= 0 ? params.maxDepth : int(8 + 1.3 * std::log2(double(N)));
  builder.build(refs, bounds, 0);
  buildSeconds = timer.seconds();
}

}  // namespace bt
