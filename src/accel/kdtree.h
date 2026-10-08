// SAH kd-tree (Havran-style), shared by the beam tracer and the ray tracer.
// Leaf triangle lists are sorted by surface area, largest first (paper Sec. 3.4).
#pragma once

#include <cstdint>
#include <vector>

#include "core/vec.h"
#include "scene/scene.h"

namespace bt {

struct KdNode {
  Real split = 0;
  uint32_t data = 3;    // low 2 bits: axis (0..2) or 3 = leaf; high bits: right child / tri count
  uint32_t offset = 0;  // leaf: first index into KdTree::triIndices
  BT_HD bool isLeaf() const { return (data & 3u) == 3u; }
  BT_HD int axis() const { return int(data & 3u); }
  BT_HD uint32_t right() const { return data >> 2; }  // left child is (this index + 1)
  BT_HD uint32_t count() const { return data >> 2; }
};

struct KdBuildParams {
  Real costTraverse = 1;
  Real costIntersect = Real(1.5);
  Real emptyBonus = Real(0.8);
  int maxDepth = -1;          // -1: 8 + 1.3 log2(N)
  int binThreshold = 2048;    // nodes larger than this use binned SAH
  int bins = 64;
  bool clipBounds = true;     // "perfect splits": clip straddling triangles to child boxes
  bool singleLeaf = false;    // degenerate tree with one leaf (brute force reference)
};

class KdTree {
 public:
  void build(const Scene& scene, const KdBuildParams& params = KdBuildParams());

  std::vector<KdNode> nodes;
  std::vector<uint32_t> triIndices;
  AABB bounds;
  // For stackless (restart-trail) traversal: each node's parent (root: ~0u)
  // and box, computed exactly as a descent from `bounds` computes them.
  std::vector<uint32_t> parent;
  std::vector<AABB> nodeBox;

  // Build statistics.
  int numLeaves = 0;
  int maxDepthReached = 0;
  double buildSeconds = 0;
  size_t numRefs = 0;
};

// Bounds of triangle `tri` clipped to `box`, rounded outward, never larger
// than `box` or `fallback` (the unclipped bounds). Exposed for tests.
AABB clippedTriangleBounds(const Scene& scene, uint32_t tri, const AABB& box, const AABB& fallback);

}  // namespace bt
