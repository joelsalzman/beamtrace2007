// Counters reported in the paper's tables (Figs. 5 and 12).
#pragma once

#include <cstdint>

#include "core/real.h"

namespace bt {

struct TraceStats {
  uint64_t kdSteps = 0;       // inner kd-node visits
  uint64_t leafVisits = 0;    // leaf visits
  uint64_t triTests = 0;      // beam-triangle (or ray-triangle) intersection tests
  uint64_t hits = 0;          // rays: ray-triangle hits found; beams: final hit beams
  uint64_t splits = 0;        // beam-triangle tests that split the beam
  uint64_t beams = 0;         // final beams output (hit + miss)
  uint64_t rootBeams = 0;     // beams handed to the tracer
  uint64_t presplitBeams = 0; // extra beams created by the direction-sign pre-split
  uint64_t mailboxSkips = 0;  // triangles skipped by the Post Office
  uint64_t fiveSplits = 0;    // pentagons split into quad + triangle (4-corner limit)
  uint64_t visibleTris = 0;   // distinct triangles hit, summed over beam trees
  uint64_t rays = 0;          // rays traced
  double droppedArea = 0;     // cross-section area lost to degenerate slivers (plane units)

  BT_HD void add(const TraceStats& o) {
    kdSteps += o.kdSteps;
    leafVisits += o.leafVisits;
    triTests += o.triTests;
    hits += o.hits;
    splits += o.splits;
    beams += o.beams;
    rootBeams += o.rootBeams;
    presplitBeams += o.presplitBeams;
    mailboxSkips += o.mailboxSkips;
    fiveSplits += o.fiveSplits;
    visibleTris += o.visibleTris;
    rays += o.rays;
    droppedArea += o.droppedArea;
  }
};

}  // namespace bt
