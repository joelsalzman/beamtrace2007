# beamtrace2007

A from-scratch implementation of

> Ryan Overbeck, Ravi Ramamoorthi, William R. Mark.
> **A Real-time Beam Tracer with Application to Exact Soft Shadows.**
> Eurographics Symposium on Rendering 2007.
> [paper](https://cseweb.ucsd.edu/~ravir/beamtrace.pdf)

This repository implements the paper's beam tracer:
- beam–triangle splitting (Sec. 3.2);
- beam–kd-tree traversal (Sec. 3.3);
- "Post Office" mailboxing and leaf triangles sorted by area (Sec. 3.4);
- its three applications: real-time primary visibility, point-light shadows (Sec. 4), and exact soft shadows from area lights (Sec. 5).

It also includes the paper's comparison baseline, a one-ray-at-a-time kd-tree ray tracer, and a benchmark harness that reproduces the paper's tables and figures (Figs. 5–13).

The CPU version follows the paper's design. It uses C++17 and keeps the four corner rays of a beam in one SIMD register: SSE `__m128` for float, AVX2 `__m256d` for double. The CUDA port in [`cuda/`](cuda/) runs the same algorithm code on the GPU, reorganized as a wavefront of sub-beams in queues (see [CUDA port](#cuda-port)).

## Quick start

```sh
git clone https://github.com/joelsalzman/beamtrace2007.git && cd beamtrace2007
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # add -DBT_CUDA=ON to also build the GPU port
cmake --build build -j
./build/bt_tests_f && ./build/bt_tests_d               # correctness tests (float and double)
./build/cuda/bt_tests_cuda                             # GPU vs CPU tests (with -DBT_CUDA=ON)
scripts/fetch_scenes.sh                                # Sponza, Conference, Armadillo (~8 MB)
scripts/run_paper_benchmarks.sh                        # full benchmark -> results/REPORT.md
```

Requirements:
- CMake ≥ 3.18 and a C++17 compiler (GCC or Clang).
- x86-64 with SSE2. `-march=native` is on by default; turn it off with `-DBT_NATIVE=OFF`. `-DBT_SIMD=OFF` gives a portable scalar build.
- Python 3 for the report; plots also need matplotlib.
- The CUDA port needs the CUDA toolkit (tested with 11.8) and a matching host compiler. The GPU architecture defaults to the local GPU (CMake ≥ 3.24); override it with e.g. `-DCMAKE_CUDA_ARCHITECTURES=86`.

Rendering a single image:

```sh
./build/bt_render --config configs/sponza.cfg --mode primary     --method beam --res 1024x1024 --out sponza --wire
./build/bt_render --config configs/building.cfg --mode pointshadow --method beam --res 1024x1024 --out building
./build/bt_render --config configs/plant.cfg  --mode softshadow  --method beam --res 512x512  --out plant_beam --pfm
./build/bt_render --config configs/plant.cfg  --mode softshadow  --method ray --samples 256 --res 512x512 --out plant_ray --pfm
./build/bt_compare plant_beam_vis.pfm plant_ray_vis.pfm   # error of 256-ray sampling vs the exact beams
```

Useful options:
- `--views all` with `--csv FILE` averages over the camera path and writes one CSV row per view.
- `--threads N` parallelizes soft shadows; the paper is single-threaded.
- `--exact` integrates the visible light polygons exactly with Lambert's formula, instead of the paper's eq. 2 light-center approximation.
- `--light-scale S` scales the area light (used for Fig. 13).
- `--no-mailbox` turns off the Post Office mailboxing.
- `--trail` uses the stackless restart-trail continuation instead of the frame stack (identical output; see below).
- `bt_render_d` is the double-precision build.
- `cuda/bt_render_cuda` takes the same options and runs on the GPU (`--device cpu` switches back). `--warmup` renders the first view once untimed, so GPU allocations are not timed.

## Running the benchmarks

```sh
git pull
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBT_CUDA=ON && cmake --build build -j   # drop -DBT_CUDA=ON without a GPU
./build/bt_tests_f && ./build/bt_tests_d && ./build/cuda/bt_tests_cuda
THREADS=$(nproc) scripts/run_paper_benchmarks.sh    # QUICK=1 for a ~2-minute smoke run
```

Results land in `results/`: CSV files, PNG/PFM images and `REPORT.md`, which puts our numbers next to the paper's.
- The CUDA runs are added automatically when `build/cuda/bt_render_cuda` exists and a GPU is present (`CUDA=0` skips them).
- `THREADS` adds a multi-threaded soft-shadow run.
- A full run takes about 10 minutes on an 8-core desktop.

## Scenes

| config | triangles | paper scene | source |
|---|---|---|---|
| `room` | 0.9K | Erw6 (816) | procedural stand-in (Erw6 is not public) |
| `building` | 2.1M | Soda Hall (2.2M) | procedural 7-floor office building (Soda Hall is not public) |
| `conference` | 125K | Conference (282K) | McGuire Computer Graphics Archive version |
| `armadillo` | 346K | Armadillo (345K) | Stanford 3D Scanning Repository |
| `sponza` | 66K | Sponza (76K) | McGuire Computer Graphics Archive (Dabrovic) |
| `plant` | 5.4K | plant (5,245) | procedural stand-in |
| `tess_sweep`, `analytic` | | | tessellation sweeps; closed-form soft-shadow test |

`scripts/fetch_scenes.sh` downloads the public scenes into `scenes/downloaded/`. Each config file sets a camera path, a point light and an area light.

## How it works

**Beams (Sec. 3.1).** A beam is 3 or 4 corner rays sharing one apex. The corner directions are stored as 2D points on a plane:
- the image plane for primary beams;
- the receiver triangle's plane for point-light shadow beams;
- the light's plane for area-light beams.

Every geometric test is a 2D line evaluated at the 4 corners at once.

**Beam–triangle intersection (Sec. 3.2).** The beam is split along the triangle's edges, one edge at a time, into miss pieces and a hit piece:
- Trivial cases (a)–(c) of Fig. 2 are handled first.
- Corners within ε of a line don't vote (the paper's fuzzy tests).
- Pentagons are split into a quad and a triangle, which keeps every beam at 4 corners or fewer.

The edge lines are computed as planes through the apex, with each shared edge in canonical vertex order. This is equivalent to projecting the triangle onto the plane, but it needs no near-plane clip, and two triangles sharing an edge produce bitwise-identical (negated) lines, so no cracks open between them.

For primary visibility, a beam that already hits T1 and meets T2 is also split by one more line: the projection of the line where T1's and T2's planes meet, which separates "T2 closer" from "T1 closer". This resolves depth exactly, including interpenetrating triangles.

**Beam–kd traversal (Sec. 3.3).** Each beam keeps near/far distances per axis (the paper's 3 near and 3 far planes) and decides near/far/both from its corners. All comparisons are cross-multiplied, which avoids `InvDirs` and so avoids inf/NaN. Because each test is a half-plane in the beam's 2D domain, deciding at the corners is conservative for every ray inside the beam.

Hit beams keep traversing until the next cell lies wholly beyond the hit (Fig. 4d). Sub-beams continue from the leaf where they were created; this is the paper's `KDStack`/`MissStack` scheme, implemented as a persistent stack of pending far nodes.

The same continuation can also be kept without a stack (`--trail`): one bit per kd depth marks a pending far child, and parent links stored with the tree lead back to it (a restart trail, after Laine's for BVHs, HPG 2010). It visits the same cells in the same order, so output and statistics are bitwise identical; on the CPU it is 1–3% slower, but it lets a sub-beam carry its whole continuation in 12 bytes, which the GPU port needs.

**Post Office mailboxing (Sec. 3.4).** Each leaf visit creates a mailbox id whose parent is the beam's id. Triangles are skipped if they were stamped by the beam or one of its 3 nearest ancestors.

**Applications.**
- **Primary visibility** produces the exact visible surface as a list of convex polygons. They are rasterized with 6× antialiasing (on the CPU here; the paper used the GPU) and shaded per sample.
- **Point-light shadows** connect each primary hit beam to the light with a new beam. Its hit list is the shadowed part and its miss list the lit part.
- **Soft shadows** trace one beam per pixel, from the shading point to the light quad. The visible fraction is the miss area divided by the light area (paper eq. 2).

## Deviations from the paper

1. **Direction-sign pre-split.** Before traversal, a beam is split where a ray-direction component changes sign, so each axis has a single direction sign. This makes the per-axis near/far planes well defined for every ray in the beam. The extra beams are counted separately (`presplit`).
2. **Edge planes through the apex** replace "project the triangle onto the image plane". Both give the same 2D lines; see above.
3. **Exact depth split** for overlapping and interpenetrating triangles, as described above. The paper does not say how it resolves depth inside a beam.
4. **No-op splits are discarded.** A triangle splits a beam only if some piece actually changes status. Clipping starts with the edge that leaves the smallest remainder, which reduces fragmentation.
5. **Horizon clip.** For soft shadows, the light polygon is clipped by the shading point's tangent plane before tracing, and the ray tracer applies the same rule.
6. **CPU rasterization** instead of the GPU. Rasterization time is reported separately (`raster_s`) and is not part of the trace times.
7. **No MLRT comparison.** MLRT is not available. As in the paper's soft-shadow section, we compare against our own optimized ray tracer.
8. **Procedural stand-ins** replace Erw6, Soda Hall and the plant, and the McGuire Conference model is coarser than the paper's. Absolute numbers are therefore not comparable; compare ratios and trends.
9. **Separate SAH tuning per method.** Each method builds its own kd-tree. Beams prefer larger leaves (SAH intersection cost 0.4); rays use the standard 1.5. `--kd-ci` overrides either.

## CUDA port

The GPU port reorganizes the tracer as a **wavefront of sub-beams** ([`cuda/wavefront.cu`](cuda/wavefront.cu)), following Laine, Karras & Aila, "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs" (HPG 2013), with the persistent threads of Aila & Laine, "Understanding the Efficiency of Ray Traversal on GPUs" (HPG 2009). The paper's own algorithm is single-threaded and leaves this part open.

**Sub-beams as records.** Every sub-beam is a self-contained 64-byte record: its corners, its hit status, the root it belongs to, and a restart-trail continuation (kd node, depth, pending far children, index of the next triangle in the leaf). Any thread can pick up any record from a queue. The per-record steps live in [`src/beam/wavefront_core.h`](src/beam/wavefront_core.h) and are shared by the GPU and by a host reference engine ([`src/beam/wavefront.cpp`](src/beam/wavefront.cpp)).

**Rounds.** Each round runs four kernels. Every kernel is a persistent grid: warps take queue entries with one `atomicAdd` for the whole warp, and appends are warp-aggregated so siblings stay next to each other.

| kernel | one thread = | one warp = | what it does |
|---|---|---|---|
| schedule | — (1 thread) | — | resets counters; admits new roots while their budgets fit |
| admit | one root: a pixel (soft shadows), a 16×16 tile (primary), a primary hit beam (point shadows) | 32 roots; for soft shadows an 8×4 pixel tile | sets up the root beam, pre-splits it by direction sign, queues the pieces |
| trace | one queued sub-beam | 32 sub-beams from the queue | one leaf per iteration (climb / descend to it, test its triangles), until the sub-beam is finished or a triangle needs a real clip; lanes are refilled from the queue as soon as their sub-beam leaves the loop |
| split | one sub-beam that needs a clip | 32 such sub-beams | clips it; reserves its pieces' slots with one atomic per warp and writes them into the next round's trace queue |

Cheap tests and expensive clips are separate kernels because only 7–13% of triangle tests clip: done inline, nearly every warp would stall on some lane's clip. A warp never shares one beam across a leaf's triangles, because leaves average only 2–4 tests per visit.

**Bounded memory without waiting.** A root is admitted only if its budget of live sub-beams (128 for shadows, 4096 for primary tiles) fits in the queues. A root that would exceed it, or overflow a fixed per-thread list, is abandoned and recomputed on the CPU, so the output is always exact; at the default sizes no root in the benchmark scenes is abandoned. Soft-shadow visibility accumulates in 64-bit fixed point, so results do not depend on scheduling: two runs are bitwise identical.

**Other GPU choices.**
- No mailboxing (re-tests are idempotent).
- Leaves reference packed 64-byte triangles (one wide load per test).
- Fast division and square root (`--use_fast_math`). Two triangles sharing an edge still compute it from identical inputs, so beams stay crack-free.

**Validation.**
- The host reference engine runs the same steps in the same rounds. Its output pieces are bitwise identical to the depth-first tracer's, and so are all its counters, for primary visibility, point shadows and soft shadows, in float and double (`tests/test_wavefront.cpp`).
- `bt_tests_cuda` checks the GPU against the CPU:
  - Soft-shadow visibility matches to about 1e-6 on average; the maximum is 6e-3, where float rounding differs.
  - Primary visibility and point-shadow masks match at every interior pixel.
  - Two runs are identical.
  - Tiny budgets force thousands of pixels onto the CPU fallback, and the result stays exact.

**Performance** (RTX 2060 SUPER vs Ryzen 7 5800X3D, view 0; `results/REPORT.md` has the full tables):

Soft shadows, 512², including the GPU primary rays:

| scene | wavefront GPU | first GPU port | CPU, 16 threads | CPU, 1 thread |
|---|---|---|---|---|
| plant | 0.29 s | 1.82 s | 0.50 s | 4.8 s |
| conference | 0.19 s | 0.90 s | 0.23 s | 2.2 s |
| sponza | 0.041 s | 0.166 s | 0.069 s | 0.61 s |
| building | 0.016 s | 0.040 s | 0.080 s | 0.32 s |

Primary visibility, 1024² (the paper's CPU numbers are single-threaded):

| scene | wavefront GPU | first GPU port | CPU, 1 thread |
|---|---|---|---|
| armadillo | 163 ms | 1102 ms | 1238 ms |
| sponza | 36 ms | 243 ms | 168 ms |
| conference | 49 ms | 286 ms | 102 ms |
| plant | 13 ms | 47 ms | 19 ms |
| building | 12 ms | 61 ms | 2.3 ms |
| room | 6 ms | 12 ms | 1.2 ms |

What these numbers show:
- **Against the first port:** the wavefront engine is 2–7× faster.
- **Against the CPU:** it beats all 16 CPU threads on soft shadows (1.2–5×) and one thread on primary visibility for the larger scenes.
- **Small scenes:** on the smallest ones it is slower than one CPU core. Each round costs a few kernel launches, and a 1024² frame is 4,096 tile roots, so a frame has a floor of a few milliseconds.
- **What limits it:** the trace kernel is limited by memory latency. Its per-lane state allows only 16 warps per SM, and even with lane refill only part of each warp is active in the descent and triangle-test loops. Tracing and clipping take roughly equal time. On Conference, 29% of the clips turn out to change nothing.

The first port, which ran one thread per root beam with ~12 KB of local memory each, is still available with `BT_GPU_ENGINE=v1`. Diagnostics:
- `BT_GPU_DEBUG=1` prints rounds, queue sizes and abandoned roots.
- `BT_WF_PROFILE=1` prints per-kernel times.
- `BT_WF_BUDGET` and `BT_WF_CAPACITY` override the budget and queue size.

## Correctness tests

`bt_tests_f` and `bt_tests_d` run the same suite in float and double:
- **Split geometry.** Area is conserved, pieces stay convex with 4 corners or fewer, and neighbouring edges agree bitwise.
- **kd traversal.** A fuzz test checks that, for every ray inside a random beam, the near/far/both decision and the box cull are conservative. A brute-force ray test and a regression test for near-planar triangles cover kd construction.
- **Primary visibility vs. ray tracing**, checked at random image points. Scenes: interpenetrating triangle soups, cameras inside geometry or on split planes, coplanar overlaps, and occluders in later kd cells. A crack test uses a closed sphere.
- **Beam output.** Re-testing every final beam against every triangle changes nothing (idempotency). Results with and without mailboxing are identical, and the kd-tree agrees with a brute-force single leaf.
- **Soft shadows** match the closed form for a square occluder under a square light to about 1e-7 in double, and match 9,216-sample ray tracing on the plant. The exact Lambert integration matches quadrature.
- **Point-light shadows** match shadow rays.
- **Continuations and wavefront.** The restart trail reproduces the frame stack bitwise, and so does the wavefront reference engine, root by root (pieces and counters). Roots that exceed their sub-beam budget are reported, never silently wrong.

`bt_golden_f` / `bt_golden_d CONFIG...` print hashes of the CPU tracer's exact output and counters, for checking that a refactoring changes nothing. Host code is compiled without implicit FMA contraction (`-ffp-contract=off`), so these hashes do not depend on how the compiler inlines.

## Repository layout

```
src/core     Real/Vec3 and the 4-lane R4 SIMD type
src/beam     the beam algorithm (beam_core.h, host+device), CPU tracer (beam_tracer.*), 2D polygon geometry (beam_geom.h),
             wavefront steps (wavefront_core.h, host+device) and the host reference engine (wavefront.*)
src/accel    SAH kd-tree
src/ray      the baseline ray tracer (ray_core.h, host+device)
src/render   camera, rasterizer, primary / point-shadow / soft-shadow pipelines
src/scene    meshes, OBJ/PLY loaders, procedural scenes, config files
apps         bt_render, bt_compare, bt_golden
tests        correctness tests
scripts      fetch_scenes.sh, run_paper_benchmarks.sh, make_report.py
configs      scene configs (camera paths, lights)
cuda         GPU port: wavefront engine (wavefront.cu), primary rays and first port (gpu_render.cu), GPU-vs-CPU tests
```

## License

No license file yet. The downloaded scenes keep their own licenses (see `scripts/fetch_scenes.sh`).
