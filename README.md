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

The CPU version follows the paper's design. It uses C++17 and keeps the four corner rays of a beam in one SIMD register: SSE `__m128` for float, AVX2 `__m256d` for double. The CUDA port in [`cuda/`](cuda/) runs the same algorithm code on the GPU (see [CUDA port](#cuda-port)).

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
- `bt_render_d` is the double-precision build.
- `cuda/bt_render_cuda` takes the same options and runs on the GPU (`--device cpu` switches back).

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

[`cuda/gpu_render.cu`](cuda/gpu_render.cu) runs the CPU tracer's algorithm on the GPU. `BeamCore` (in [`src/beam/beam_core.h`](src/beam/beam_core.h)) is a host/device template over a storage policy:
- On the CPU, the storage is growable `std::vector`s.
- On the GPU, each thread has fixed-size arrays in local memory. Their sizes come from measured high-water marks.

If a thread runs out of storage, it flags its work item. The host then recomputes that item with the CPU tracer, so the output stays exact. A full 512² frame typically has 0–20 such pixels.

Three kernels:
- **Soft shadows:** one thread per pixel, after a GPU primary-ray kernel. The kernel uses persistent warps: each warp fetches 8×4 pixel tiles so its lanes trace similar beams.
- **Primary visibility:** one thread per 16×16 pixel tile (each tile is a root beam).
- **Point-light shadows:** one thread per primary hit beam.

Two GPU-specific settings:
- **No mailboxing.** With small per-thread tables in local memory, the Post Office costs more than it saves.
- **Recompute triangle setup** instead of caching it.

**Validation.** `bt_tests_cuda` compares GPU and CPU output:
- Soft-shadow visibility agrees to about 1e-6 on average; the maximum difference is 6e-3, at a few pixels where float rounding differs.
- Primary visibility and point-shadow masks agree at every interior pixel.

**Performance, honestly.** On an RTX 2060 SUPER, soft shadows run 2–4× faster than one CPU core (Ryzen 5800X3D), but slower than all 16 CPU threads. Primary visibility per tile is slower than the CPU.

The algorithm keeps deep, irregular per-beam state: a beam pool, a work stack, a frame stack and polygon scratch space, about 12 KB per thread. Lanes in a warp follow different paths through the kd-tree and the splitting code. As a result:
- Local-memory traffic dominates.
- Divergence serializes the warp.
- Adding warps per SM does not help.

A fast GPU version would need a warp-cooperative redesign. For example, a warp could test a leaf's triangles in parallel against a shared beam, or process a pixel's beam pieces as a batch. That is future work. The paper itself is a CPU algorithm and used the GPU only for rasterization.

## Correctness tests

`bt_tests_f` and `bt_tests_d` run the same suite in float and double:
- **Split geometry.** Area is conserved, pieces stay convex with 4 corners or fewer, and neighbouring edges agree bitwise.
- **kd traversal.** A fuzz test checks that, for every ray inside a random beam, the near/far/both decision and the box cull are conservative. A brute-force ray test and a regression test for near-planar triangles cover kd construction.
- **Primary visibility vs. ray tracing**, checked at random image points. Scenes: interpenetrating triangle soups, cameras inside geometry or on split planes, coplanar overlaps, and occluders in later kd cells. A crack test uses a closed sphere.
- **Beam output.** Re-testing every final beam against every triangle changes nothing (idempotency). Results with and without mailboxing are identical, and the kd-tree agrees with a brute-force single leaf.
- **Soft shadows** match the closed form for a square occluder under a square light to about 1e-7 in double, and match 9,216-sample ray tracing on the plant. The exact Lambert integration matches quadrature.
- **Point-light shadows** match shadow rays.

## Repository layout

```
src/core     Real/Vec3 and the 4-lane R4 SIMD type
src/beam     the beam algorithm (beam_core.h, host+device), CPU tracer (beam_tracer.*), 2D polygon geometry (beam_geom.h)
src/accel    SAH kd-tree
src/ray      the baseline ray tracer (ray_core.h, host+device)
src/render   camera, rasterizer, primary / point-shadow / soft-shadow pipelines
src/scene    meshes, OBJ/PLY loaders, procedural scenes, config files
apps         bt_render, bt_compare
tests        correctness tests
scripts      fetch_scenes.sh, run_paper_benchmarks.sh, make_report.py
configs      scene configs (camera paths, lights)
cuda         GPU port (gpu_render.cu) and GPU-vs-CPU tests
```

## License

No license file yet. The downloaded scenes keep their own licenses (see `scripts/fetch_scenes.sh`).
