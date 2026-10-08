# beamtrace2007

A from-scratch implementation of

> Ryan Overbeck, Ravi Ramamoorthi, William R. Mark.
> **A Real-time Beam Tracer with Application to Exact Soft Shadows.**
> Eurographics Symposium on Rendering 2007.
> ([paper](https://cseweb.ucsd.edu/~ravir/beamtrace.pdf))

Work in progress. See below for build instructions; the full write-up
(design, deviations from the paper, benchmark instructions) is added as the
implementation lands.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
