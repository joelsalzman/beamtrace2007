#!/usr/bin/env bash
# Reproduces the paper's experiments (EGSR 2007, Figs. 5-13) and writes
# CSV files, images and results/REPORT.md.
#
#   scripts/run_paper_benchmarks.sh            # full run (single-threaded, like the paper)
#   QUICK=1 scripts/run_paper_benchmarks.sh    # small resolutions / few views (smoke test)
#   THREADS=16 scripts/run_paper_benchmarks.sh # also time multi-threaded soft shadows
#
# Environment:
#   BUILD=build      build directory (configured and built if missing)
#   OUT=results      output directory
#   QUICK=0|1        quick mode
#   THREADS=1        extra thread count for the soft-shadow scaling run (1 = skip)
#   NO_FETCH=1       do not download scenes (scenes that are missing are skipped)
#   DEVICE=cpu|cuda  also run the CUDA port when built with -DBT_CUDA=ON (cuda adds GPU columns)
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
OUT=${OUT:-results}
QUICK=${QUICK:-0}
THREADS=${THREADS:-1}
DEVICE=${DEVICE:-cpu}

if [ ! -x "$BUILD/bt_render" ]; then
  cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD" -j
fi
if [ "${NO_FETCH:-0}" != 1 ]; then scripts/fetch_scenes.sh || echo "warning: scene download failed; continuing with what is available"; fi

mkdir -p "$OUT/img"
R="$BUILD/bt_render"
have() {  # config -> true if its mesh files exist
  local f
  for f in $(awk '$1=="mesh" && ($2=="obj" || $2=="ply") {print $3}' "configs/$1.cfg"); do
    [ -f "$f" ] || { echo "skipping $1 (missing $f; run scripts/fetch_scenes.sh)"; return 1; }
  done
  return 0
}

if [ "$QUICK" = 1 ]; then
  RES_P=512x512; RES_S=192x192; VIEWS="--view 0"; SWEEP="0.5 1 2"
else
  RES_P=1024x1024; RES_S=512x512; VIEWS="--views all"; SWEEP="0.5 0.75 1 1.25 1.5 2"
fi

echo "== machine"
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //')"
  echo "cores: $(nproc 2>/dev/null || echo ?)"
  echo "compiler: $(c++ --version 2>/dev/null | head -1)"
  echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "quick: $QUICK"
  command -v nvidia-smi >/dev/null && echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" || true
} | tee "$OUT/machine.txt"

# ---------------------------------------------------------------- Fig. 5-7
# Primary visibility (diffuse shading, point light at the camera), beams vs rays.
rm -f "$OUT/primary.csv"
for sc in room building conference armadillo sponza plant; do
  have $sc || continue
  echo "== primary: $sc"
  $R --config configs/$sc.cfg --mode primary --method beam --res $RES_P --aa 6 $VIEWS --csv "$OUT/primary.csv" \
     --out "$OUT/img/primary_${sc}_beam" --quiet
  $R --config configs/$sc.cfg --mode primary --method ray --res $RES_P $VIEWS --csv "$OUT/primary.csv" --quiet
done
# Wireframe beams for the first view of each scene.
for sc in room building sponza; do
  have $sc || continue
  $R --config configs/$sc.cfg --mode primary --method beam --res $RES_P --view 0 --wire --out "$OUT/img/wire_$sc" --quiet
done

# ---------------------------------------------------------------- Figs. 9-10
# Real-time point-light shadows in the building (Soda Hall stand-in).
rm -f "$OUT/pointshadow.csv"
echo "== point-light shadows"
$R --config configs/building.cfg --mode pointshadow --method beam --res $RES_P --aa 6 --view 0 --wire \
   --out "$OUT/img/pointshadow_building_beam" --csv "$OUT/pointshadow.csv" --quiet
$R --config configs/building.cfg --mode pointshadow --method ray --res $RES_P --view 0 \
   --out "$OUT/img/pointshadow_building_ray" --csv "$OUT/pointshadow.csv" --quiet
$R --config configs/building.cfg --mode pointshadow --method beam --res $RES_P --aa 6 $VIEWS --csv "$OUT/pointshadow.csv" --quiet --tag sweep
$R --config configs/building.cfg --mode pointshadow --method ray --res $RES_P $VIEWS --csv "$OUT/pointshadow.csv" --quiet --tag sweep

# ---------------------------------------------------------------- Figs. 11-12
# Soft shadows: exact beams (reference) vs 256 jittered shadow rays vs equal-time rays.
rm -f "$OUT/softshadow.csv" "$OUT/softshadow_errors.csv"
echo "scene,method,samples,rmse,mae,max,frac_gt_0.01,frac_gt_0.05" > "$OUT/softshadow_errors.csv"
for sc in plant sponza conference building; do
  have $sc || continue
  echo "== soft shadows: $sc"
  $R --config configs/$sc.cfg --mode softshadow --method beam --res $RES_S --view 0 \
     --out "$OUT/img/soft_${sc}_beam" --pfm --csv "$OUT/softshadow.csv" --quiet
  for spp in 256 4 9 16 25; do
    $R --config configs/$sc.cfg --mode softshadow --method ray --samples $spp --res $RES_S --view 0 \
       --out "$OUT/img/soft_${sc}_ray$spp" --pfm --csv "$OUT/softshadow.csv" --quiet
    m=$("$BUILD/bt_compare" "$OUT/img/soft_${sc}_beam_vis.pfm" "$OUT/img/soft_${sc}_ray${spp}_vis.pfm" \
        "$OUT/img/soft_${sc}_diff$spp.pfm")
    echo "$sc,ray,$spp,$(echo "$m" | awk '{print $2","$4","$6","$8","$10}')" >> "$OUT/softshadow_errors.csv"
  done
  # Exact integration of the visible light polygons (instead of eq. 2's center approximation).
  $R --config configs/$sc.cfg --mode softshadow --method beam --exact --res $RES_S --view 0 \
     --out "$OUT/img/soft_${sc}_beam_exact" --csv "$OUT/softshadow.csv" --quiet --tag exact
done

# ---------------------------------------------------------------- Fig. 13
# Beam soft-shadow time vs light source area (Sponza).
rm -f "$OUT/lightsweep.csv"
if have sponza; then
  echo "== light-area sweep (sponza)"
  for s in $SWEEP; do
    $R --config configs/sponza.cfg --mode softshadow --method beam --res $RES_S --view 0 --light-scale $s \
       --csv "$OUT/lightsweep.csv" --quiet
  done
  $R --config configs/sponza.cfg --mode softshadow --method ray --samples 256 --res $RES_S --view 0 \
     --light-scale 2 --csv "$OUT/lightsweep.csv" --quiet --tag ray256
fi

# ---------------------------------------------------------------- extras
rm -f "$OUT/threads.csv"
if [ "$THREADS" -gt 1 ] && have sponza; then
  echo "== multi-threaded soft shadows ($THREADS threads)"
  for m in beam ray; do
    $R --config configs/sponza.cfg --mode softshadow --method $m --res $RES_S --view 0 --threads $THREADS \
       --csv "$OUT/threads.csv" --quiet
  done
fi
if [ "$DEVICE" = cuda ] && [ -x "$BUILD/bt_render_cuda" ]; then
  echo "== CUDA"
  rm -f "$OUT/cuda.csv"
  for sc in plant sponza conference building; do
    have $sc || continue
    "$BUILD/bt_render_cuda" --config configs/$sc.cfg --mode softshadow --res $RES_S --view 0 \
       --out "$OUT/img/soft_${sc}_cuda" --pfm --csv "$OUT/cuda.csv" --quiet
    m=$("$BUILD/bt_compare" "$OUT/img/soft_${sc}_beam_vis.pfm" "$OUT/img/soft_${sc}_cuda_vis.pfm")
    echo "$sc,cuda,0,$(echo "$m" | awk '{print $2","$4","$6","$8","$10}')" >> "$OUT/softshadow_errors.csv"
  done
fi

echo "== report"
python3 scripts/make_report.py "$OUT"
echo "done: $OUT/REPORT.md"
