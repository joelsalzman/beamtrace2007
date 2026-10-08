#!/usr/bin/env python3
"""Builds results/REPORT.md from the CSV files written by run_paper_benchmarks.sh.

Only the Python standard library is required; plots are added when matplotlib
is available.
"""
import csv
import math
import os
import sys
from collections import OrderedDict, defaultdict

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"

# Values printed in the paper (3.0 GHz Pentium 4, single thread), for orientation.
# Fig. 11 labels: beam time, ray time with 256 shadow rays, and the equal-time ray run.
PAPER_SOFT = {
    "plant": ("5 s", "108 s", "5 s / 9 rays"),
    "sponza": ("9 s", "81 s", "9 s / 25 rays"),
    "conference": ("3 s", "99 s", "3 s / 4 rays"),
    "building": ("1.78 s (Soda Hall)", "79 s", "1.84 s / 4 rays"),
}
PAPER_PRIMARY_FPS = {"room": "180 (Erw6)", "building": "25 (Soda Hall)", "conference": "4", "armadillo": "0.35"}
STANDIN = {"room": "Erw6 stand-in", "building": "Soda Hall stand-in", "plant": "plant stand-in"}


def read(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    return f"{x:.{d}g}"


lines = []
w = lines.append

w("# Benchmark report: Overbeck, Ramamoorthi & Mark, EGSR 2007\n")
mt = os.path.join(OUT, "machine.txt")
if os.path.exists(mt):
    w("```")
    w(open(mt).read().strip())
    w("```\n")
w("All timings are single-threaded unless noted, like the paper. Beam timings cover the visibility "
  "computation only; the final rasterization (done on the GPU in the paper) is timed separately as "
  "`raster_s` in the CSV files. Paper numbers come from a 3.0 GHz Pentium 4 and differently "
  "tessellated versions of the scenes, so compare trends and ratios, not absolute values.\n")

# ------------------------------------------------------------------ primary
prim = read("primary.csv")
plots = []
if prim:
    w("## Primary visibility (paper Figs. 5-7)\n")
    w("Averages over each scene's camera path, 1 sample per pixel for rays and exact beams (rasterized with "
      "6x AA afterwards). Statistics are per pixel of the frame, as in the paper.\n")
    w("| scene | tris | visible tris | beams: kd steps/px | beams: isect/px | beams: hits/px | hit beams per visible tri | "
      "rays: kd steps/px | rays: isect/px | beam FPS | ray FPS | speed-up | paper FPS (beams) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = OrderedDict()
    for r in prim:
        by.setdefault(r["scene"], {}).setdefault(r["method"], []).append(r)
    scatter = []
    for sc, d in by.items():
        b, ry = d.get("beam", []), d.get("ray", [])
        px = num(b[0] if b else ry[0], "W") * num(b[0] if b else ry[0], "H")
        def per(rows, k):
            return mean([num(r, k) / px for r in rows])
        bt = mean([num(r, "trace_s") for r in b]) if b else float("nan")
        rt = mean([num(r, "trace_s") for r in ry]) if ry else float("nan")
        vis = mean([num(r, "visible_tris") for r in b]) if b else float("nan")
        ratio = mean([num(r, "hit_beams") / max(1, num(r, "visible_tris")) for r in b]) if b else float("nan")
        name = sc + (f" ({STANDIN[sc]})" if sc in STANDIN else "")
        tris = int(num((b or ry)[0], "tris"))
        w(f"| {name} | {tris:,} | {fmt(vis)} | {fmt(per(b, 'kd_steps'))} | {fmt(per(b, 'tri_tests'))} | "
          f"{fmt(per(b, 'hits'))} | {fmt(ratio)} | {fmt(per(ry, 'kd_steps'))} | {fmt(per(ry, 'tri_tests'))} | "
          f"{fmt(1 / bt) if bt > 0 else '-'} | {fmt(1 / rt) if rt > 0 else '-'} | "
          f"{fmt(rt / bt) if bt > 0 and rt > 0 else '-'}x | {PAPER_PRIMARY_FPS.get(sc, '-')} |")
        for r in b:
            scatter.append((sc, num(r, "visible_tris"), num(r, "hit_beams"), num(r, "trace_s")))
    w("")
    w("The paper reports 5.5-6.5 hit beams per visible triangle (Fig. 6) and running time proportional to "
      "the number of visible triangles (Fig. 7).\n")
    if scatter:
        xs = [s[1] for s in scatter]
        ys = [s[2] for s in scatter]
        ts = [s[3] for s in scatter]
        def fit(x, y):  # least squares through the origin
            sxx = sum(a * a for a in x)
            return sum(a * b for a, b in zip(x, y)) / sxx if sxx else float("nan")
        def r2(x, y, k):
            my = mean(y)
            ss = sum((b - my) ** 2 for b in y)
            return 1 - sum((b - k * a) ** 2 for a, b in zip(x, y)) / ss if ss else float("nan")
        kb, kt = fit(xs, ys), fit(xs, ts)
        w(f"Across all {len(scatter)} views: hit beams = {kb:.2f} x visible triangles (R^2 = {r2(xs, ys, kb):.3f}); "
          f"seconds per frame = {kt * 1e6:.2f} us x visible triangles (R^2 = {r2(xs, ts, kt):.3f}).\n")
        plots.append(("fig6_hitbeams.png", "Visible triangles vs hit beams (paper Fig. 6)", scatter, 2, "hit beams"))
        plots.append(("fig7_time.png", "Visible triangles vs seconds per frame (paper Fig. 7)", scatter, 3,
                      "seconds per frame"))

# ------------------------------------------------------------------ point shadows
ps = read("pointshadow.csv")
if ps:
    w("## Point-light shadows (paper Figs. 9-10)\n")
    w("Primary visibility plus one shadow beam per hit beam (building, the Soda Hall stand-in). The paper "
      "renders its Fig. 9 view at 21 FPS (MLRT: 5 FPS).\n")
    w("| method | views | mean trace time | FPS | kd steps/px | isect/px |")
    w("|---|---|---|---|---|---|")
    for m in ("beam", "ray"):
        rows = [r for r in ps if r["method"] == m and r["tag"] == "sweep"] or [r for r in ps if r["method"] == m]
        if not rows:
            continue
        px = num(rows[0], "W") * num(rows[0], "H")
        t = mean([num(r, "trace_s") for r in rows])
        w(f"| {m} | {len(rows)} | {t * 1000:.1f} ms | {fmt(1 / t)} | {fmt(mean([num(r, 'kd_steps') / px for r in rows]))} | "
          f"{fmt(mean([num(r, 'tri_tests') / px for r in rows]))} |")
    w("\nImages: `img/pointshadow_building_beam.png`, `img/pointshadow_building_beam_wire.png`, "
      "`img/pointshadow_building_ray.png`.\n")

# ------------------------------------------------------------------ soft shadows
ss = read("softshadow.csv")
errs = read("softshadow_errors.csv")
if ss:
    w("## Soft shadows (paper Figs. 11-12)\n")
    w("One exact shadow beam per pixel (apex at the shading point, base = the light quad). Times include "
      "the ray-traced primary visibility, as in the paper. Errors are the visible light fraction of the ray "
      "tracer measured against the exact beam result.\n")
    w("| scene | beam time | rays (256) time | speed-up | beams: kd steps/px | beams: isect/px | beams: hits/px | "
      "visible occluder tris/px | rays (256): kd steps/px | rays (256): isect/px | equal-time rays | paper: beam / 256 rays / equal time |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    scenes = OrderedDict()
    for r in ss:
        scenes.setdefault(r["scene"], []).append(r)
    errmap = {(e["scene"], e["samples"]): e for e in errs}
    for sc, rows in scenes.items():
        beam = [r for r in rows if r["method"] == "beam" and r["tag"] == ""]
        r256 = [r for r in rows if r["method"] == "ray" and r["samples"] == "256"]
        if not beam:
            continue
        b = beam[0]
        px = num(b, "W") * num(b, "H")
        bt = num(b, "trace_s") + num(b, "primary_s")
        rt = num(r256[0], "trace_s") + num(r256[0], "primary_s") if r256 else float("nan")
        # equal-time ray run: the slowest ray run that is no slower than the beams
        eq = "-"
        cands = [r for r in rows if r["method"] == "ray" and r["samples"] != "256"]
        cands = sorted(cands, key=lambda r: num(r, "trace_s") + num(r, "primary_s"))
        best = None
        for r in cands:
            if num(r, "trace_s") + num(r, "primary_s") <= bt * 1.05:
                best = r
        if best is None and cands:
            best = cands[0]
        if best is not None:
            e = errmap.get((sc, best["samples"]))
            eq = (f"{best['samples']} rays, {num(best, 'trace_s') + num(best, 'primary_s'):.2f} s"
                  + (f", RMSE {float(e['rmse']):.3f}" if e else ""))
        p = PAPER_SOFT.get(sc)
        w(f"| {sc} | {bt:.2f} s | {rt:.2f} s | {rt / bt:.1f}x | {fmt(num(b, 'kd_steps') / px)} | "
          f"{fmt(num(b, 'tri_tests') / px)} | {fmt(num(b, 'hits') / px)} | {fmt(num(b, 'shadow_vis_tris') / px)} | "
          f"{fmt(num(r256[0], 'kd_steps') / px) if r256 else '-'} | {fmt(num(r256[0], 'tri_tests') / px) if r256 else '-'} | "
          f"{eq} | {' / '.join(p) if p else '-'} |")
    w("")
    if errs:
        w("Error of ray-traced visibility against the exact beam result:\n")
        w("| scene | method | samples | RMSE | mean abs | max | pixels > 0.01 | pixels > 0.05 |")
        w("|---|---|---|---|---|---|---|---|")
        for e in errs:
            w(f"| {e['scene']} | {e['method']} | {e['samples']} | {float(e['rmse']):.4f} | {float(e['mae']):.4f} | "
              f"{float(e['max']):.3f} | {100 * float(e['frac_gt_0.01']):.2f}% | {100 * float(e['frac_gt_0.05']):.2f}% |")
        w("")
    ex = [r for r in ss if r["tag"] == "exact"]
    if ex:
        w("Exact integration (Lambert's formula over the visible light polygons instead of eq. 2's "
          "light-center approximation) costs: " +
          ", ".join(f"{r['scene']} {num(r, 'trace_s') + num(r, 'primary_s'):.2f} s" for r in ex) + ".\n")
    w("Images: `img/soft_<scene>_beam.png` (exact), `img/soft_<scene>_ray256.png`, `img/soft_<scene>_ray<N>.png`; "
      "visibility as PFM in `img/*_vis.pfm`; absolute differences in `img/soft_<scene>_diff<N>.pfm`.\n")

# ------------------------------------------------------------------ light sweep
ls = read("lightsweep.csv")
sweep = []
if ls:
    base_area = None
    try:
        for line in open("configs/sponza.cfg"):
            t = line.split()
            if t and t[0] == "area_light":
                u = list(map(float, t[4:7]))
                v = list(map(float, t[7:10]))
                c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
                base_area = math.sqrt(sum(x * x for x in c))
    except OSError:
        pass
    w("## Time vs light source area (paper Fig. 13, Sponza)\n")
    w("The paper reports nearly linear growth (1.8 s, 9.0 s and 38.4 s for its three marked sizes).\n")
    w("| light area | beam time | rays (256) time |")
    w("|---|---|---|")
    for r in ls:
        s = num(r, "light_scale")
        area = base_area * s * s if base_area else float("nan")
        t = num(r, "trace_s") + num(r, "primary_s")
        if r["method"] == "beam":
            sweep.append((area, t))
            w(f"| {fmt(area)} | {t:.2f} s | |")
        else:
            w(f"| {fmt(area)} | | {t:.2f} s |")
    w("")

th = read("threads.csv")
if th:
    w("## Multi-threaded soft shadows (extra, not in the paper)\n")
    w("| scene | method | threads | time |")
    w("|---|---|---|---|")
    for r in th:
        w(f"| {r['scene']} | {r['method']} | {r['threads']} | {num(r, 'trace_s') + num(r, 'primary_s'):.2f} s |")
    w("")

cu = read("cuda.csv")
if cu:
    w("## CUDA port\n")
    w("| scene | GPU soft-shadow time | kd steps/px | isect/px |")
    w("|---|---|---|---|")
    for r in cu:
        px = num(r, "W") * num(r, "H")
        w(f"| {r['scene']} | {num(r, 'trace_s') + num(r, 'primary_s'):.3f} s | {fmt(num(r, 'kd_steps') / px)} | "
          f"{fmt(num(r, 'tri_tests') / px)} |")
    w("")

# ------------------------------------------------------------------ plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID, S1, S2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"

    def style(ax, title, xl, yl):
        ax.set_facecolor(SURF)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(xl, color=INK2)
        ax.set_ylabel(yl, color=INK2)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)

    made = []
    for fname, title, data, idx, ylab in plots:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURF)
        x = [d[1] for d in data]
        y = [d[idx] for d in data]
        ax.scatter(x, y, s=36, color=S1, edgecolors=SURF, linewidths=1.5, zorder=3)
        # direct labels: one per scene, at its largest point
        last = {}
        for d in data:
            if d[0] not in last or d[1] > last[d[0]][1]:
                last[d[0]] = d
        for sc, d in last.items():
            ax.annotate(sc, (d[1], d[idx]), xytext=(6, 4), textcoords="offset points", color=INK2, fontsize=9)
        style(ax, title, "visible triangles per frame", ylab)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=130)
        plt.close(fig)
        made.append((fname, title))
    if sweep:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURF)
        sweep.sort()
        ax.plot([a for a, _ in sweep], [t for _, t in sweep], color=S1, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
        style(ax, "Beam soft-shadow time vs light area, Sponza (paper Fig. 13)", "light area (scene units^2)",
              "seconds per frame")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig13_lightarea.png"), dpi=130)
        plt.close(fig)
        made.append(("fig13_lightarea.png", "Beam soft-shadow time vs light area"))
    if made:
        w("## Plots\n")
        for f, t in made:
            w(f"![{t}]({f})\n")
except ImportError:
    w("_(matplotlib not available: plots skipped)_\n")

with open(os.path.join(OUT, "REPORT.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", os.path.join(OUT, "REPORT.md"))
