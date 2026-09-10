#!/usr/bin/env python3
"""
Plot combined scalability benchmark: Fortran (CPU serial) vs LLM JAX translations (GPU).

Run from the kessler project root:
    python3 _compare_results/plot_Fortran_vs_LLMs_scalability.py

Prerequisites — the two input JSON files (the paper's are archived under
_compare_results/outputs/scalability/; regenerate with):
  1. LLMs JAX benchmark (one GPU):  qsub ../LAFT/pbsJobs/compare_scalability_LLMs.sh
     Writes: _compare_results/outputs/scalability/LLMs_scalability_results.json
  2. Fortran serial benchmark (CPU): qsub ../LAFT/pbsJobs/compare_scalability_fortran.sh
     Writes: _compare_results/outputs/scalability/fortran_scalability_results.json

Reads:
    _compare_results/outputs/scalability/LLMs_scalability_results.json
        — JAX timings for Claude, GPT, Gemini, Qwen (from scalability_benchmark_LLMs.py)
    _compare_results/outputs/scalability/fortran_scalability_results.json
        — Fortran serial CPU timings (from data/exp2_jd/src/scalability_benchmark_fortran.F90)

Outputs:
    _compare_results/outputs/plots/plot_Fortran_vs_LLMs_scalability.png
        — Side-by-side linear and log-log scalability plots (Fortran vs all LLM translations)

    _compare_results/outputs/scalability/scalability_timings.csv
        — Tab-separated table of median ± std runtimes (ms) for every model at each ncol

    _compare_results/outputs/scalability/scalability_speedup_ncol10000.csv
        — Tab-separated speedup of each LLM translation relative to Fortran at ncol=10,000
"""

import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent
OUT_DIR  = HERE / 'outputs'
JAX_JSON  = OUT_DIR / 'scalability' / 'LLMs_scalability_results.json'
FTN_JSON  = OUT_DIR / 'scalability' / 'fortran_scalability_results.json'
PLOT_OUT  = OUT_DIR / 'plots'      / 'plot_Fortran_vs_LLMs_scalability.png'
CSV_OUT   = OUT_DIR / 'scalability' / 'scalability_timings.csv'
SPEED_OUT = OUT_DIR / 'scalability' / 'scalability_speedup_ncol10000.csv'

PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f'WARNING: {path} not found — skipping')
        return None
    with open(path) as f:
        return json.load(f)

jax_data = load_json(JAX_JSON)
ftn_data = load_json(FTN_JSON)

if jax_data is None and ftn_data is None:
    print('ERROR: no data files found. Run the benchmarks first.')
    sys.exit(1)

# Merge all results under one dict: model_key -> list of row dicts
all_results = {}
if jax_data:
    all_results.update(jax_data['results'])
if ftn_data:
    all_results.update(ftn_data['results'])

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STYLE = {
    'claude':  {'color': '#f97316', 'marker': 'o', 'ls': '-',  'label': 'Claude (JAX/GPU)'},
    'gpt':     {'color': '#10b981', 'marker': 's', 'ls': '-',  'label': 'GPT (JAX/GPU)'},
    'gemini':  {'color': '#3b82f6', 'marker': '^', 'ls': '-',  'label': 'Gemini (JAX/GPU)'},
    'qwen':    {'color': '#8b5cf6', 'marker': 'D', 'ls': '-',  'label': 'Qwen (JAX/GPU)'},
    'fortran': {'color': '#ef4444', 'marker': 'X', 'ls': '--', 'label': 'Fortran (CPU serial)'},
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'kessler_run Scalability: Fortran (CPU serial) vs JAX LLM translations (GPU)\n'
    f'nz={jax_data["config"]["nz"]}, dt=60 s, {5} reps — median ± std',
    fontsize=13, fontweight='bold',
)

for model, rows in all_results.items():
    good    = [r for r in rows if 'error' not in r and 'skipped' not in r]
    if not good:
        continue
    ncols   = [r['ncol']     for r in good]
    medians = [r['median_s'] * 1e3 for r in good]   # ms
    stds    = [r['std_s']    * 1e3 for r in good]

    sty = STYLE.get(model, {'color': 'gray', 'marker': 'x', 'ls': '-', 'label': model})
    kw  = dict(color=sty['color'], marker=sty['marker'], linestyle=sty['ls'],
               label=sty['label'], linewidth=1.8, markersize=7)

    axes[0].errorbar(ncols, medians, yerr=stds, **kw)
    axes[1].errorbar(ncols, medians, yerr=stds, **kw)

for ax, xscale, yscale, title in zip(
    axes,
    ['linear', 'log'],
    ['linear', 'log'],
    ['Runtime vs Grid Size (linear)', 'Runtime vs Grid Size (log-log)'],
):
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel('ncol (number of columns)', fontsize=10)
    ax.set_ylabel('Median runtime (ms)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=150, bbox_inches='tight')
print(f'Plot saved -> {PLOT_OUT}')

# ---------------------------------------------------------------------------
# Build per-model lookup: model -> {ncol: (median_ms, std_ms)}
# ---------------------------------------------------------------------------
MODEL_ORDER = ['fortran', 'claude', 'gpt', 'gemini', 'qwen']
LABEL = {
    'fortran': 'Fortran (CPU serial)',
    'claude':  'Claude (JAX/GPU)',
    'gpt':     'GPT (JAX/GPU)',
    'gemini':  'Gemini (JAX/GPU)',
    'qwen':    'Qwen (JAX/GPU)',
}

model_data = {}   # model -> {ncol: (median_ms, std_ms)}
for model, rows in all_results.items():
    good = [r for r in rows if 'error' not in r and 'skipped' not in r]
    model_data[model] = {r['ncol']: (r['median_s'] * 1e3, r['std_s'] * 1e3) for r in good}

# Collect all ncols present across all models, sorted
all_ncols = sorted({ncol for md in model_data.values() for ncol in md})

# Present models in display order (skip any not in the data)
present_models = [m for m in MODEL_ORDER if m in model_data]

# ---------------------------------------------------------------------------
# CSV 1 — all timings:  ncol | model1 (median ± std) | model2 | ...
# ---------------------------------------------------------------------------
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(CSV_OUT, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(['ncol'] + [LABEL[m] for m in present_models])
    for ncol in all_ncols:
        row = [ncol]
        for m in present_models:
            if ncol in model_data[m]:
                med, std = model_data[m][ncol]
                row.append(f'{med:.3f} ± {std:.3f}')
            else:
                row.append('N/A')
        writer.writerow(row)
print(f'Timings CSV saved -> {CSV_OUT}')

# ---------------------------------------------------------------------------
# CSV 2 — speedup vs Fortran at ncol=10,000
# ---------------------------------------------------------------------------
TARGET_NCOL = 10_000
if 'fortran' in model_data and TARGET_NCOL in model_data['fortran']:
    ftn_med = model_data['fortran'][TARGET_NCOL][0]   # ms
    with open(SPEED_OUT, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['model', f'runtime_ms (ncol={TARGET_NCOL})', 'speedup_vs_fortran'])
        for m in present_models:
            if TARGET_NCOL in model_data[m]:
                med, std = model_data[m][TARGET_NCOL]
                speedup = ftn_med / med
                writer.writerow([LABEL[m], f'{med:.3f} ± {std:.3f}', f'{speedup:.2f}×'])
            else:
                writer.writerow([LABEL[m], 'N/A', 'N/A'])
    print(f'Speedup CSV saved -> {SPEED_OUT}')
else:
    print(f'WARNING: Fortran data at ncol={TARGET_NCOL} not found — speedup CSV skipped')
