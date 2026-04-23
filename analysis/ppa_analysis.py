"""
ppa_analysis.py  --  EEL6764 HNSW PPA Analysis
Uses REAL McPAT v1.3 numbers (45nm HP process) — NOT analytical estimates.

McPAT was run on gem5 stats.txt for each ROI configuration.
Area is circuit-level (SRAM + logic + interconnect), power is full chip.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CLOCK_HZ   = 3.0e9   # 3 GHz (gem5 config)
BASELINE_A = 39.21   # mm² — real McPAT area for baseline (all SW configs share this)

# Real McPAT results for each configuration
# (label, IPC_ROI, area_mm2, runtime_power_W, total_power_W, category)
configs = [
    ("Baseline\n(256kB L2)", 0.932,  39.21,  6.060,  9.970,  "baseline"),
    ("Graph\nReorder",       0.986,  39.21,  5.780,  9.691,  "sw_fix"),
    ("Scalar Quant\n(int8)", 1.026,  39.21,  5.973,  9.884,  "sw_fix"),
    ("Reorder +\nQuant",     1.039,  39.21,  6.025,  9.936,  "sw_fix"),
    ("L2=2MB\n(HW only)",   1.036,  53.53,  6.558, 11.335,  "hw"),
]

BASELINE_IPC   = configs[0][1]
BASELINE_AREA  = configs[0][2]
BASELINE_POWER = configs[0][4]

CAT_COLOR = {
    "baseline": "#888888",
    "sw_fix":   "#FF9800",
    "hw":       "#F44336",
}
CAT_LABEL = {
    "baseline": "Baseline",
    "sw_fix":   "SW-only fix (0% area cost)",
    "hw":       "HW change (silicon cost)",
}

# ─── Figure: 3-panel PPA using real McPAT numbers ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("HNSW PPA Analysis — Real McPAT v1.3 Numbers (45nm HP, 3 GHz)",
             fontsize=12, fontweight="bold")

labels  = [c[0]                                           for c in configs]
ipcs    = [c[1]                                           for c in configs]
gains   = [(c[1] - BASELINE_IPC)/BASELINE_IPC*100        for c in configs]
areas   = [c[2] / BASELINE_AREA                          for c in configs]
colors  = [CAT_COLOR[c[5]]                               for c in configs]

# Legend patches
legend_patches = [
    mpatches.Patch(color=v, label=CAT_LABEL[k]) for k, v in CAT_COLOR.items()
]

# ── Panel 1: IPC bar chart ──────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(range(len(configs)), ipcs, color=colors, edgecolor="black", linewidth=0.8)
ax.axhline(BASELINE_IPC, color="black", linestyle="--", linewidth=1, label="Baseline IPC")
for i, (bar, ipc) in enumerate(zip(bars, ipcs)):
    ax.text(bar.get_x() + bar.get_width()/2, ipc + 0.005, f"{ipc:.3f}",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("IPC (ROI — search phase)")
ax.set_title("IPC Comparison\n(real gem5 stats)")
ax.set_ylim(0.80, 1.15)
ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color="black", linestyle="--", label="Baseline")
], fontsize=7, loc="upper left")
ax.set_facecolor("#F9F9F9")

# ── Panel 2: IPC gain vs relative chip area (real McPAT) ───────────────────
ax = axes[1]
non_base = [(areas[i], gains[i], colors[i], labels[i])
            for i in range(len(configs)) if configs[i][5] != "baseline"]
nb_areas, nb_gains, nb_colors, nb_labels = zip(*non_base)

ax.scatter(nb_areas, nb_gains, c=nb_colors, s=180,
           edgecolors="black", linewidths=0.8, zorder=5)
for a, g, lbl in zip(nb_areas, nb_gains, nb_labels):
    ax.annotate(lbl.replace("\n", " "), (a, g),
                xytext=(8, 4), textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="No area cost")
ax.axhline(0,   color="black", linewidth=0.5)
ax.set_xlabel("Relative Chip Area  (McPAT mm² / Baseline 39.21 mm²)", fontsize=9)
ax.set_ylabel("IPC Gain vs Baseline  (%)")
ax.set_title("Performance vs Area Cost\n(real McPAT area — NOT estimated)")
ax.set_xlim(0.90, 1.50)
ax.set_ylim(-2, 14)
ax.legend(handles=legend_patches, fontsize=7, loc="upper right")
ax.set_facecolor("#F9F9F9")
ax.grid(True, linestyle="--", alpha=0.35, color="gray")

# Annotate L2=2MB explicitly with real area
hw_idx = next(i for i in range(len(configs)) if configs[i][5] == "hw")
ax.annotate(
    f"L2=2MB:\n{areas[hw_idx]:.2f}× area\n+{gains[hw_idx]:.1f}% IPC",
    xy=(areas[hw_idx], gains[hw_idx]),
    xytext=(-65, -28), textcoords="offset points",
    fontsize=7.5, color="#B71C1C",
    arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2),
    bbox=dict(boxstyle="round", fc="#FFEBEE", ec="#F44336", alpha=0.9)
)

# ── Panel 3: Total power (real McPAT watts) ─────────────────────────────────
ax = axes[2]
total_powers_W = [c[4] for c in configs]
bars = ax.bar(range(len(configs)), total_powers_W, color=colors,
              edgecolor="black", linewidth=0.8)
ax.axhline(BASELINE_POWER, color="black", linestyle="--", linewidth=1)
for i, (bar, pw) in enumerate(zip(bars, total_powers_W)):
    ax.text(bar.get_x() + bar.get_width()/2, pw + 0.05, f"{pw:.2f}W",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("Total Power  (W)  — McPAT 45nm HP")
ax.set_title("Total Chip Power\n(runtime dynamic + leakage)")
ax.set_ylim(0, 13.5)
ax.legend(handles=legend_patches, fontsize=7, loc="upper left")
ax.set_facecolor("#F9F9F9")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "results", "ppa_analysis.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"[saved] {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"  {'Config':<26} {'IPC':>6} {'Gain':>8} {'Area(mm²)':>10} {'Rel.Area':>9} "
      f"{'TotPwr(W)':>10} {'IPC/W':>8}")
print("="*90)
for label, ipc, area, rp, tp, cat in configs:
    name     = label.replace("\n", " ")
    gain     = (ipc - BASELINE_IPC) / BASELINE_IPC * 100
    rel_a    = area / BASELINE_AREA
    ipc_w    = ipc / tp
    gain_str = f"{gain:+.1f}%" if abs(gain) > 0.01 else "baseline"
    print(f"  {name:<26} {ipc:>6.3f} {gain_str:>8} {area:>10.2f} {rel_a:>8.2f}× "
          f"{tp:>10.3f} {ipc_w:>8.4f}")
print("="*90)
print(f"\n  Baseline: 39.21 mm², 9.970 W, IPC=0.932")
print(f"  L2=2MB:  53.53 mm² (+1.37×), 11.335 W (+13.7%), IPC=1.036 — NOT Pareto optimal")
print(f"  R+Q:     39.21 mm² (1.00×),   9.936 W ( -0.3%), IPC=1.039 — Pareto optimal")
