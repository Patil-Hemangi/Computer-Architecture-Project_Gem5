"""
ppa_analysis.py  --  EEL6764 HNSW PPA Cost Analysis
Analytical Power-Performance-Area model comparing all sweep configs.
"""
import os
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

configs = [
    # IPC values from re-run after searchKnn bug fix (redundant l2sq eliminated)
    ("Baseline\n256kB ROB128", 1.004,  0.9170, "baseline"),
    ("L2=512kB",  1.012,  0.9161, "l2"),
    ("L2=1MB",    1.046,  0.9161, "l2"),
    ("L2=2MB",    1.110,  0.9158, "l2"),
    ("ROB=32",    0.8694, 0.9196, "rob"),
    ("ROB=64",    1.009,  0.9161, "rob"),
    ("ROB=256",   1.011,  0.9160, "rob"),
    ("HW Prefetch\n256kB", 1.004,  0.9170, "prefetch"),  # confirmed 0% delta
    ("HW Prefetch\n512kB", 1.012,  0.9161, "prefetch"),  # confirmed 0% delta
    ("HW Prefetch\n2MB",   1.110,  0.9158, "prefetch"),  # confirmed 0% delta
    ("DDR5-6400",      0.9023, 0.8725, "mem"),
    ("HBM 1.0",        1.004,  0.8769, "mem"),  # neutral vs DDR4 baseline
    ("SW Prefetch",    0.9774, 0.9207, "mem"),
    ("SW+HBM",         1.024,  0.8753, "mem"),
    # Phase 5-7: algorithmic SW fixes — zero hardware cost, 1.00x area
    ("Graph Reorder",  1.022,  0.8864, "sw_fix"),  # BFS node renumbering
    ("Scalar Quant",   1.041,  0.8458, "sw_fix"),  # float32→int8 (4× vec size)
    ("Reorder+Quant",  1.106,  0.8456, "sw_fix"),  # combined — first YES in PPA
    ("Reorder+L2=2MB", 1.121,  0.8844, "sw_fix"),  # best absolute IPC
]

# Baseline IPC is derived from configs[0] so it stays in sync automatically.
BASELINE_IPC = configs[0][1]

# PPA model constants — CACTI-inspired estimates for a 7 nm-class server chip:
#   L2 cache ≈ 20% of die area (scales linearly with size)
#   ROB       ≈  5% of die area (scales linearly with entry count vs 128-entry base)
#   Prefetcher logic ≈ 2% added area overhead
#   Dynamic power ∝ Area^0.8 (sub-linear: voltage scales down with area at same perf)
CACHE_CHIP_FRACTION    = 0.20
ROB_CHIP_FRACTION      = 0.05
PREFETCH_CHIP_FRACTION = 0.02
POWER_EXPONENT         = 0.80

def relative_area(label):
    # Pure-SW and pure-mem configs override everything — check first to avoid
    # spurious prefetcher/cache additions (e.g. "SW Prefetch" contains "Prefetch").
    if label in ("DDR5-6400", "HBM 1.0", "SW Prefetch", "SW+HBM"):
        return 1.0  # DRAM is off-chip; same die area as baseline
    if label in ("Graph Reorder", "Scalar Quant", "Reorder+Quant"):
        return 1.0  # pure software: zero silicon cost

    if   "512kB" in label: cache_scale = 2.0
    elif "1MB"   in label: cache_scale = 4.0
    elif "2MB"   in label: cache_scale = 8.0
    else:                  cache_scale = 1.0
    area = 1.0 + CACHE_CHIP_FRACTION * (cache_scale - 1.0)

    if   "ROB=32"  in label: area += ROB_CHIP_FRACTION * (32/128  - 1.0)
    elif "ROB=64"  in label: area += ROB_CHIP_FRACTION * (64/128  - 1.0)
    elif "ROB=256" in label: area += ROB_CHIP_FRACTION * (256/128 - 1.0)

    if "Prefetch" in label: area += PREFETCH_CHIP_FRACTION
    return area

def relative_power(area_rel):
    return area_rel ** POWER_EXPONENT

print("\n" + "="*85)
print(f"  {'Config':<22} {'IPC':>6} {'IPC gain':>9} {'Rel Area':>9} {'Rel Power':>10} {'Perf/Area':>10} {'Worth it?':>10}")
print("="*85)

for label, ipc, l2_mr, cat in configs:
    name = label.replace("\n", " ")
    ipc_gain = (ipc - BASELINE_IPC) / BASELINE_IPC * 100
    area     = relative_area(name)
    power    = relative_power(area)
    eff      = ipc_gain / ((area - 1.0) * 100) if area > 1.0 else float("inf")
    # YES: >5% gain at <30% area cost — meaningful improvement, reasonable silicon budget
    # MARGINAL: 2–5% gain or area 1.3–2×
    # NO: <2% gain (noise-level) or negative
    worth    = "YES" if (ipc_gain > 5.0 and area < 1.3) else ("NO" if ipc_gain < 2.0 else "MARGINAL")
    gain_str = f"{ipc_gain:+.1f}%" if abs(ipc_gain) > 0.01 else "baseline"
    print(f"  {name:<22} {ipc:>6.4f} {gain_str:>9} {area:>8.2f}x {power:>9.2f}x {eff:>9.2f} {worth:>10}")

print("="*85)
print(f"\n  Baseline IPC = {BASELINE_IPC:.4f}  (L2=256kB, ROB=128, Width=4, DDR4-2400)")
print(f"  Prefetcher result: IPC unchanged — confirms irregular pointer-chasing bottleneck")
print()

if HAS_PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("HNSW PPA Analysis — gem5 EEL6764", fontsize=13, fontweight="bold")

    labels = [c[0] for c in configs]
    ipcs   = [c[1] for c in configs]
    gains  = [(c[1] - BASELINE_IPC) / BASELINE_IPC * 100 for c in configs]
    areas  = [relative_area(c[0].replace("\n", " ")) for c in configs]
    powers = [relative_power(a) for a in areas]
    CAT_COLOR = {
        "baseline": "gray",
        "l2":       "steelblue",
        "rob":      "tomato",
        "prefetch": "mediumseagreen",
        "mem":      "darkorchid",
        "sw_fix":   "darkorange",
    }
    colors = [CAT_COLOR.get(c[3], "black") for c in configs]

    # Legend patches for all category colors
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLOR.items()]

    axes[0].bar(range(len(configs)), ipcs, color=colors, edgecolor="black")
    axes[0].axhline(BASELINE_IPC, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(configs)))
    axes[0].set_xticklabels(labels, fontsize=6)
    axes[0].set_ylabel("IPC"); axes[0].set_title("IPC Comparison"); axes[0].set_ylim(0.7, 1.3)
    axes[0].legend(handles=legend_patches, fontsize=6, loc="upper left")

    non_baseline = [(a, g, col, c[0]) for a, g, col, c in zip(areas, gains, colors, configs)
                    if c[3] != "baseline"]
    nb_areas  = [x[0] for x in non_baseline]
    nb_gains  = [x[1] for x in non_baseline]
    nb_colors = [x[2] for x in non_baseline]
    nb_labels = [x[3] for x in non_baseline]
    axes[1].scatter(nb_areas, nb_gains, c=nb_colors, s=100, edgecolors="black", zorder=5)
    for a, g, lbl in zip(nb_areas, nb_gains, nb_labels):
        axes[1].annotate(lbl.replace("\n", " "), (a, g),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Relative Chip Area"); axes[1].set_ylabel("IPC Gain vs Baseline (%)")
    axes[1].set_title("Performance vs Area Cost")
    axes[1].legend(handles=legend_patches, fontsize=6, loc="upper left")

    axes[2].bar(range(len(configs)), powers, color=colors, edgecolor="black")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xticks(range(len(configs)))
    axes[2].set_xticklabels(labels, fontsize=6)
    axes[2].set_ylabel("Relative Power"); axes[2].set_title("Power Overhead")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "ppa_analysis.png")
    plt.savefig(out, dpi=150)
    print(f"[plot] Saved: {out}")
