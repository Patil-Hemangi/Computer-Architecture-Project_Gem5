"""
advanced_analysis.py  --  EEL6764 HNSW Advanced Microarchitectural Analysis
Generates three publication-quality figures:
  Fig 1 — CPI Stack Decomposition (bar chart)
  Fig 2 — Roofline Model with latency ceiling
  Fig 3 — L2 MPKI comparison across all configurations

Usage:
    python advanced_analysis.py
    python advanced_analysis.py --baseline-stats results/hnsw_l2_256kB/stats.txt
Outputs: results/cpi_stack.png, results/roofline.png, results/mpki_comparison.png
"""
import os
import re
import sys
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not found — pip install matplotlib")

# ---------------------------------------------------------------------------
# Fallback baseline values (used when stats.txt cannot be found)
# All values from hnsw_l2_256kB_roi/stats.txt  — ROI (search phase only).
# Updated to match actual ROI stats; do NOT use full-run values here.
# ---------------------------------------------------------------------------
_FALLBACK = {
    "SIM_INSTS":        8_397_180,
    "SIM_SECONDS":      0.003003,
    "IPC":              0.932087,
    "CPI":              1.072861,
    "C_0ISSUE":         2_956_759,
    "C_1ISSUE":         1_118_273,
    "C_2ISSUE":         812_170,
    "C_3ISSUE":         1_625_506,
    "C_4ISSUE":         2_390_049,
    "L1D_HITS":         3_226_146,
    "L1D_MISSES":       68_119,
    "L1I_MISSES":       1_411,
    "L2_HITS":          6_317,
    "L2_MISSES":        70_564,
    "DRAM_READ_BURSTS": 75_691,
    "BP_LOOKUPS":       1_284_717,
    "BP_MISPREDICTS":   23_605,
}

def _parse_stat(filepath, pattern):
    """Search for a regex pattern in stats.txt and return the first captured float."""
    try:
        with open(filepath) as f:
            text = f.read()
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            return float(m.group(1))
    except (FileNotFoundError, OSError):
        pass
    return None

def load_baseline(stats_path=None):
    """
    Try to load baseline values from a stats.txt file.
    Falls back gracefully to _FALLBACK values for any key not found.
    Returns a dict with all required keys.
    """
    if stats_path is None:
        # Try default locations relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "results", "hnsw_l2_256kB", "stats.txt"),
            os.path.join(script_dir, "..", "results", "hnsw_baseline", "stats.txt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                stats_path = c
                break

    d = {}
    if stats_path and os.path.isfile(stats_path):
        print(f"[info] Loading baseline stats from: {stats_path}")
        pat = {
            "SIM_INSTS":        r"simInsts\s+([\d.e+]+)",
            "SIM_SECONDS":      r"simSeconds\s+([\d.e+]+)",
            "IPC":              r"board\.processor\.cores\.core\.ipc\s+([\d.e+\-]+)",
            "CPI":              r"board\.processor\.cores\.core\.cpi\s+([\d.e+\-]+)",
            "C_0ISSUE":         r"board\.processor\.cores\.core\.numIssuedDist::0\s+([\d.e+]+)",
            "C_1ISSUE":         r"board\.processor\.cores\.core\.numIssuedDist::1\s+([\d.e+]+)",
            "C_2ISSUE":         r"board\.processor\.cores\.core\.numIssuedDist::2\s+([\d.e+]+)",
            "C_3ISSUE":         r"board\.processor\.cores\.core\.numIssuedDist::3\s+([\d.e+]+)",
            "C_4ISSUE":         r"board\.processor\.cores\.core\.numIssuedDist::4\s+([\d.e+]+)",
            "L1D_HITS":         r"board\.cache_hierarchy\.l1d-cache-0\.overallHits::total\s+([\d.e+]+)",
            "L1D_MISSES":       r"board\.cache_hierarchy\.l1d-cache-0\.overallMisses::total\s+([\d.e+]+)",
            "L1I_MISSES":       r"board\.cache_hierarchy\.l1i-cache-0\.overallMisses::total\s+([\d.e+]+)",
            "L2_HITS":          r"board\.cache_hierarchy\.l2-cache-0\.overallHits::total\s+([\d.e+]+)",
            "L2_MISSES":        r"board\.cache_hierarchy\.l2-cache-0\.overallMisses::total\s+([\d.e+]+)",
            "DRAM_READ_BURSTS": r"board\.memory\.mem_ctrl\.readBursts\s+([\d.e+]+)",
            "BP_LOOKUPS":       r"board\.processor\.cores\.core\.branchPred\.condPredicted\s+([\d.e+]+)",
            "BP_MISPREDICTS":   r"board\.processor\.cores\.core\.branchPred\.condIncorrect\s+([\d.e+]+)",
        }
        for key, pattern in pat.items():
            val = _parse_stat(stats_path, pattern)
            if val is not None:
                d[key] = val
            else:
                print(f"[warn] '{key}' not found in stats.txt — using fallback value")
                d[key] = _FALLBACK[key]
    else:
        print("[warn] No stats.txt found — using fallback baseline values")
        d = dict(_FALLBACK)
    return d

# ---------------------------------------------------------------------------
# Load baseline — use CLI arg if provided, else auto-discover
# ---------------------------------------------------------------------------
import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument("--baseline-stats", default=None)
_args, _ = _p.parse_known_args()
_B = load_baseline(_args.baseline_stats)

# Unpack into module-level names for backward compat with all functions below
SIM_INSTS        = int(_B["SIM_INSTS"])
SIM_SECONDS      = _B["SIM_SECONDS"]
CLOCK_GHZ        = 3.0
ISSUE_WIDTH      = 4

IPC              = _B["IPC"]
CPI              = _B["CPI"]
TOTAL_CYCLES     = SIM_INSTS * CPI

C_0ISSUE         = int(_B["C_0ISSUE"])
C_1ISSUE         = int(_B["C_1ISSUE"])
C_2ISSUE         = int(_B["C_2ISSUE"])
C_3ISSUE         = int(_B["C_3ISSUE"])
C_4ISSUE         = int(_B["C_4ISSUE"])

L1D_HITS         = int(_B["L1D_HITS"])
L1D_MISSES       = int(_B["L1D_MISSES"])
L1I_MISSES       = int(_B["L1I_MISSES"])
L2_HITS          = int(_B["L2_HITS"])
L2_MISSES        = int(_B["L2_MISSES"])
DRAM_READ_BURSTS = int(_B["DRAM_READ_BURSTS"])

BP_LOOKUPS       = int(_B["BP_LOOKUPS"])
BP_MISPREDICTS   = int(_B["BP_MISPREDICTS"])

# Memory constants — configurable via environment or keep as documented defaults
DRAM_BURST_BYTES   = 64
DDR4_PEAK_BW_GBS   = float(os.environ.get("DDR4_PEAK_BW_GBS", "19.2"))
DDR4_LATENCY_NS    = float(os.environ.get("DDR4_LATENCY_NS",  "55.0"))
HBM_LATENCY_NS     = 10.0
BP_PENALTY_CYCLES  = 12

# ---------------------------------------------------------------------------
# Derived scalars
# ---------------------------------------------------------------------------
DRAM_READ_BYTES = DRAM_READ_BURSTS * DRAM_BURST_BYTES
EFF_BW_GBS      = DRAM_READ_BYTES / SIM_SECONDS / 1e9       # 1.614 GB/s
BW_UTIL_PCT     = EFF_BW_GBS / DDR4_PEAK_BW_GBS * 100       # 8.4%
SERIAL_CEIL_GBS = DRAM_BURST_BYTES / (DDR4_LATENCY_NS * 1e-9) / 1e9  # 1.164 GB/s

MPKI_L1D        = L1D_MISSES / SIM_INSTS * 1000   # 7.60
MPKI_L1I        = L1I_MISSES / SIM_INSTS * 1000   # 0.15
MPKI_L2         = L2_MISSES  / SIM_INSTS * 1000   # 7.80
MPKI_DRAM       = DRAM_READ_BURSTS / SIM_INSTS * 1000  # 8.37

F_MEM           = C_0ISSUE / TOTAL_CYCLES          # 0.323 — memory stall fraction

# ---------------------------------------------------------------------------
# CPI Stack values
# ---------------------------------------------------------------------------
CPI_IDEAL       = 1.0 / ISSUE_WIDTH                                   # 0.250
CPI_MEM_STALL   = C_0ISSUE / SIM_INSTS                                # 0.322
CPI_BRANCH      = BP_MISPREDICTS * BP_PENALTY_CYCLES / SIM_INSTS      # 0.032
CPI_OTHER       = CPI - CPI_IDEAL - CPI_MEM_STALL - CPI_BRANCH       # 0.392

# ---------------------------------------------------------------------------
# MPKI across all configurations (L2 MPKI derived from miss rate × baseline accesses)
# Format: (label, l2_miss_rate, color)
# ---------------------------------------------------------------------------
BASELINE_L2_ACCESSES = L2_HITS + L2_MISSES  # 76,963

CONFIGS_MPKI = [
    ("Baseline\n256kB",  0.9170, "gray"),
    ("L2=512kB",         0.9161, "steelblue"),
    ("L2=1MB",           0.9161, "steelblue"),
    ("L2=2MB",           0.9158, "steelblue"),
    ("ROB=32",           0.9196, "tomato"),
    ("ROB=64",           0.9161, "tomato"),
    ("ROB=256",          0.9160, "tomato"),
    ("HWpf+256kB",       0.9170, "mediumseagreen"),
    ("HWpf+512kB",       0.9161, "mediumseagreen"),
    ("HWpf+2MB",         0.9158, "mediumseagreen"),
    ("DDR5",             0.8725, "darkorchid"),
    ("HBM",              0.8769, "darkorchid"),
    ("SW-pf",            0.9207, "darkorchid"),
    ("SW+HBM",           0.8753, "darkorchid"),
    ("Reorder",          0.8864, "darkorange"),
    ("SQ(int8)",         0.8458, "darkorange"),
    ("Reorder+Quant",    0.8456, "darkorange"),
    ("Reorder+2MB",      0.8844, "darkorange"),
]

# ---------------------------------------------------------------------------
# Amdahl sweep data
# ---------------------------------------------------------------------------
K_VALS  = np.array([1, 2, 3, 5, 10, 20, 50, 100, 1e6])
SPEEDUP = 1.0 / ((1.0 - F_MEM) + F_MEM / K_VALS)


# ---------------------------------------------------------------------------
# Print summary to console
# ---------------------------------------------------------------------------
def print_advanced_summary():
    print("\n" + "="*65)
    print("  HNSW Advanced Microarchitectural Analysis — Baseline")
    print("="*65)

    print(f"\n  Cache Hierarchy (MPKI = Misses Per Kilo Instructions):")
    print(f"    L1D MPKI   {MPKI_L1D:>8.2f}  ({L1D_MISSES:,} misses / {SIM_INSTS/1e6:.2f}M insts)")
    print(f"    L1I MPKI   {MPKI_L1I:>8.2f}  ({L1I_MISSES:,} misses)")
    print(f"    L2  MPKI   {MPKI_L2:>8.2f}  ({L2_MISSES:,} misses, {L2_HITS:,} hits)")
    print(f"    DRAM MPKI  {MPKI_DRAM:>8.2f}  ({DRAM_READ_BURSTS:,} × 64B read bursts)")

    print(f"\n  Effective DRAM Bandwidth:")
    print(f"    Achieved        {EFF_BW_GBS:.3f} GB/s  ({BW_UTIL_PCT:.1f}% of {DDR4_PEAK_BW_GBS} GB/s peak)")
    print(f"    Serial ceiling  {SERIAL_CEIL_GBS:.3f} GB/s  (1 req / {DDR4_LATENCY_NS:.0f} ns DDR4 latency)")
    print(f"    Peak DDR4-2400 {DDR4_PEAK_BW_GBS:.1f}   GB/s  (2.4 GT/s × 8B)")
    print(f"    ⟹  Pointer-chasing limits MLP to ~1 outstanding request")
    print(f"       achieving only {EFF_BW_GBS:.2f}/{DDR4_PEAK_BW_GBS:.1f} = {BW_UTIL_PCT:.1f}% of bandwidth")

    print(f"\n  CPI Stack Decomposition (total CPI = {CPI:.4f}):")
    print(f"    Ideal 4-wide base  {CPI_IDEAL:.4f}  ({CPI_IDEAL/CPI*100:5.1f}% of CPI)")
    print(f"    Memory stall       {CPI_MEM_STALL:.4f}  ({CPI_MEM_STALL/CPI*100:5.1f}%)  ← dominant (0-issue cycles)")
    print(f"    Branch mispredict  {CPI_BRANCH:.4f}  ({CPI_BRANCH/CPI*100:5.1f}%)  ({BP_MISPREDICTS:,} × {BP_PENALTY_CYCLES} cycles)")
    print(f"    Other (ILP<4,      {CPI_OTHER:.4f}  ({CPI_OTHER/CPI*100:5.1f}%)  (partial-issue + decode pressure)")
    print(f"       decode stalls)")

    print(f"\n  Amdahl's Law — Memory Stall = {F_MEM*100:.1f}% of cycles:")
    print(f"    k=∞  (eliminate stalls)  {1/(1-F_MEM):.3f}×  IPC ceiling = {IPC/(1-F_MEM):.3f}")
    print(f"    k=2  (HBM 2× latency)   {1/((1-F_MEM)+F_MEM/2):.3f}×")
    print(f"    k=10 (L3/near-mem)       {1/((1-F_MEM)+F_MEM/10):.3f}×")
    print(f"    k=20 (PIM)               {1/((1-F_MEM)+F_MEM/20):.3f}×")
    print(f"    ⟹  Even perfect memory gives only {(1/(1-F_MEM)-1)*100:.1f}% max IPC gain")
    print(f"       because {(1-F_MEM)*100:.1f}% of cycles are compute or decode-limited")
    print("="*65)


# ---------------------------------------------------------------------------
# Fig 1: CPI Stack Bar Chart
# ---------------------------------------------------------------------------
def plot_cpi_stack(out_dir):
    if not HAS_PLOT:
        return
    labels = ["Baseline\n(256kB, ROB128)"]
    ideal  = [CPI_IDEAL]
    mem    = [CPI_MEM_STALL]
    branch = [CPI_BRANCH]
    other  = [CPI_OTHER]

    fig, ax = plt.subplots(figsize=(5, 5))
    x = range(len(labels))

    b1 = ax.bar(x, ideal,  color="steelblue",     edgecolor="black", label="Ideal base (4-wide)", width=0.5)
    b2 = ax.bar(x, mem,    bottom=ideal,           color="tomato",        edgecolor="black", label="Memory stall (0-issue)", width=0.5)
    b3 = ax.bar(x, branch, bottom=[i+m for i,m in zip(ideal, mem)],
                color="gold",          edgecolor="black", label="Branch mispredict", width=0.5)
    b4 = ax.bar(x, other,  bottom=[i+m+b for i,m,b in zip(ideal, mem, branch)],
                color="lightgray",     edgecolor="black", label="Other (ILP<4, decode)", width=0.5)

    # Annotate percentages
    cumsum = [0] * len(labels)
    for vals, c in [(ideal, "white"), (mem, "white"), (branch, "black"), (other, "black")]:
        for i, v in enumerate(vals):
            if v > 0.01:
                ax.text(i, cumsum[i] + v/2, f"{v:.3f}\n({v/CPI*100:.0f}%)",
                        ha="center", va="center", fontsize=8, color=c, fontweight="bold")
            cumsum[i] += v

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CPI (cycles per instruction)", fontsize=10)
    ax.set_title("HNSW CPI Stack Decomposition\ngem5 X86O3CPU Baseline", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(CPI, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(0.52, CPI + 0.01, f"Actual CPI = {CPI:.4f}", fontsize=7, color="black")

    plt.tight_layout()
    out = os.path.join(out_dir, "cpi_stack.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 2: Roofline Model
# ---------------------------------------------------------------------------
def plot_roofline(out_dir):
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(7, 5))

    # Compute measured arithmetic intensity
    # FP instructions: FP Add + SIMD FP (each SIMD = 4-wide SSE)
    fp_scalar = 2_104_353  # FP Add instructions
    fp_simd   = 2_003_199  # SIMD FP instructions × 4 FLOPs each
    total_flop = fp_scalar * 1 + fp_simd * 4  # 10.1 MFLOP
    ai_measured = total_flop / DRAM_READ_BYTES  # FLOP/byte with real DRAM traffic

    # Peak compute ceiling: 4-wide × 3 GHz (scalar FP)
    peak_compute_gflops = CLOCK_GHZ * ISSUE_WIDTH  # 12 GFLOP/s

    # AI sweep for roofline shape
    ai_range = np.logspace(-2, 3, 500)

    # Standard roofline (peak bandwidth)
    bw_ceil_peak   = DDR4_PEAK_BW_GBS * ai_range
    compute_ceil   = np.full_like(ai_range, peak_compute_gflops)
    roofline_peak  = np.minimum(bw_ceil_peak, compute_ceil)

    # Latency-limited roofline (effective serialized bandwidth)
    bw_ceil_serial  = EFF_BW_GBS * ai_range
    roofline_serial = np.minimum(bw_ceil_serial, compute_ceil)

    # HNSW achieved compute throughput
    achieved_gflops = total_flop / SIM_SECONDS / 1e9  # GFLOP/s

    ax.loglog(ai_range, roofline_peak, "b-",  linewidth=2, label=f"Peak roofline (DDR4 {DDR4_PEAK_BW_GBS:.0f} GB/s + {peak_compute_gflops:.0f} GFLOP/s)")
    ax.loglog(ai_range, roofline_serial, "r--", linewidth=2, label=f"Latency-limited ({EFF_BW_GBS:.2f} GB/s eff., serial pointer-chase)")

    # Ridge point
    ridge_ai_peak   = peak_compute_gflops / DDR4_PEAK_BW_GBS
    ridge_ai_serial = peak_compute_gflops / EFF_BW_GBS
    ax.axvline(ridge_ai_peak,   color="blue",  linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(ridge_ai_serial, color="red",   linestyle=":", linewidth=1, alpha=0.7)
    ax.text(ridge_ai_peak * 1.1,  0.015, f"Ridge\n{ridge_ai_peak:.2f}", fontsize=7, color="blue")
    ax.text(ridge_ai_serial * 0.5, 0.015, f"Ridge\n{ridge_ai_serial:.1f}", fontsize=7, color="red", ha="right")

    # HNSW operating point
    ax.scatter([ai_measured], [achieved_gflops], color="darkorange", s=120, zorder=5,
               marker="*", label=f"HNSW operating point\n(AI={ai_measured:.2f} FLOP/B, {achieved_gflops:.2f} GFLOP/s)")
    ax.annotate(f"HNSW\n({ai_measured:.2f}, {achieved_gflops:.2f} GFLOP/s)",
                (ai_measured, achieved_gflops),
                xytext=(ai_measured * 3, achieved_gflops * 2),
                arrowprops=dict(arrowstyle="->", color="darkorange"),
                fontsize=8, color="darkorange")

    ax.set_xlabel("Arithmetic Intensity (FLOP / DRAM byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=10)
    ax.set_title("HNSW Roofline Model — Pointer-Chasing Serialization\n"
                 "Standard roofline vs latency-limited effective ceiling", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.01, 100)

    # Annotation box explaining the gap
    ax.text(0.02, 50, f"HNSW appears compute-bound on\nstandard roofline (AI={ai_measured:.2f} > ridge {ridge_ai_peak:.2f})\n"
            f"but achieves only {achieved_gflops:.2f} GFLOP/s\n"
            f"because effective BW = {EFF_BW_GBS:.2f} GB/s\n"
            f"({BW_UTIL_PCT:.1f}% of peak — serialized by\npointer-chasing, MLP ≈ 1)",
            fontsize=7, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(out_dir, "roofline.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3: L2 MPKI Comparison
# ---------------------------------------------------------------------------
def plot_mpki(out_dir):
    if not HAS_PLOT:
        return
    labels = [c[0] for c in CONFIGS_MPKI]
    # L2 MPKI approximation: assumes total L2 accesses ≈ baseline (76,963) for all configs.
    # Exact formula is miss_rate × actual_L2_accesses / (sim_insts/1000); we only have miss_rate,
    # so baseline_L2_accesses is used as a constant approximation across configs.
    mpki_vals = [c[1] * BASELINE_L2_ACCESSES / (SIM_INSTS / 1000) for c in CONFIGS_MPKI]
    colors    = [c[2] for c in CONFIGS_MPKI]

    n = len(CONFIGS_MPKI)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.8), 5))
    x = range(n)
    ax.bar(x, mpki_vals, color=colors, edgecolor="black", width=0.7)
    ax.axhline(MPKI_L2, color="black", linestyle="--", linewidth=1, label=f"Baseline L2 MPKI = {MPKI_L2:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("L2 MPKI (Misses Per Kilo Instructions)", fontsize=10)
    ax.set_title("L2 MPKI Across All Configurations\n(lower = better cache utilization)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(5.5, 8.5)

    cat_colors = {
        "Baseline": "gray", "L2 sweep": "steelblue", "ROB sweep": "tomato",
        "HW Prefetch": "mediumseagreen", "DRAM/SW": "darkorchid", "SW fix": "darkorange",
    }
    patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(out_dir, "mpki_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 4: Amdahl's Law speedup curve
# ---------------------------------------------------------------------------
def plot_amdahl(out_dir):
    if not HAS_PLOT:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    k_range = np.logspace(0, 3, 500)
    speedup = 1.0 / ((1.0 - F_MEM) + F_MEM / k_range)

    ax.semilogx(k_range, speedup, "b-", linewidth=2)
    ax.axhline(1.0 / (1.0 - F_MEM), color="red", linestyle="--",
               label=f"Theoretical max = {1/(1-F_MEM):.2f}× (k→∞)", linewidth=1.5)

    # Mark specific configs
    marked = [
        (1,   "Baseline",  "gray"),
        (2,   "HBM (~2×)", "darkorchid"),
        (20,  "PIM (~20×)","darkorange"),
    ]
    for k, lbl, col in marked:
        s = 1.0 / ((1.0 - F_MEM) + F_MEM / k)
        ax.scatter([k], [s], color=col, s=80, zorder=5)
        ax.annotate(f"{lbl}\n{s:.2f}×", (k, s), xytext=(k * 1.4, s - 0.03),
                    fontsize=8, color=col)

    ax.set_xlabel("Memory latency improvement factor k", fontsize=10)
    ax.set_ylabel("Speedup over baseline", fontsize=10)
    ax.set_title(f"Amdahl's Law — HNSW Memory Bottleneck\n"
                 f"f_mem = {F_MEM*100:.1f}% of cycles are memory stalls",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.9, 1.6)

    plt.tight_layout()
    out = os.path.join(out_dir, "amdahl_bound.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEL6764 HNSW advanced analysis")
    parser.add_argument("--baseline-stats", default=None,
                        help="Path to baseline stats.txt (auto-discovers if omitted)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots (default: <project_root>/results)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(script_dir, "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    print_advanced_summary()
    plot_cpi_stack(out_dir)
    plot_roofline(out_dir)
    plot_mpki(out_dir)
    plot_amdahl(out_dir)

    print(f"\nAll plots saved to: {os.path.abspath(out_dir)}/")
