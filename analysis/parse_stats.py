"""
parse_stats.py  --  EEL6764 GEM5 Stats Analyzer
Parses m5out/stats.txt and prints a clean summary + generates plots.

Usage:
    python parse_stats.py --stats m5out/stats.txt
    python parse_stats.py --sweep-dir results/  --param l1d_size

Requires: matplotlib  (pip install matplotlib)
"""

import argparse
import os
import re
import sys

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not found -- plots disabled. pip install matplotlib")


# ---------------------------------------------------------------------------
# Key stats to extract (regex key -> friendly label)
# ---------------------------------------------------------------------------
STAT_KEYS = {
    # Performance
    r"simInsts\s+([\d.e+]+)":                          "Simulated Instructions",
    r"simTicks\s+([\d.e+]+)":                          "Simulated Ticks",
    r"hostSeconds\s+([\d.e+]+)":                       "Host Wall-Clock (s)",
    r"simSeconds\s+([\d.e+]+)":                        "Simulated Seconds",

    # O3CPU IPC/CPI (gem5 v24)
    r"board\.processor\.cores\.core\.ipc\s+([\d.e+\-]+)":   "IPC",
    r"board\.processor\.cores\.core\.cpi\s+([\d.e+\-]+)":   "CPI",
    r"board\.processor\.cores\.core\.branchPred\.condIncorrect\s+([\d.e+]+)": "Branch Mispredicts",
    r"board\.processor\.cores\.core\.branchPred\.condPredicted\s+([\d.e+]+)": "Branch Predictions",
    r"board\.processor\.cores\.core\.branchPred\.lookups_0::total\s+([\d.e+]+)": "Branch Lookups",

    # L1D Cache (gem5 v24 naming)
    r"board\.cache_hierarchy\.l1d-cache-0\.overallHits::total\s+([\d.e+]+)":      "L1D Hits",
    r"board\.cache_hierarchy\.l1d-cache-0\.overallMisses::total\s+([\d.e+]+)":    "L1D Misses",
    r"board\.cache_hierarchy\.l1d-cache-0\.overallMissRate::total\s+([\d.e+]+)":  "L1D Miss Rate",

    # L1I Cache
    r"board\.cache_hierarchy\.l1i-cache-0\.overallHits::total\s+([\d.e+]+)":      "L1I Hits",
    r"board\.cache_hierarchy\.l1i-cache-0\.overallMisses::total\s+([\d.e+]+)":    "L1I Misses",
    r"board\.cache_hierarchy\.l1i-cache-0\.overallMissRate::total\s+([\d.e+]+)":  "L1I Miss Rate",

    # L2 Cache
    r"board\.cache_hierarchy\.l2-cache-0\.overallHits::total\s+([\d.e+]+)":       "L2 Hits",
    r"board\.cache_hierarchy\.l2-cache-0\.overallMisses::total\s+([\d.e+]+)":     "L2 Misses",
    r"board\.cache_hierarchy\.l2-cache-0\.overallMissRate::total\s+([\d.e+]+)":   "L2 Miss Rate",

    # Memory bandwidth (approx)
    r"board\.memory\.mem_ctrl\.readBursts\s+([\d.e+]+)":       "DRAM Read Bursts",
    r"board\.memory\.mem_ctrl\.writeBursts\s+([\d.e+]+)":      "DRAM Write Bursts",

    # Pipeline stall analysis
    r"board\.processor\.cores\.core\.numIssuedDist::mean\s+([\d.e+\-]+)":        "Issue Rate (mean)",
    r"board\.processor\.cores\.core\.numIssuedDist::0\s+([\d.e+]+)":             "Stall Cycles (0-issue)",
    r"board\.processor\.cores\.core\.numIssuedDist::1\s+([\d.e+]+)":             "1-issue Cycles",
    r"board\.processor\.cores\.core\.numIssuedDist::2\s+([\d.e+]+)":             "2-issue Cycles",
    r"board\.processor\.cores\.core\.numIssuedDist::3\s+([\d.e+]+)":             "3-issue Cycles",
    r"board\.processor\.cores\.core\.numIssuedDist::4\s+([\d.e+]+)":             "Full-Width Cycles (4-issue)",
    r"board\.processor\.cores\.core\.decode\.blockedCycles\s+([\d.e+]+)":         "Decode Blocked Cycles",
    r"board\.processor\.cores\.core\.fetch\.cycles\s+([\d.e+]+)":                 "Fetch Active Cycles",
    r"board\.processor\.cores\.core\.rename\.ROBFullEvents\s+([\d.e+]+)":         "ROB Full Events",
    r"board\.processor\.cores\.core\.rename\.SQFullEvents\s+([\d.e+]+)":          "SQ Full Events",
    r"board\.processor\.cores\.core\.rename\.LQFullEvents\s+([\d.e+]+)":          "LQ Full Events",
    r"board\.processor\.cores\.core\.rename\.IQFullEvents\s+([\d.e+]+)":          "IQ Full Events (rename)",
    r"board\.processor\.cores\.core\.rename\.blockCycles\s+([\d.e+]+)":           "Rename Block Cycles",
    r"board\.processor\.cores\.core\.commit\.commitSquashedInsts\s+([\d.e+]+)":   "Commit Squashed Insts",

    # Instruction type breakdown — gem5 v24 uses statIssuedInstType_0
    r"board\.processor\.cores\.core\.statIssuedInstType_0::MemRead\s+([\d.e+]+)":      "Insts: Load",
    r"board\.processor\.cores\.core\.statIssuedInstType_0::MemWrite\s+([\d.e+]+)":     "Insts: Store",
    r"board\.processor\.cores\.core\.statIssuedInstType_0::IntAlu\s+([\d.e+]+)":       "Insts: IntALU",
    r"board\.processor\.cores\.core\.statIssuedInstType_0::FloatAdd\s+([\d.e+]+)":     "Insts: FP Add",
    r"board\.processor\.cores\.core\.statIssuedInstType_0::SimdFloatAdd\s+([\d.e+]+)": "Insts: SIMD FP",
    r"board\.processor\.cores\.core\.statIssuedInstType_0::IntMult\s+([\d.e+]+)":      "Insts: IntMult",
}


def parse_stats_file(filepath):
    """Return dict of {label: value} from a stats.txt file."""
    if not os.path.isfile(filepath):
        print(f"[error] File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r") as f:
        text = f.read()

    results = {}
    for pattern, label in STAT_KEYS.items():
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            results[label] = float(m.group(1))
        else:
            results[label] = None

    return results


def print_summary(stats, title="GEM5 Simulation Summary"):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    for label, val in stats.items():
        if val is not None:
            print(f"  {label:<35} {val:>15.4g}")
        else:
            print(f"  {label:<35} {'N/A':>15}")

    # Derived: miss rate if individual counts available
    hits   = stats.get("L1D Hits")
    misses = stats.get("L1D Misses")
    if hits is not None and misses is not None and (hits + misses) > 0:
        mr = misses / (hits + misses) * 100
        print(f"  {'L1D Miss Rate (computed)':<35} {mr:>14.2f}%")

    bp_lookups  = stats.get("Branch Lookups")
    bp_wrong    = stats.get("Branch Mispredicts")
    bp_correct  = stats.get("Branch Predictions")
    if bp_lookups is not None and bp_wrong is not None and bp_lookups > 0:
        bpr = bp_wrong / bp_lookups * 100
        print(f"  {'Branch Mispredict Rate':<35} {bpr:>14.2f}%")

    # --- MPKI Analysis ---
    sim_insts = stats.get("Simulated Instructions")
    l1d_m = stats.get("L1D Misses")
    l1i_m = stats.get("L1I Misses")
    l2_m  = stats.get("L2 Misses")
    dram_r = stats.get("DRAM Read Bursts")
    if sim_insts is not None and sim_insts > 0:
        print(f"\n  MPKI (Misses Per Kilo Instructions):")
        if l1d_m is not None:
            print(f"    L1D MPKI                 {l1d_m / sim_insts * 1000:>8.2f}")
        if l1i_m is not None:
            print(f"    L1I MPKI                 {l1i_m / sim_insts * 1000:>8.2f}")
        if l2_m is not None:
            print(f"    L2  MPKI                 {l2_m  / sim_insts * 1000:>8.2f}")
        if dram_r is not None:
            print(f"    DRAM Read MPKI           {dram_r / sim_insts * 1000:>8.2f}  (each burst = 64 B)")

    # --- Effective DRAM Bandwidth ---
    sim_s = stats.get("Simulated Seconds")
    if dram_r is not None and sim_s is not None and sim_s > 0:
        eff_bw_gbs = dram_r * 64 / sim_s / 1e9
        peak_bw    = 19.2  # DDR4-2400 single-channel theoretical peak (GB/s)
        util_pct   = eff_bw_gbs / peak_bw * 100
        print(f"\n  Effective DRAM Bandwidth:")
        print(f"    Read bandwidth (actual)  {eff_bw_gbs:>8.3f} GB/s")
        print(f"    DDR4-2400 peak           {peak_bw:>8.1f} GB/s")
        print(f"    Bandwidth utilization    {util_pct:>7.1f}%  ← serialized by pointer-chasing")
        print(f"    Serial DRAM ceiling      {64/55e-9/1e9:>8.3f} GB/s  (1 req / 55 ns DDR4 latency)")

    # --- CPI Stack Decomposition ---
    ipc  = stats.get("IPC")
    cpi  = stats.get("CPI")
    stall_0 = stats.get("Stall Cycles (0-issue)")
    full_4  = stats.get("Full-Width Cycles (4-issue)")
    if cpi is not None and sim_insts is not None and stall_0 is not None:
        total_issue_cycles = stats.get("Fetch Active Cycles")
        cpi_ideal   = 0.25  # 4-wide machine
        cpi_mem     = stall_0 / sim_insts
        # Branch penalty: condIncorrect * avg_penalty_cycles / committed_insts
        bp_miscnt   = stats.get("Branch Mispredicts")
        bp_predicted = stats.get("Branch Predictions")
        if bp_miscnt is not None and bp_predicted is not None and bp_predicted > 0:
            misrate = bp_miscnt / bp_predicted
        else:
            misrate = 0.0
        cpi_branch  = (bp_miscnt or 0) * 12 / sim_insts  # 12-cycle avg penalty
        cpi_other   = cpi - cpi_ideal - cpi_mem - cpi_branch
        print(f"\n  CPI Stack Decomposition (total CPI = {cpi:.4f}):")
        print(f"    Ideal (4-wide base)      {cpi_ideal:>8.4f}  ({cpi_ideal/cpi*100:5.1f}%)")
        print(f"    Memory stall (0-issue)   {cpi_mem:>8.4f}  ({cpi_mem/cpi*100:5.1f}%)  ← dominant")
        print(f"    Branch mispredict        {cpi_branch:>8.4f}  ({cpi_branch/cpi*100:5.1f}%)")
        print(f"    Other (partial-issue,    {cpi_other:>8.4f}  ({cpi_other/cpi*100:5.1f}%)")
        print(f"       decode/rename pressure)")

    # --- Amdahl's Law Bound ---
    if stall_0 is not None and sim_insts is not None and cpi is not None:
        total_cyc = sim_insts * cpi
        f_mem = stall_0 / total_cyc  # fraction of total cycles that are memory stalls
        speedup_max  = 1.0 / (1.0 - f_mem)
        ipc_max      = (ipc or 1.0) * speedup_max
        speedup_hbm  = 1.0 / ((1.0 - f_mem) + f_mem / 2.0)
        speedup_pim  = 1.0 / ((1.0 - f_mem) + f_mem / 20.0)
        print(f"\n  Amdahl's Law — Memory Bottleneck Bound:")
        print(f"    Memory stall fraction    {f_mem*100:>7.1f}% of all cycles")
        print(f"    Max speedup (k→∞)        {speedup_max:>7.3f}x  (IPC ceiling = {ipc_max:.3f})")
        print(f"    HBM speedup (k=2)        {speedup_hbm:>7.3f}x  (2× lower DRAM latency)")
        print(f"    PIM speedup (k=20)       {speedup_pim:>7.3f}x  (near-memory compute)")

    # Pipeline stall summary
    total_cycles = stats.get("Fetch Active Cycles")
    stall_c  = stats.get("Stall Cycles (0-issue)")
    full_c   = stats.get("Full-Width Cycles (4-issue)")
    rob_full = stats.get("ROB Full Events")
    sq_full  = stats.get("SQ Full Events")
    lq_full  = stats.get("LQ Full Events")
    iq_full  = stats.get("IQ Full Events (rename)")
    if total_cycles is not None and stall_c is not None:
        print(f"\n  Pipeline Utilization:")
        print(f"    0-issue (stall) cycles   {stall_c/total_cycles*100:>6.1f}%  ({stall_c:.3g})")
        if full_c is not None:
            print(f"    4-issue (full)  cycles   {full_c/total_cycles*100:>6.1f}%  ({full_c:.3g})")
        if rob_full is not None:
            print(f"    ROB full events          {rob_full:.3g}")
        if sq_full is not None:
            print(f"    SQ full events           {sq_full:.3g}  ← dominant stall")
        if lq_full is not None:
            print(f"    LQ full events           {lq_full:.3g}")
        if iq_full is not None:
            print(f"    IQ full events (rename)  {iq_full:.3g}")

    # Instruction type breakdown
    inst_types = {
        "Load":     stats.get("Insts: Load"),
        "Store":    stats.get("Insts: Store"),
        "IntALU":   stats.get("Insts: IntALU"),
        "FP Add":   stats.get("Insts: FP Add"),
        "SIMD FP":  stats.get("Insts: SIMD FP"),
        "IntMult":  stats.get("Insts: IntMult"),
    }
    present = {k: v for k, v in inst_types.items() if v is not None}
    if present:
        total = sum(present.values())
        print(f"\n  Instruction Type Breakdown (total={total:.0f}):")
        for k, v in present.items():
            print(f"    {k:<10} {v:>12.0f}  ({v/total*100:5.1f}%)")
    print(f"{'='*55}\n")


def _categorize(name):
    """Map a result-directory name to (short_label, color)."""
    n = name.replace("hnsw_", "")
    if n.startswith("l2_"):
        return n.replace("l2_", "L2="), "steelblue"
    if n.startswith("rob_"):
        return "ROB=" + n[4:], "tomato"
    if n.startswith("prefetch_l2_"):
        return "HWpf+" + n[12:], "mediumseagreen"
    if n.startswith("mem_"):
        return n[4:].upper(), "darkorchid"
    if n == "swprefetch":
        return "SW-pf", "darkorchid"
    if n == "swprefetch_hbm":
        return "SW-pf+HBM", "darkorchid"
    if n == "reorder":
        return "Reorder", "darkorange"
    if n == "reorder_l2_2MB":
        return "Reorder+2MB", "darkorange"
    if n == "quant":
        return "SQ(int8)", "darkorange"
    if n == "reorder_quant":
        return "Reorder+SQ", "darkorange"
    return n, "gray"


def sweep_plot(results_dir, param_name):
    """
    Sweep mode: results_dir contains sub-folders named by param value.
    Plots IPC and L2 miss rate vs parameter, colored by category.
    """
    entries = []
    for entry in sorted(os.listdir(results_dir)):
        stats_path = os.path.join(results_dir, entry, "stats.txt")
        if os.path.isfile(stats_path):
            stats = parse_stats_file(stats_path)
            ipc = stats.get("IPC")
            mr  = stats.get("L2 Miss Rate")
            if ipc is not None:
                label, color = _categorize(entry)
                entries.append((label, ipc, mr if mr is not None else 0.0, color))

    if not entries:
        print("[error] No stats.txt files found in sweep directory.")
        return

    labels = [e[0] for e in entries]
    ipc_vals = [e[1] for e in entries]
    l2_miss  = [e[2] for e in entries]
    colors   = [e[3] for e in entries]

    print(f"\nSweep results ({param_name}):")
    for lbl, ipc, mr, _ in entries:
        print(f"  {lbl:>14} -> IPC = {ipc:.4f}   L2 Miss Rate = {mr:.6f}")

    if HAS_PLOT:
        import matplotlib.patches as mpatches
        n = len(entries)
        fig, axes = plt.subplots(1, 2, figsize=(max(14, n * 0.8), 5))
        fig.suptitle("HNSW gem5 Sweep — All Configs", fontsize=13, fontweight="bold")

        x = range(n)
        BASELINE_IPC = 1.004

        axes[0].bar(x, ipc_vals, color=colors, edgecolor="black", width=0.7)
        axes[0].axhline(BASELINE_IPC, color="black", linestyle="--", linewidth=1, label="Baseline")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("IPC")
        axes[0].set_title("IPC by Configuration")
        axes[0].set_ylim(0.75, 1.20)

        axes[1].bar(x, l2_miss, color=colors, edgecolor="black", width=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("L2 Miss Rate")
        axes[1].set_title("L2 Miss Rate by Configuration")
        axes[1].set_ylim(0.75, 1.0)

        cat_colors = {
            "L2 sweep": "steelblue", "ROB sweep": "tomato",
            "HW Prefetch": "mediumseagreen", "DRAM/SW": "darkorchid",
            "SW fix": "darkorange", "Baseline": "gray",
        }
        patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()]
        axes[0].legend(handles=patches, fontsize=7, loc="upper left")

        plt.tight_layout()
        out = os.path.join(results_dir, "sweep_all.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n[plot] Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEL6764 GEM5 stats parser")
    parser.add_argument("--stats",      default=None, help="Path to stats.txt")
    parser.add_argument("--sweep-dir",  default=None, help="Directory of sweep runs")
    parser.add_argument("--param",      default="parameter", help="Parameter name for sweep plot")
    args = parser.parse_args()

    if args.stats:
        stats = parse_stats_file(args.stats)
        print_summary(stats)
        if HAS_PLOT:
            inst_types = {
                "Load":    stats.get("Insts: Load"),
                "Store":   stats.get("Insts: Store"),
                "IntALU":  stats.get("Insts: IntALU"),
                "FP Add":  stats.get("Insts: FP Add"),
                "SIMD FP": stats.get("Insts: SIMD FP"),
                "IntMult": stats.get("Insts: IntMult"),
            }
            present = {k: v for k, v in inst_types.items() if v is not None and v > 0}
            if present:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(list(present.values()), labels=list(present.keys()),
                       autopct="%1.1f%%", startangle=90,
                       colors=["steelblue","tomato","mediumseagreen","darkorchid","darkorange","gold"])
                ax.set_title("HNSW Instruction Type Distribution (gem5 baseline)", fontweight="bold")
                out = os.path.join(os.path.dirname(args.stats), "inst_type_pie.png")
                plt.tight_layout()
                plt.savefig(out, dpi=150)
                print(f"[plot] Saved: {out}")

    elif args.sweep_dir:
        sweep_plot(args.sweep_dir, args.param)

    else:
        print("Usage: python parse_stats.py --stats m5out/stats.txt")
        print("       python parse_stats.py --sweep-dir results/ --param l1d_size")
