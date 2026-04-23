#!/usr/bin/env python3
"""
deep_pointer_analysis.py -- deeper HNSW pointer-chasing analysis

Compares a list of gem5 result directories and summarizes:
  - CPI stack (ideal / memory-stall / branch / other)
  - issue phases (0-issue / partial / full-width)
  - stall events (SQ / LQ / IQ / ROB full)
  - instruction mix buckets
  - cache miss rates and miss latencies
  - memory traffic and bandwidth

Example:
  python3 analysis/deep_pointer_analysis.py \
      --results-dir results \
      --configs hnsw_searchroi_baseline hnsw_roi_baseline_ef50 hnsw_roi_prefetch_ef50 \
                hnsw_roi_reorder_ef50 hnsw_roi_quant_ef50 hnsw_roi_rq_ef50
"""

import argparse
import csv
import math
import os
import re
from typing import Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


STAT_PATTERNS = {
    "sim_insts": r"simInsts\s+([\d.e+\-]+)",
    "sim_seconds": r"simSeconds\s+([\d.e+\-]+)",
    "ipc": r"board\.processor\.cores\.core\.ipc\s+([\d.e+\-]+)",
    "cpi": r"board\.processor\.cores\.core\.cpi\s+([\d.e+\-]+)",
    "num_cycles": r"board\.processor\.cores\.core\.numCycles\s+([\d.e+\-]+)",
    "issue0": r"board\.processor\.cores\.core\.numIssuedDist::0\s+([\d.e+\-]+)",
    "issue1": r"board\.processor\.cores\.core\.numIssuedDist::1\s+([\d.e+\-]+)",
    "issue2": r"board\.processor\.cores\.core\.numIssuedDist::2\s+([\d.e+\-]+)",
    "issue3": r"board\.processor\.cores\.core\.numIssuedDist::3\s+([\d.e+\-]+)",
    "issue4": r"board\.processor\.cores\.core\.numIssuedDist::4\s+([\d.e+\-]+)",
    "sq_full": r"board\.processor\.cores\.core\.rename\.SQFullEvents\s+([\d.e+\-]+)",
    "lq_full": r"board\.processor\.cores\.core\.rename\.LQFullEvents\s+([\d.e+\-]+)",
    "iq_full": r"board\.processor\.cores\.core\.rename\.IQFullEvents\s+([\d.e+\-]+)",
    "rob_full": r"board\.processor\.cores\.core\.rename\.ROBFullEvents\s+([\d.e+\-]+)",
    "cond_wrong": r"board\.processor\.cores\.core\.branchPred\.condIncorrect\s+([\d.e+\-]+)",
    "cond_pred": r"board\.processor\.cores\.core\.branchPred\.condPredicted\s+([\d.e+\-]+)",
    "l1d_hits": r"board\.cache_hierarchy\.l1d-cache-0\.overallHits::total\s+([\d.e+\-]+)",
    "l1d_misses": r"board\.cache_hierarchy\.l1d-cache-0\.overallMisses::total\s+([\d.e+\-]+)",
    "l1d_miss_rate": r"board\.cache_hierarchy\.l1d-cache-0\.overallMissRate::total\s+([\d.e+\-]+)",
    "l2_hits": r"board\.cache_hierarchy\.l2-cache-0\.overallHits::total\s+([\d.e+\-]+)",
    "l2_misses": r"board\.cache_hierarchy\.l2-cache-0\.overallMisses::total\s+([\d.e+\-]+)",
    "l2_miss_rate": r"board\.cache_hierarchy\.l2-cache-0\.overallMissRate::total\s+([\d.e+\-]+)",
    "l1d_lat": r"board\.cache_hierarchy\.l1d-cache-0\.demandAvgMissLatency::processor\.cores\.core\.data\s+([\d.e+\-]+)",
    "l2_lat": r"board\.cache_hierarchy\.l2-cache-0\.demandAvgMissLatency::processor\.cores\.core\.data\s+([\d.e+\-]+)",
    "mem_read_bursts": r"board\.memory\.mem_ctrl\.readBursts\s+([\d.e+\-]+)",
    "bytes_read": r"board\.memory\.mem_ctrl\.bytesReadSys\s+([\d.e+\-]+)",
    "bytes_written": r"board\.memory\.mem_ctrl\.bytesWrittenSys\s+([\d.e+\-]+)",
    "int_alu": r"board\.processor\.cores\.core\.statIssuedInstType_0::IntAlu\s+([\d.e+\-]+)",
    "mem_read": r"board\.processor\.cores\.core\.statIssuedInstType_0::MemRead\s+([\d.e+\-]+)",
    "mem_write": r"board\.processor\.cores\.core\.statIssuedInstType_0::MemWrite\s+([\d.e+\-]+)",
    "float_mem_read": r"board\.processor\.cores\.core\.statIssuedInstType_0::FloatMemRead\s+([\d.e+\-]+)",
    "float_mem_write": r"board\.processor\.cores\.core\.statIssuedInstType_0::FloatMemWrite\s+([\d.e+\-]+)",
    "float_add": r"board\.processor\.cores\.core\.statIssuedInstType_0::FloatAdd\s+([\d.e+\-]+)",
    "simd_float_add": r"board\.processor\.cores\.core\.statIssuedInstType_0::SimdFloatAdd\s+([\d.e+\-]+)",
    "simd_float_mult": r"board\.processor\.cores\.core\.statIssuedInstType_0::SimdFloatMult\s+([\d.e+\-]+)",
    "issued_total": r"board\.processor\.cores\.core\.statIssuedInstType_0::total\s+([\d.e+\-]+)",
}


def parse_stats_file(stats_path: str) -> Dict[str, float]:
    with open(stats_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    out: Dict[str, float] = {}
    for key, pattern in STAT_PATTERNS.items():
        match = re.search(pattern, text, re.MULTILINE)
        out[key] = float(match.group(1)) if match else 0.0
    return out


def derive_metrics(raw: Dict[str, float], label: str) -> Dict[str, float]:
    sim_insts = raw["sim_insts"] or 1.0
    sim_seconds = raw["sim_seconds"] or 1.0
    num_cycles = raw["num_cycles"] or 1.0
    cpi = raw["cpi"]

    branch_rate = (raw["cond_wrong"] / raw["cond_pred"] * 100.0) if raw["cond_pred"] else 0.0
    branch_cpi = raw["cond_wrong"] * 12.0 / sim_insts
    mem_cpi = raw["issue0"] / sim_insts
    ideal_cpi = 0.25
    other_cpi = max(cpi - ideal_cpi - mem_cpi - branch_cpi, 0.0)

    partial_issue_pct = (raw["issue1"] + raw["issue2"] + raw["issue3"]) / num_cycles * 100.0
    zero_issue_pct = raw["issue0"] / num_cycles * 100.0
    full_issue_pct = raw["issue4"] / num_cycles * 100.0

    mem_inst = raw["mem_read"] + raw["mem_write"] + raw["float_mem_read"] + raw["float_mem_write"]
    vector_fp = raw["float_add"] + raw["simd_float_add"] + raw["simd_float_mult"]
    other_inst = max(raw["issued_total"] - raw["int_alu"] - mem_inst - vector_fp, 0.0)
    issued_total = raw["issued_total"] or 1.0

    read_bw_gbs = raw["bytes_read"] / sim_seconds / 1e9 if sim_seconds else 0.0
    write_bw_gbs = raw["bytes_written"] / sim_seconds / 1e9 if sim_seconds else 0.0

    return {
        "label": label,
        "ipc": raw["ipc"],
        "cpi": cpi,
        "branch_rate_pct": branch_rate,
        "ideal_cpi": ideal_cpi,
        "mem_cpi": mem_cpi,
        "branch_cpi": branch_cpi,
        "other_cpi": other_cpi,
        "zero_issue_pct": zero_issue_pct,
        "partial_issue_pct": partial_issue_pct,
        "full_issue_pct": full_issue_pct,
        "sq_full": raw["sq_full"],
        "lq_full": raw["lq_full"],
        "iq_full": raw["iq_full"],
        "rob_full": raw["rob_full"],
        "l1d_miss_rate_pct": raw["l1d_miss_rate"] * 100.0,
        "l2_miss_rate_pct": raw["l2_miss_rate"] * 100.0,
        "l1d_lat_kticks": raw["l1d_lat"] / 1000.0,
        "l2_lat_kticks": raw["l2_lat"] / 1000.0,
        "read_bw_gbs": read_bw_gbs,
        "write_bw_gbs": write_bw_gbs,
        "mem_share_pct": mem_inst / issued_total * 100.0,
        "int_share_pct": raw["int_alu"] / issued_total * 100.0,
        "vector_share_pct": vector_fp / issued_total * 100.0,
        "other_share_pct": other_inst / issued_total * 100.0,
    }


def normalize_label(name: str) -> str:
    label = name.replace("hnsw_", "")
    label = label.replace("searchroi_", "")
    label = label.replace("_roi", "")
    return label


def write_csv(metrics: List[Dict[str, float]], out_path: str) -> None:
    if not metrics:
        return
    fieldnames = list(metrics[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def plot_cpi_stack(metrics: List[Dict[str, float]], out_path: str) -> None:
    if not HAS_PLOT or not metrics:
        return
    labels = [m["label"] for m in metrics]
    x = range(len(metrics))
    ideal = [m["ideal_cpi"] for m in metrics]
    mem = [m["mem_cpi"] for m in metrics]
    branch = [m["branch_cpi"] for m in metrics]
    other = [m["other_cpi"] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.2), 5))
    ax.bar(x, ideal, color="#2563EB", label="Ideal (4-wide)")
    ax.bar(x, mem, bottom=ideal, color="#DC2626", label="Memory stall (0-issue)")
    ax.bar(x, branch, bottom=[a + b for a, b in zip(ideal, mem)], color="#F59E0B", label="Branch")
    ax.bar(
        x,
        other,
        bottom=[a + b + c for a, b, c in zip(ideal, mem, branch)],
        color="#9CA3AF",
        label="Other",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("CPI")
    ax.set_title("CPI stack across pointer-chasing experiments")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_issue_phases(metrics: List[Dict[str, float]], out_path: str) -> None:
    if not HAS_PLOT or not metrics:
        return
    labels = [m["label"] for m in metrics]
    x = range(len(metrics))
    zero = [m["zero_issue_pct"] for m in metrics]
    partial = [m["partial_issue_pct"] for m in metrics]
    full = [m["full_issue_pct"] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.2), 5))
    ax.bar(x, zero, color="#DC2626", label="0-issue")
    ax.bar(x, partial, bottom=zero, color="#F59E0B", label="1-3 issue")
    ax.bar(x, full, bottom=[a + b for a, b in zip(zero, partial)], color="#16A34A", label="4-issue")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Cycles (%)")
    ax.set_title("Issue phases: where the pipeline spends time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stall_events(metrics: List[Dict[str, float]], out_path: str) -> None:
    if not HAS_PLOT or not metrics:
        return
    labels = [m["label"] for m in metrics]
    x = list(range(len(metrics)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 1.3), 5))
    ax.bar([i - 1.5 * width for i in x], [m["sq_full"] for m in metrics], width, label="SQ full")
    ax.bar([i - 0.5 * width for i in x], [m["lq_full"] for m in metrics], width, label="LQ full")
    ax.bar([i + 0.5 * width for i in x], [m["iq_full"] for m in metrics], width, label="IQ full")
    ax.bar([i + 1.5 * width for i in x], [m["rob_full"] for m in metrics], width, label="ROB full")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Events")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Stall-related queue pressure events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_instruction_mix(metrics: List[Dict[str, float]], out_path: str) -> None:
    if not HAS_PLOT or not metrics:
        return
    labels = [m["label"] for m in metrics]
    x = range(len(metrics))
    mem = [m["mem_share_pct"] for m in metrics]
    ints = [m["int_share_pct"] for m in metrics]
    vec = [m["vector_share_pct"] for m in metrics]
    other = [m["other_share_pct"] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 1.2), 5))
    ax.bar(x, ints, color="#2563EB", label="Int/ctrl")
    ax.bar(x, mem, bottom=ints, color="#DC2626", label="Memory")
    ax.bar(x, vec, bottom=[a + b for a, b in zip(ints, mem)], color="#16A34A", label="FP/SIMD")
    ax.bar(
        x,
        other,
        bottom=[a + b + c for a, b, c in zip(ints, mem, vec)],
        color="#9CA3AF",
        label="Other",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Issued instructions (%)")
    ax.set_title("Instruction-phase mix across configurations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(metrics: List[Dict[str, float]]) -> None:
    print("\n=== Deep Pointer-Chasing Summary ===")
    header = (
        f"{'Config':<24} {'IPC':>7} {'MemCPI':>8} {'BrCPI':>7} "
        f"{'0iss%':>7} {'Part%':>7} {'Full%':>7} {'L2Miss%':>9} {'L2Lat(k)':>9}"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m['label']:<24} {m['ipc']:>7.3f} {m['mem_cpi']:>8.3f} {m['branch_cpi']:>7.3f} "
            f"{m['zero_issue_pct']:>7.2f} {m['partial_issue_pct']:>7.2f} {m['full_issue_pct']:>7.2f} "
            f"{m['l2_miss_rate_pct']:>9.2f} {m['l2_lat_kticks']:>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deeper HNSW pointer-chasing analysis")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--configs", nargs="+", required=True, help="Result directory names under results-dir")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    metrics: List[Dict[str, float]] = []
    for config in args.configs:
        stats_path = os.path.join(args.results_dir, config, "stats.txt")
        raw = parse_stats_file(stats_path)
        metrics.append(derive_metrics(raw, normalize_label(config)))

    print_summary(metrics)

    csv_path = os.path.join(output_dir, "deep_pointer_summary.csv")
    write_csv(metrics, csv_path)

    plot_cpi_stack(metrics, os.path.join(output_dir, "deep_cpi_stack_compare.png"))
    plot_issue_phases(metrics, os.path.join(output_dir, "deep_issue_phases.png"))
    plot_stall_events(metrics, os.path.join(output_dir, "deep_stall_events.png"))
    plot_instruction_mix(metrics, os.path.join(output_dir, "deep_instruction_mix.png"))

    print(f"\nWrote summary CSV to: {csv_path}")
    if HAS_PLOT:
        print(f"Wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
