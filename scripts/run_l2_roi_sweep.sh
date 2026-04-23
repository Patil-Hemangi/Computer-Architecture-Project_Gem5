#!/usr/bin/env bash
# run_l2_roi_sweep.sh -- ROI-based L2 size sweep for HNSW search-phase analysis
#
# Purpose:
#   Run only the cache-capacity experiment using the ROI-enabled benchmark mode:
#     search_roi_nogt
#
# Why this exists:
#   Earlier L2-size results were gathered before the search ROI measurement fix.
#   This script reruns the L2 sweep using stats reset/dump around the search phase
#   so the cache conclusions are based on the correct region of interest.
#
# Output directories:
#   /workspace/results/hnsw_l2_256kB_roi
#   /workspace/results/hnsw_l2_512kB_roi
#   /workspace/results/hnsw_l2_1MB_roi
#   /workspace/results/hnsw_l2_2MB_roi

set -euo pipefail

GEM5_BIN="/opt/gem5/build/X86/gem5.opt"
CONFIG="/workspace/configs/run_benchmark.py"
BINARY="/workspace/benchmarks/hnsw_gem5"
RESULTS="/workspace/results"
MAXTICK=3000000000
BIN_ARGS="500 20 /workspace/bigann/sift/ search_roi_nogt"

echo "============================================================"
echo " ROI L2 Size Sweep (search phase only)"
echo "============================================================"
echo "Binary:  $BINARY"
echo "Args:    $BIN_ARGS"
echo "Maxtick: $MAXTICK"
echo ""

if [ ! -f "$BINARY" ]; then
    echo "ERROR: benchmark binary not found at $BINARY"
    exit 1
fi

for L2 in 256kB 512kB 1MB 2MB; do
    OUT="$RESULTS/hnsw_l2_${L2}_roi"
    mkdir -p "$OUT"

    echo ">>> Running L2=$L2"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size "$L2" \
        --rob-size 128 --cpu-width 4 \
        --maxtick $MAXTICK | tee "$OUT/run.log"

    echo "    Done -> $OUT"
    echo "    ROI markers:"
    grep "\[roi\]" "$OUT/run.log" || true
    echo ""
done

echo "============================================================"
echo " ROI L2 sweep complete."
echo " Next: extract key stats from hnsw_l2_*_roi directories."
echo "============================================================"
