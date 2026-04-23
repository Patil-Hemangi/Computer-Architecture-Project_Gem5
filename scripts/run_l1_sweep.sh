#!/usr/bin/env bash
# run_l1_sweep.sh  --  EEL6764 L1 Data Cache Size Sweep
#
# Professor requirement: "CPI vs L1 cache size"
# This is the only experiment from the formal instructions not previously run.
#
# Sweep: L1D size = 8kB, 16kB, 32kB (baseline), 64kB, 128kB
#   - L1I fixed at 32kB (instruction footprint is small, not the bottleneck)
#   - L2 fixed at 256kB baseline
#   - All other params at baseline
#
# Hypothesis: HNSW has pointer-chasing with low MLP — L1D enlargement may
#   reduce compulsory misses for the first traversal pass, but since reuse
#   is zero (confirmed by replacement policy sweep), larger L1D may provide
#   diminishing returns beyond fitting the working-set hot path.
#
# Usage: ./run_l1_sweep.sh
# Results: results/hnsw_l1d_<SIZE>/
#
# NOTE: Run this inside the gem5 container where gem5.opt is at /opt/gem5/

GEM5_BIN="/opt/gem5/build/X86/gem5.opt"
CONFIG="/workspace/configs/run_benchmark.py"
BINARY="/workspace/benchmarks/hnsw_gem5"
RESULTS="/workspace/results"
MAXTICK=3000000000   # 3B ticks — same as cache focus sweep for consistency
BIN_ARGS="500 20 /workspace/bigann/sift/ search_roi_nogt"

set -e

# ---------------------------------------------------------------------------
# Build binary if not present
# ---------------------------------------------------------------------------
if [ ! -f "$BINARY" ]; then
    echo ">>> Binary not found at $BINARY — building now..."
    make -C /workspace/benchmarks -f Makefile.hnsw hnsw_gem5
    echo "    Build complete."
fi

# ---------------------------------------------------------------------------
# L1D Size Sweep
# L1I stays at 32kB (cold instruction misses are negligible vs data misses)
# L2 stays at 256kB baseline (isolates L1D effect)
# ---------------------------------------------------------------------------
echo "============================================================"
echo " L1D Cache Size Sweep (L1I=32kB, L2=256kB, LRU, 3B ticks)"
echo "============================================================"

for L1D in 8kB 16kB 32kB 64kB 128kB; do
    OUT="$RESULTS/hnsw_l1d_${L1D}"
    mkdir -p "$OUT"
    echo ">>> L1D size = $L1D"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size "$L1D" --l1i-size 32kB --l2-size 256kB \
        --l2-assoc 16 --l2-replacement lru \
        --rob-size 128 --cpu-width 4 \
        --maxtick $MAXTICK
    echo "    Done -> $OUT"
    echo ""
done

echo "============================================================"
echo " L1D sweep complete. Analyze with:"
echo "   python3 analysis/enhanced_cache_analysis.py --results-dir results/ --l1-sweep"
echo "============================================================"
