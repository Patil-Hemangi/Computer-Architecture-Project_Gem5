#!/usr/bin/env bash
# run_cache_focus_sweep.sh  --  EEL6764 focused cache analysis sweep
#
# Professor direction: narrow focus to cache only.
# Question: what KIND of cache misses does HNSW create?
#   - Compulsory? Capacity? Conflict?
#
# Sweep 1: L2 associativity (fixed 256kB size, LRU policy)
#   → does IPC/miss-rate change as associativity decreases?
#   → if yes: conflict misses exist; if no: capacity/cold misses dominate
#
# Sweep 2: L2 replacement policy (fixed 256kB, 16-way)
#   → does choice of eviction policy matter?
#   → if all same: no exploitable locality; if LRU wins: some reuse exists
#
# Reduced to 1B ticks (from 3B) per professor suggestion — still captures
# the traversal phase and gives stable miss-rate measurements.
#
# Usage: ./run_cache_focus_sweep.sh
# Results: results/hnsw_assoc_N/   results/hnsw_repl_POLICY/

GEM5_BIN="/opt/gem5/build/X86/gem5.opt"
CONFIG="/workspace/configs/run_benchmark.py"
BINARY="/workspace/benchmarks/hnsw_gem5"
RESULTS="/workspace/results"
MAXTICK=3000000000   # 3B ticks — 1B only captures init/build phase, not search
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
# Sweep 1: L2 Associativity  (256kB, LRU — only assoc varies)
# Baseline is 16-way; test lower to detect conflict misses
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Sweep 1: L2 Associativity (256kB, LRU, 1B ticks)"
echo "============================================================"
for ASSOC in 1 2 4 8 16; do
    OUT="$RESULTS/hnsw_assoc_${ASSOC}way"
    mkdir -p "$OUT"
    echo ">>> L2 assoc=${ASSOC}-way"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --l2-assoc "$ASSOC" --l2-replacement lru \
        --rob-size 128 --cpu-width 4 \
        --maxtick $MAXTICK
    echo "    Done -> $OUT"
done

echo ""

# ---------------------------------------------------------------------------
# Sweep 2: L2 Replacement Policy  (256kB, 16-way — only policy varies)
# Baseline is LRU; compare to random/fifo/brrip
# If all policies give similar miss rate: no exploitable locality
# If LRU beats random: some temporal reuse exists
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Sweep 2: L2 Replacement Policy (256kB, 16-way, 1B ticks)"
echo "============================================================"
for POLICY in lru random fifo brrip; do
    OUT="$RESULTS/hnsw_repl_${POLICY}"
    mkdir -p "$OUT"
    echo ">>> L2 replacement=${POLICY}"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --l2-assoc 16 --l2-replacement "$POLICY" \
        --rob-size 128 --cpu-width 4 \
        --maxtick $MAXTICK
    echo "    Done -> $OUT"
done

echo ""
echo "============================================================"
echo " All cache-focus runs complete."
echo " Analyze with: python3 analysis/cache_focus_analysis.py --results-dir results/"
echo "============================================================"
