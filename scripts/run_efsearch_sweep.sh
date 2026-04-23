#!/usr/bin/env bash
# run_efsearch_sweep.sh -- ROI sweep of HNSW search depth
#
# Varies efSearch to stress pointer-chasing depth and expose how search breadth
# changes stalls, miss latency, and issue behavior.
#
# Outputs:
#   /workspace/results/hnsw_ef_<EF>

set -euo pipefail

GEM5_BIN="/opt/gem5/build/X86/gem5.opt"
CONFIG="/workspace/configs/run_benchmark.py"
BENCH_DIR="/workspace/benchmarks"
RESULTS="/workspace/results"
MAXTICK="${MAXTICK:-3000000000}"
NUM_BASE="${NUM_BASE:-500}"
NUM_QUERIES="${NUM_QUERIES:-20}"
DATA_DIR="${DATA_DIR:-/workspace/bigann/sift/}"
MODE="${MODE:-search_roi_nogt}"
TOP_K="${TOP_K:-10}"
EFS="${EFS:-10 25 50 100}"

echo "============================================================"
echo " ROI efSearch Sweep (pointer-chasing depth)"
echo "============================================================"
echo "Base vectors : $NUM_BASE"
echo "Queries      : $NUM_QUERIES"
echo "Mode         : $MODE"
echo "efSearch set : $EFS"
echo "Top-K        : $TOP_K"
echo "Maxtick      : $MAXTICK"
echo ""

make -C "$BENCH_DIR" -f Makefile.hnsw hnsw_gem5

for ef in $EFS; do
    outdir="$RESULTS/hnsw_ef_${ef}"
    mkdir -p "$outdir"
    echo ">>> Running efSearch=$ef"
    "$GEM5_BIN" --outdir="$outdir" "$CONFIG" \
        --binary "$BENCH_DIR/hnsw_gem5" \
        --bin-args "$NUM_BASE $NUM_QUERIES $DATA_DIR $MODE $ef $TOP_K" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --rob-size 128 --cpu-width 4 \
        --maxtick "$MAXTICK" | tee "$outdir/run.log"
    echo "    Done -> $outdir"
    echo ""
done

echo "============================================================"
echo " ROI efSearch sweep complete."
echo "============================================================"
