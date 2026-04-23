#!/usr/bin/env bash
# run_pointer_variant_roi_sweep.sh -- ROI comparison of pointer-chasing mitigations
#
# Compares the baseline HNSW search path against software-side changes that
# attack locality and footprint more directly than cache-policy tuning.
#
# Outputs:
#   /workspace/results/hnsw_roi_baseline_ef<EF>
#   /workspace/results/hnsw_roi_prefetch_ef<EF>
#   /workspace/results/hnsw_roi_reorder_ef<EF>
#   /workspace/results/hnsw_roi_quant_ef<EF>
#   /workspace/results/hnsw_roi_rq_ef<EF>
#   /workspace/results/hnsw_roi_packed_ef<EF>
#   /workspace/results/hnsw_roi_deep_ef<EF>
#   /workspace/results/hnsw_roi_hybrid_ef<EF>
#   /workspace/results/hnsw_roi_hybrid_deep_ef<EF>

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
EF_SEARCH="${EF_SEARCH:-50}"
TOP_K="${TOP_K:-10}"

declare -a BINARIES=(
    "baseline:hnsw_gem5"
    "prefetch:hnsw_gem5_prefetch"
    "reorder:hnsw_gem5_reorder"
    "quant:hnsw_gem5_quant"
    "rq:hnsw_gem5_rq"
    "packed:hnsw_gem5_packed"
    "deep:hnsw_gem5_deep"
    "hybrid:hnsw_gem5_hybrid"
    "hybrid_deep:hnsw_gem5_hybrid_deep"
)

echo "============================================================"
echo " ROI Variant Sweep (pointer-chasing mitigations)"
echo "============================================================"
echo "Base vectors : $NUM_BASE"
echo "Queries      : $NUM_QUERIES"
echo "Mode         : $MODE"
echo "efSearch     : $EF_SEARCH"
echo "Top-K        : $TOP_K"
echo "Maxtick      : $MAXTICK"
echo ""

make -C "$BENCH_DIR" -f Makefile.hnsw all

for entry in "${BINARIES[@]}"; do
    label="${entry%%:*}"
    binary="${entry##*:}"
    outdir="$RESULTS/hnsw_roi_${label}_ef${EF_SEARCH}"
    mkdir -p "$outdir"

    echo ">>> Running $label"
    "$GEM5_BIN" --outdir="$outdir" "$CONFIG" \
        --binary "$BENCH_DIR/$binary" \
        --bin-args "$NUM_BASE $NUM_QUERIES $DATA_DIR $MODE $EF_SEARCH $TOP_K" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --rob-size 128 --cpu-width 4 \
        --maxtick "$MAXTICK" | tee "$outdir/run.log"
    echo "    Done -> $outdir"
    echo ""
done

echo "============================================================"
echo " ROI variant sweep complete."
echo "============================================================"
