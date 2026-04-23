#!/usr/bin/env bash
# run_sweep_hnsw.sh  --  EEL6764 HNSW sweep: L2 size + ROB size
#
# Usage: ./run_sweep_hnsw.sh
# Results go to: /workspace/results/

GEM5_BIN="/opt/gem5/build/X86/gem5.opt"
CONFIG="/workspace/configs/run_benchmark.py"
BINARY="/workspace/benchmarks/hnsw_gem5"
RESULTS="/workspace/results"
MAXTICK=3000000000   # 3B ticks per run (~10 min each)
BIN_ARGS="500 20 /workspace/bigann/sift/ search_roi_nogt"

# ---------------------------------------------------------------------------
# Phase 1: L2 cache sweep  (ROB=128, width=4 fixed)
# ---------------------------------------------------------------------------
echo "=== Phase 1: L2 Cache Sweep ==="
for L2 in 256kB 512kB 1MB 2MB; do
    OUT="$RESULTS/hnsw_l2_${L2}"
    mkdir -p "$OUT"
    echo ">>> L2=$L2"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size "$L2" \
        --rob-size 128 --cpu-width 4 \
        --maxtick $MAXTICK
done

# ---------------------------------------------------------------------------
# Phase 2: ROB size sweep  (L2=512kB fixed — cost-effective point from Phase 1; best IPC was L2=2MB)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 2: ROB Size Sweep ==="
for ROB in 32 64 128 256; do
    OUT="$RESULTS/hnsw_rob_${ROB}"
    mkdir -p "$OUT"
    echo ">>> ROB=$ROB"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 512kB \
        --rob-size "$ROB" --cpu-width 4 \
        --maxtick $MAXTICK
done

# ---------------------------------------------------------------------------
# Phase 3: HW fix — stride prefetcher (bottleneck: DRAM latency from
# pointer-chasing + distance-compute stride pattern)
# Run 3 variants: prefetcher only, prefetcher + L2=512kB, prefetcher + L2=2MB
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 3: Stride Prefetcher (HW Bottleneck Fix) ==="
for L2 in 256kB 512kB 2MB; do
    OUT="$RESULTS/hnsw_prefetch_l2_${L2}"
    mkdir -p "$OUT"
    echo ">>> Prefetcher + L2=$L2"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size "$L2" \
        --rob-size 128 --cpu-width 4 \
        --prefetcher \
        --maxtick $MAXTICK
done

# ---------------------------------------------------------------------------
# Phase 4a: Faster DRAM — DDR4-3200 and LPDDR5-6400 (pure HW latency fix)
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 4a: Faster DRAM ==="
for MEM in ddr5 hbm; do
    OUT="$RESULTS/hnsw_mem_${MEM}"
    mkdir -p "$OUT"
    echo ">>> DRAM=$MEM"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --rob-size 128 --cpu-width 4 \
        --memory "$MEM" \
        --maxtick $MAXTICK
done

# ---------------------------------------------------------------------------
# Phase 4b: Software prefetch binary + baseline DRAM
# Recompile with prefetch hints first:
#   cd /workspace/benchmarks
#   g++ -O2 -static -march=x86-64 -std=c++17 -o hnsw_gem5_prefetch hnsw_gem5_benchmark.cpp
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 4b: Software Prefetch Binary ==="
OUT="$RESULTS/hnsw_swprefetch"
mkdir -p "$OUT"
echo ">>> SW prefetch binary + DDR4-2400"
$GEM5_BIN --outdir="$OUT" "$CONFIG" \
    --binary "${BINARY}_prefetch" --bin-args "$BIN_ARGS" \
    --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
    --rob-size 128 --cpu-width 4 \
    --maxtick $MAXTICK

# Best combo: SW prefetch + HBM
OUT="$RESULTS/hnsw_swprefetch_hbm"
mkdir -p "$OUT"
echo ">>> SW prefetch binary + HBM"
$GEM5_BIN --outdir="$OUT" "$CONFIG" \
    --binary "${BINARY}_prefetch" --bin-args "$BIN_ARGS" \
    --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
    --rob-size 128 --cpu-width 4 \
    --memory hbm \
    --maxtick $MAXTICK

# ---------------------------------------------------------------------------
# Phase 5: Graph BFS reordering (SW fix — improves spatial locality)
# Nodes are renumbered so graph-neighbors are contiguous in memory.
# Build: g++ -O2 -static -march=x86-64 -std=c++17 -DGRAPH_REORDER
#        -o hnsw_gem5_reorder hnsw_gem5_benchmark.cpp
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 5: Graph BFS Reordering ==="
OUT="$RESULTS/hnsw_reorder"
mkdir -p "$OUT"
echo ">>> Graph reorder + DDR4-2400 baseline"
$GEM5_BIN --outdir="$OUT" "$CONFIG" \
    --binary "${BINARY}_reorder" --bin-args "$BIN_ARGS" \
    --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
    --rob-size 128 --cpu-width 4 \
    --maxtick $MAXTICK

# Reorder + larger L2 to see combined benefit
OUT="$RESULTS/hnsw_reorder_l2_2MB"
mkdir -p "$OUT"
echo ">>> Graph reorder + L2=2MB"
$GEM5_BIN --outdir="$OUT" "$CONFIG" \
    --binary "${BINARY}_reorder" --bin-args "$BIN_ARGS" \
    --l1d-size 32kB --l1i-size 32kB --l2-size 2MB \
    --rob-size 128 --cpu-width 4 \
    --maxtick $MAXTICK

# ---------------------------------------------------------------------------
# Phase 6: Scalar quantization float32 → int8 (SW fix — 4× vec size reduction)
# Vectors shrink from 512 B to 128 B — 4× more fit in L2, reducing DRAM traffic.
# Build: g++ -O2 -static -march=x86-64 -std=c++17 -DSCALAR_QUANT
#        -o hnsw_gem5_quant hnsw_gem5_benchmark.cpp
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 6: Scalar Quantization (float32 → int8) ==="
OUT="$RESULTS/hnsw_quant"
mkdir -p "$OUT"
echo ">>> Scalar quant + DDR4-2400 baseline"
$GEM5_BIN --outdir="$OUT" "$CONFIG" \
    --binary "${BINARY}_quant" --bin-args "$BIN_ARGS" \
    --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
    --rob-size 128 --cpu-width 4 \
    --maxtick $MAXTICK

# ---------------------------------------------------------------------------
# Phase 7: Reorder + quantization combined (best SW fix)
# Build: g++ -O2 -static -march=x86-64 -std=c++17 -DGRAPH_REORDER -DSCALAR_QUANT
#        -o hnsw_gem5_rq hnsw_gem5_benchmark.cpp
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 7: Reorder + Quantization Combined ==="
OUT="$RESULTS/hnsw_reorder_quant"
mkdir -p "$OUT"
echo ">>> Reorder + scalar quant + DDR4-2400"
$GEM5_BIN --outdir="$OUT" "$CONFIG" \
    --binary "${BINARY}_rq" --bin-args "$BIN_ARGS" \
    --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
    --rob-size 128 --cpu-width 4 \
    --maxtick $MAXTICK

# ---------------------------------------------------------------------------
# Phase 8: L2 MSHR count sweep  (baseline binary, L2=256kB, ROB=128)
# Tests whether more outstanding-miss slots improve MLP for HNSW.
# Expected result: no effect — inter-hop serial dependency limits MLP to ~1.
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 8: L2 MSHR Count Sweep ==="
for MSHRS in 20 40 64 128; do
    OUT="$RESULTS/hnsw_mshr_${MSHRS}"
    mkdir -p "$OUT"
    echo ">>> MSHRs=$MSHRS"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "$BINARY" --bin-args "$BIN_ARGS" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --rob-size 128 --cpu-width 4 \
        --l2-mshrs "$MSHRS" \
        --maxtick $MAXTICK
done

# ---------------------------------------------------------------------------
# Phase 9: Query-level multithreading — TLP sweep (SW fix)
#
# HNSW queries are embarrassingly parallel across queries; each search thread
# uses thread_local visited arrays so there is no shared mutable state.
# ROB/MSHR sweeps showed ILP and MLP are both saturated (serial RAW chain).
# TLP is the only remaining source of parallelism — N threads → ~N× throughput.
#
# Binary: compile with -DMULTITHREAD -pthread
#   g++ -O2 -static -march=x86-64 -std=c++17 -DMULTITHREAD -pthread \
#       -o hnsw_gem5_mt hnsw_gem5_benchmark.cpp
#
# gem5 uses per-core private L1+L2 (PrivateL1PrivateL2CacheHierarchy).
# argv[4] = numThreads must match --num-cores.
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 9: Query-Level Multithreading (TLP) ==="
for CORES in 2 4; do
    OUT="$RESULTS/hnsw_mt_${CORES}c"
    mkdir -p "$OUT"
    echo ">>> MT: $CORES cores / $CORES threads"
    $GEM5_BIN --outdir="$OUT" "$CONFIG" \
        --binary "${BINARY}_mt" --bin-args "500 20 /workspace/bigann/sift/ $CORES" \
        --l1d-size 32kB --l1i-size 32kB --l2-size 256kB \
        --rob-size 128 --cpu-width 4 \
        --num-cores "$CORES" \
        --maxtick $MAXTICK
done

echo ""
echo "=== Sweep Complete ==="
echo "Analyze:"
echo "  python3 /workspace/analysis/parse_stats.py --sweep-dir $RESULTS --param sweep"
