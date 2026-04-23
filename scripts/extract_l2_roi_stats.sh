#!/usr/bin/env bash
# extract_l2_roi_stats.sh -- print key ROI L2 sweep stats in a compact table

set -euo pipefail

RESULTS="/workspace/results"

printf "%-14s %-10s %-10s %-12s %-12s %-10s\n" \
  "Config" "IPC" "CPI" "L1D_miss" "L2_miss" "ZeroIssue"
printf "%-14s %-10s %-10s %-12s %-12s %-10s\n" \
  "--------------" "----------" "----------" "------------" "------------" "----------"

for L2 in 256kB 512kB 1MB 2MB; do
    STATS="$RESULTS/hnsw_l2_${L2}_roi/stats.txt"
    if [ ! -f "$STATS" ]; then
        printf "%-14s %-10s %-10s %-12s %-12s %-10s\n" "L2=$L2" "MISSING" "-" "-" "-" "-"
        continue
    fi

    IPC=$(grep -m1 "board.processor.cores.core.ipc" "$STATS" | awk '{print $2}')
    CPI=$(grep -m1 "board.processor.cores.core.cpi" "$STATS" | awk '{print $2}')
    L1D=$(grep -m1 "board.cache_hierarchy.l1d-cache-0.overallMissRate::total" "$STATS" | awk '{print $2}')
    L2M=$(grep -m1 "board.cache_hierarchy.l2-cache-0.overallMissRate::total" "$STATS" | awk '{print $2}')
    ZERO=$(grep -m1 "board.processor.cores.core.numIssuedDist::0" "$STATS" | awk '{print $3}')

    printf "%-14s %-10s %-10s %-12s %-12s %-10s\n" \
      "L2=$L2" "$IPC" "$CPI" "$L1D" "$L2M" "$ZERO"
done

