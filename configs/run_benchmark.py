"""
run_benchmark.py  --  EEL6764 GEM5 SE config for HNSW
gem5 v24.1  branch: class  https://github.com/EEL6764/gem5/tree/class

Sweep dimensions:
  Cache:  --l1d-size  --l1i-size  --l2-size
  CPU:    --rob-size  --cpu-width

Usage:
    gem5.opt configs/run_benchmark.py \\
        --binary benchmarks/hnsw_gem5 \\
        --bin-args "500 20" \\
        --l2-size 512kB --rob-size 128 \\
        --maxtick 3000000000
"""

import argparse
import os

import m5
from m5.objects import X86O3CPU, TournamentBP

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import (
    SingleChannelDDR4_2400,
    SingleChannelHBM,
    DIMM_DDR5_6400,
    DIMM_DDR5_8400,
)
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="EEL6764 HNSW gem5 config")
parser.add_argument("--binary",    required=True, help="Path to static binary")
parser.add_argument("--bin-args",  default="",    help="Args passed to binary e.g. '500 20'")
parser.add_argument("--l1i-size",  default="32kB")
parser.add_argument("--l1d-size",  default="32kB")
parser.add_argument("--l2-size",   default="256kB")
parser.add_argument("--rob-size",  type=int, default=128,  help="ROB entries (64/128/256)")
parser.add_argument("--cpu-width", type=int, default=4,    help="Pipeline width (1/2/4/8)")
parser.add_argument("--num-int-regs", type=int, default=256)
parser.add_argument("--num-fp-regs",  type=int, default=256)
parser.add_argument("--maxtick",   type=int, default=None, help="Max simulation ticks")
parser.add_argument("--prefetcher", action="store_true", default=False,
                    help="Attach StridePrefetcher to L2 cache (HW fix for memory bottleneck)")
parser.add_argument("--l2-mshrs", type=int, default=20,
                    help="L2 cache MSHR count (default=20). More MSHRs = more outstanding misses in parallel.")
parser.add_argument("--imp", action="store_true", default=False,
                    help="Attach IndirectMemoryPrefetcher to L2 — targets A[B[i]] pointer-chasing pattern.")
parser.add_argument("--lq-size", type=int, default=32,
                    help="Load Queue entries (default=32). Larger LQ allows more in-flight loads.")
parser.add_argument("--memory", default="ddr4",
                    choices=["ddr4", "ddr5", "ddr5_fast", "hbm"],
                    help="Memory type: ddr4=DDR4-2400 (baseline), ddr5=DDR5-6400, ddr5_fast=DDR5-8400, hbm=HBM")
parser.add_argument("--num-cores", type=int, default=1,
                    help="Number of CPU cores. Use with MULTITHREAD binary for query-level TLP sweep.")
parser.add_argument("--l2-assoc", type=int, default=16,
                    help="L2 cache associativity (default=16). Sweep: 1,2,4,8,16 to distinguish conflict vs capacity misses.")
parser.add_argument("--l2-replacement", default="lru",
                    choices=["lru", "random", "fifo", "brrip"],
                    help="L2 replacement policy (default=lru). Options: lru, random, fifo, brrip.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Custom O3 core with configurable ROB + width
# ---------------------------------------------------------------------------
class HNSWCore(BaseCPUCore):
    def __init__(self, width, rob_size, num_int_regs, num_fp_regs, lq_size=32):
        core = X86O3CPU()
        core.fetchWidth    = width
        core.decodeWidth   = width
        core.renameWidth   = width
        core.dispatchWidth = width
        core.issueWidth    = width
        core.wbWidth       = width
        core.commitWidth   = width
        core.numROBEntries    = rob_size
        core.numPhysIntRegs   = num_int_regs
        core.numPhysFloatRegs = num_fp_regs
        core.LQEntries        = lq_size
        core.branchPred = TournamentBP()
        super().__init__(core, ISA.X86)

class HNSWProcessor(BaseCPUProcessor):
    def __init__(self, width, rob_size, num_int_regs, num_fp_regs, lq_size=32, num_cores=1):
        super().__init__(
            cores=[HNSWCore(width, rob_size, num_int_regs, num_fp_regs, lq_size)
                   for _ in range(num_cores)]
        )

# ---------------------------------------------------------------------------
# Configurable cache hierarchy — patches L2 inside incorporate_cache_hierarchy
# so attributes exist and m5.instantiate() hasn't been called yet.
# This is the correct gem5 hook point; modifying _l2_caches after Simulator()
# is too late (m5.instantiate has already frozen the object graph).
# ---------------------------------------------------------------------------
_RP_MAP = {
    "lru":    "LRURP",
    "random": "RandomRP",
    "fifo":   "FIFORP",
    "brrip":  "BRRIPRP",
}

class ConfigurableL2CacheHierarchy(PrivateL1PrivateL2CacheHierarchy):
    def __init__(self, l1d_size, l1i_size, l2_size,
                 l2_mshrs=20, stride_prefetcher=False, imp=False,
                 l2_assoc=16, l2_replacement="lru"):
        super().__init__(l1d_size=l1d_size, l1i_size=l1i_size, l2_size=l2_size)
        self._cfg_mshrs       = l2_mshrs
        self._cfg_stride      = stride_prefetcher
        self._cfg_imp         = imp
        self._cfg_assoc       = l2_assoc
        self._cfg_replacement = l2_replacement

    def incorporate_cache_hierarchy(self, board):
        super().incorporate_cache_hierarchy(board)
        # _l2_caches is guaranteed to exist here (parent just created it)
        l2s = []
        if hasattr(self, "_l2_caches"):
            l2s = self._l2_caches if isinstance(self._l2_caches, list) else [self._l2_caches]
        elif hasattr(self, "_l2_cache"):
            l2s = [self._l2_cache]

        if not l2s:
            print("[config] WARNING: could not locate L2 cache objects inside incorporate_cache_hierarchy")
            return

        for l2 in l2s:
            if self._cfg_mshrs != 20:
                l2.mshrs = self._cfg_mshrs
                print(f"[config] L2 MSHRs set to {self._cfg_mshrs}")
            if self._cfg_assoc != 16:
                l2.assoc = self._cfg_assoc
                print(f"[config] L2 associativity set to {self._cfg_assoc}-way")
            if self._cfg_replacement != "lru":
                rp_cls_name = _RP_MAP[self._cfg_replacement]
                import importlib
                m5_objects = importlib.import_module("m5.objects")
                rp_cls = getattr(m5_objects, rp_cls_name)
                l2.replacement_policy = rp_cls()
                print(f"[config] L2 replacement policy set to {rp_cls_name}")
            if self._cfg_imp:
                from m5.objects import IndirectMemoryPrefetcher
                l2.prefetcher = IndirectMemoryPrefetcher()
                print("[config] IndirectMemoryPrefetcher attached to L2")
            elif self._cfg_stride:
                from m5.objects import StridePrefetcher
                l2.prefetcher = StridePrefetcher()
                print("[config] StridePrefetcher attached to L2")


# ---------------------------------------------------------------------------
# Build system
# ---------------------------------------------------------------------------
cache_hierarchy = ConfigurableL2CacheHierarchy(
    l1d_size=args.l1d_size,
    l1i_size=args.l1i_size,
    l2_size=args.l2_size,
    l2_mshrs=args.l2_mshrs,
    stride_prefetcher=args.prefetcher,
    imp=args.imp,
    l2_assoc=args.l2_assoc,
    l2_replacement=args.l2_replacement,
)

_mem_map = {
    "ddr4":     (SingleChannelDDR4_2400, "4GB"),
    "ddr5":     (DIMM_DDR5_6400,        "4GB"),
    "ddr5_fast":(DIMM_DDR5_8400,        "4GB"),
    "hbm":      (SingleChannelHBM,      "256MiB"),  # HBM default max is 256MiB
}
_mem_cls, _mem_size = _mem_map[args.memory]
memory = _mem_cls(size=_mem_size)

processor = HNSWProcessor(
    width=args.cpu_width,
    rob_size=args.rob_size,
    num_int_regs=args.num_int_regs,
    num_fp_regs=args.num_fp_regs,
    lq_size=args.lq_size,
    num_cores=args.num_cores,
)

board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# L2 configuration (MSHRs, prefetcher, IMP) is applied inside
# ConfigurableL2CacheHierarchy.incorporate_cache_hierarchy() above.

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
bin_args = args.bin_args.split() if args.bin_args else []
board.set_se_binary_workload(
    binary=BinaryResource(local_path=args.binary),
    arguments=bin_args,
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
simulator = Simulator(board=board)

outdir = m5.options.outdir
simulator.add_text_stats_output(os.path.join(outdir, "stats.txt"))
simulator.add_json_stats_output(os.path.join(outdir, "stats.json"))

if args.maxtick is not None:
    simulator.run(max_ticks=args.maxtick)
else:
    simulator.run()

print("\n[gem5] Simulation complete.")
print("[gem5] Stats: " + os.path.join(outdir, "stats.txt"))
