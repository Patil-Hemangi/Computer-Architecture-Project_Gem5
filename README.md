**Microarchitectural Analysis of HNSW Vector Search (gem5)**

Course project for EEL6764 (Computer Architecture), University of South Florida, Spring 2026.

**Overview**

This project investigates why HNSW graph search performs far below the theoretical peak of a 4-wide out-of-order processor.

Using gem5 simulation, we show that the performance gap is not due to insufficient compute resources, but due to memory latency caused by HNSW’s access pattern. Even though only ~19% of instructions are memory operations, they dominate execution time because most of them miss in cache and stall the pipeline.

**The key result is simple:
**HNSW is fundamentally memory-latency bound due to pointer-chasing, not compute-limited.

**Key Findings**
Memory stalls dominate performance
~32.8% of total CPI comes from memory stalls
Each L2 miss costs ~165 cycles
Extremely high L2 miss rate
~91.78% L2 miss rate at baseline
Misses are mostly compulsory (cold), not capacity or conflict
Out-of-order execution cannot hide latency
Store Queue Full events dominate (3M+), blocking rename and stalling the pipeline
Back-pressure is a consequence of long-latency loads, not a separate issue
Hardware scaling gives limited benefit
Increasing L2 from 256 kB → 1 MB barely changes miss rate (~0.7 pp)
Larger caches do not help because data is never reused
Root cause: HNSW access pattern
Pointer-chasing → no prefetching possible
Random heap layout → no spatial locality
Visited-set → no temporal reuse
Software/data layout changes work better than hardware
BFS graph reordering: +5.8% IPC
int8 quantization: +10.1% IPC
Combined: +11.5% IPC with no hardware cost
Core Insight

This project is not about “bad hardware utilization.”
It shows that:

The processor is behaving correctly.
The workload is what prevents high performance.

HNSW creates a worst-case scenario for caches:

Every access is new (cold)
Addresses are unpredictable
Dependencies serialize execution

Because of this, cache tuning (size, associativity, policy) cannot fix the problem.

**Methodolog**y
gem5 SE-mode simulation (x86 O3 CPU)
ROI-based measurement of HNSW search phase
Metrics collected:
CPI stack breakdown
Cache miss rates (L1, L2)
Stall events (SQFull, ROBFull, etc.)
Instruction mix
DRAM access behavior

**We systematically evaluated:**

L1 cache size (8 kB → 128 kB)
L2 cache size (256 kB → 2 MB)
Associativity (attempted; limited by config issue)
Replacement policies (attempted; limited by config issue)
Data layout optimizations (reordering, quantization)
Repository Structure
benchmarks/ – HNSW benchmark implementation
configs/ – gem5 simulation scripts
results/ – raw simulation outputs
analysis/ – parsing + plotting scripts
report/ – final report and figures
bigann/ – dataset inputs
Workload Description
HNSW graph search using SIFT-style 128D vectors
Key characteristics:
Pointer-chasing graph traversal
Distance computation (SIMD-friendly)
Irregular memory accesses dominate behavior
Simulator Configuration
gem5 SE mode, x86 ISA
4-wide out-of-order CPU (X86O3CPU)
3 GHz clock
32 kB L1 caches (I/D)
256 kB L2 (baseline)
DDR4 memory (~165 cycle latency)
Running the Project
Build benchmark in benchmarks/

**Run simulation:**

configs/run_benchmark.py

or

run_sweep_hnsw.sh

**Analyze results:**

analysis/

(Exact commands depend on your local gem5 setup.)
**
Report
**
Full analysis and results:
report/EEL6764 Final Project Report.pdf

**Limitations**
Dataset size is small (500 vectors)
Larger datasets would further reinforce memory-bound behavior
Associativity and replacement sweeps were affected by a gem5 config issue
Conclusions are supported by other experiments and workload analysis

**Takeaway**

If you try to “fix” HNSW with bigger caches or better hardware knobs, you’ll get marginal gains.
If you reduce data movement per hop, you get real improvement.

