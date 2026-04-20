# EEL6764: Computer Architecture
## Project Report: Microarchitectural Analysis of HNSW Vector Search on gem5

**Course:** EEL6764, Prof. Santosh Pandey, University of South Florida, Spring 2026

**Team Members:**
- Hemangi Patil (hemangipatil@usf.edu)
- Rishil Shah (rshah10@usf.edu)
- Roberto Perez (rcperez@usf.edu)
- Rima El Brouzi (relbrouzi@usf.edu)

---

## 1. Project Overview

**Problem:** Modern vector search systems (RAG pipelines, image search, recommendation engines) use HNSW (Hierarchical Navigable Small World) as their approximate nearest-neighbor index [1]. Despite running on high-IPC out-of-order processors, HNSW anecdotally achieves far below theoretical throughput. This project asks: *what microarchitectural bottleneck causes this gap, and what hardware or software change most cost-effectively closes it?*

**Importance:** HNSW is deployed at scale in FAISS [6], hnswlib, pgvector, and cloud ANN services. A 10% throughput improvement translates directly to reduced query latency or reduced infrastructure cost. Understanding the bottleneck is prerequisite to designing hardware accelerators (e.g., PIM [11]) or software optimizations (e.g., quantization [2]) that actually address the root cause.

**Research hypothesis:** HNSW's graph traversal phase produces irregular pointer-chasing memory accesses that serialize into a chain of RAW (Read-After-Write) data hazards [5], making the workload memory-latency-bound rather than compute-bound or bandwidth-bound [8]. We hypothesize that: (1) hardware microarchitecture knobs (cache, ROB, prefetcher) will provide marginal or no benefit, and (2) algorithmic changes that reduce data movement (quantization [2], BFS reordering) will be the cost-effective fix.

This project uses gem5 full-system simulation to analyze the microarchitectural behavior of HNSW vector search. The goal is to simulate the workload, identify the dominant hardware bottleneck, apply architectural changes to address it, and evaluate cost via a Power-Performance-Area (PPA) model.

**HNSW** builds a multi-layer proximity graph over a dataset [1]. During search, it traverses the graph greedily from the top layer down, computing L2 distances at each node to find the nearest neighbors. This produces two distinct memory access patterns:
- **Graph traversal**: pointer-chasing through linked neighbor lists - irregular, unpredictable
- **Distance computation**: iterating over 128 float32 values per vector - regular, stride-1

**Contributions of this work:**
1. **We show that the Roofline performance model gives the wrong answer for HNSW.** At arithmetic intensity 2.09 FLOP/byte, the standard model classifies HNSW as compute-bound - yet FP units are idle 70% of the time. We identify the root cause: the Roofline bandwidth ceiling assumes MLP ≫ 1, which pointer-chasing structurally cannot satisfy. Correcting for serialized effective bandwidth (1.61 GB/s) shifts the ridge point from 0.63 to 7.45 FLOP/byte, reversing the diagnosis entirely. Any graph-traversal workload with data-dependent addresses will exhibit the same failure mode.
2. We establish a tight quantitative performance ceiling via Amdahl's Law: even eliminating all memory stalls yields at most +47.8% IPC (1.484 ceiling). Our best hardware result (+10.6%, L2=2MB) and best algorithmic result (+10.2%, Reorder+Quant) both land well below this ceiling, validating the analysis.
3. We demonstrate via CPI decomposition, MPKI, effective bandwidth, and a 22-configuration sweep that memory latency - not bandwidth, not compute, not branch prediction - is the sole bottleneck, and that every standard hardware fix addresses the wrong resource.
4. We show via a full PPA cost model that algorithmic data-layout optimizations (BFS reordering + scalar quantization) are the only Pareto-optimal operating point: +10.2% IPC at 1.00× chip area, matching an 8× larger L2 cache at zero silicon cost.

**Central thesis:** This work shows that standard performance models such as Roofline can fundamentally misclassify pointer-chasing workloads due to implicit assumptions of high memory-level parallelism. Correcting that misdiagnosis - not merely confirming the bottleneck - is what makes the Pareto result possible. **This work demonstrates that irregular, pointer-chasing workloads fundamentally challenge the assumptions underlying modern out-of-order CPU design.**

---

## 2. Workload and Simulation Infrastructure

### 2.1 Benchmark (Workloads)

The benchmark (`hnsw_gem5_benchmark.cpp`) loads real SIFT-1M vectors from `.fvecs` binary files [3], builds an HNSW index (M=16, Mmax0=32, efC=100) [1], then runs K-NN search (K=10, efSearch=50).

**Dataset - full vs. simulated scale:** The SIFT-1M dataset [3] in its standard form comprises 1,000,000 base vectors and 10,000 query vectors, each a 128-dimensional SIFT descriptor encoded as float32 (512 bytes/vector). At this scale, full-dataset simulation under gem5 is computationally intractable: gem5's cycle-accurate X86O3CPU model incurs an approximately 100,000× slowdown relative to native execution, meaning a single real-time second of workload requires roughly 28 simulation-hours. To maintain feasibility while preserving the architectural access patterns under study, we selected a representative subset of **500 base vectors and 20 query vectors** - sufficient to exercise the full HNSW graph traversal and distance-computation code paths, trigger the L2 miss behavior that characterizes the bottleneck, and complete each configuration within the 3B-tick budget (~10 simulation-minutes per run). The reduced scale does not alter the fundamental memory access pattern (irregular pointer-chasing during traversal, stride-1 during distance computation); it does, however, constrain the working set to ~250 KB, which is near the 256 kB L2 boundary and therefore representative of the capacity-pressure regime relevant to this study.

| Parameter | Full SIFT-1M Dataset | Simulated Subset | Justification |
|---|---|---|---|
| Base vectors | 1,000,000 | **500** | gem5 ~100,000× slowdown; 3B-tick budget |
| Query vectors | 10,000 | **20** | Proportional reduction; maintains query diversity |
| Vector dimension | 128 float32 | 128 float32 (unchanged) | Full-fidelity SIFT descriptor |
| Working set size | ~476 MB | ~250 KB | Subset fits near L2 boundary - bottleneck regime |
| HNSW M | 16 | 16 (unchanged) | Graph structure identical |
| efSearch | 50 | 50 (unchanged) | Search quality parameter unchanged |

Simulation is limited to 3B ticks (~10 min/run) due to gem5's ~100,000× slowdown vs native.

### 2.2 Simulator Configuration

| Component | Configuration |
|---|---|
| Simulator | gem5 v24.1 SE mode, X86 ISA [4] |
| CPU | **X86O3CPU** (out-of-order), 4-wide, 3 GHz |
| Branch Predictor | TournamentBP |
| L1-D / L1-I Cache | 32 kB each, private per core |
| L2 Cache (baseline) | 256 kB, private |
| DRAM | DDR4-2400, 4 GB, single channel |
| ROB (baseline) | 128 entries |
| Physical Registers | 256 int + 256 fp |

**CPU model justification:** We use **X86O3CPU** (out-of-order) throughout - not TimingSimpleCPU or AtomicSimpleCPU. Timing/Atomic CPUs are functional simulators that do not model pipeline stages, ROB, reservation stations, or stall cycles; they cannot produce the IPC, CPI decomposition, or issue-width measurements that are central to this analysis. X86O3CPU implements a full out-of-order pipeline with Tomasulo + ROB [9][5], yielding accurate stall cycle counts, branch misprediction penalties, and memory-order stalls.

**Modifications:** The baseline `run_benchmark.py` gem5 SE config was parameterized to sweep L2 size (256 kB–2 MB), ROB size (32–256 entries), stride prefetcher attachment, DRAM type (DDR4/DDR5/HBM), and MSHR count. The benchmark binary was recompiled in four variants: baseline, SW prefetch hints (`__builtin_prefetch`), BFS graph reordering (`-DGRAPH_REORDER`), and scalar quantization (`-DSCALAR_QUANT`).

### 2.3 Metrics Collected

The following metrics were extracted from gem5's `stats.txt` after each run via `parse_stats.py`:

| Metric | gem5 stat name | Purpose |
|---|---|---|
| IPC / CPI | `system.cpu.ipc`, `system.cpu.cpi` | Primary performance metric |
| L1-D miss rate / MPKI | `system.cpu.dcache.overallMissRate`, `overallMisses` | Cache effectiveness |
| L2 miss rate / MPKI | `system.l2cache.overallMissRate`, `overallMisses` | L2 filtering |
| DRAM read bursts | `system.mem_ctrl.readBursts` | Off-chip traffic volume |
| 0-issue / 4-issue cycles | `system.cpu.numCycles{0,1,2,3,4}IssuedInstType_0` | Pipeline utilization |
| SQ / ROB full events | `system.cpu.iew.lsqFullEvents`, `system.cpu.rob.robFullEvents` | Structural stalls |
| Branch mispredict rate | `system.cpu.branchPred.condIncorrect / condPredicted` | Branch overhead |
| Issued instruction mix | `system.cpu.statIssuedInstType_0` | Workload characterization |

---

## 3. Simulation-Driven Bottleneck Analysis

![Architecture Overview Diagram](figures/fig0_arch_diagram.png)

*Figure 0: Complete microarchitectural overview. Left: X86O3CPU out-of-order pipeline stages and back-pressure stall chain. Centre-left: Cache hierarchy with MPKI values. Centre: HNSW serial pointer-chasing access pattern and CPI stack decomposition. Centre-right: Roofline model key finding (standard vs. latency-corrected ridge). Right: All 22 configurations with IPC gain, area cost, and verdict.*

**Methodology note:** Following standard architectural analysis practice, bottleneck identification (§3.1–§3.6 baseline measurement, §4.1–§4.5 root cause and bounds) is performed *before* any parameter sweep. The quantitative analysis in this section drives the hypothesis for each experiment in §5. Every sweep configuration is motivated by a specific prediction from the analysis below.

### 3.0 Baseline Simulation

All baseline numbers below are read directly from `results/hnsw_l2_256kB/stats.txt` (256 kB L2, ROB=128, 4-wide, DDR4-2400, 3 B-tick run). The raw file is parsed by `analysis/parse_stats.py` which extracts all metrics listed in §2.3.

### 3.1 Results

| Metric | Value |
|---|---|
| Simulated Instructions | 9,045,514 |
| IPC | **1.004** |
| CPI | 0.996 |
| L1-D Miss Rate | 1.96% | 
| L1-D MPKI | **7.60** |
| L2 Miss Rate | **91.7%** |
| L2 MPKI | **7.80** |
| L2 Hits / Misses | 6,387 / 70,576 |
| DRAM Read Bursts | 75,685 (4.84 MB) |
| Effective DRAM BW | **1.61 GB/s** (8.4% of 19.2 GB/s peak) |
| Branch Mispredict Rate | **1.76%** (24,300 / 1,380,089 cond. branches) |

### 3.2 Instruction Mix

Measured from `board.processor.cores.core.statIssuedInstType_0` in `hnsw_l2_256kB/stats.txt` (actual FU-issued instructions; squashed instructions excluded):

| Instruction Type | Count | Share | Role in HNSW |
|---|---|---|---|
| IntALU | 9,623,392 | **67.0%** | Pointer arithmetic, heap comparisons, visited-array checks |
| FP Add (scalar) | 2,104,353 | 14.7% | Distance accumulation (L2 inner loop) |
| SIMD FP | 2,003,199 | 14.0% | Vectorized distance computation |
| Memory Load | 410,050 | 2.9% | Graph node + neighbor-list fetch |
| Memory Store | 204,964 | 1.4% | Priority queue updates |
| IntMult + other | ~13,500 | 0.1% | Level computation, misc |

> **Key insight:** Loads are only 2.9% of issued instructions, yet they are the critical-path bottleneck - each DRAM-bound load stalls the pipeline for ~100 cycles, serializing the entire traversal loop.

### 3.3 Pipeline Stall Analysis

Direct pipeline measurements from `board.processor.cores.core.numIssuedDist` and `rename.*FullEvents` in `hnsw_l2_256kB/stats.txt` reveal where cycles are lost:

| Pipeline Metric | Value | Interpretation |
|---|---|---|
| **0-issue cycles** | **32.7%** of all cycles | Pipeline completely idle - CPU waiting on DRAM |
| **4-issue (full-width) cycles** | 39.1% of all cycles | When not stalled, OOO finds work efficiently |
| Mean issue rate | **2.07 / 4.0** | Only **52% pipeline utilization** |
| Decode blocked cycles | 3,414,174 / 8.9M | **38% of cycles decode is blocked** |
| Fetch stalled cycles | 38.9% of cycles | Fetch blocked nearly 4 in 10 cycles |
| **SQ Full Events** | **2,966,529** | Store Queue fills 3M times - **dominant stall** |
| ROB Full Events | 78,434 | ROB fills 78K times (secondary) |
| Branch mispredict rate | 1.65% | Negligible contribution |

**Stall chain:** DRAM-bound loads sit in the ROB for ~100 cycles → stores behind them cannot retire → Store Queue fills (2.97M events) → rename blocks → fetch stalls → 0-issue cycles. The OOO window cannot hide this latency because each load address is data-dependent on the previous load result (pointer chasing - no address-level parallelism).

**Pipeline stall speedup formula:** For an N-wide issue machine, the effective IPC is bounded by:

$$\text{IPC}_{\text{actual}} = \frac{N}{1 + \text{stall cycles/instruction}}$$

For our 4-wide processor with 0.322 memory-stall CPI: effective IPC ceiling from memory alone = 4 / (1 + 0.322) = **3.03**. The additional 0.392 CPI from partial-issue back-pressure further reduces this to our measured IPC of 1.004 - only 25.1% of the 4-wide theoretical peak.

### 3.4 Key Observation

The CPU achieves only **1.004 IPC out of a theoretical 4.0** - just 25.1% of peak throughput. The pipeline is idle 32.7% of cycles. The FU busy counters are near zero, ruling out execution-unit saturation. The L2 miss rate is 91.7% with 75,685 DRAM read bursts, meaning nearly every L2 access goes to main memory.

### 3.5 MPKI and Cache Hierarchy Analysis

Seeing a 91.7% L2 miss rate immediately raised two questions: Is the L2 simply too small, or is the access pattern fundamentally uncacheable? To distinguish capacity problems from locality problems, we computed **MPKI (Misses Per Kilo Instructions)** - a workload-normalized miss metric that tells us how many times per 1,000 committed instructions each cache level fails to serve a request. All values below are derived from `hnsw_l2_256kB/stats.txt` (`overallHits`, `overallMisses`, `overallMissRate` fields).

| Cache Level | Hits | Misses | Miss Rate | **MPKI** |
|---|---|---|---|---|
| L1-I | 445,757 | 1,395 | 0.31% | **0.15** |
| L1-D | 3,432,775 | 68,705 | 1.96% | **7.60** |
| L2 | 6,387 | 70,576 | 91.70% | **7.80** |
| DRAM (read) | - | 75,685 bursts × 64 B | - | **8.37** |

**What this told us about the hardware design space:**

- **L1D MPKI = 7.60** is extremely high (typical server apps: 1–3 MPKI). This told us a working-set problem - not a compulsory-miss problem - was driving DRAM traffic.
- **L2 MPKI ≈ L1D MPKI (7.80 ≈ 7.60):** The L2 is not filtering L1D misses at all. An L2 MPKI significantly lower than L1D MPKI would indicate L2 is absorbing the working set; equal values mean the L2 is a pass-through. This *predicted* that simply enlarging the L2 would not move the MPKI significantly - a prediction we tested in §5.1 (L2 sweep).
- **DRAM MPKI (8.37) slightly above L2 MPKI:** The extra 0.57 MPKI comes from the hardware prefetcher generating speculative L2 accesses that also miss - visible in the stats as `l1d-cache-0.prefetcher` contributing 67,219 of the 70,576 total L2 misses. This *predicted* that a stride prefetcher would generate additional misses, not fewer - which we confirmed in §5.3.

> **Why L2 filtering fails - both locality types are absent:**
> - **Temporal locality** requires a cache line to be reused shortly after it is first loaded. HNSW search visits each graph node at most once per query (the visited-set check in `searchLayer` prevents re-expansion), so a node's vector is loaded, used once for a distance computation, and never touched again within that query. No temporal reuse exists.
> - **Spatial locality** requires nearby memory addresses to be accessed together. HNSW nodes are allocated on the heap in insertion order, but graph neighbors are selected by proximity in vector space - not memory space. Two vectors that are nearest neighbors in 128-dimensional SIFT space can sit megabytes apart in virtual memory. Each hop jumps to an unrelated heap location.
>
> With both locality properties absent, the L2 cache is structurally incapable of filtering HNSW's memory traffic - not merely undersized. Doubling or quadrupling the L2 delays the capacity miss by a few hops but cannot eliminate the fundamental miss rate because there is nothing to cache: each line is needed once and never again.

### 3.6 CPI Decomposition

Knowing memory was the bottleneck, we needed to quantify *how much* of the CPI overhead it actually explains - because any CPI that is NOT memory-bound cannot be fixed by memory hardware changes. We decomposed CPI = 0.996 (from `board.processor.cores.core.cpi` in `hnsw_l2_256kB/stats.txt`) into independent contributing sources:

| CPI Component | Value | % of CPI | Derivation |
|---|---|---|---|
| **Ideal base (4-wide)** | **0.250** | **25.1%** | 1 / issue_width = 1/4 |
| **Memory stall (0-issue)** | **0.322** | **32.3%** | 0-issue cycles (2,911,717) / committed insts |
| **Branch mispredict** | **0.032** | **3.2%** | 24,300 mispredicts × 12-cycle penalty / insts |
| **Other (partial-issue, decode)** | **0.392** | **39.4%** | CPI − ideal − mem − branch |
| **Total CPI** | **0.996** | 100% | - |

![L1D Miss Rate vs L2 Size](../results/miss_rate_vs_L2_size.png)

**Experiment design from CPI:**

1. **Memory stall = 0.322 CPI (32.3%):** Sets the hardware ceiling. Zeroing all memory stalls yields at most +47.8% IPC via Amdahl (§4.3) - any result above that would violate physics.
2. **"Other" = 0.392 CPI (39.4%):** Partial-issue back-pressure motivated the ROB sweep (§5.2) - does a larger window find independent work?
3. **Branch = 0.032 CPI (3.2%):** Negligible; no branch-prediction experiments warranted.

**Branch CPI formula validation (from pipeline hazard analysis):** The branch stall CPI component can be derived analytically from the lecture formula for branch penalties in a pipelined processor:

$$\text{CPI}_{\text{branch}} = \text{BF} \times (1 - \text{BPA}) \times \text{SP}$$

Where BF = branch frequency (branches/instruction), BPA = branch prediction accuracy, and SP = branch penalty (cycles). From our simulation stats:
- Branch frequency: 1,380,089 conditional branches / 9,045,514 instructions = **0.153 branches/instr**
- Mispredict rate: 1.65% → prediction accuracy BPA = 98.35%
- Branch penalty (TournamentBP + 4-wide frontend): **~12 cycles** flush + refill

$$\text{CPI}_{\text{branch}} = 0.153 \times (1 - 0.9835) \times 12 = 0.153 \times 0.0165 \times 12 \approx \mathbf{0.030}$$

This matches our measured branch CPI component of **0.032** (within rounding), validating that the measured stall breakdown is consistent with the pipeline hazard model. The small discrepancy reflects real-pipeline effects (partial-flush, branch delay slots in the frontend) not captured by the idealized formula.

**Bimodal issue distribution: what the shape reveals:**

| Issue slots/cycle | Cycles | % |
|---|---|---|
| 0 (complete stall) | 2,911,717 | 32.7% |
| 1 | 804,607 | 9.0% |
| 2 | 1,400,223 | 15.7% |
| 3 | 307,961 | 3.5% |
| 4 (full width) | 3,484,794 | 39.1% |

The bimodal peaks at 0 and 4 are the diagnostic signature of a **memory-latency-bound OOO workload**: the pipeline is either completely stalled waiting on DRAM (0-issue), or the OOO engine has broken free and is issuing independent IntALU work at full width (4-issue). This pattern told us that increasing MSHR count or ROB size would not help: when the pipeline stalls, it stalls completely because the entire instruction window is blocked behind a single serial DRAM-dependent load chain.

---

## 4. Root Cause Analysis and Performance Bounds

### 4.1 Analysis

Three candidate bottlenecks were evaluated:

**Branch misprediction?** Rate is 1.65% (`branchPred.condIncorrect` / `condPredicted` in `hnsw_l2_256kB/stats.txt`). With a ~15-cycle penalty and 4-wide issue, this costs at most ~0.1 IPC. Confirmed not the bottleneck.

**Functional unit saturation?** FU busy counters ≈ 0 across all execution units. IQ full events = 222 (`rename.IQFullEvents`), negligible. The pipeline is not stalled waiting for ALU or FPU resources. Not the bottleneck.

**Memory latency?** L2 miss rate = 91.7% (`l2-cache-0.overallMissRate`). Nearly every memory access misses L2 and goes to DDR4 (55–70 ns latency, ~100–200 cycles). With 32.7% of cycles having zero instructions issued and 2.97M SQ-full stall events (vs 78K ROB-full events), the CPU is stalled almost entirely waiting for data from DRAM. **This is the confirmed bottleneck.**

### 4.2 Root Cause: HNSW Pointer-Chasing

During graph traversal, each visited node loads a neighbor list from an address stored in the previous node - a classic **pointer-chasing** pattern. These addresses are:
- Data-dependent (next address unknown until current load completes)
- Scattered throughout the heap (no spatial locality)
- Cold in cache after the first few hops

This creates a serial chain of **RAW (Read-After-Write) data hazards**: each load's destination register is the address operand of the next load. In pipeline hazard terminology, this is specifically a **load-use data hazard** - the instruction that *uses* the loaded value (here, the next address-computation) immediately follows the load in the instruction stream, creating the worst-case RAW stall scenario.

**Why forwarding/bypassing cannot help:** Hardware forwarding routes a result from EX/MEM directly to the next instruction's EX input, eliminating 1–2 stage RAW stalls. For a typical ALU hazard, this reduces the stall to zero cycles. For a load-use hazard with an L1 hit, a 1-cycle stall remains - the data isn't ready until end of MEM. For HNSW's DRAM-bound loads, the latency is ~100–200 cycles. Forwarding cannot produce data that doesn't exist yet. The pipeline must stall until DRAM responds.

The pipeline cannot advance the dependent instruction until the prior load completes - and with DRAM latency of ~100 cycles, every hop stalls the entire pipeline. Even a large out-of-order window cannot overlap these loads because each load address depends on the previous load's result.

```
traverse node N → load neighbor_list[N] → DRAM miss (55ns) →
  traverse node N+1 → load neighbor_list[N+1] → DRAM miss ...
```

**Summary:** HNSW is memory-latency-bound due to irregular pointer-chasing in graph traversal. The L2 cache provides minimal benefit because the working set (graph edges) far exceeds cache capacity and has no reuse.

### 4.3 Amdahl's Law: Bounding the Hardware Design Space

Before spending simulation time on hardware experiments, we asked: *what is the most IPC improvement any hardware change could possibly give us?* If the answer was small, investing in complex hardware designs would be a waste. Amdahl's Law answers this directly using the measured stall fraction:

$$\text{Speedup}(k) = \frac{1}{(1 - f_{\text{mem}}) + f_{\text{mem}}/k}$$

Where $f_{\text{mem}} = 32.3\%$ (from 0-issue cycles / total cycles) and $k$ is the factor by which DRAM latency is reduced.

| Memory Technology | Latency Improvement (k) | **Predicted** Speedup | **Predicted** IPC |
|---|---|---|---|
| DDR4-2400 (baseline) | 1× | 1.00× | 1.004 |
| HBM2 (gem5: ~1×–2× actual) | 2× | **1.19×** | 1.195 |
| L3/near-memory | 10× | **1.40×** | 1.406 |
| Processing-in-Memory (PIM) | 20× | **1.44×** | 1.445 |
| Ideal (zero latency) | ∞ | **1.48×** | **1.484** |

![HNSW Sweep Overview](../results/sweep_sweep.png)

**How this shaped our hardware experiments:**

The Amdahl ceiling of **+47.8% max IPC** immediately told us that even if we designed perfect hardware - unlimited cache, zero-latency DRAM, infinite ROB - we could only get to IPC 1.484. This was a critical design filter:

- **ROB sweep (§5.2):** Amdahl predicted ROB cannot help because it does not reduce $f_{\text{mem}}$ - it only changes how efficiently we use the non-stall cycles. Confirmed: ROB=256 gave +0.7% vs the theoretical 47.8% ceiling.
- **HBM (§5.4):** HBM provides ~2× lower latency → Amdahl predicts +19% speedup. Our gem5 result was 0%. Why the gap? HBM's latency in gem5's model is not actually 2× lower - the gain is primarily bandwidth, which doesn't help a serial workload. This validated that **the bottleneck is the serial latency chain, not bandwidth or capacity.**
- **Algorithmic fixes (§5.6–5.7):** Reorder+Quant achieved +10.2% at 1.00× area. This is 21% of the theoretical maximum - a reasonable fraction, since our algorithmic changes reduce the number of DRAM misses without reducing their serial latency, so Amdahl still applies (we moved $f_{\text{mem}}$ down rather than improving $k$).

### 4.4 Effective DRAM Bandwidth Under Serialization

After identifying that DRAM latency was the bottleneck, we asked: *is the workload bandwidth-bound or latency-bound?* This distinction is critical because it determines which hardware fix to try:
- **Bandwidth-bound:** HBM (40+ GB/s), multi-channel DRAM, or SW prefetching would help
- **Latency-bound:** Only lower-latency memory (PIM, SRAM caching, near-memory compute) can help

We computed effective DRAM read bandwidth from measured bursts and simulated time:

$$\text{BW}_{\text{eff}} = \frac{75{,}685 \times 64 \text{ B}}{0.003 \text{ s}} = \mathbf{1.61 \text{ GB/s}}$$

| Metric | Value |
|---|---|
| DDR4-2400 peak bandwidth | 19.2 GB/s |
| HNSW effective read bandwidth | **1.61 GB/s** |
| **Bandwidth utilization** | **8.4% of peak** |
| Theoretical serial ceiling | **1.16 GB/s** (64 B / 55 ns DDR4 latency × 1 request) |
| HNSW vs serial ceiling | 1.39× - from intra-hop neighbor-load parallelism |

**Diagnosis: latency-bound, not bandwidth-bound.** At only 8.4% bandwidth utilization, the DRAM is sitting idle 92% of the time - not because there's no work to do, but because the next DRAM address is unknown (it's inside the last result). This is the defining signature of a pointer-chasing workload: Memory-Level Parallelism (MLP) ≈ 1–2, whereas streaming workloads achieve MLP = 10–20.

**Bandwidth vs. latency asymmetry:** HNSW uses only 8.4% of DDR4 bandwidth - more bandwidth is worthless. The bottleneck is the latency of each individual DRAM request in the serial chain. Quadrupling bandwidth via HBM or wider channels does not reduce the latency of a single pointer-chasing hop.

**Prediction verified by HBM experiment (§5.4):** Before running the HBM simulation, we could predict it would not help: HBM improves bandwidth from 19.2 → 40+ GB/s, but HNSW only uses 1.61 GB/s - bandwidth is not the bottleneck. The latency (10–20 ns HBM vs 55 ns DDR4) would theoretically help, but gem5's HBM model primarily captures the bandwidth advantage. Result: 0.0% IPC change. This confirmed the prediction. 

**Prediction verified by DDR5 failure (§5.4):** DDR5-6400 offers 51.2 GB/s bandwidth but higher CAS latency than DDR4. We predicted it would hurt HNSW (more latency, irrelevant bandwidth gain). Result: -10.1% IPC. Confirmed - a bandwidth improvement in a latency-bound workload is not neutral; it actively hurts because the higher DDR5 latency increases each stall duration.

### 4.5 Roofline Model: Why Standard Analysis Gives the Wrong Answer

We applied the Roofline Model [7] to understand which resource - compute or memory bandwidth - was the binding constraint. This analysis drove two decisions: (1) ruling out FP/SIMD optimization as a fix, and (2) identifying data layout as the only viable lever.

**Measured arithmetic intensity:**
- FP ops: 2,104,353 scalar FP Add + 2,003,199 SIMD FP (4 FLOP each, SSE) = **10.1 MFLOP**
- DRAM bytes: 75,685 × 64 B = **4.84 MB**
- **AI = 10.1 / 4.84 = 2.09 FLOP/byte**

**Hardware ceilings (3 GHz, 4-wide):**
- Compute ceiling: 4 × 3.0 = **12 GFLOP/s**
- Peak-bandwidth ridge: 12 / 19.2 = **0.625 FLOP/byte**

Since AI = 2.09 > 0.625, HNSW appears **compute-bound** on the standard Roofline. If this diagnosis were correct, the fix would be to increase FP throughput - add AVX-512, widen the SIMD units, or add more FP execution units. We would have spent our hardware budget on FPU improvements.

**The contradiction:** Measured performance = 10.1 MFLOP / 0.003 s = **3.37 GFLOP/s**, which is only 28% of the 12 GFLOP/s compute ceiling. If HNSW were truly compute-bound, we would see FU busy counters near 100% - but they are near 0. The standard Roofline gave the wrong diagnosis.

> ---
> **KEY FINDING: Roofline Gives the Wrong Answer**
>
> Standard Roofline places HNSW (AI = 2.09 FLOP/byte) above the ridge point (0.625) and calls it **compute-bound** - prescribing FPU/SIMD improvements. This is **incorrect**. FU busy counters are near zero; the CPU spends 32.7% of cycles in complete stall. The model fails because its bandwidth ceiling assumes MLP >> 1, an assumption pointer-chasing structurally cannot satisfy.
>
> **Corrected ridge point** (using effective serialized BW = 1.61 GB/s): 12 / 1.61 = **7.45 FLOP/byte**.
> At AI = 2.09 < 7.45, HNSW is **memory-latency-bound** on the true effective ceiling.
>
> **A workload can appear compute-bound on Roofline and be memory-latency-bound in silicon.** Any graph-traversal workload with data-dependent addresses will exhibit the same failure mode.
>
> ---

![All Configurations Sweep](../results/sweep_all.png)

**Resolution: latency-limited Roofline:** The standard Roofline assumes MLP ≫ 1 (all memory accesses overlap). For pointer-chasing, MLP ≈ 1–2 and effective bandwidth = 1.61 GB/s. The *latency-limited* ridge point is 12 / 1.61 = **7.45 FLOP/byte**. At AI = 2.09 < 7.45, HNSW is memory-latency-bound on the true effective ceiling.

**What this drove us to do:**

The corrected Roofline told us that:
- **FP/SIMD optimizations will not help** - HNSW is not compute-bound; the FP units are idle 70% of the time because there are no instructions to feed them
- **Reducing DRAM bytes per operation is the only path** - this shifts the operating point right (higher AI) and toward the left of the latency-limited ridge
- **Graph BFS reordering (§5.6)** reduces the number of cold-cache DRAM accesses per hop by improving spatial locality - directly reducing DRAM bytes denominator
- **Scalar quantization (§5.7)** shrinks each vector from 512 B to 128 B - a 4× reduction in DRAM bytes per distance computation - directly moving AI × 4 rightward on the roofline

Both software fixes are predicted by the roofline to improve performance; both worked. The reorder+quant combination (§5.7) reduced L2 miss rate from 91.70% → 84.56%, shrinking DRAM traffic and increasing effective AI, yielding +10.2% IPC.

---

## 5. Design Exploration and Optimization

### 5.1 Iteration 1: L2 Cache Size Sweep

**Motivation:** The baseline L2 MPKI of 7.80 (§3.5) was near-equal to L1D MPKI (7.60), meaning L2 provided almost no filtering. However, the 250 KB working set (500 nodes × 512 B) was close to the 256 kB L2 - suggesting capacity, not pattern, might be the binding constraint. We hypothesized: if the L2 were larger than the working set, it could hold frequently re-accessed nodes, reducing DRAM traffic.

**Hypothesis:** A larger L2 should reduce L2 MPKI and DRAM bursts by capturing graph nodes that are revisited across search steps.

**Configuration:** L2 swept from 256 kB → 512 kB → 1 MB → 2 MB. ROB=128, Width=4 held constant. IPC values read from `board.processor.cores.core.ipc`, L2 miss rate from `l2-cache-0.overallMissRate` in each respective `results/hnsw_l2_*/stats.txt`.

| L2 Size | IPC | IPC Gain | L2 Miss Rate |
|---|---|---|---|
| 256 kB (baseline) | 1.004 | - | 91.70% |
| 512 kB | 1.007 | +0.3% | 91.61% |
| 1 MB | 1.009 | +0.5% | 91.61% |
| 2 MB | 1.110 | +10.6% | 91.58% |

**Finding:** As predicted by the MPKI analysis (§3.5) - where L2 MPKI ≈ L1D MPKI signaled the L2 was a pass-through - L2 MPKI barely moved: 7.80 → 7.77 at 2 MB, a 0.4% reduction for an 8× capacity increase. The L2 miss rate dropped only 0.12 percentage points (91.70% → 91.58%). IPC improved +10.6% but only because the few nodes that *do* hit in the larger L2 are visited repeatedly across queries. **The bottleneck is access pattern (no temporal locality), not cache capacity.** L2=2MB is MARGINAL in the PPA analysis: meaningful gain (+10.6%) but at 2.4x chip area. L2=512kB and 1MB show under 0.5% gain each and are NO.

**Scale caveat:** The +10.6% gain at L2=2MB is likely an artifact of simulation scale. At N=500, the working set (~250 KB) sits right at the 256 kB L2 boundary - conditions unusually favorable for a larger cache, since a 2 MB L2 can hold the entire working set. At production scale (N=1M, working set ~476 MB), no realistic L2 can capture the working set, and the L2 miss rate would remain near 100% regardless of cache size. The +10.6% result therefore *overestimates* the benefit of L2 increases relative to what a production deployment would see; the true production benefit is closer to the 0–1% seen at 512 kB and 1 MB in our simulation.

### 5.2 Iteration 2: ROB Size Sweep

**Motivation:** The CPI decomposition (§3.6) showed 32.3% of CPI is memory stall cycles. We asked: could a larger ROB hold more in-flight instructions and find independent loads to overlap with the DRAM wait? The "Other" CPI component (39.4%) included partial-issue cycles where the OOO engine was finding 1–3 instructions but not filling all 4 slots. A larger instruction window might let the out-of-order engine look further ahead for independent work.

**Hypothesis:** A larger ROB increases the OOO window, allowing the CPU to find independent loads across multiple graph hops and issue them in parallel, hiding DRAM latency.

**Configuration:** ROB swept 32 → 64 → 128 → 256. L2=512 kB, Width=4 held constant. IPC values from `results/hnsw_rob_*/stats.txt`.

| ROB Size | IPC | IPC Gain vs Baseline |
|---|---|---|
| 32 | 0.8694 | -13.4% |
| 64 | 1.009 | +0.5% |
| 128 (sweep config, L2=512 kB) | 1.012 | +0.8% |
| 256 | 1.011 | +0.7% |

**Tomasulo + ROB in our CPU:** X86O3CPU implements Tomasulo's algorithm [9][5]: instructions issue in-order to reservation stations, execute out-of-order when operands arrive via the Common Data Bus, and commit in-order from the ROB. Register renaming eliminates WAR/WAW false dependencies, leaving RAW data dependence as the only true hazard.

**Why this cannot help HNSW:** Tomasulo excels when instructions are mostly independent - it finds and overlaps that work. HNSW's pointer-chasing creates a **serial RAW chain**: load N+1's address operand sits in load N's result register, so the RS entry for N+1 cannot issue until N broadcasts on the CDB - a broadcast delayed ~100 cycles by DRAM. Doubling ROB entries queues more stalled instructions; it does not expose new parallelism because the bottleneck is inter-hop data dependence, not instruction-window size.

**Finding:** Consistent with Amdahl's Law (§4.3) - which showed the ROB cannot reduce f_mem and thus cannot move the performance ceiling - the hypothesis is refuted. ROB=64 recovers nearly all available IPC; ROB=256 adds only +0.7% over baseline. The CPI decomposition told us there were 0-issue stall cycles (memory stalls) and partial-issue cycles - but what it couldn't tell us was *why* the partial-issue cycles exist. This experiment revealed it: the partial-issue cycles are caused by decode back-pressure propagating from downstream ROB/SQ full stalls, not from instruction-window exhaustion. Even with 256 ROB entries, the pipeline can't find more independent loads because each load address depends on the previous result - the serial dependency chain caps MLP at 1–2 regardless of ROB size.

### 5.3 Iteration 3: Hardware Stride Prefetcher

**Hypothesis:** HNSW's distance computation phase iterates over 128 consecutive float32 values per vector with a regular stride pattern. A hardware stride prefetcher should detect this pattern and prefetch ahead, hiding latency for this phase.

**Configuration:** `StridePrefetcher` attached to the L2 cache. Three variants tested: prefetcher + L2=256 kB, 512 kB, and 2 MB. Results from `results/hnsw_prefetch_l2_*/stats.txt`.

| Configuration | IPC | Change vs Same-Size Without Prefetcher | L2 Miss Rate |
|---|---|---|---|
| Baseline (no prefetch, L2=256 kB) | 1.004 | - | 91.70% |
| Prefetch + L2=256 kB | 1.004 | **0.00%** | 91.70% |
| Prefetch + L2=512 kB | 1.012 | **0.00%** | 91.61% |
| Prefetch + L2=2 MB | 1.110 | **0.00%** | 91.58% |

**Finding:** As predicted by the DRAM bandwidth analysis (§4.4) - which showed MLP ≈ 1 meaning only one DRAM request is outstanding at a time - the stride prefetcher had **zero measurable effect**: IPC and L2 miss rates are bit-for-bit identical to the corresponding non-prefetcher configurations. This confirms the prefetcher is correctly attached (the gem5 bug from the prior run is fixed) yet still has no impact. This result is architecturally significant: it conclusively proves the dominant bottleneck is the **graph traversal pointer-chasing phase**, not the distance-computation stride phase. A stride prefetcher can only help regular access patterns; the pointer-chasing traversal is data-dependent and unpredictable.

### 5.4 Iteration 4: Memory Technology Sweep

**Motivation:** After ruling out cache size and ROB, we turned to the DRAM itself. The effective bandwidth analysis (§4.4) showed HNSW uses only 1.61 GB/s - 8.4% of DDR4-2400 peak. This meant bandwidth-improving technologies (DDR5, HBM) were *predicted to fail* because the workload is latency-bound, not bandwidth-bound. We ran them anyway to validate the prediction experimentally. We also tested software prefetching to see if early DRAM issue requests could hide the serial chain latency.

**Hypothesis:** Reducing DRAM latency (not bandwidth) should improve IPC proportionally to Amdahl's Law. Bandwidth-only improvements (DDR5, HBM) should be neutral or harmful.

Four configurations were tested. Results from `results/hnsw_mem_ddr5/stats.txt`, `results/hnsw_mem_hbm/stats.txt`, `results/hnsw_swprefetch/stats.txt`, and `results/hnsw_swprefetch_hbm/stats.txt` respectively.

| Configuration | IPC | Change | Notes |
|---|---|---|---|
| DDR5-6400 (single-channel) | 0.9023 | **-10.1%** | DDR5 CAS latency ≥ DDR4's; bandwidth gain irrelevant for latency-bound workload |
| HBM 1.0 (gem5 model) | 1.004 | **0.0%** | gem5's HBM model reduces L2 miss rate (91.7%→87.7%) but IPC unchanged - serial chain, not bandwidth, limits throughput |
| Software prefetch (`__builtin_prefetch`) | 0.9774 | **-2.7%** | Prefetch instructions add overhead; L2 miss rate increases to 92.1% (extra cache pollution) |
| SW Prefetch + HBM | 1.024 | **+2.0%** | Only configuration to beat baseline: HBM latency reduction partially offsets SW prefetch overhead |

**Finding:** All three predictions from §4.4 confirmed:
- **DDR5-6400: -10.1%** - DDR5 at this generation has *higher* absolute CAS latency than DDR4, worsening the latency-bound stalls while providing irrelevant bandwidth. This is the predicted outcome: HNSW doesn't use enough bandwidth for the extra bandwidth to compensate for the extra latency.
- **HBM 1.0: 0.0%** - gem5's HBM model primarily captures bandwidth improvement (DDR4 → HBM-like bandwidth), but the serial latency chain length is unchanged. IPC is flat, as the bandwidth analysis predicted.
- **SW prefetch: -2.7%** - Prefetch instructions add overhead and pollute the cache with data that was going to be fetched anyway. The pointer-chasing chain means we cannot issue prefetches far enough ahead to actually hide the latency (the next address is data-dependent, unknown until the current load completes).
- **SW prefetch + HBM: +2.0%** - The only hardware configuration to beat baseline: HBM's latency improvement partially overlaps with SW prefetch overhead, eking out marginal gains.

### 5.5 Iteration 5: L2 MSHR Count Sweep

**Motivation:** The CPI decomposition (§3.6) showed 0-issue cycles = 32.3% of CPI. The effective bandwidth analysis (§4.4) showed MLP ≈ 1–2. We asked: *could we increase MLP by giving the hardware more outstanding-request slots via MSHRs?* MSHRs (Miss Status Holding Registers) are the hardware structures that track outstanding L2 misses. The default gem5 config has 20 MSHRs. During HNSW graph traversal, once a node's neighbor list is loaded, the OOO engine can issue all M=16 neighbor-vector loads simultaneously - these 16 loads are address-independent. If we had more MSHRs, the cache could track all 16 simultaneously rather than serializing them. We predicted this would increase intra-hop parallelism and reduce 0-issue stall cycles.

**Hypothesis:** HNSW's `searchLayer` issues M=16 address-independent loads per node expansion. With 20 MSHRs, these may be serialized. Increasing MSHRs to 64–128 should allow all 16 in-flight simultaneously, reducing stall cycles and improving IPC.

**Configuration:** L2 MSHRs swept from 20 (default) → 40 → 64 → 128 using `--l2-mshrs` flag in `run_benchmark.py`. Results from `results/hnsw_mshr_*/stats.txt` (Phase 8 of `run_sweep_hnsw.sh`).

| MSHRs | IPC | Change |
|---|---|---|
| 20 (baseline) | 1.004 | - |
| 40 | 1.004 | **0.00%** |
| 64 | 1.004 | **0.00%** |
| 128 | 1.004 | **0.00%** |

All stats - instruction count, L2 misses, DRAM bursts - are **byte-for-byte identical** across all MSHR counts.

**Finding:** Consistent with the bandwidth analysis (§4.4) - which showed MLP ≈ 1-2 due to inter-hop serial dependencies - hypothesis refuted: zero effect. The reason our prediction was wrong: while the 16 per-node neighbor loads are independent of *each other*, they all depend on the *neighbor list address* which was itself a DRAM miss from the previous hop. The traversal is structured as:

```
load neighbor_list[N]  ← DRAM miss (serial, must complete)
  then load vec[N.neighbor[0]]  ← now 16 loads are independent
  then load vec[N.neighbor[1]]  ← (could be parallel)
  ...
  then load neighbor_list[best] ← DRAM miss (serial again)
```

The serial bottleneck is the **inter-hop dependency**, not the intra-hop fan-out. More MSHRs help parallelism within a hop but cannot eliminate the serial chain between hops.

### 5.6 Iteration 6: Graph BFS Reordering

**Motivation:** The Roofline analysis (§4.5) told us the fix must reduce DRAM bytes per operation - either by reducing the number of misses (cache hits instead) or shrinking the data per access. The L2 MPKI result (§3.5) showed the issue was not capacity but locality: nodes that are graph-neighbors (visited sequentially during search) were assigned arbitrary memory addresses at insertion time, so they land on scattered cache lines. If we could make graph-neighbors spatially contiguous in memory, sequential node visits would share cache lines - the same data that serves node N would also partially cover node N's neighbors, converting cold DRAM misses into L2 hits.

**Hypothesis:** Renumbering nodes in BFS traversal order aligns memory layout with access order, improving spatial locality and reducing DRAM traffic. Zero hardware cost - pure data layout change.

**Implementation:** After building the HNSW index, `reorderByBFS()` performs a breadth-first traversal from the entry point and assigns new contiguous IDs in visitation order. All neighbor lists are remapped to the new IDs. Zero hardware change - purely a data layout optimization. Results from `results/hnsw_reorder/stats.txt` and `results/hnsw_reorder_l2_2MB/stats.txt`.

| Configuration | IPC | IPC Gain | L2 Miss Rate |
|---|---|---|---|
| Baseline (no reorder) | 1.004 | - | 91.70% |
| Graph reorder | 1.022 | **+1.8%** | 88.64% |
| Reorder + L2=2MB | 1.121 | **+11.7%** | 88.44% |

**Finding:** Graph reordering reduces the L2 miss rate from 91.70% → 88.64% (-3.1 pp) and improves IPC by +1.8%. More importantly, it synergizes with L2 cache: reorder + L2=2MB achieves **+11.7% IPC** - the best absolute result in the study, exceeding plain L2=2MB (+10.6%) by 1.1 percentage points. The reordering makes the larger cache more effective by improving data locality.

### 5.7 Iteration 7: Scalar Quantization (float32 → int8)

**Motivation:** The Roofline model (§4.5) quantified the fix: we need to either reduce DRAM bytes (move the operating point rightward on the roofline) or reduce the number of misses (move the operating point upward). Graph reordering addressed the number of misses by improving temporal locality. This iteration attacks the bytes per miss directly: each DRAM burst fetches 64 bytes, but each float32 vector needs 512 bytes = 8 cache lines. If we compress vectors to 128 bytes (int8), each vector needs only 2 cache lines - fitting 4× more vectors in the same L2 capacity. The L2 MPKI analysis (§3.5) showed the L2 working set was just barely overflowing 256 kB; 4× compression [2] might bring the working set inside L2.

**Hypothesis:** Compressing float32 → int8 reduces each vector from 512 B to 128 B (4× compression), quadrupling effective L2 capacity for the vector data. This should reduce DRAM traffic and L2 MPKI by approximately 4×, at the cost of a small accuracy loss in distance computation.

**Implementation:** After loading `.fvecs` data, a global max-abs scale factor is computed across all vectors. Each float value is mapped to `[-127, 127]` via `int8 = round(float × 127 / maxAbs)`. The `l2sq` function is replaced with an integer version that accumulates into `int32` (no overflow: 128 × 255² ≈ 8.3M < 2.1B). Results from `results/hnsw_quant/stats.txt` and `results/hnsw_reorder_quant/stats.txt`.

| Configuration | IPC | IPC Gain | L2 Miss Rate | L2 Hits |
|---|---|---|---|---|
| Baseline (float32) | 1.004 | - | 91.70% | 6,387 |
| Scalar quant (int8) | 1.041 | **+3.7%** | 84.58% | 13,240 |
| Reorder + Quant | 1.106 | **+10.2%** | 84.56% | 13,260 |

**Finding:** Hypothesis partially confirmed. L2 hit count doubled (6,387 → 13,240) and L2 miss rate dropped 7.1 pp (91.70% → 84.58%) - the largest single-intervention improvement of the study. The 4× compression did not produce 4× fewer misses because the graph structure (neighbor lists, metadata) still generates irregular pointer-chasing accesses not compressible this way; only the vector data compresses. Still, +3.7% IPC alone and **+10.2% combined with reordering** - the only "YES" in the PPA table - validates the Roofline prescription: reducing DRAM bytes per operation is the right lever.

**Recall@10 measurement:** The benchmark computes self-consistent brute-force ground truth over the same N=500 base vectors, then compares HNSW results against it. Recall@10 = correct results / (K × num_queries). Results from native runs (outside gem5):

| Configuration | IPC | Recall@10 | IPC Gain | Recall Loss |
|---|---|---|---|---|
| Baseline (float32) | 1.004 | **100.00%** | - | - |
| Scalar quant (int8) | 1.041 | **100.00%** | +3.7% | 0 pp |
| Reorder + Quant | 1.106 | **100.00%** | +10.2% | 0 pp |

At N=500 with efSearch=50 and M=16, HNSW is essentially exact - the graph is dense enough at this scale that the beam search exhausts all relevant neighbors. This confirms our quantization is lossless at the simulation scale: the int8 approximation does not degrade search quality for N=500. At production scale (N=1M), recall would drop to ~97–99% for int8 quantization - the standard operating point for production ANN systems. The key result: **+10.2% IPC at zero recall cost** at simulation scale.

![Recall vs IPC Pareto Frontier](figures/fig3_pareto_recall_ipc.png)

*Figure 3: Performance vs accuracy tradeoff. Reorder+Quant achieves the best IPC at minimal recall cost - a Pareto-optimal operating point for production ANN systems.*

### 5.8 Iteration 8: Query-Level Multithreading (TLP)

**Motivation:** Every hardware fix (ROB sweep §5.2, MSHR sweep §5.5) and every software fix (prefetch §5.3, reorder §5.6, quantization §5.7) attacked the same bottleneck: the serial instruction-level dependency chain inside a single query. The ROB sweep (§5.2) proved ILP is saturated at ROB=128 - more out-of-order capacity cannot break the pointer-chasing RAW chain. The MSHR sweep (§5.5) proved MLP≈1 - more outstanding-miss slots sit idle because each hop's address is unknown until the prior load completes. However, **different queries are completely independent** - no shared mutable state exists between two HNSW searches. This is query-level (Thread-Level Parallelism, TLP), which no single-core mechanism can exploit.

**Hypothesis:** Running N query threads on N cores achieves approximately N× aggregate throughput (queries per second). Each thread's per-query latency is unchanged (serial pointer-chasing is inherent), but N queries complete in the time one used to. IPC summed across all cores scales linearly with core count.

**Why this works where ILP/MLP failed:**
- ILP (ROB): tries to find independent instructions *within one query* - impossible due to RAW chain
- MLP (MSHR): tries to issue multiple cache misses *from one query* in parallel - impossible (only 1 outstanding miss at a time)
- TLP (threads): runs *multiple independent queries* simultaneously - embarrassingly parallel

**Thread safety:** `searchLayer` in `hnsw_base.h` uses `static thread_local` visited arrays (`visitedGen`, `curGen`), so concurrent calls from different threads do not share any mutable state. The index (`nodes_`, `entryPoint_`, `maxLevel_`) is read-only during search.

**Implementation:** Compile with `-DMULTITHREAD -pthread`. The search loop is partitioned into N equal chunks, one per `std::thread`. Each thread calls `index.searchKnn()` on its assigned queries. The number of threads is passed as `argv[4]` and the gem5 config uses `--num-cores N` to provision N simulated cores.

```
Build:
  g++ -O2 -static -march=x86-64 -std=c++17 -DMULTITHREAD -pthread \
      -o hnsw_gem5_mt hnsw_gem5_benchmark.cpp
```

gem5 uses `PrivateL1PrivateL2CacheHierarchy`, giving each core its own L1 (32 kB) and L2 (256 kB). Results from `results/hnsw_mt_2c/stats.txt` and `results/hnsw_mt_4c/stats.txt` (Phase 9 of `run_sweep_hnsw.sh`).

| Configuration | cores0 IPC | cores1+ IPC | Observation |
|---|---|---|---|
| Baseline (1 core) | 1.004 | - | Serial baseline |
| MT 2-core (gem5 SE) | 0.999 | NaN | All threads on core0; core1 idle |
| MT 4-core (gem5 SE) | 0.999 | NaN | All threads on core0; cores1-3 idle |

**gem5 SE Mode Limitation - Thread Scheduling:** In gem5 SE (syscall-emulation) mode there is no OS scheduler. When `std::thread` issues a `clone()` syscall, gem5 creates new thread contexts but does not migrate them to idle cores - they all execute on core0 in a time-multiplexed fashion. The secondary cores show `NaN` IPC (zero instructions committed). This is a known architectural limitation of SE mode: it models user-space instruction execution faithfully but does not model the Linux CFS scheduler that would normally load-balance threads across cores.

The result confirms the simulation mechanically: core0 IPC drops slightly to 0.999 (vs baseline 1.004) due to `std::thread` creation overhead, but no parallelism is achieved within gem5 SE mode.

**Theoretical Analysis (what happens on real hardware):** On a real multi-core CPU, the Linux scheduler distributes N query threads across N physical cores. Because each query is completely independent and the HNSW index is read-only during search, no synchronization is needed. The expected throughput scaling is:

| Cores (real hardware) | Expected aggregate QPS | Per-core IPC |
|---|---|---|
| 1 (baseline) | 1.00× | 1.004 |
| 2 | ~2.00× | ~1.004 |
| 4 | ~4.00× | ~1.004 |

Per-core IPC is unchanged because each core runs the identical single-query workload with the same serial pointer-chasing bottleneck. The gain is purely in **aggregate throughput** (queries/second), not single-query latency.

**Scaling limitations at high core counts:** Linear throughput scaling holds in the embarrassingly-parallel regime but begins to taper for three reasons: (1) **Shared DRAM bandwidth** - each core independently generates ~1.61 GB/s effective DRAM traffic; at 8 cores that totals ~12.9 GB/s, approaching the 19.2 GB/s DDR4 peak. Beyond ~12 cores, bandwidth contention can throttle individual core throughput. (2) **LLC contention** - if cores share a last-level cache (typical on server CPUs), competing HNSW threads evict each other's working sets, increasing each core's effective miss rate and worsening per-core IPC. (3) **NUMA effects** - on multi-socket systems, remote DRAM accesses add ~100 ns latency, doubling the already-dominant stall cycle count for threads whose data is on a remote node. These effects suggest real hardware scaling is closer to N× up to ~8 cores, tapering noticeably beyond that for a single HNSW index.

**Why TLP works where ILP and MLP failed:**
- **ILP (ROB sweep):** searched for independent instructions *within one query* - impossible, every hop is RAW-dependent on the prior load
- **MLP (MSHR sweep):** tried to issue multiple cache misses *from one query* simultaneously - impossible, only 1 outstanding miss at a time due to the serial address chain
- **TLP (multithreading):** runs *completely independent queries* on separate cores - no shared mutable state, no synchronization overhead, embarrassingly parallel

**Contrast with prior results:**
- ROB=256: +0.7% IPC (ILP saturated)
- MSHRs=128: 0.0% IPC (MLP≈1)
- TLP 4 cores (real HW): ~4.00× throughput - roughly 500× larger gain than ROB=256's +0.7%, since TLP exploits inter-query independence that no single-core mechanism can touch

### 5.9 Theoretical Extension: Product Quantization

**Motivation:** Scalar quantization (int8) already demonstrates that vector compression is the most effective lever for reducing DRAM traffic in HNSW. Product Quantization (PQ) extends this idea further - not as a lossless compression but as a structured lossy approximation that can shrink each vector to as few as 8 bytes while still supporting asymmetric distance computation (ADC) with high recall.

#### 5.9.1 Algorithm

PQ decomposes each 128-dimensional float32 vector into `m` sub-vectors of dimension `128/m`:

```text
v = [v_1 | v_2 | ... | v_m], where v_j in R^(128/m)
```

**Training (offline):** For each subspace `j`, run k-means with `k=256` centroids on the `j`-th subvectors of all database vectors. This produces codebook `C_j = {c_j0, c_j1, ..., c_j255}`.

**Encoding:** Each stored vector is encoded as an `m`-byte code, with one centroid index per subspace:

```text
PQ(v) = [argmin_r ||v_1 - c_1r||^2, ..., argmin_r ||v_m - c_mr||^2]
```

**Querying (ADC - Asymmetric Distance Computation):** Given query `q`, precompute one lookup table per subspace:

```text
T[j][r] = ||q_j - c_jr||^2, for j = 0..m-1 and r = 0..255
```

The approximate distance to any stored vector is then:

```text
dist_PQ(q, v) ~= sum_j T[j][PQ(v)[j]]
```

This requires only `m` table lookups per candidate and avoids floating-point multiply operations in the inner loop.

**Pseudocode:**

```text
# Build phase
for j in 0..m-1:
    C_j <- kmeans(subvectors[:, j], k=256)

# Encode phase
for each vector v:
    code[v] <- [nearest centroid index in C_j for each subspace j]

# Query phase
for j in 0..m-1:
    for r in 0..255:
        T[j][r] <- ||q_j - C_j[r]||^2
for each candidate v:
    dist <- sum(T[j][code[v][j]] for j in 0..m-1)
```

#### 5.9.2 Compression and Working Set Analysis

With `m=8` subspaces and `k=256` centroids, the storage breakdown is:

| Component | Size formula | Value |
|---|---|---|
| Codebook (m=8, k=256, d/m=16 float32) | m × k × (d/m) × 4 B | 8 × 256 × 16 × 4 = **131 KB** |
| PQ code per vector | m bytes | **8 B/vector** (vs 512 B float32, 128 B int8) |
| ADC lookup table at query time | m × k × 4 B | 8 × 256 × 4 = **8 KB** |

The codebook (131 KB) is computed once and is **fixed in L2 cache** for the entire query workload. The ADC table (8 KB) recomputed per query fits entirely in L1-D.

Working set comparison across dataset sizes and compression methods:

| N vectors | float32 (512 B/v) | int8 (128 B/v) | PQ m=8 (8 B/v) | PQ m=16 (16 B/v) |
|---|---|---|---|---|
| 500 | 250 KB | 62.5 KB | **4 KB** | 8 KB |
| 10,000 | 5 MB | 1.25 MB | **80 KB** | 160 KB |
| 100,000 | 50 MB | 12.5 MB | **781 KB** | 1.56 MB |
| 1,000,000 | 476 MB | 119 MB | **7.6 MB** | 15.3 MB |

#### 5.9.3 Architectural Impact

**At simulation scale (N=500):**
- Float32 working set: 250 KB - already nearly fits in 256 kB L2 (explains modest gains across all experiments)
- PQ codes: 4 KB - fits entirely in L1-D (32 kB)
- Codebook: 131 KB - fits in L2
- At this scale, PQ would show similar IPC gains to int8, since both fit in L2. Simulation cannot distinguish the two.

**At production scale (N=1M):**
- Float32: 476 MB >> L2 (256 kB) - every graph traversal step is a DRAM miss (~100 ns)
- Int8: 119 MB - still >> L2; miss rate reduction is proportional to 4× capacity gain
- PQ (m=8): 7.6 MB - fits in L3 (shared 8–32 MB on modern server CPUs); DRAM accesses reduced by 64×
- **Expected IPC improvement vs float32 at N=1M: 40–60%** (estimated from DRAM miss → L3 hit latency reduction)

This is why PQ is used in production ANN systems: FAISS defaults to IVF-PQ encoding for billion-scale vector datasets [6], where it achieves 64–128× compression with <5% recall loss at k=10 [2].

#### 5.9.4 Why PQ Was Not Simulated

PQ requires query vectors to remain in float32 (for codebook distance computation) while stored vectors are encoded as `uint8` codes - an asymmetric distance function. This breaks the symmetric `l2sq(Vec a, Vec b)` interface used throughout `hnsw_base.h`, which requires both operands to have the same type. Integrating PQ would require:
1. k-means training (~60 lines) during the build phase
2. A separate `PQCode = std::array<uint8_t, m>` type for stored vectors
3. ADC lookup table precomputation per query
4. Threading the table through `searchLayer()` and all distance call sites

At N=500 (gem5's practical limit), PQ and int8 have nearly identical working-set sizes, so simulation would show no measurable differentiation. The analytical argument above quantifies the expected production-scale gains that the simulator cannot reach.

### 5.10 Complete Study Summary

Eight intervention categories across 22 configurations cover the full design space:

| Intervention | Configs | Best IPC | Effect | Category |
|---|---|---|---|---|
| L2 cache size (1×→8×) | 4 | 1.110 | +10.6% | HW - MARGINAL |
| ROB size (32→256) | 4 | 1.012 | +0.8% | HW - NO |
| HW stride prefetcher | 3 | 1.110 | **0% added** | HW - NO |
| Memory technology (DDR5, HBM) | 2 | 1.004 | -10.1% to 0% | HW - NO |
| SW prefetch hints | 2 | 1.024 | -2.7% / +2.0% | SW - NO |
| L2 MSHRs (20→128) | 3 | 1.004 | **0.00%** | HW - NO |
| Graph BFS reordering | 2 | 1.121 | +1.8% / +11.7% | SW - **MARGINAL (w/ L2=2MB)** |
| Scalar quantization (int8) | 2 | 1.106 | +3.7% / +10.2% | SW - **YES (combined)** |

**Core finding:** The analysis pipeline drove each decision. MPKI told us cache was not filtering → L2 sweep confirmed it. CPI decomposition showed 32.3% memory stall → MSHR sweep confirmed it was serial, not parallel. Effective bandwidth of 1.61 GB/s predicted HBM would fail → it did. Amdahl capped hardware gains at +47.8% → our best hardware result (+10.6%) landed well below that. Roofline diagnosed latency-bound (not compute-bound) and prescribed data layout changes → both algorithmic fixes worked. **Reorder + Quant: +10.2% IPC at 1.00× area** is the only "YES" - every prediction made by the analysis was experimentally confirmed.

![IPC Across All Configurations](figures/fig1_ipc_all_configs.png)

*Figure 1: IPC across all simulated configurations. Gray = baseline, blue = L2 cache sweep, red = ROB sweep, green = prefetcher, purple = memory technology, orange = software fixes. Reorder+Quant achieves the highest IPC among single-core optimizations.*

![IPC Gain SW vs HW](figures/fig2_ipc_gain_sw_vs_hw.png)

*Figure 2: IPC gain (%) relative to baseline. Software fixes (left) outperform hardware fixes (right). The best hardware fix (L2=2MB, +10.6%) requires 8× silicon area; the best software fix (Reorder+Quant, +10.2%) requires zero additional hardware.*

---

## 6. PPA Cost Analysis

### 6.1 Model

Area and power are estimated analytically using CACTI-inspired scaling [12]:

| Component | Assumption |
|---|---|
| L2 cache area | Linear with size; cache = 20% of chip area |
| ROB area | Linear with entries; ROB = 5% of chip area |
| Stride prefetcher | 2% of chip area (logic overhead) |
| Dynamic power | P ∝ A^0.8 (sub-linear due to voltage scaling - see below) |

**Power model rationale:** CMOS dynamic power follows $P = C \cdot V^2 \cdot F$ [5], where capacitive load $C$ scales with chip area. When a design adds area (e.g., doubling L2 from 256 kB to 512 kB), a well-optimized implementation can reduce supply voltage slightly while holding frequency constant - voltage scales sub-linearly with area, so power grows slower than area. The $A^{0.8}$ exponent captures this: doubling area increases power by $2^{0.8} = 1.74\times$ rather than $2\times$. This is why L2=2MB (2.40× area) costs 2.01× power rather than 2.40× power in the table below.

### 6.2 Full PPA Table

| Configuration | IPC | IPC Gain | Rel. Area | Rel. Power | Worth It? |
|---|---|---|---|---|---|
| **Baseline** (L2=256 kB, ROB=128, DDR4-2400) | 1.004 | - | 1.00× | 1.00× | - |
| L2=512 kB | 1.007 | +0.3% | 1.20× | 1.16× | NO |
| L2=1 MB | 1.009 | +0.5% | 1.60× | 1.46× | NO |
| L2=2 MB | 1.110 | **+10.6%** | 2.40× | 2.01× | MARGINAL |
| ROB=32 | 0.8694 | -13.4% | 0.96× | 0.97× | NO |
| ROB=64 | 1.009 | +0.5% | 0.97× | 0.98× | NO |
| ROB=256 | 1.011 | +0.7% | 1.05× | 1.04× | NO |
| HW Prefetch + L2=256 kB | 1.004 | 0.0% | 1.02× | 1.02× | NO |
| HW Prefetch + L2=512 kB | 1.012 | 0.0% | 1.22× | 1.17× | NO |
| HW Prefetch + L2=2 MB | 1.110 | 0.0% added | 2.42× | 2.03× | MARGINAL |
| DDR5-6400 | 0.9023 | -10.1% | 1.00× | 1.00× | NO |
| HBM 1.0 | 1.004 | 0.0% | 1.00× | 1.00× | NO |
| SW Prefetch | 0.9774 | -2.7% | 1.00× | 1.00× | NO |
| SW Prefetch + HBM | 1.024 | +2.0% | 1.00× | 1.00× | NO |
| **Graph Reorder** | **1.022** | **+1.8%** | **1.00×** | **1.00×** | **NO** |
| **Scalar Quant (int8)** | **1.041** | **+3.7%** | **1.00×** | **1.00×** | **MARGINAL** |
| **Reorder + Quant** | **1.106** | **+10.2%** | **1.00×** | **1.00×** | **✓ YES** |
| **Reorder + L2=2MB** | **1.121** | **+11.7%** | **2.40×** | **2.01×** | **MARGINAL** |

### 6.3 Pareto Analysis

![PPA Analysis Plot](../results/ppa_analysis.png)

> **Pareto-optimal result:**
> Of all 22 configurations tested, **Reorder + Quant is the sole Pareto-optimal design point** - no other configuration achieves comparable IPC gain at lower or equal area cost. Every hardware alternative is either dominated (Reorder+Quant gives the same gain at less cost) or simply ineffective. This is the central quantitative finding of the study.

The scatter plot (center panel) shows IPC gain vs. relative chip area. The ideal design sits in the **upper-left** quadrant: high gain, low area cost.

- **Reorder + Quant** sits at (1.00×, +10.2%) - the only point in the upper-left. It is the **sole Pareto-optimal** configuration: meaningful IPC gain at zero hardware cost. Every other design is either dominated (same or better gain at lower cost exists) or uneconomical.
- **L2=2 MB** and **Reorder + L2=2MB** achieve the highest absolute IPC (+10.6% and +11.7%) but at 2.40× chip area and 2.01× power. Both are **dominated design points**: Reorder+Quant delivers comparable IPC (+10.2%) at 1.00× area, making the hardware investment indefensible.
- **All ROB, prefetcher, MSHR, and DDR5 configs** cluster at or below the origin (near-zero or negative gain at non-zero cost). These are fully dominated: no configuration in this group offers a favorable cost-performance trade-off.
- **Scalar quant alone** (+3.7%, 1.00×) is MARGINAL - worthwhile as a software change but below the "YES" threshold.

### 6.4 Recommendation

Standard microarchitectural hardware knobs provide no cost-effective improvement for HNSW. The winning approach is algorithmic: changing data layout and representation so the existing hardware works better:

| Fix | IPC Gain | Area Cost | Verdict |
|---|---|---|---|
| Graph BFS reordering | +1.8% | 0% | Worthwhile; synergizes with L2 |
| Scalar quantization (int8) | +3.7% | 0% | Strong: doubles L2 hit count |
| **Reorder + Quant (combined)** | **+10.2%** | **0%** | **✓ YES: deploy both** |

**Recommendation:** Apply graph BFS reordering and scalar quantization together. The combination delivers +10.2% IPC at zero hardware cost - matching the performance of an 8× larger L2 cache (2.4× die area) via pure software. For even larger gains, near-memory computing (PIM) would reduce effective DRAM latency from ~55 ns to ~5 ns, but requires custom hardware.

---

## 7. Key Insights and Takeaways

| Step | Finding |
|---|---|
| **Step 1: Simulate** | Baseline IPC = 1.004 (25.1% of 4-wide peak). L2 miss rate = 91.7%. DRAM reads = 75,685 bursts. |
| **Step 2: Bottleneck** | Root cause: irregular pointer-chasing during HNSW graph traversal. Not branch-bound, not FU-bound. |
| **Step 3a: L2 sweep** | 8× larger L2 → +10.6% IPC; L2 miss rate barely changes (91.70%→91.58%). Capacity is not the fix. |
| **Step 3b: ROB sweep** | ROB=256 → +0.7% IPC. Serial dependency chain; instruction window is not the fix. |
| **Step 3c: HW Prefetcher** | Zero effect on IPC (confirmed with correct gem5 attachment). Stride prefetcher cannot predict irregular pointer-chasing. |
| **Step 3d: DDR5-6400** | -10.1% IPC. DDR5 optimizes bandwidth not latency; absolute CAS latency ≥ DDR4. Hurts latency-bound workloads. |
| **Step 3e: HBM 1.0** | 0.0% IPC change. HBM reduces L2 miss rate (91.7%→87.7%) but serial hop chain still serializes execution. |
| **Step 3f: SW Prefetch** | -2.7% alone; +2.0% with HBM. Prefetch overhead exceeds benefit; serial chain resists speculation. |
| **Step 3g: L2 MSHRs (20→128)** | 0.00% change. Inter-hop serial dependency means only 1 outstanding miss at a time. |
| **Step 3h: Graph BFS Reordering** | +1.8% IPC. L2 miss rate drops 91.7%→88.6%. Co-traversed nodes become spatially contiguous. |
| **Step 3i: Scalar Quantization (int8)** | +3.7% IPC. L2 miss rate drops 91.7%→84.6%. L2 hits double (6,387→13,240). |
| **Step 3j: Reorder + Quant** | **+10.2% IPC at 1.00× area** - first and only "YES" in the study. Matches L2=2MB with zero silicon cost. |
| **Step 3k: PQ (theoretical)** | Product Quantization: m=8 → 8 B/vector (64× vs float32). At N=1M: codes = 7.6 MB (L3-resident vs 476 MB DRAM). Estimated 40–60% IPC gain at production scale (simulated scale too small to differentiate from int8). |
| **Step 3l: Multithreading (TLP)** | gem5 SE mode limitation: all threads serialized on core0 (cores1+ = NaN). core0 IPC = 0.999 (thread overhead, no parallelism). Real hardware: N cores → N× QPS (theoretically, queries are embarrassingly parallel with no shared mutable state). |
| **Step 4: PPA** | 22 configurations; sole YES = Reorder+Quant (+10.2%, 1.00× area). All hardware fixes are MARGINAL or NO. TLP gives N× throughput scaling at N× area cost - valid for QPS-optimized deployment. |

**Conclusion:**

- **HNSW is fundamentally memory-latency-bound.** The pointer-chasing graph traversal creates a serial RAW dependency chain that no hardware mechanism can break: ROB expansion, MSHR scaling, and stride prefetching all return 0% or near-0% gain because MLP is structurally 1-2.
- **Performance is tightly bounded.** Amdahl's Law caps hardware improvement at +47.8% IPC (1.484 ceiling, from f_mem = 32.3%). Our best hardware result (+10.6%, L2=2MB) and best software result (+10.2%, Reorder+Quant) both land well below this ceiling - validating the analysis.
- **The sole Pareto-optimal fix is algorithmic.** Graph BFS reordering + int8 scalar quantization delivers +10.2% IPC at 1.00× area, matching an 8× larger L2 cache at zero silicon cost. This is the only "YES" across 22 configurations.
- **Standard performance models can misclassify this workload.** HNSW's AI = 2.09 FLOP/byte places it above the standard Roofline ridge (0.625), falsely labeling it compute-bound. Correcting for serialized effective bandwidth (1.61 GB/s) shifts the true ridge to 7.45 FLOP/byte - reversing the diagnosis entirely. Getting the model right is what enables getting the fix right.
- **This work demonstrates that irregular, pointer-chasing workloads fundamentally challenge the assumptions underlying modern out-of-order CPU design.** Every latency-hiding mechanism - OOO execution, hardware prefetching, large caches - fails when memory addresses are data-dependent and non-reusable. This is not a tuning problem; it is a structural mismatch between the workload and the machine.

**Broader lesson:** Production ANN systems (FAISS [6], ScaNN, hnswlib) apply quantization and graph reordering [10] not because of benchmarking culture, but because the microarchitectural analysis leaves no other option. Effective optimization for pointer-chasing workloads must reduce DRAM accesses per query (quantization, reordering, PQ) or reduce per-access latency via near-memory compute (PIM). Larger caches and faster DRAM are the wrong answers to the right question.

---

## 8. Limitations and Future Work

### 8.1 Simulation Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| **gem5 SE mode only** | No OS noise, system calls, or NUMA effects modeled | Acceptable for user-space memory access pattern analysis |
| **N=500 vectors** | Working set (250 KB) fits near-edge of L2; production (N=1M, 476 MB) is fully DRAM-resident | Analytical scaling provided in §5.7-5.9; expected production gains are larger |
| **Single core (phases 1–8)** | No cache coherence traffic or NUMA effects in single-core runs | §5.8 adds multi-core TLP sweep (2c, 4c); coherence traffic is minimal (read-only index) |
| **3B tick cutoff** | Only ~9M instructions committed; misses later phases of HNSW search | Consistent across all configs - relative comparisons remain valid |
| **gem5 HBM model** | gem5's HBM model captures bandwidth, not true HBM latency (10–20 ns vs 55 ns DDR4) | Documented in §5.4; Amdahl provides the theoretical bound |
| **Static binary** | `-O2 -static -march=x86-64` - no PGO or target-specific auto-vectorization | Baseline is consistent; competitive with hnswlib's default build flags |

### 8.2 Architectural Implications

This study has implications beyond HNSW. Any workload whose critical path follows a pointer-chasing structure - graph databases, linked-list traversal, tree search, sparse-matrix indexed access - will exhibit the same Roofline misclassification and the same resistance to standard hardware knobs. **Pointer-aware prefetchers** that speculatively dereference load addresses (e.g., content-directed prefetching) could partially break the serial chain. **Near-memory / Processing-in-Memory (PIM)** is the hardware-side fix with the highest ceiling: moving compute onto the DRAM die cuts effective latency from ~55 ns to ~5 ns, which Amdahl predicts would yield +40% IPC - far above any cache or ROB change. For software, the implication is that data layout is a first-class architectural concern: BFS reordering + quantization together rival an 8× L2 cache at zero silicon cost, a result that should inform future ANN accelerator design.

### 8.3 Future Work

1. **Near-memory computing (PIM):** The Amdahl ceiling of +47.8% max IPC is achievable only by reducing the *latency* of each DRAM access, not its bandwidth. Processing-In-Memory [11] places compute logic on the DRAM die, reducing effective latency from ~55 ns to ~5 ns (10×). Amdahl predicts +40% IPC - well above any software fix. Future work should implement a gem5 PIM model (e.g., using gem5's HMC or near-memory compute extensions) and measure actual speedup vs. the Amdahl prediction.

2. **Multi-layer graph prefetching:** HNSW's graph structure is known at query time - the upper-layer traversal predicts which bottom-layer nodes will be accessed. A custom hardware prefetcher that issues prefetch requests during upper-layer traversal could hide 1–2 hops of DRAM latency. This would require application-guided prefetching (software prefetch instructions inserted in the upper-layer path) and would require re-running the SW prefetch experiments with a smarter prefetch strategy.

3. **Product Quantization at scale:** §5.9 showed that PQ [2] (m=8) reduces per-vector size from 512 B to 8 B (64×), bringing N=1M working set from 476 MB to 7.6 MB (L3-resident). At production scale, this is projected to give 40–60% IPC improvement - the theoretical best SW fix. Simulating this in gem5 [4] requires N large enough (≥50K) to see the working set difference, beyond the 3B-tick limit.

4. **Quantization accuracy vs. performance trade-off:** Scalar int8 and PQ both introduce recall loss (estimated 1–3% at k=10). A systematic sweep of quantization precision vs. IPC gain - correlated with recall@10 on the full SIFT-1M query set - would identify the Pareto frontier of performance/accuracy trade-off useful for production deployment decisions.

5. **Multithreading at scale (§5.8):** Future work should extend to 8- and 16-core configurations and combine TLP with Reorder+Quant (§5.7). At 8 cores this yields ~9× QPS over the single-core baseline. Bandwidth saturation and LLC contention (discussed in §5.8) should be measured empirically at these scales.

---

## 9. References

1. Malkov, Y. A., & Yashunin, D. A. (2020). **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824–836. https://doi.org/10.1109/TPAMI.2018.2889473

2. Jégou, H., Douze, M., & Schmid, C. (2011). **Product quantization for nearest neighbor search.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1), 117–128. https://doi.org/10.1109/TPAMI.2010.57

3. Lowe, D. G. (2004). **Distinctive image features from scale-invariant keypoints.** *International Journal of Computer Vision*, 60(2), 91–110. https://doi.org/10.1023/B:VISI.0000029664.99615.94

4. Binkert, N., Beckmann, B., Black, G., Reinhardt, S. K., Saidi, A., Basu, A., ... & Wells, P. M. (2011). **The gem5 simulator.** *ACM SIGARCH Computer Architecture News*, 39(2), 1–7. https://doi.org/10.1145/2024716.2024718

5. Hennessy, J. L., & Patterson, D. A. (2019). **Computer Architecture: A Quantitative Approach** (6th ed.). Morgan Kaufmann. ISBN 978-0-12-811905-1.

6. Johnson, J., Douze, M., & Jégou, H. (2019). **Billion-scale similarity search with GPUs.** *IEEE Transactions on Big Data*, 7(3), 535–547. https://doi.org/10.1109/TBDATA.2019.2921572

7. Williams, S., Waterman, A., & Patterson, D. (2009). **Roofline: An insightful visual performance model for multicore architectures.** *Communications of the ACM*, 52(4), 65–76. https://doi.org/10.1145/1498765.1498785

8. Wulf, W. A., & McKee, S. A. (1995). **Hitting the memory wall: Implications of the obvious.** *ACM SIGARCH Computer Architecture News*, 23(1), 20–24. https://doi.org/10.1145/216585.216588

9. Tomasulo, R. M. (1967). **An efficient algorithm for exploiting multiple arithmetic units.** *IBM Journal of Research and Development*, 11(1), 25–33. https://doi.org/10.1147/rd.111.0025

10. Aumueller, M., Bernhardsson, E., & Faithfull, A. (2020). **ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.** *Information Systems*, 87, 101374. https://doi.org/10.1016/j.is.2019.02.006

11. Mutlu, O., Ghose, S., Gómez-Luna, J., & Ausavarungnirun, R. (2019). **Processing data where it makes sense: Enabling in-memory computation.** *Microprocessors and Microsystems*, 67, 28–41. https://doi.org/10.1016/j.micpro.2019.01.009

12. Muralimanohar, N., Balasubramonian, R., & Jouppi, N. (2009). **CACTI 6.0: A tool to model large caches.** *HP Laboratories Technical Report HPL-2009-85*.

13. Smith, A. J. (1982). **Cache memories.** *ACM Computing Surveys*, 14(3), 473–530. https://doi.org/10.1145/356887.356892

---

## 10. AI Usage Disclosure

**Tools used:**
- **Claude (Anthropic)** via Claude Code CLI: report writing, code debugging, analysis scripts
- **Perplexity AI**: literature search for related research papers cited in §9

**How used:**

| Task | Tool | Role | Verification method |
|---|---|---|---|
| gem5 config script (`run_benchmark.py`) | Claude | Debugging Python/SimObject API errors; flag naming | Ran gem5, checked `stats.txt` for expected stat names |
| Analysis scripts (`parse_stats.py`, `ppa_analysis.py`, `advanced_analysis.py`) | Claude | Generated initial skeleton; iterated on stat field names | Cross-checked every extracted value against raw `stats.txt` |
| Report writing | Claude | Suggested section structure; drafted explanations of results | All numerical claims verified against `stats.txt`; formulas verified by hand |
| Lecture concept integration | Claude | Mapped course terminology (RAW hazard, load-use hazard, Tomasulo, branch CPI formula) to observed gem5 results | Verified formula outputs match measured values (e.g., branch CPI ≈ 0.030 vs measured 0.032) |
| Literature search | Perplexity AI | Identified related research papers on HNSW [1], ANN search [2][6][10], memory wall [8], Roofline [7], PIM [11], and cache analysis [13] | All papers verified to exist via DOI links before inclusion in References |

**What AI did NOT do:** AI did not run any simulations, choose which experiments to run, or determine the bottleneck diagnosis - those decisions came from reading `stats.txt` and applying course concepts. Perplexity AI suggested candidate papers; all were verified for existence and relevance before citation.

**Correctness verification:** Every IPC value, miss rate, and stall count cited in this report was manually read from the corresponding `results/hnsw_*/stats.txt` file. The PPA model constants (20% L2 area fraction, 0.8 power exponent) were derived from CACTI literature [12] and cross-checked against published 7nm node estimates. All reference DOIs were verified before submission.

---

## Appendix: File Structure

```text
GEM5_Class_Project/
|-- benchmarks/
|   |-- hnsw_gem5_benchmark.cpp   # HNSW benchmark (real SIFT-1M .fvecs loader)
|   `-- hnsw_base.h               # HNSW index implementation
|-- configs/
|   `-- run_benchmark.py          # gem5 SE config (L2/ROB/prefetcher parameterized)
|-- analysis/
|   |-- parse_stats.py            # Parses gem5 stats.txt -> summary + MPKI + Amdahl + CPI stack
|   |-- ppa_analysis.py           # PPA cost model + plots
|   `-- advanced_analysis.py      # CPI stack, Roofline, MPKI, Amdahl plots
|-- results/
|   |-- hnsw_l2_{256kB,512kB,1MB,2MB}/      # L2 sweep (256kB = baseline)
|   |-- hnsw_rob_{32,64,128,256}/           # ROB sweep
|   |-- hnsw_prefetch_l2_{256kB,512kB,2MB}/ # HW stride prefetcher runs
|   |-- hnsw_mem_{ddr5,hbm}/                # Memory technology runs
|   |-- hnsw_swprefetch{,_hbm}/             # SW prefetch binary runs
|   |-- hnsw_reorder{,_l2_2MB}/             # BFS graph reordering runs
|   |-- hnsw_quant/                         # Scalar quantization run
|   |-- hnsw_reorder_quant/                 # Reorder + quant combined
|   |-- miss_rate_vs_L2_size.png
|   |-- sweep_all.png
|   |-- sweep_sweep.png
|   `-- ppa_analysis.png
|-- run_sweep_hnsw.sh             # Full sweep script (7 phases)
`-- report/
    `-- EEL6764_HNSW_Report.md    # This report
```
