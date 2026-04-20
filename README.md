# Microarchitectural Analysis of HNSW Vector Search on gem5

Course project for EEL6764 (Computer Architecture), University of South Florida, Spring 2026.

## Overview

This project studies the microarchitectural behavior of HNSW (Hierarchical Navigable Small World) vector search using gem5 full-system simulation. The goal is to understand why HNSW underutilizes modern out-of-order CPUs and to evaluate whether hardware changes or algorithmic optimizations improve performance more effectively.

The core finding is that HNSW is primarily limited by memory latency from irregular pointer-chasing accesses, not by floating-point throughput or raw memory bandwidth. Across the explored design space, standard hardware scaling provides limited gains, while software/data-layout optimizations such as graph reordering and scalar quantization offer better performance-per-cost tradeoffs.

## Team

- Hemangi Patil
- Rishil Shah
- Roberto Perez
- Rima El Brouzi

## Repository Structure

- `benchmarks/` - HNSW benchmark source and build inputs
- `configs/` - gem5 simulation configuration scripts
- `results/` - collected gem5 outputs and sweep results
- `analysis/` - scripts for parsing statistics and generating plots
- `report/` - project report sources
- `bigann/` - dataset-related inputs used by the benchmark

## Workload

The benchmark loads SIFT-style vectors, builds an HNSW index, and runs K-NN search under gem5 using an out-of-order x86 CPU model. The simulated subset is reduced for feasibility, but it preserves the access patterns that matter for architectural analysis:

- irregular graph traversal with pointer-chasing memory accesses
- regular distance computation over 128-dimensional vectors

## Simulator Setup

Baseline simulation configuration:

- gem5 v24.1 SE mode, x86 ISA
- `X86O3CPU` at 3 GHz
- private 32 kB L1 instruction and data caches
- private 256 kB L2 cache baseline
- DDR4 memory baseline

Sweeps in this repo explore changes such as:

- L2 cache size
- reorder buffer size
- prefetching
- memory type
- MSHR count
- software-prefetch, graph reordering, and quantization variants

## Main Results

- HNSW behaves as a memory-latency-bound workload.
- Increasing conventional hardware resources gives only modest gains.
- Algorithmic reductions in data movement provide the best cost-effective improvements.
- The project compares performance using IPC/CPI analysis, stall breakdowns, and power-performance-area tradeoffs.

## Running the Project

Exact build and run steps depend on your local gem5 setup, but the repo workflow is organized around:

1. Building the benchmark variant in `benchmarks/`
2. Running gem5 through `configs/run_benchmark.py` or `run_sweep_hnsw.sh`
3. Parsing and plotting results with scripts in `analysis/`

## Report

The primary written report is:

- `report/EEL6764_HNSW_Report.md`

## Notes

- Large generated outputs are stored under `results/`.
- This repository contains both source code and experiment artifacts used for the final analysis.
