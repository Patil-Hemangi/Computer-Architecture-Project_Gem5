// =============================================================================
// hnsw_gem5_benchmark.cpp  —  HNSW benchmark for gem5 SE mode
//
// Loads real SIFT-1M vectors from .fvecs files.
// Files must be accessible at the path given as argv[3] (default:
//   /workspace/bigann/sift/)
//
// Build (inside container):
//   Baseline:
//     g++ -O2 -static -march=x86-64 -std=c++17 -o hnsw_gem5 hnsw_gem5_benchmark.cpp
//   SW prefetch hints:
//     g++ -O2 -static -march=x86-64 -std=c++17 -DSW_PREFETCH -o hnsw_gem5_prefetch hnsw_gem5_benchmark.cpp
//   Graph BFS reordering (Iteration 6):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DGRAPH_REORDER -o hnsw_gem5_reorder hnsw_gem5_benchmark.cpp
//   Scalar quantization float32→int8 (Iteration 7):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DSCALAR_QUANT -o hnsw_gem5_quant hnsw_gem5_benchmark.cpp
//   Reorder + quantization combined:
//     g++ -O2 -static -march=x86-64 -std=c++17 -DGRAPH_REORDER -DSCALAR_QUANT -o hnsw_gem5_rq hnsw_gem5_benchmark.cpp
//   Multithreaded query dispatch (Iteration 9 — TLP):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DMULTITHREAD -pthread -o hnsw_gem5_mt hnsw_gem5_benchmark.cpp
//   argv: <numBase> <numQueries> <dataDir> [numThreads=1]
//
// Run natively:
//   ./hnsw_gem5 500 20 /workspace/bigann/sift/
//
// Run in gem5 (single-core baseline):
//   gem5.opt configs/run_benchmark.py
//       --binary benchmarks/hnsw_gem5
//       --bin-args "500 20 /workspace/bigann/sift/"
//
// Run in gem5 (multithreaded, 4 cores):
//   gem5.opt configs/run_benchmark.py
//       --binary benchmarks/hnsw_gem5_mt
//       --bin-args "500 20 /workspace/bigann/sift/ 4"
//       --num-cores 4
// =============================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef MULTITHREAD
#include <thread>
#endif

#include "hnsw_base.h"

// ---------------------------------------------------------------------------
// .fvecs loader — always returns FloatVec regardless of compile-time Vec type.
// Format: [dim:int32][f0:float32]...[f_{dim-1}:float32] repeated per vector
// Quantization (if enabled) happens after loading, not here.
// ---------------------------------------------------------------------------
static std::vector<FloatVec> loadFvecs(const char* path, int maxVecs) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("[error] Cannot open: %s\n", path);
        std::exit(1);
    }

    std::vector<FloatVec> vecs;
    vecs.reserve(maxVecs);

    for (int i = 0; i < maxVecs; ++i) {
        int dim = 0;
        if (fread(&dim, sizeof(int), 1, f) != 1) break;
        if (dim != kDim) {
            printf("[error] Expected dim=%d, got %d in %s\n", kDim, dim, path);
            std::exit(1);
        }
        FloatVec v;
        if (fread(v.data(), sizeof(float), kDim, f) != (size_t)kDim) break;
        vecs.push_back(v);
    }

    fclose(f);
    printf("    Loaded %d vectors from %s\n", (int)vecs.size(), path);
    return vecs;
}

#ifdef SCALAR_QUANT
// ---------------------------------------------------------------------------
// Scalar quantization: float32 → int8  (512 B/vec → 128 B/vec, 4× smaller)
//
// Global max-abs scale so the full dynamic range maps to [-127, 127].
// All vectors share one scale factor — simple, lossless in range, slightly
// lossy in precision. Sufficient for approximate nearest-neighbor search.
// ---------------------------------------------------------------------------
static std::vector<Vec> quantize(const std::vector<FloatVec>& fvecs) {
    float maxAbs = 1e-9f;
    for (const auto& v : fvecs)
        for (float x : v)
            if (std::fabs(x) > maxAbs) maxAbs = std::fabs(x);

    const float scale = 127.0f / maxAbs;
    printf("    Quantizing %zu vectors: scale=%.4f  (%zu B/vec → %zu B/vec)\n",
           fvecs.size(), scale, sizeof(FloatVec), sizeof(Vec));

    std::vector<Vec> out(fvecs.size());
    for (size_t i = 0; i < fvecs.size(); ++i)
        for (int d = 0; d < kDim; ++d) {
            float s = fvecs[i][d] * scale;
            out[i][d] = (int8_t)std::max(-127.0f, std::min(127.0f, std::roundf(s)));
        }
    return out;
}
#endif

int main(int argc, char** argv) {
    int numBase    = (argc > 1) ? std::atoi(argv[1]) : 500;
    int numQueries = (argc > 2) ? std::atoi(argv[2]) : 20;
    const char* dataDir  = (argc > 3) ? argv[3] : "/workspace/bigann/sift/";
#ifdef MULTITHREAD
    int numThreads = (argc > 4) ? std::atoi(argv[4]) : 1;
#endif

    // Build file paths
    char basePath[512], queryPath[512];
    std::snprintf(basePath,  sizeof(basePath),  "%ssift_base.fvecs",  dataDir);
    std::snprintf(queryPath, sizeof(queryPath), "%ssift_query.fvecs", dataDir);

    printf("=== HNSW Benchmark for GEM5 (SIFT-1M) ===\n");
    printf("Base: %d  Queries: %d\n", numBase, numQueries);
    printf("Data dir: %s\n\n", dataDir);

    // -------------------------------------------------------------------------
    // [1] Load SIFT vectors (always as float32 from .fvecs format)
    // -------------------------------------------------------------------------
    printf("[1] Loading SIFT vectors...\n");
    std::vector<FloatVec> floatBase    = loadFvecs(basePath,  numBase);
    std::vector<FloatVec> floatQueries = loadFvecs(queryPath, numQueries);

#ifdef SCALAR_QUANT
    // Convert float32 → int8 (4× size reduction: 512 B/vec → 128 B/vec)
    std::vector<Vec> base    = quantize(floatBase);
    std::vector<Vec> queries = quantize(floatQueries);
    floatBase.clear();    floatBase.shrink_to_fit();
    floatQueries.clear(); floatQueries.shrink_to_fit();
#else
    // Vec == FloatVec when not quantizing — no copy needed
    std::vector<Vec>& base    = floatBase;
    std::vector<Vec>& queries = floatQueries;
#endif

    printf("    Done.  Vector storage: %zu B/vec\n\n", sizeof(Vec));

    // -------------------------------------------------------------------------
    // [2] Build HNSW index
    // -------------------------------------------------------------------------
    printf("[2] Building HNSW index (M=16, Mmax0=32, efC=100)...\n");
    HNSWIndex index(16, 32, 100);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < (int)base.size(); ++i) {
        index.insert(base[i]);
        if ((i + 1) % 1000 == 0)
            printf("    Inserted %d / %d\n", i + 1, (int)base.size());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double buildSec = std::chrono::duration<double>(t1 - t0).count();
    printf("    Build: %.3f sec  (%.0f inserts/sec)\n\n",
           buildSec, base.size() / buildSec);

    const int K        = 10;   // top-K neighbors
    const int efSearch = 50;   // beam width for HNSW search

    // -------------------------------------------------------------------------
    // [3] Build self-consistent ground truth via brute-force BEFORE reordering.
    //     Must run here: reorderByBFS() remaps node IDs, so GT computed after
    //     reorder would use mismatched IDs and give incorrect recall values.
    //     base[i] → GT records original index i; searchKnn returns these same
    //     indices only when the index has not yet been reordered.
    // -------------------------------------------------------------------------
    printf("[3] Computing brute-force ground truth (N=%d, Q=%d, K=%d)...\n",
           numBase, (int)queries.size(), K);
    std::vector<std::vector<int>> groundTruth(queries.size());
    {
        for (int q = 0; q < (int)queries.size(); ++q) {
            std::vector<std::pair<float,int>> dists;
            dists.reserve(base.size());
            for (int i = 0; i < (int)base.size(); ++i)
                dists.push_back({l2sq(queries[q], base[i]), i});
            std::partial_sort(dists.begin(), dists.begin() + K, dists.end());
            groundTruth[q].resize(K);
            for (int k = 0; k < K; ++k)
                groundTruth[q][k] = dists[k].second;
        }
    }
    printf("    Done.\n\n");

    // Free base vectors — no longer needed after GT computation and index build
    { std::vector<Vec>().swap(base); }

#ifdef GRAPH_REORDER
    // newToOld[newId] = originalInsertionIndex (== base[] index used in GT)
    std::vector<int> newToOld;
    auto tr = std::chrono::high_resolution_clock::now();
    index.reorderByBFS(&newToOld);
    auto te2 = std::chrono::high_resolution_clock::now();
    printf("    BFS reorder: %.3f sec\n\n",
           std::chrono::duration<double>(te2 - tr).count());
#endif

    // -------------------------------------------------------------------------
    // [4] Search
    // -------------------------------------------------------------------------
    int nq = (int)queries.size();
    std::vector<std::vector<HNSWIndex::Result>> results(nq);

#ifdef MULTITHREAD
    // Query-level (TLP) parallelism: independent queries dispatched across threads.
    // searchKnn is thread-safe — visitedGen/curGen are thread_local in hnsw_base.h.
    // The index (nodes_, entryPoint_, maxLevel_) is read-only during search.
    if (numThreads < 1) numThreads = 1;
    printf("[4] Querying (%d queries, K=%d, efSearch=%d, threads=%d)...\n",
           nq, K, efSearch, numThreads);

    auto ts = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::thread> workers;
        workers.reserve(numThreads);
        int chunk = (nq + numThreads - 1) / numThreads;
        for (int t = 0; t < numThreads; ++t) {
            int start = t * chunk;
            int end   = std::min(start + chunk, nq);
            if (start >= end) break;
            workers.emplace_back([&, start, end]() {
                for (int q = start; q < end; ++q)
                    results[q] = index.searchKnn(queries[q], K, efSearch);
            });
        }
        for (auto& w : workers) w.join();
    }
    auto te = std::chrono::high_resolution_clock::now();
#else
    printf("[4] Querying (%d queries, K=%d, efSearch=%d)...\n",
           nq, K, efSearch);

    auto ts = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < nq; ++q)
        results[q] = index.searchKnn(queries[q], K, efSearch);
    auto te = std::chrono::high_resolution_clock::now();
#endif

    double searchSec = std::chrono::duration<double>(te - ts).count();
    printf("    Search: %.3f sec  (%.0f QPS)\n\n",
           searchSec, nq / searchSec);

    // -------------------------------------------------------------------------
    // [5] Recall@K
    // -------------------------------------------------------------------------
    if (!groundTruth.empty()) {
        int hits = 0, total = 0;
        for (int q = 0; q < nq && q < (int)groundTruth.size(); ++q) {
            const auto& gt = groundTruth[q];
            int gtK = std::min((int)gt.size(), K);
            for (const auto& r : results[q]) {
                // Translate returned ID back to original insertion index.
                // After reorderByBFS, newToOld[newId] == original base[] index.
#ifdef GRAPH_REORDER
                int origId = newToOld[r.id];
#else
                int origId = r.id;
#endif
                for (int j = 0; j < gtK; ++j) {
                    if (origId == gt[j]) { ++hits; break; }
                }
            }
            total += gtK;
        }
        double recall = (total > 0) ? (100.0 * hits / total) : 0.0;
        printf("[5] Recall@%d = %.2f%%  (%d / %d correct)\n\n", K, recall, hits, total);
    } else {
        printf("[5] Recall@%d = N/A  (ground truth file not found)\n\n", K);
    }

    // -------------------------------------------------------------------------
    // [6] Summary
    // -------------------------------------------------------------------------
    printf("Index: nodes=%d  entry=%d  maxLevel=%d\n",
           index.size(), index.entryPoint(), index.maxLevel());
    printf("=== Benchmark Complete ===\n");
    return 0;
}
