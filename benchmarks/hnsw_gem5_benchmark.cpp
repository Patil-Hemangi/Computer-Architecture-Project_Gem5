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
//   Packed search graph (freeze heap-based adjacency into contiguous arrays):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DPACKED_SEARCH -o hnsw_gem5_packed hnsw_gem5_benchmark.cpp
//   Deep combined layout fix (packed graph + reorder + quantization):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DPACKED_SEARCH -DGRAPH_REORDER -DSCALAR_QUANT -o hnsw_gem5_deep hnsw_gem5_benchmark.cpp
//   Quantized search + exact rerank (compress hot path, load float only for shortlist):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DSCALAR_QUANT -DHYBRID_RERANK -o hnsw_gem5_hybrid hnsw_gem5_benchmark.cpp
//   Deepest variant in this repo: packed graph + quantized search + exact rerank + reorder
//     g++ -O2 -static -march=x86-64 -std=c++17 -DPACKED_SEARCH -DGRAPH_REORDER -DSCALAR_QUANT -DHYBRID_RERANK -o hnsw_gem5_hybrid_deep hnsw_gem5_benchmark.cpp
//   Multithreaded query dispatch (Iteration 9 — TLP):
//     g++ -O2 -static -march=x86-64 -std=c++17 -DMULTITHREAD -pthread -o hnsw_gem5_mt hnsw_gem5_benchmark.cpp
//   argv (single-thread): <numBase> <numQueries> <dataDir> [mode] [efSearch] [K] [rerankWidth]
//     mode:
//       full              = build + GT + search (legacy behavior)
//       search_roi        = reset/dump gem5 stats around search only
//       search_roi_nogt   = search ROI + skip brute-force GT
//       search_only_nogt  = skip brute-force GT, no ROI markers
//     efSearch:
//       beam width for HNSW search (default 50)
//     K:
//       top-K results returned (default 10)
//     rerankWidth:
//       HYBRID_RERANK only. Coarse candidate count reranked with full float vectors.
//       Default: max(4*K, efSearch)
//   argv (multithread): <numBase> <numQueries> <dataDir> [numThreads=1]
//
// Run natively:
//   ./hnsw_gem5 500 20 /workspace/bigann/sift/ search_roi_nogt 50 10
//
// Run in gem5 (single-core baseline):
//   gem5.opt configs/run_benchmark.py
//       --binary benchmarks/hnsw_gem5
//       --bin-args "500 20 /workspace/bigann/sift/ search_roi_nogt"
//
// Run in gem5 (multithreaded, 4 cores):
//   gem5.opt configs/run_benchmark.py
//       --binary benchmarks/hnsw_gem5_mt
//       --bin-args "500 20 /workspace/bigann/sift/ 4"
//       --num-cores 4
// =============================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef MULTITHREAD
#include <thread>
#endif

#if __has_include(<gem5/m5ops.h>)
#include <gem5/m5ops.h>
#define HNSW_HAS_M5OPS 1
#else
#define HNSW_HAS_M5OPS 0
static inline void m5_work_begin(uint64_t, uint64_t) {}
static inline void m5_work_end(uint64_t, uint64_t) {}
static inline void m5_reset_stats(uint64_t, uint64_t) {}
static inline void m5_dump_stats(uint64_t, uint64_t) {}
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

struct RunOptions {
    bool useSearchROI = false;
    bool skipGroundTruth = false;
};

static RunOptions parseRunMode(const char* mode) {
    RunOptions opts;
    if (!mode || std::strcmp(mode, "full") == 0) return opts;

    if (std::strcmp(mode, "search_roi") == 0) {
        opts.useSearchROI = true;
    } else if (std::strcmp(mode, "search_roi_nogt") == 0) {
        opts.useSearchROI = true;
        opts.skipGroundTruth = true;
    } else if (std::strcmp(mode, "search_only_nogt") == 0) {
        opts.skipGroundTruth = true;
    } else {
        printf("[warn] Unknown mode '%s' -- using full mode\n", mode);
    }
    return opts;
}

#ifdef SCALAR_QUANT
// ---------------------------------------------------------------------------
// Scalar quantization: float32 → int8  (512 B/vec → 128 B/vec, 4× smaller)
//
// Global max-abs scale so the full dynamic range maps to [-127, 127].
// All vectors share one scale factor — simple, lossless in range, slightly
// lossy in precision. Sufficient for approximate nearest-neighbor search.
// ---------------------------------------------------------------------------
struct ScalarQuantizer {
    float scale = 1.0f;

    static ScalarQuantizer fit(const std::vector<FloatVec>& base,
                               const std::vector<FloatVec>& queries) {
        float maxAbs = 1e-9f;
        auto updateMax = [&](const std::vector<FloatVec>& vecs) {
            for (const auto& v : vecs)
                for (float x : v)
                    if (std::fabs(x) > maxAbs) maxAbs = std::fabs(x);
        };
        updateMax(base);
        updateMax(queries);

        ScalarQuantizer q;
        q.scale = 127.0f / maxAbs;
        return q;
    }

    Vec quantizeOne(const FloatVec& src) const {
        Vec out;
        for (int d = 0; d < kDim; ++d) {
            float s = src[d] * scale;
            out[d] = (int8_t)std::max(-127.0f, std::min(127.0f, std::roundf(s)));
        }
        return out;
    }

    std::vector<Vec> quantizeAll(const std::vector<FloatVec>& fvecs) const {
        std::vector<Vec> out(fvecs.size());
        for (size_t i = 0; i < fvecs.size(); ++i)
            out[i] = quantizeOne(fvecs[i]);
        return out;
    }
};

static float l2sqFloat(const FloatVec& a, const FloatVec& b) {
    float sum = 0.0f;
    for (int i = 0; i < kDim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static std::vector<HNSWIndex::Result> exactRerank(
        const std::vector<HNSWIndex::Result>& coarse,
        const FloatVec& query,
        const std::vector<FloatVec>& baseFloat,
        int k,
#ifdef GRAPH_REORDER
        const std::vector<int>* newToOld
#else
        const std::vector<int>* /*newToOld*/
#endif
) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(coarse.size());
    for (const auto& cand : coarse) {
        int origId = cand.id;
#ifdef GRAPH_REORDER
        origId = (*newToOld)[cand.id];
#endif
        ranked.push_back(std::make_pair(l2sqFloat(query, baseFloat[origId]), cand.id));
    }
    std::sort(ranked.begin(), ranked.end());

    std::vector<HNSWIndex::Result> out;
    out.reserve(k);
    for (int i = 0; i < k && i < (int)ranked.size(); ++i)
        out.push_back({ ranked[i].second, ranked[i].first });
    return out;
}
#endif

int main(int argc, char** argv) {
    int numBase    = (argc > 1) ? std::atoi(argv[1]) : 500;
    int numQueries = (argc > 2) ? std::atoi(argv[2]) : 20;
    const char* dataDir  = (argc > 3) ? argv[3] : "/workspace/bigann/sift/";
    int efSearch = 50;
    int K        = 10;
    int rerankWidth = 0;
#ifdef MULTITHREAD
    int numThreads = (argc > 4) ? std::atoi(argv[4]) : 1;
#else
    const char* mode = (argc > 4) ? argv[4] : "full";
    efSearch = (argc > 5) ? std::atoi(argv[5]) : 50;
    K        = (argc > 6) ? std::atoi(argv[6]) : 10;
    rerankWidth = (argc > 7) ? std::atoi(argv[7]) : 0;
#endif

#ifndef MULTITHREAD
    RunOptions runOpts = parseRunMode(mode);
    if (efSearch < 1) efSearch = 1;
    if (K < 1) K = 1;
#ifdef HYBRID_RERANK
    if (rerankWidth < K) rerankWidth = std::max(efSearch, 4 * K);
#endif
#else
    RunOptions runOpts;
#endif

    // Build file paths
    char basePath[512], queryPath[512];
    std::snprintf(basePath,  sizeof(basePath),  "%ssift_base.fvecs",  dataDir);
    std::snprintf(queryPath, sizeof(queryPath), "%ssift_query.fvecs", dataDir);

    printf("=== HNSW Benchmark for GEM5 (SIFT-1M) ===\n");
    printf("Base: %d  Queries: %d\n", numBase, numQueries);
    printf("Data dir: %s\n\n", dataDir);
#ifndef MULTITHREAD
    printf("Mode: %s\n", mode);
    printf("Search params: efSearch=%d  K=%d\n", efSearch, K);
#ifdef HYBRID_RERANK
    printf("Hybrid rerank: width=%d (quantized search -> exact float rerank)\n", rerankWidth);
#endif
    if (runOpts.useSearchROI)
        printf("ROI stats: search phase only\n");
    if (runOpts.skipGroundTruth)
        printf("Ground truth: skipped\n");
    printf("\n");
#endif

    // -------------------------------------------------------------------------
    // [1] Load SIFT vectors (always as float32 from .fvecs format)
    // -------------------------------------------------------------------------
    printf("[1] Loading SIFT vectors...\n");
    std::vector<FloatVec> floatBase    = loadFvecs(basePath,  numBase);
    std::vector<FloatVec> floatQueries = loadFvecs(queryPath, numQueries);

#ifdef SCALAR_QUANT
    // Use one shared scale for base and query vectors. Separate scales would
    // distort distances and weaken quantized search quality.
    ScalarQuantizer quantizer = ScalarQuantizer::fit(floatBase, floatQueries);
    printf("    Quantizing %zu base + %zu query vectors: scale=%.4f  (%zu B/vec -> %zu B/vec)\n",
           floatBase.size(), floatQueries.size(), quantizer.scale, sizeof(FloatVec), sizeof(Vec));
    std::vector<Vec> base    = quantizer.quantizeAll(floatBase);
    std::vector<Vec> queries = quantizer.quantizeAll(floatQueries);
#ifndef HYBRID_RERANK
    floatBase.clear();    floatBase.shrink_to_fit();
    floatQueries.clear(); floatQueries.shrink_to_fit();
#endif
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

    // -------------------------------------------------------------------------
    // [3] Build self-consistent ground truth via brute-force BEFORE reordering.
    //     Must run here: reorderByBFS() remaps node IDs, so GT computed after
    //     reorder would use mismatched IDs and give incorrect recall values.
    //     base[i] → GT records original index i; searchKnn returns these same
    //     indices only when the index has not yet been reordered.
    // -------------------------------------------------------------------------
    std::vector<std::vector<int>> groundTruth(queries.size());
    if (!runOpts.skipGroundTruth) {
        printf("[3] Computing brute-force ground truth (N=%d, Q=%d, K=%d)...\n",
               numBase, (int)queries.size(), K);
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
        printf("    Done.\n\n");
    } else {
        groundTruth.clear();
        printf("[3] Skipping brute-force ground truth for search-focused measurement.\n\n");
    }

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

#ifdef PACKED_SEARCH
    {
        auto tp0 = std::chrono::high_resolution_clock::now();
        index.finalizeForSearch();
        auto tp1 = std::chrono::high_resolution_clock::now();
        printf("    Packed search layout: %.3f sec\n\n",
               std::chrono::duration<double>(tp1 - tp0).count());
    }
#endif

    // -------------------------------------------------------------------------
    // [4] Search
    // -------------------------------------------------------------------------
    int nq = (int)queries.size();
    std::vector<std::vector<HNSWIndex::Result>> results(nq);

    if (runOpts.useSearchROI) {
#if HNSW_HAS_M5OPS
        printf("[roi] Resetting gem5 stats at start of search phase.\n");
        m5_work_begin(0, 0);
        m5_reset_stats(0, 0);
#else
        printf("[roi] gem5 m5ops header not available at compile time; ROI disabled.\n");
#endif
    }

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
    for (int q = 0; q < nq; ++q) {
#ifdef HYBRID_RERANK
        auto coarse = index.searchKnn(queries[q], rerankWidth, std::max(efSearch, rerankWidth));
        results[q] = exactRerank(
            coarse,
            floatQueries[q],
            floatBase,
            K,
#ifdef GRAPH_REORDER
            &newToOld
#else
            nullptr
#endif
        );
#else
        results[q] = index.searchKnn(queries[q], K, efSearch);
#endif
    }
    auto te = std::chrono::high_resolution_clock::now();
#endif

    double searchSec = std::chrono::duration<double>(te - ts).count();
    printf("    Search: %.3f sec  (%.0f QPS)\n\n",
           searchSec, nq / searchSec);

    if (runOpts.useSearchROI) {
#if HNSW_HAS_M5OPS
        m5_dump_stats(0, 0);
        m5_work_end(0, 0);
        printf("[roi] Dumped gem5 stats at end of search phase.\n\n");
#endif
    }

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
    } else if (runOpts.skipGroundTruth) {
        printf("[5] Recall@%d = skipped  (ground truth disabled in this mode)\n\n", K);
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
