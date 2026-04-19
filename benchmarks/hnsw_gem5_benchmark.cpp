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
//
// Run natively:
//   ./hnsw_gem5 500 20 /workspace/bigann/sift/
//
// Run in gem5:
//   gem5.opt configs/run_benchmark.py \
//       --binary benchmarks/hnsw_gem5 \
//       --bin-args "500 20 /workspace/bigann/sift/"
// =============================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    const char* dataDir = (argc > 3) ? argv[3] : "/workspace/bigann/sift/";

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

#ifdef GRAPH_REORDER
    // Renumber nodes in BFS traversal order from the entry point so that
    // graph-neighbors are contiguous in memory — improves spatial locality.
    auto tr = std::chrono::high_resolution_clock::now();
    index.reorderByBFS();
    auto te2 = std::chrono::high_resolution_clock::now();
    printf("    BFS reorder: %.3f sec\n\n",
           std::chrono::duration<double>(te2 - tr).count());
#endif

    // Free base vectors after index is built
    { std::vector<Vec>().swap(base); }

    // -------------------------------------------------------------------------
    // [3] Search
    // -------------------------------------------------------------------------
    const int K        = 10;   // top-K neighbors to return (standard ANN benchmark value)
    const int efSearch = 50;  // beam width ≥ K; higher = better recall, more compute
    printf("[3] Querying (%d queries, K=%d, efSearch=%d)...\n",
           (int)queries.size(), K, efSearch);

    auto ts = std::chrono::high_resolution_clock::now();
    for (int q = 0; q < (int)queries.size(); ++q)
        index.searchKnn(queries[q], K, efSearch);
    auto te = std::chrono::high_resolution_clock::now();
    double searchSec = std::chrono::duration<double>(te - ts).count();
    printf("    Search: %.3f sec  (%.0f QPS)\n\n",
           searchSec, queries.size() / searchSec);

    // -------------------------------------------------------------------------
    // [4] Summary
    // -------------------------------------------------------------------------
    printf("Index: nodes=%d  entry=%d  maxLevel=%d\n",
           index.size(), index.entryPoint(), index.maxLevel());
    printf("=== Benchmark Complete ===\n");
    return 0;
}
