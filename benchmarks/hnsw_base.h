#pragma once
// =============================================================================
// hnsw_base.h  —  Clean baseline HNSW implementation
//
// Paper: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
//        search using Hierarchical Navigable Small World graphs",
//        IEEE TPAMI 2020.
//
// This file contains ONLY the core HNSW algorithm.
// No SIMD, no quantization, no parallelism, no optimization flags.
// This is the starting point we build on top of.
//
// Dataset: SIFT-128 (128-dimensional float32 vectors)
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

// =============================================================================
// Constants — sized for SIFT-1M (1 000 000 vectors, 128 dims)
// =============================================================================

static constexpr int kDim      = 128;   // vector dimension (SIFT-128)
static constexpr int kMaxNodes = 1100000; // max insertable vectors (1M + headroom)
static constexpr int kMaxLevels = 6;    // ceil(log₁₆(1M)) + 1 headroom; formula gives ≈5, 6 is safe upper bound

// FloatVec is always float — used by the .fvecs file loader regardless of mode.
using FloatVec = std::array<float, kDim>;

// =============================================================================
// Vec type and distance function
//
// SCALAR_QUANT: compress stored vectors to int8 (128 B/vec vs 512 B/vec).
//   l2sq accumulates into int32 then casts to float so all callers stay typed.
// Default: float32 vectors, standard L2 distance.
// =============================================================================

#ifdef SCALAR_QUANT
using Vec = std::array<int8_t, kDim>;
inline float l2sq(const Vec& a, const Vec& b) {
    int32_t sum = 0;
    for (int i = 0; i < kDim; ++i) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        sum += d * d;
    }
    return (float)sum;
}
#else
using Vec = FloatVec;
inline float l2sq(const Vec& a, const Vec& b) {
    float sum = 0.0f;
    for (int i = 0; i < kDim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
#endif

// =============================================================================
// Node — one vector in the graph
// =============================================================================

struct Node {
    Vec  vec;
    int  level;                                        // max layer this node lives on
    std::vector<int> neighbors[kMaxLevels];            // adjacency list per layer

    Node() : level(0) { vec.fill(0); }   // 0 works for both float and int8_t
};

// =============================================================================
// HNSW Index
//
// Key parameters (passed to constructor):
//   M              — max connections per node per layer (default 16)
//   Mmax0          — max connections at layer 0 (default 2*M = 32)
//   efConstruction — beam width during index build (default 100)
// =============================================================================

class HNSWIndex {
public:
    int M, Mmax0, efConstruction;

    explicit HNSWIndex(int M_ = 16, int Mmax0_ = 32, int efC_ = 100)
        : M(M_), Mmax0(Mmax0_), efConstruction(efC_),
          nodeCount_(0), entryPoint_(-1), maxLevel_(-1),
          levelMul_(1.0f / std::log(static_cast<float>(M_)))
    {
        nodes_.reserve(kMaxNodes);
    }

    // -------------------------------------------------------------------------
    // insert — Algorithm 1 from the paper
    // -------------------------------------------------------------------------
    void insert(const Vec& v) {
        if (nodeCount_ >= kMaxNodes)
            throw std::runtime_error("index full");

        int id    = nodeCount_++;
        int level = randomLevel();
        nodes_.emplace_back();          // allocate node on demand
        nodes_[id].vec   = v;
        nodes_[id].level = level;
        // neighbors already empty from Node() constructor

        // First node — just set as entry point
        if (entryPoint_ < 0) {
            entryPoint_ = id;
            maxLevel_   = level;
            return;
        }

        // Greedy descent from top layer down to level+1 (ef=1, coarse nav)
        std::vector<int> ep = { entryPoint_ };
        for (int lc = maxLevel_; lc > level; --lc) {
            ep = searchLayer(v, ep, 1, lc);
        }

        // From min(level, maxLevel) down to 0 — insert at each layer
        for (int lc = std::min(level, maxLevel_); lc >= 0; --lc) {
            std::vector<int> W = searchLayer(v, ep, efConstruction, lc);
            std::vector<int> neighbors = selectNeighbors(id, W, mmax(lc), lc);

            // Connect new node to its selected neighbors
            nodes_[id].neighbors[lc] = neighbors;

            // Backlink: add id to each neighbor, then prune if over limit
            for (int nid : neighbors) {
                auto& nbrs = nodes_[nid].neighbors[lc];
                if (std::find(nbrs.begin(), nbrs.end(), id) == nbrs.end())
                    nbrs.push_back(id);
                if ((int)nbrs.size() > mmax(lc)) {
                    std::vector<int> pruned = selectNeighbors(nid, nbrs, mmax(lc), lc);
                    nodes_[nid].neighbors[lc] = pruned;
                }
            }

            ep = W;  // carry forward winners to next layer
        }

        // Promote entry point if this node lives higher
        if (level > maxLevel_) {
            entryPoint_ = id;
            maxLevel_   = level;
        }
    }

    // -------------------------------------------------------------------------
    // searchKnn — Algorithm 5 from the paper
    // Returns top-k (id, dist_sq) pairs sorted by distance ascending
    // -------------------------------------------------------------------------
    struct Result {
        int   id;
        float dist;
    };

    // -------------------------------------------------------------------------
    // searchKnn — Algorithm 5 (K-NN-SEARCH) from the paper
    //
    // Pseudocode:
    //   Input:  hnsw, q (query), K, ef
    //   1  W  ← ∅
    //   2  ep ← enter-point of hnsw
    //   3  L  ← level of ep   (top layer)
    //   4  for lc ← L .. 1:
    //   5      W  ← SEARCH-LAYER(q, ep, ef=1, lc)
    //   6      ep ← nearest element in W to q
    //   7  W  ← SEARCH-LAYER(q, ep, ef, lc=0)
    //   8  return K nearest elements from W to q
    // -------------------------------------------------------------------------
    std::vector<Result> searchKnn(const Vec& query, int k, int efSearch) const {
        // Line 1: W = ∅  (allocated inside searchLayer)
        // Line 2: ep ← enter-point
        if (entryPoint_ < 0) return {};
        std::vector<int> ep = { entryPoint_ };

        // Line 3: L ← level of entry point  (= maxLevel_)

        // Lines 4-6: greedy descent through layers L .. 1 with ef=1
        //   Each call returns the 1 nearest element found at that layer.
        //   We set ep to that single nearest element before descending further.
        for (int lc = maxLevel_; lc >= 1; --lc) {
            std::vector<int> W = searchLayer(query, ep, /*ef=*/1, lc);
            // Line 6: ep ← nearest element in W to q
            // (W has exactly 1 element when ef=1, so just take W[0])
            ep = { nearestIn(W, query) };
        }

        // Line 7: full beam search at layer 0 with the requested ef
        std::vector<int> W = searchLayer(query, ep, std::max(efSearch, k), /*lc=*/0);

        // Line 8: return K nearest elements from W to q
        // Compute distances once, sort pairs, then emit — avoids O(n log n)
        // redundant l2sq calls that the sort lambda would otherwise cause.
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(W.size());
        for (int id : W)
            ranked.push_back({ l2sq(nodes_[id].vec, query), id });
        std::sort(ranked.begin(), ranked.end());

        std::vector<Result> out;
        out.reserve(k);
        for (int i = 0; i < k && i < (int)ranked.size(); ++i)
            out.push_back({ ranked[i].second, ranked[i].first });
        return out;
    }

    int  size()       const { return nodeCount_; }
    int  entryPoint() const { return entryPoint_; }
    int  maxLevel()   const { return maxLevel_; }

    // -------------------------------------------------------------------------
    // reorderByBFS — renumber nodes in BFS traversal order from the entry point
    //
    // GRAPH_REORDER optimization: nodes that are graph-neighbors (and thus
    // frequently accessed together during search) are assigned contiguous
    // memory addresses. This improves spatial locality — when one node is
    // fetched from DRAM, its likely-to-be-visited neighbors are on the same
    // or adjacent cache lines, turning cold DRAM misses into warm L2 hits.
    // -------------------------------------------------------------------------
    void reorderByBFS() {
        if (nodeCount_ == 0) return;

        std::vector<int> oldToNew(nodeCount_, -1);
        std::vector<int> newOrder;
        newOrder.reserve(nodeCount_);

        // BFS from entry point across all layers — visit order mirrors search order
        std::queue<int> q;
        q.push(entryPoint_);
        oldToNew[entryPoint_] = 0;
        newOrder.push_back(entryPoint_);

        while (!q.empty()) {
            int cur = q.front(); q.pop();
            for (int lc = nodes_[cur].level; lc >= 0; --lc) {
                for (int nbr : nodes_[cur].neighbors[lc]) {
                    if (oldToNew[nbr] == -1) {
                        oldToNew[nbr] = (int)newOrder.size();
                        newOrder.push_back(nbr);
                        q.push(nbr);
                    }
                }
            }
        }

        // Handle any nodes not reachable from entry point (should not happen
        // in a well-connected HNSW, but guard against degenerate cases)
        for (int i = 0; i < nodeCount_; ++i)
            if (oldToNew[i] == -1) { oldToNew[i] = (int)newOrder.size(); newOrder.push_back(i); }

        // Rebuild node array in BFS order, remapping all neighbor IDs
        std::vector<Node> reordered(nodeCount_);
        for (int newId = 0; newId < nodeCount_; ++newId) {
            reordered[newId] = std::move(nodes_[newOrder[newId]]);
            for (int lc = 0; lc <= reordered[newId].level; ++lc)
                for (int& nbr : reordered[newId].neighbors[lc])
                    nbr = oldToNew[nbr];
        }

        nodes_      = std::move(reordered);
        entryPoint_ = 0;   // entry point was assigned new ID 0 above
    }

private:
    std::vector<Node> nodes_;
    int nodeCount_;
    int entryPoint_, maxLevel_;
    float levelMul_;

    // Max connections at layer lc
    int mmax(int lc) const { return (lc == 0) ? Mmax0 : M; }

    // Return the id in `candidates` that is nearest to query q.
    // Used in Algorithm 5 line 6: ep ← nearest element in W to q.
    int nearestIn(const std::vector<int>& candidates, const Vec& q) const {
        int   best  = candidates[0];
        float bestD = l2sq(nodes_[best].vec, q);
        for (int i = 1; i < (int)candidates.size(); ++i) {
            float d = l2sq(nodes_[candidates[i]].vec, q);
            if (d < bestD) { bestD = d; best = candidates[i]; }
        }
        return best;
    }

    // Random level — geometric distribution (Algorithm 1 line 4)
    int randomLevel() const {
        static thread_local unsigned seed = 0u;
        if (seed == 0u) seed = 12345u ^ (unsigned)(uintptr_t)&seed;
        seed = seed * 1103515245u + 12345u;
        double r = (double)(seed & 0x7fffffffu) / (double)0x7fffffffu + 1e-9;
        int lv = (int)(-std::log(r) * levelMul_);
        return std::min(lv, kMaxLevels - 1);
    }

    // -------------------------------------------------------------------------
    // searchLayer — Algorithm 2 from the paper
    //
    // Input:  query vector q, entry point set ep, beam width ef, layer lc
    // Output: ef nearest neighbors found (as node ids)
    //
    // Uses two priority queues:
    //   candidates — min-heap (closest first) — nodes to expand next
    //   W          — max-heap (farthest first) — current result set of size ef
    // -------------------------------------------------------------------------
    std::vector<int> searchLayer(const Vec& q,
                                 const std::vector<int>& entryPoints,
                                 int ef,
                                 int lc) const {
        // visited: generation counter avoids clearing a large array each call
        static thread_local std::vector<int> visitedGen;
        static thread_local int              curGen = 1;
        if ((int)visitedGen.size() < kMaxNodes)
            visitedGen.assign(kMaxNodes, 0);
        ++curGen;

        // candidates: min-heap on distance
        using Pair = std::pair<float, int>;
        std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cands;

        // W: max-heap on distance (result set, kept at size ef)
        std::priority_queue<Pair> W;

        for (int ep : entryPoints) {
            float d = l2sq(nodes_[ep].vec, q);
            cands.push({d, ep});
            W.push({d, ep});
            visitedGen[ep] = curGen;
        }

        while (!cands.empty()) {
            float cDist = cands.top().first;
            int   cId   = cands.top().second;
            cands.pop();

            // If the closest unvisited candidate is farther than worst in W,
            // we can't improve W — stop.
            if (cDist > W.top().first) break;

#ifdef SW_PREFETCH
            // 1-hop: prefetch neighbor vectors before distance computation.
            for (int nid : nodes_[cId].neighbors[lc])
                __builtin_prefetch(nodes_[nid].vec.data(), 0, 1);
            // 2-hop: prefetch neighbors' neighbor lists for the next iteration.
            for (int nid : nodes_[cId].neighbors[lc]) {
                if (!nodes_[nid].neighbors[lc].empty())
                    __builtin_prefetch(nodes_[nid].neighbors[lc].data(), 0, 0);
            }
#endif

            for (int nid : nodes_[cId].neighbors[lc]) {
                if (visitedGen[nid] == curGen) continue;
                visitedGen[nid] = curGen;

                float nd = l2sq(nodes_[nid].vec, q);
                if ((int)W.size() < ef || nd < W.top().first) {
                    cands.push({nd, nid});
                    W.push({nd, nid});
                    if ((int)W.size() > ef) W.pop();  // evict farthest
                }
            }
        }

        // Extract result set
        std::vector<int> out;
        out.reserve(W.size());
        while (!W.empty()) {
            out.push_back(W.top().second);
            W.pop();
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // selectNeighbors — Algorithm 4 (heuristic) from the paper
    //
    // Prefers diverse neighbors: among candidates, keeps a neighbor only if
    // it is closer to the new node than to any already-selected neighbor.
    // This produces better-connected graphs than simple top-M selection.
    // -------------------------------------------------------------------------
    std::vector<int> selectNeighbors(int id,
                                     const std::vector<int>& candidates,
                                     int M_,
                                     int lc) const {
        // Sort candidates by distance to the new node
        std::vector<std::pair<float, int>> sorted;
        sorted.reserve(candidates.size());
        for (int cid : candidates)
            sorted.push_back({ l2sq(nodes_[id].vec, nodes_[cid].vec), cid });
        std::sort(sorted.begin(), sorted.end());

        std::vector<int> result;
        result.reserve(M_);
        std::vector<int> discarded;

        for (int si = 0; si < (int)sorted.size(); ++si) {
            float dToBase = sorted[si].first;
            int   cid     = sorted[si].second;
            if ((int)result.size() >= M_) break;

            // Keep if cid is closer to base than to any already-selected neighbor
            bool keep = true;
            for (int rid : result) {
                if (l2sq(nodes_[cid].vec, nodes_[rid].vec) < dToBase) {
                    keep = false;
                    break;
                }
            }
            if (keep) result.push_back(cid);
            else      discarded.push_back(cid);
        }

        // Backfill from discarded to reach M_ connections if possible
        for (int cid : discarded) {
            if ((int)result.size() >= M_) break;
            result.push_back(cid);
        }

        return result;
    }

};
