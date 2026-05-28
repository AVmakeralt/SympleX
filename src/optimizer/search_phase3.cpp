// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/optimizer/search_phase3.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

// Portable high-resolution timing
#include <time.h>

namespace symplex::optimizer {

namespace {

/// Compute a "naive" baseline latency for comparison.
/// The naive config is the smallest tile that fits the Tensor Core
/// native dimensions (m, n, k) with no further optimisation.
double naive_baseline_latency(
    const hardware::HardwareTarget& target,
    size_t ndim
) {
    const auto& tc = target.gpu.tensor_core;

    ExtendedTileConfig naive;
    if (ndim >= 3) {
        naive.inner_tiles = {tc.m, tc.n, tc.k};
        naive.outer_tiles = {tc.m, tc.n, tc.k};
        naive.compute_ops  = compute_matmul_ops(tc.m, tc.n, tc.k);
        naive.bytes_moved  = compute_matmul_bytes(
            tc.m, tc.n, tc.k, target.bytes_per_element);
    } else {
        naive.inner_tiles = {tc.m, tc.n};
        naive.outer_tiles = {tc.m, tc.n};
        int64_t vol = tc.m * tc.n;
        naive.compute_ops  = 2 * vol;
        naive.bytes_moved  = 3 * vol * target.bytes_per_element;
    }
    naive.operational_intensity = naive.calc_operational_intensity();

    return estimate_latency_ns(naive, target);
}

/// Re-estimate latency with an occupancy-aware model.
/// The roofline model gives a lower bound; low occupancy degrades the
/// achievable fraction of peak.  We apply an occupancy derating factor:
///   effective_flops = peak_flops * min(1.0, occupancy / max_warps)
double occupancy_derated_latency(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    double peak_flops = target.peak_flops_fp16();
    double peak_bw    = target.gpu.memory.global_bw_gbps * 1e9;

    // Occupancy derating: if we don't saturate the SM, we achieve
    // a fraction of peak proportional to active-warps / max-warps.
    double occ_ratio = 1.0;
    if (target.gpu.sm.max_warps > 0 && cfg.occupancy > 0) {
        occ_ratio = std::min(
            static_cast<double>(cfg.occupancy) /
            static_cast<double>(target.gpu.sm.max_warps),
            1.0);
    } else if (cfg.occupancy == 0) {
        occ_ratio = 0.1;  // Extremely low – penalise heavily
    }

    // Pipeline overlap: with software pipelining, memory and compute
    // can partially overlap.  The effective latency approaches
    // max(compute, memory) * (1 - overlap_factor) + min(compute, memory) * overlap_factor
    // For a 2-stage pipeline with perfect overlap on the steady state:
    double pipeline_overlap = (target.pipeline_stages > 1) ? 0.15 : 0.0;

    double compute_time_s = (peak_flops > 0.0 && occ_ratio > 0.0)
        ? static_cast<double>(cfg.compute_ops) / (peak_flops * occ_ratio)
        : std::numeric_limits<double>::max();
    double memory_time_s  = (peak_bw > 0.0)
        ? static_cast<double>(cfg.bytes_moved) / peak_bw
        : std::numeric_limits<double>::max();

    // Apply pipeline overlap reduction
    double roofline_time = std::max(compute_time_s, memory_time_s);
    double min_time      = std::min(compute_time_s, memory_time_s);
    double effective_time = roofline_time * (1.0 - pipeline_overlap)
                          + min_time * pipeline_overlap;

    return effective_time * 1e9;  // ns
}

} // anonymous namespace

// ── Empirical profiling (CPU micro-benchmark) ───────────────────────────

namespace {

/// Portable monotonic clock reader – returns nanoseconds since an arbitrary epoch.
inline int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL +
           static_cast<int64_t>(ts.tv_nsec);
}

/// Run a single tiled matmul benchmark: C += A*B on CPU buffers.
/// Returns elapsed time in nanoseconds, or -1 on failure (e.g. OOM).
int64_t benchmark_tiled_matmul(
    int64_t tm, int64_t tn, int64_t tk,
    int64_t warmup_iters, int64_t measure_iters
) {
    // Validate dimensions – keep the problem small enough to fit in cache/L2
    // so we measure compute throughput rather than main-memory bandwidth.
    const int64_t max_tile_vol = 512;   // safety cap for single-tile elements
    if (tm <= 0 || tn <= 0 || tk <= 0) return -1;
    if (tm > max_tile_vol || tn > max_tile_vol || tk > max_tile_vol) return -1;

    size_t a_elems = static_cast<size_t>(tm * tk);
    size_t b_elems = static_cast<size_t>(tk * tn);
    size_t c_elems = static_cast<size_t>(tm * tn);

    // Allocate FP16-like buffers (use int16_t as a stand-in for __fp16 on CPU).
    // For benchmarking the memory/compute pattern, the actual data type doesn't
    // matter – what matters is the arithmetic intensity and data movement.
    auto* a_buf = static_cast<float*>(std::malloc(a_elems * sizeof(float)));
    auto* b_buf = static_cast<float*>(std::malloc(b_elems * sizeof(float)));
    auto* c_buf = static_cast<float*>(std::malloc(c_elems * sizeof(float)));

    if (!a_buf || !b_buf || !c_buf) {
        std::free(a_buf);
        std::free(b_buf);
        std::free(c_buf);
        return -1;  // OOM
    }

    // Initialise with deterministic data
    for (size_t i = 0; i < a_elems; ++i) a_buf[i] = static_cast<float>(i % 7) * 0.1f;
    for (size_t i = 0; i < b_elems; ++i) b_buf[i] = static_cast<float>(i % 11) * 0.1f;
    for (size_t i = 0; i < c_elems; ++i) c_buf[i] = 0.0f;

    // ── Warmup ────────────────────────────────────────────────────────
    for (int64_t w = 0; w < warmup_iters; ++w) {
        for (int64_t i = 0; i < tm; ++i) {
            for (int64_t j = 0; j < tn; ++j) {
                float sum = c_buf[i * tn + j];
                for (int64_t kk = 0; kk < tk; ++kk) {
                    sum += a_buf[i * tk + kk] * b_buf[kk * tn + j];
                }
                c_buf[i * tn + j] = sum;
            }
        }
    }

    // Reset C for measurement
    for (size_t i = 0; i < c_elems; ++i) c_buf[i] = 0.0f;

    // ── Measure ───────────────────────────────────────────────────────
    std::vector<int64_t> samples;
    samples.reserve(static_cast<size_t>(measure_iters));

    for (int64_t it = 0; it < measure_iters; ++it) {
        int64_t t0 = now_ns();

        for (int64_t i = 0; i < tm; ++i) {
            for (int64_t j = 0; j < tn; ++j) {
                float sum = c_buf[i * tn + j];
                for (int64_t kk = 0; kk < tk; ++kk) {
                    sum += a_buf[i * tk + kk] * b_buf[kk * tn + j];
                }
                c_buf[i * tn + j] = sum;
            }
        }

        int64_t t1 = now_ns();
        samples.push_back(t1 - t0);
    }

    // ── Compute median ────────────────────────────────────────────────
    std::sort(samples.begin(), samples.end());
    int64_t median = samples[samples.size() / 2];

    std::free(a_buf);
    std::free(b_buf);
    std::free(c_buf);

    return median;
}

} // anonymous namespace

double empirical_profile_latency(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
) {
    // Extract tile dimensions
    int64_t tm = (cfg.inner_tiles.size() > 0) ? cfg.inner_tiles[0] : 16;
    int64_t tn = (cfg.inner_tiles.size() > 1) ? cfg.inner_tiles[1] : 16;
    int64_t tk = (cfg.inner_tiles.size() > 2) ? cfg.inner_tiles[2] : 16;

    // Run the CPU micro-benchmark
    const int64_t warmup_iters  = 3;
    const int64_t measure_iters = 11;   // odd → clean median
    int64_t measured_ns = benchmark_tiled_matmul(tm, tn, tk,
                                                  warmup_iters, measure_iters);

    if (measured_ns > 0) {
        // Scale the CPU measurement to approximate GPU latency.
        // The CPU benchmark gives us a *relative* ranking of tile configs
        // that captures compute intensity, register pressure, and cache
        // behaviour.  We scale it so the median matches the analytical
        // estimate for the best case, preserving relative ordering while
        // keeping values in the right ballpark.
        double analytical_ns = cfg.estimated_latency_ns;
        if (analytical_ns > 0.0) {
            // Compute a scaling factor from the naive tile (smallest Tensor
            // Core dimensions).  This makes the empirical measurement
            // proportional to the analytical model while using real
            // measured ratios between configs.
            double naive_ns_est = static_cast<double>(
                target.gpu.tensor_core.m * target.gpu.tensor_core.n *
                target.gpu.tensor_core.k * 2.0);  // rough FLOP count
            double bench_ratio = static_cast<double>(measured_ns) /
                                 std::max(naive_ns_est, 1.0);
            return analytical_ns * bench_ratio;
        }
        return static_cast<double>(measured_ns);
    }

    // Benchmarking failed (OOM, invalid dimensions, etc.) – fall back
    // to the analytical estimate.
    return cfg.estimated_latency_ns;
}

// ── Phase 3 implementation ───────────────────────────────────────────────

SearchPhase3Result phase3_occupancy_sieve(
    std::vector<ExtendedTileConfig> candidates,
    const hardware::HardwareTarget& target,
    size_t top_n
) {
    SearchPhase3Result result;

    if (candidates.empty()) {
        // No candidates survived earlier phases – return empty result.
        return result;
    }

    // Step 1: take only the top_n candidates (they are already sorted
    //         by phase 2 score).
    if (candidates.size() > top_n) {
        candidates.resize(top_n);
    }

    // Step 2: compute occupancy-aware latency for each candidate.
    size_t ndim = candidates.front().inner_tiles.size();

    for (auto& cfg : candidates) {
        // Re-estimate occupancy if not already populated
        if (cfg.occupancy <= 0) {
            cfg.occupancy = estimate_occupancy(cfg, target);
        }

        cfg.estimated_latency_ns = occupancy_derated_latency(cfg, target);

        // Optionally override with empirical profiling (currently a no-op
        // that returns the analytical value).
        double empirical_ns = empirical_profile_latency(cfg, target);
        // Blend: 80% analytical + 20% empirical when empirical is available
        // (currently both are the same, so no change).
        cfg.estimated_latency_ns = 0.8 * cfg.estimated_latency_ns
                                 + 0.2 * empirical_ns;
    }

    // Step 3: rank by estimated latency (ascending = fastest first).
    std::sort(candidates.begin(), candidates.end(),
        [](const ExtendedTileConfig& a, const ExtendedTileConfig& b) {
            return a.estimated_latency_ns < b.estimated_latency_ns;
        });

    result.ranked_candidates = std::move(candidates);
    result.best_config = result.ranked_candidates.front();

    // Step 4: compute speedup vs naive baseline.
    double baseline_ns = naive_baseline_latency(target, ndim);
    if (baseline_ns > 0.0 && result.best_config.estimated_latency_ns > 0.0) {
        result.estimated_speedup_vs_baseline =
            baseline_ns / result.best_config.estimated_latency_ns;
    }

    return result;
}

} // namespace symplex::optimizer
