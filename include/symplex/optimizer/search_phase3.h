// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/optimizer/tile_config.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>
#include <cstddef>

namespace symplex::optimizer {

/// Result of the Phase 3 occupancy sieve.
struct SearchPhase3Result {
    ExtendedTileConfig best_config;                 ///< Best candidate found
    std::vector<ExtendedTileConfig> ranked_candidates; ///< All candidates, ranked by latency
    double estimated_speedup_vs_baseline = 0.0;     ///< Speedup vs naive single-tile config
};

/// Phase 3: Hardware Occupancy Sieve (Micro-Benchmarking)
///
/// Takes the top-N candidates from Phase 2 and refines the ranking using
/// an analytical occupancy model and latency estimation:
///
///   latency = max(compute_time, memory_time)
///   compute_time = compute_ops / peak_flops
///   memory_time  = bytes_moved / peak_bandwidth
///
/// The candidate with the lowest estimated latency wins.  An interface
/// for empirical profiling is provided; the current implementation
/// returns the analytical estimate (future: integrate CUDA events).
///
/// \param candidates  Scored configs from phase 2 (moved in).
/// \param target      Hardware target specification.
/// \param top_n       Maximum candidates to evaluate (default 20).
/// \return            Phase 3 result with best config and ranking.
SearchPhase3Result phase3_occupancy_sieve(
    std::vector<ExtendedTileConfig> candidates,
    const hardware::HardwareTarget& target,
    size_t top_n = 20
);

/// Empirical profiling placeholder.
/// In a full deployment this would launch a micro-kernel with the given
/// tile configuration and measure wall-clock time via CUDA events.
/// For now it returns the analytical estimate stored in
/// cfg.estimated_latency_ns.
double empirical_profile_latency(
    const ExtendedTileConfig& cfg,
    const hardware::HardwareTarget& target
);

} // namespace symplex::optimizer
