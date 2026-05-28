// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/schedule/tiling.h"
#include <cstdint>
#include <string>
#include <sstream>
#include <algorithm>

namespace symplex::optimizer {

/// ExtendedTileConfig: augments the base TileConfig with superoptimizer
/// cost-model annotations.  Every phase of the 3-phase search populates
/// additional fields so that later phases can refine the ranking without
/// recomputing earlier metrics.
struct ExtendedTileConfig : schedule::TileConfig {
    double estimated_latency_ns  = 0.0;   // Estimated kernel latency (ns)
    double operational_intensity = 0.0;   // FLOPS per byte (roofline metric)
    int64_t compute_ops          = 0;     // Total FMA operations in the tile
    int64_t bytes_moved          = 0;     // Total bytes transferred (HBM ↔ SRAM)
    int64_t occupancy            = 0;     // Active warps per SM for this config
    bool    compute_bound        = false; // Does OI exceed the roofline knee?
    double  score                = 0.0;   // Composite score for ranking

    ExtendedTileConfig() = default;

    /// Construct from a base TileConfig, leaving cost fields at defaults.
    explicit ExtendedTileConfig(schedule::TileConfig base)
        : TileConfig(std::move(base)) {}

    // ── Derived helpers ──────────────────────────────────────────────

    /// Compute operational intensity from stored fields.
    [[nodiscard]] double calc_operational_intensity() const {
        if (bytes_moved <= 0) return 0.0;
        return static_cast<double>(compute_ops) /
               static_cast<double>(bytes_moved);
    }

    /// SRAM utilisation as a fraction of max_sram_bytes.
    [[nodiscard]] double sram_utilization(int64_t max_sram_bytes) const {
        if (max_sram_bytes <= 0) return 0.0;
        // Use the TileConfig base-class method with double-buffering
        size_t fp = sram_footprint(3, 2, true);
        return std::min(static_cast<double>(fp) /
                        static_cast<double>(max_sram_bytes), 1.0);
    }

    /// Fraction of the inner tile that is a multiple of the Tensor Core
    /// native dimensions (m, n, k).  Returns 1.0 for perfect alignment.
    [[nodiscard]] double tensor_core_alignment(
        int64_t tc_m, int64_t tc_n, int64_t tc_k
    ) const {
        if (inner_tiles.size() < 3) return 0.0;
        double aligned = 0.0;
        if (tc_m > 0 && inner_tiles[0] % tc_m == 0) aligned += 1.0;
        if (tc_n > 0 && inner_tiles[1] % tc_n == 0) aligned += 1.0;
        if (tc_k > 0 && inner_tiles[2] % tc_k == 0) aligned += 1.0;
        return aligned / 3.0;
    }

    /// String representation with cost annotations.
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "ExtTile{inner=[";
        for (size_t i = 0; i < inner_tiles.size(); ++i) {
            if (i > 0) oss << "x";
            oss << inner_tiles[i];
        }
        oss << "], outer=[";
        for (size_t i = 0; i < outer_tiles.size(); ++i) {
            if (i > 0) oss << "x";
            oss << outer_tiles[i];
        }
        oss << "], oi=" << operational_intensity
            << ", ops=" << compute_ops
            << ", bytes=" << bytes_moved
            << ", latency_ns=" << estimated_latency_ns
            << ", occupancy=" << occupancy
            << ", compute_bound=" << (compute_bound ? "yes" : "no")
            << ", score=" << score
            << "}";
        return oss.str();
    }
};

// Cost-model helper utilities shared across search phases.
int64_t compute_matmul_ops(int64_t tm, int64_t tn, int64_t tk);
int64_t compute_matmul_bytes(
    int64_t tm, int64_t tn, int64_t tk, int64_t bytes_per_element);
double compute_operational_intensity(
    int64_t tm, int64_t tn, int64_t tk, int64_t bytes_per_element);
int64_t estimate_occupancy(
    const ExtendedTileConfig& cfg, const hardware::HardwareTarget& target);
double estimate_latency_ns(
    const ExtendedTileConfig& cfg, const hardware::HardwareTarget& target);

} // namespace symplex::optimizer
