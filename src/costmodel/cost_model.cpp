// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/costmodel/cost_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>

namespace symplex::costmodel {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

CostModel::CostModel(hardware::HardwareTarget target, CostModelLevel level)
    : target_(std::move(target))
    , level_(level)
    , roofline_(target_)
    , analytical_(target_)
    , empirical_(target_)
{}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const hardware::HardwareTarget& CostModel::target() const {
    return target_;
}

CostModelLevel CostModel::level() const {
    return level_;
}

// ---------------------------------------------------------------------------
// Unified estimation
// ---------------------------------------------------------------------------

CostEstimate CostModel::estimate(
    const polyhedral::IterationSpace& ispace,
    const schedule::TileConfig& tile
) {
    CostEstimate est{};
    est.latency_ns  = 0.0;
    est.confidence  = 0.0;
    est.model_used  = level_;
    est.details     = "";

    // ── Determine problem type from iteration space name ──────────────
    // We inspect the name to decide whether this is a matmul or conv2d.
    // The iteration space factories set names like "matmul" or "conv2d".
    const std::string& name = ispace.name();
    bool is_matmul = (name.find("matmul") != std::string::npos);
    bool is_conv2d = (name.find("conv2d") != std::string::npos);

    // ── Dispatch based on model level ─────────────────────────────────
    switch (level_) {

    // ── ROOFLINE_ONLY ─────────────────────────────────────────────────
    case CostModelLevel::ROOFLINE_ONLY: {
        // Compute ops and bytes from the iteration space and tile.
        // For matmul: ops = 2 * M * N * K, bytes = (M*K + K*N + M*N) * bpe
        // For general: we approximate from the iteration space volume.
        int64_t compute_ops = 0;
        int64_t bytes_moved = 0;

        if (is_matmul && tile.inner_tiles.size() >= 3) {
            // For matmul, use tile-specific roofline analysis
            int64_t tm = tile.inner_tiles[0];
            int64_t tn = tile.inner_tiles[1];
            int64_t tk = tile.inner_tiles[2];

            // Use outer tiles as problem dimensions (fallback to inner if outer empty)
            int64_t M = (tile.outer_tiles.size() > 0 && tile.outer_tiles[0] > 0)
                        ? tile.outer_tiles[0] : tm;
            int64_t N = (tile.outer_tiles.size() > 1 && tile.outer_tiles[1] > 0)
                        ? tile.outer_tiles[1] : tn;
            int64_t K = (tile.outer_tiles.size() > 2 && tile.outer_tiles[2] > 0)
                        ? tile.outer_tiles[2] : tk;

            compute_ops = 2 * M * N * K;
            int64_t bpe = target_.bytes_per_element;
            bytes_moved = (M * K + K * N + M * N) * bpe;

            // Use the per-tile roofline for a more precise answer
            RooflinePoint rp = roofline_.analyze_matmul_tile(tm, tn, tk);

            est.latency_ns = rp.total_time_ns;
            est.confidence = 0.3;  // Roofline is a coarse model

            std::ostringstream oss;
            oss << "Roofline: OI=" << rp.operational_intensity
                << " GFLOPS=" << rp.achievable_gflops
                << " eff=" << rp.efficiency_percent << "%"
                << " bound=" << (rp.is_compute_bound ? "compute" : "memory");
            est.details = oss.str();
        } else {
            // General roofline: estimate from iteration space volume
            // Compute ops ≈ 2 * total_elements (assuming FMA per element)
            int64_t total_elements = tile.inner_volume();
            for (size_t i = 0; i < ispace.num_statements(); ++i) {
                const auto& stmt = ispace.statement(i);
                // Use the domain to get a rough iteration count
                int64_t stmt_iters = stmt.domain.count_points();
                if (stmt_iters > 0) {
                    total_elements = std::max(total_elements, stmt_iters);
                }
            }

            compute_ops = 2 * total_elements;
            int64_t bpe = target_.bytes_per_element;
            // Rough bytes: 3 tensors (2 reads + 1 write) per element
            bytes_moved = 3 * total_elements * bpe;

            RooflinePoint rp = roofline_.analyze(compute_ops, bytes_moved);

            est.latency_ns = rp.total_time_ns;
            est.confidence = 0.2;  // Even lower confidence for unknown op types

            std::ostringstream oss;
            oss << "Roofline(generic): OI=" << rp.operational_intensity
                << " GFLOPS=" << rp.achievable_gflops
                << " eff=" << rp.efficiency_percent << "%"
                << " bound=" << (rp.is_compute_bound ? "compute" : "memory");
            est.details = oss.str();
        }
        break;
    }

    // ── ANALYTICAL ────────────────────────────────────────────────────
    case CostModelLevel::ANALYTICAL: {
        if (is_matmul && tile.inner_tiles.size() >= 3) {
            int64_t M = (tile.outer_tiles.size() > 0 && tile.outer_tiles[0] > 0)
                        ? tile.outer_tiles[0] : tile.inner_tiles[0];
            int64_t N = (tile.outer_tiles.size() > 1 && tile.outer_tiles[1] > 0)
                        ? tile.outer_tiles[1] : tile.inner_tiles[1];
            int64_t K = (tile.outer_tiles.size() > 2 && tile.outer_tiles[2] > 0)
                        ? tile.outer_tiles[2] : tile.inner_tiles[2];

            AnalyticalEstimate ae = analytical_.estimate_matmul(M, N, K, tile);

            est.latency_ns = ae.latency_ns;
            est.confidence = 0.6;

            // Boost confidence if occupancy is healthy
            double occ_ratio = (target_.gpu.sm.max_warps > 0)
                ? static_cast<double>(ae.occupancy_warps) /
                  static_cast<double>(target_.gpu.sm.max_warps)
                : 0.0;
            if (occ_ratio >= 0.5) est.confidence = 0.7;
            if (occ_ratio >= 0.75) est.confidence = 0.8;

            std::ostringstream oss;
            oss << "Analytical(matmul): compute=" << ae.compute_ns << "ns"
                << " hbm_load=" << ae.hbm_load_ns << "ns"
                << " hbm_store=" << ae.hbm_store_ns << "ns"
                << " overlap=" << (ae.overlap_factor * 100.0) << "%"
                << " sram=" << (ae.sram_bytes_used / 1024) << "KB"
                << " occupancy=" << ae.occupancy_warps << " warps";
            est.details = oss.str();
        } else if (is_conv2d) {
            // For conv2d we need to extract dimensions from outer tiles
            // or fall back to a generic estimate.
            // Expected outer_tiles: [batch, oc, oh, ow, ic, kh, kw]
            // or a subset.
            int64_t batch = 1, oc = 1, ic = 1, oh = 1, ow = 1, kh = 1, kw = 1;

            if (tile.outer_tiles.size() >= 7) {
                batch = tile.outer_tiles[0];
                oc    = tile.outer_tiles[1];
                oh    = tile.outer_tiles[2];
                ow    = tile.outer_tiles[3];
                ic    = tile.outer_tiles[4];
                kh    = tile.outer_tiles[5];
                kw    = tile.outer_tiles[6];
            } else if (tile.outer_tiles.size() >= 4) {
                // Simplified: [oc, spatial, ic, filter]
                oc      = tile.outer_tiles[0];
                oh      = tile.outer_tiles[1]; ow = 1;
                ic      = tile.outer_tiles[2];
                kh      = tile.outer_tiles[3]; kw = 1;
            }

            AnalyticalEstimate ae = analytical_.estimate_conv2d(
                batch, oc, ic, oh, ow, kh, kw, tile
            );

            est.latency_ns = ae.latency_ns;
            est.confidence = 0.5;

            double occ_ratio = (target_.gpu.sm.max_warps > 0)
                ? static_cast<double>(ae.occupancy_warps) /
                  static_cast<double>(target_.gpu.sm.max_warps)
                : 0.0;
            if (occ_ratio >= 0.5) est.confidence = 0.6;
            if (occ_ratio >= 0.75) est.confidence = 0.7;

            std::ostringstream oss;
            oss << "Analytical(conv2d): compute=" << ae.compute_ns << "ns"
                << " hbm_load=" << ae.hbm_load_ns << "ns"
                << " hbm_store=" << ae.hbm_store_ns << "ns"
                << " overlap=" << (ae.overlap_factor * 100.0) << "%"
                << " sram=" << (ae.sram_bytes_used / 1024) << "KB"
                << " occupancy=" << ae.occupancy_warps << " warps";
            est.details = oss.str();
        } else {
            // Unknown operation type: use roofline with analytical refinement
            int64_t total_elements = tile.inner_volume();
            int64_t compute_ops = 2 * total_elements;
            int64_t bytes_moved = 3 * total_elements * target_.bytes_per_element;

            RooflinePoint rp = roofline_.analyze(compute_ops, bytes_moved);
            // Refine with occupancy-based scaling
            double occ_scale = 1.0;
            int64_t regs_per_thread = 32;
            int64_t smem_per_block = tile.inner_volume() * target_.bytes_per_element * 3;
            int64_t occ_warps = target_.gpu.sm.compute_occupancy(
                regs_per_thread, smem_per_block
            );
            if (target_.gpu.sm.max_warps > 0 && occ_warps > 0) {
                occ_scale = static_cast<double>(target_.gpu.sm.max_warps) /
                            static_cast<double>(occ_warps);
            }

            est.latency_ns = rp.total_time_ns * occ_scale;
            est.confidence = 0.4;

            std::ostringstream oss;
            oss << "Analytical(generic): roofline=" << rp.total_time_ns << "ns"
                << " occ_scale=" << occ_scale
                << " bound=" << (rp.is_compute_bound ? "compute" : "memory");
            est.details = oss.str();
        }
        break;
    }

    // ── EMPIRICAL ─────────────────────────────────────────────────────
    case CostModelLevel::EMPIRICAL: {
        if (is_matmul && tile.inner_tiles.size() >= 3) {
            int64_t M = (tile.outer_tiles.size() > 0 && tile.outer_tiles[0] > 0)
                        ? tile.outer_tiles[0] : tile.inner_tiles[0];
            int64_t N = (tile.outer_tiles.size() > 1 && tile.outer_tiles[1] > 0)
                        ? tile.outer_tiles[1] : tile.inner_tiles[1];
            int64_t K = (tile.outer_tiles.size() > 2 && tile.outer_tiles[2] > 0)
                        ? tile.outer_tiles[2] : tile.inner_tiles[2];

            ProfileResult pr = empirical_.profile_matmul(M, N, K, tile);

            if (pr.valid) {
                est.latency_ns = pr.mean_latency_ns;
                // Confidence depends on variance relative to mean
                if (pr.mean_latency_ns > 0.0) {
                    double cv = pr.std_dev_ns / pr.mean_latency_ns;  // Coeff of variation
                    // Low CV → high confidence; CV > 0.2 → low confidence
                    est.confidence = std::max(0.0, std::min(1.0, 1.0 - cv * 5.0));
                } else {
                    est.confidence = 0.0;
                }

                // Boost confidence if result was measured on real hardware
                if (pr.is_measured) {
                    est.confidence = std::min(est.confidence + 0.2, 1.0);
                }
                if (!pr.high_confidence) {
                    est.confidence *= 0.7;  // Penalize low-confidence fallback estimates
                }

                std::ostringstream oss;
                oss << "Empirical(matmul): mean=" << pr.mean_latency_ns << "ns"
                    << " min=" << pr.min_latency_ns << "ns"
                    << " max=" << pr.max_latency_ns << "ns"
                    << " stddev=" << pr.std_dev_ns << "ns"
                    << " warps=" << pr.active_warps
                    << " sm_eff=" << pr.sm_efficiency_percent << "%"
                    << " measured=" << (pr.is_measured ? "yes" : "no")
                    << " high_conf=" << (pr.high_confidence ? "yes" : "no");
                est.details = oss.str();
            } else {
                // Profiling failed, fall back to analytical
                AnalyticalEstimate ae = analytical_.estimate_matmul(M, N, K, tile);
                est.latency_ns = ae.latency_ns;
                est.confidence = 0.3;
                est.model_used = CostModelLevel::ANALYTICAL;
                est.details = "Empirical failed; analytical fallback: " +
                              std::to_string(ae.latency_ns) + "ns";
            }
        } else if (is_conv2d) {
            // Conv2D empirical profiling not yet supported; use analytical
            int64_t batch = 1, oc = 1, ic = 1, oh = 1, ow = 1, kh = 1, kw = 1;

            if (tile.outer_tiles.size() >= 7) {
                batch = tile.outer_tiles[0];
                oc    = tile.outer_tiles[1];
                oh    = tile.outer_tiles[2];
                ow    = tile.outer_tiles[3];
                ic    = tile.outer_tiles[4];
                kh    = tile.outer_tiles[5];
                kw    = tile.outer_tiles[6];
            } else if (tile.outer_tiles.size() >= 4) {
                oc = tile.outer_tiles[0];
                oh = tile.outer_tiles[1]; ow = 1;
                ic = tile.outer_tiles[2];
                kh = tile.outer_tiles[3]; kw = 1;
            }

            AnalyticalEstimate ae = analytical_.estimate_conv2d(
                batch, oc, ic, oh, ow, kh, kw, tile
            );

            est.latency_ns = ae.latency_ns;
            est.confidence = 0.4;
            est.model_used = CostModelLevel::ANALYTICAL;

            std::ostringstream oss;
            oss << "Empirical(conv2d): not yet supported, analytical fallback"
                << " latency=" << ae.latency_ns << "ns";
            est.details = oss.str();
        } else {
            // Generic operation
            int64_t total_elements = tile.inner_volume();
            int64_t compute_ops = 2 * total_elements;
            int64_t bytes_moved = 3 * total_elements * target_.bytes_per_element;

            RooflinePoint rp = roofline_.analyze(compute_ops, bytes_moved);
            est.latency_ns = rp.total_time_ns;
            est.confidence = 0.2;
            est.model_used = CostModelLevel::ROOFLINE_ONLY;

            std::ostringstream oss;
            oss << "Empirical(generic): not supported, roofline fallback"
                << " latency=" << rp.total_time_ns << "ns";
            est.details = oss.str();
        }
        break;
    }

    // ── HYBRID ────────────────────────────────────────────────────────
    case CostModelLevel::HYBRID: {
        // The hybrid model blends analytical and empirical estimates
        // using a confidence-weighted average.
        //
        // When empirical data is available with low variance, it gets
        // high weight.  When variance is high or empirical is unavailable,
        // the analytical estimate dominates.

        // Step 1: Always compute the analytical estimate
        double analytical_ns = 0.0;
        double analytical_conf = 0.0;
        std::string analytical_detail;

        if (is_matmul && tile.inner_tiles.size() >= 3) {
            int64_t M = (tile.outer_tiles.size() > 0 && tile.outer_tiles[0] > 0)
                        ? tile.outer_tiles[0] : tile.inner_tiles[0];
            int64_t N = (tile.outer_tiles.size() > 1 && tile.outer_tiles[1] > 0)
                        ? tile.outer_tiles[1] : tile.inner_tiles[1];
            int64_t K = (tile.outer_tiles.size() > 2 && tile.outer_tiles[2] > 0)
                        ? tile.outer_tiles[2] : tile.inner_tiles[2];

            AnalyticalEstimate ae = analytical_.estimate_matmul(M, N, K, tile);
            analytical_ns = ae.latency_ns;

            double occ_ratio = (target_.gpu.sm.max_warps > 0)
                ? static_cast<double>(ae.occupancy_warps) /
                  static_cast<double>(target_.gpu.sm.max_warps)
                : 0.0;
            analytical_conf = 0.6 + 0.2 * std::min(occ_ratio, 1.0);

            std::ostringstream oss;
            oss << "analytical=" << ae.latency_ns << "ns"
                << " occ=" << ae.occupancy_warps << " warps"
                << " sram=" << (ae.sram_bytes_used / 1024) << "KB";
            analytical_detail = oss.str();
        } else if (is_conv2d) {
            int64_t batch = 1, oc = 1, ic = 1, oh = 1, ow = 1, kh = 1, kw = 1;

            if (tile.outer_tiles.size() >= 7) {
                batch = tile.outer_tiles[0];
                oc    = tile.outer_tiles[1];
                oh    = tile.outer_tiles[2];
                ow    = tile.outer_tiles[3];
                ic    = tile.outer_tiles[4];
                kh    = tile.outer_tiles[5];
                kw    = tile.outer_tiles[6];
            } else if (tile.outer_tiles.size() >= 4) {
                oc = tile.outer_tiles[0];
                oh = tile.outer_tiles[1]; ow = 1;
                ic = tile.outer_tiles[2];
                kh = tile.outer_tiles[3]; kw = 1;
            }

            AnalyticalEstimate ae = analytical_.estimate_conv2d(
                batch, oc, ic, oh, ow, kh, kw, tile
            );
            analytical_ns = ae.latency_ns;
            analytical_conf = 0.5;

            std::ostringstream oss;
            oss << "analytical=" << ae.latency_ns << "ns"
                << " occ=" << ae.occupancy_warps << " warps";
            analytical_detail = oss.str();
        } else {
            // Generic
            int64_t total_elements = tile.inner_volume();
            int64_t compute_ops = 2 * total_elements;
            int64_t bytes_moved = 3 * total_elements * target_.bytes_per_element;
            RooflinePoint rp = roofline_.analyze(compute_ops, bytes_moved);
            analytical_ns = rp.total_time_ns;
            analytical_conf = 0.3;
            analytical_detail = "roofline=" + std::to_string(rp.total_time_ns) + "ns";
        }

        // Step 2: Try empirical profiling
        double empirical_ns = 0.0;
        double empirical_conf = 0.0;
        bool empirical_valid = false;
        std::string empirical_detail;

        if (is_matmul && tile.inner_tiles.size() >= 3) {
            int64_t M = (tile.outer_tiles.size() > 0 && tile.outer_tiles[0] > 0)
                        ? tile.outer_tiles[0] : tile.inner_tiles[0];
            int64_t N = (tile.outer_tiles.size() > 1 && tile.outer_tiles[1] > 0)
                        ? tile.outer_tiles[1] : tile.inner_tiles[1];
            int64_t K = (tile.outer_tiles.size() > 2 && tile.outer_tiles[2] > 0)
                        ? tile.outer_tiles[2] : tile.inner_tiles[2];

            ProfileResult pr = empirical_.profile_matmul(M, N, K, tile, 5, 50);

            if (pr.valid) {
                empirical_ns = pr.mean_latency_ns;
                empirical_valid = true;

                // Confidence from coefficient of variation
                if (pr.mean_latency_ns > 0.0) {
                    double cv = pr.std_dev_ns / pr.mean_latency_ns;
                    empirical_conf = std::max(0.0, std::min(0.9, 1.0 - cv * 3.0));
                }

                if (pr.is_measured) {
                    empirical_conf = std::min(empirical_conf + 0.1, 0.95);
                }
                if (!pr.high_confidence) {
                    empirical_conf *= 0.7;  // Reduce weight of low-confidence fallback
                }

                std::ostringstream oss;
                oss << "empirical=" << pr.mean_latency_ns << "ns"
                    << " stddev=" << pr.std_dev_ns << "ns"
                    << " measured=" << (pr.is_measured ? "yes" : "no")
                    << " high_conf=" << (pr.high_confidence ? "yes" : "no");
                empirical_detail = oss.str();
            }
        }

        // Step 3: Blend the estimates
        if (empirical_valid) {
            // Weighted blend: w_emp * empirical + w_ana * analytical
            double w_emp = empirical_conf;
            double w_ana = analytical_conf * (1.0 - empirical_conf);
            double w_total = w_emp + w_ana;

            if (w_total > 0.0) {
                est.latency_ns = (w_emp * empirical_ns + w_ana * analytical_ns) / w_total;
                est.confidence = std::min(w_total, 1.0);
            } else {
                est.latency_ns = analytical_ns;
                est.confidence = analytical_conf;
            }

            std::ostringstream oss;
            oss << "Hybrid: " << empirical_detail
                << " | " << analytical_detail
                << " | blended=" << est.latency_ns << "ns"
                << " w_emp=" << w_emp << " w_ana=" << w_ana;
            est.details = oss.str();
        } else {
            // No empirical data, use analytical alone
            est.latency_ns = analytical_ns;
            est.confidence = analytical_conf * 0.8;  // Slight penalty for no empirical

            est.details = "Hybrid(analytical-only): " + analytical_detail;
        }
        break;
    }
    }

    return est;
}

} // namespace symplex::costmodel
