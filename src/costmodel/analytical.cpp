// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/costmodel/analytical.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace symplex::costmodel {

// ---------------------------------------------------------------------------
// Helper: ceiling division
// ---------------------------------------------------------------------------

static inline int64_t ceil_div(int64_t num, int64_t den) {
    if (den <= 0) return 0;
    return (num + den - 1) / den;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

AnalyticalCostModel::AnalyticalCostModel(const hardware::HardwareTarget& target)
    : target_(target)
{}

// ---------------------------------------------------------------------------
// Matrix multiplication estimation
// ---------------------------------------------------------------------------

AnalyticalEstimate AnalyticalCostModel::estimate_matmul(
    int64_t M, int64_t N, int64_t K,
    const schedule::TileConfig& tile
) const {
    AnalyticalEstimate est{};
    est.latency_ns        = 0.0;
    est.compute_ns        = 0.0;
    est.sram_load_ns      = 0.0;
    est.sram_store_ns     = 0.0;
    est.hbm_load_ns       = 0.0;
    est.hbm_store_ns      = 0.0;
    est.overlap_factor    = 0.0;
    est.sram_bytes_used   = 0;
    est.register_bytes_used = 0;
    est.occupancy_warps   = 0;

    // Guard: require at least 3 inner tile dimensions (tm, tn, tk)
    if (tile.inner_tiles.size() < 3 ||
        tile.inner_tiles[0] <= 0 ||
        tile.inner_tiles[1] <= 0 ||
        tile.inner_tiles[2] <= 0) {
        return est;
    }

    const int64_t tm = tile.inner_tiles[0];
    const int64_t tn = tile.inner_tiles[1];
    const int64_t tk = tile.inner_tiles[2];
    const int64_t bpe = target_.bytes_per_element;

    // ── SRAM footprint ────────────────────────────────────────────────
    // Per tile, we hold A[tm, tk], B[tk, tn], C[tm, tn] in SRAM.
    // With pipeline_stages * double-buffering, multiply by pipeline_stages.
    const int64_t sram_a  = tm * tk * bpe;
    const int64_t sram_b  = tk * tn * bpe;
    const int64_t sram_c  = tm * tn * bpe;
    const int64_t sram_per_stage = sram_a + sram_b + sram_c;
    est.sram_bytes_used = sram_per_stage * target_.pipeline_stages;

    // ── Register footprint ────────────────────────────────────────────
    // Each thread holds fragments of A, B, C in registers.
    // Conservative estimate: 3 * (tile_m * tile_n) * bpe for the
    // accumulator + 2 operand fragments.  We approximate the register
    // footprint as proportional to the per-warp MMA fragment sizes.
    // For Tensor Core MMA (m x n x k), each warp holds:
    //   A fragment: (m/16) * (k/16) * 16 registers  (16 regs per 16x16 tile)
    //   B fragment: (k/16) * (n/8)  * 8  registers
    //   C fragment: (m/16) * (n/8)  * 8  registers
    // Simplified model: 3 * tm * tn * bpe (A, B, C fragments per warp)
    est.register_bytes_used = 3 * tm * tn * bpe;

    // ── Number of tiles (ceil division) ───────────────────────────────
    const int64_t num_tiles_m = ceil_div(M, tm);
    const int64_t num_tiles_n = ceil_div(N, tn);
    const int64_t num_tiles_k = ceil_div(K, tk);
    const int64_t total_tiles = num_tiles_m * num_tiles_n;

    // ── Per-tile compute ──────────────────────────────────────────────
    // FMA ops per tile = tm * tn * tk, each FMA = 2 FLOPS
    const int64_t ops_per_tile = 2 * tm * tn * tk;

    // Total compute ops across the whole GEMM
    const int64_t total_ops = ops_per_tile * total_tiles * num_tiles_k;

    // Compute time: total_ops / peak_flops (seconds), then to ns
    const double peak_flops = target_.peak_flops_fp16();  // FLOPS/s
    double compute_time_s = (peak_flops > 0.0)
                            ? static_cast<double>(total_ops) / peak_flops
                            : 0.0;
    est.compute_ns = compute_time_s * 1e9;

    // ── HBM traffic ───────────────────────────────────────────────────
    // For each output tile (m_tile, n_tile), we stream over K in chunks of tk.
    // HBM loads per (m_tile, n_tile) tile:
    //   A tile: M tiles * K/tk iterations * (tm * tk * bpe) bytes
    //   B tile: N tiles * K/tk iterations * (tk * tn * bpe) bytes
    // HBM store per (m_tile, n_tile) tile:
    //   C tile: tm * tn * bpe (written once after K reduction complete)

    // Total HBM bytes loaded across all tiles and K-steps
    const int64_t hbm_load_bytes = total_tiles * num_tiles_k * (sram_a + sram_b);
    const int64_t hbm_store_bytes = total_tiles * sram_c;

    const double hbm_bw_bps = target_.gpu.memory.global_bw_gbps * 1e9;  // bytes/s

    double hbm_load_s = (hbm_bw_bps > 0.0)
                        ? static_cast<double>(hbm_load_bytes) / hbm_bw_bps
                        : 0.0;
    double hbm_store_s = (hbm_bw_bps > 0.0)
                         ? static_cast<double>(hbm_store_bytes) / hbm_bw_bps
                         : 0.0;

    est.hbm_load_ns  = hbm_load_s * 1e9;
    est.hbm_store_ns = hbm_store_s * 1e9;

    // ── SRAM traffic ──────────────────────────────────────────────────
    // SRAM load: moving data from SRAM buffers to register fragments.
    // SRAM store: writing accumulated results from registers back to SRAM.
    const double sram_bw_bps = target_.gpu.memory.shared_bw_gbps * 1e9;

    // Per tile: load A and B from SRAM buffer to registers (once per tk step)
    const int64_t sram_load_per_tile = sram_a + sram_b;   // bytes per K-step
    const int64_t total_sram_load = total_tiles * num_tiles_k * sram_load_per_tile;

    // Per tile: store C from registers back to SRAM (once per tile, after K loop)
    const int64_t total_sram_store = total_tiles * sram_c;

    double sram_load_s = (sram_bw_bps > 0.0)
                         ? static_cast<double>(total_sram_load) / sram_bw_bps
                         : 0.0;
    double sram_store_s = (sram_bw_bps > 0.0)
                          ? static_cast<double>(total_sram_store) / sram_bw_bps
                          : 0.0;

    est.sram_load_ns  = sram_load_s * 1e9;
    est.sram_store_ns = sram_store_s * 1e9;

    // ── Software pipelining overlap ───────────────────────────────────
    // With pipeline_stages stages, memory and compute can overlap.
    // The overlap factor represents the fraction of memory latency that
    // can be hidden behind compute.  With S stages, the steady-state
    // overlap hides (S-1)/S of the memory latency for a pipeline-friendly
    // kernel where memory and compute times are balanced.
    //
    //   overlap_factor = (pipeline_stages - 1) / pipeline_stages
    //
    // The actual hidden memory time is:
    //   hidden = overlap_factor * min(memory_time, compute_time)
    //
    // This only applies when both memory and compute are present.
    const double pipeline_stages = static_cast<double>(target_.pipeline_stages);
    est.overlap_factor = (pipeline_stages > 1.0)
                         ? (pipeline_stages - 1.0) / pipeline_stages
                         : 0.0;

    // ── Occupancy ─────────────────────────────────────────────────────
    // Estimate registers per thread from the register footprint and
    // assume a thread-block size of tm * tn (one thread per output element).
    const int64_t threads_per_block = std::min(
        tm * tn,
        target_.gpu.max_threads_per_block
    );
    // Each thread holds roughly: 3 * tm * tn * bpe / (tm * tn * 4) registers
    // (4 bytes per register).  Simplified model for Tensor Core MMA:
    // registers_per_thread ≈ register_bytes_used / (threads_per_block * 4)
    const int64_t regs_per_thread = (threads_per_block > 0)
        ? std::max(est.register_bytes_used / (threads_per_block * 4), int64_t(1))
        : int64_t(32);  // default conservative estimate

    est.occupancy_warps = target_.gpu.sm.compute_occupancy(
        regs_per_thread,
        est.sram_bytes_used
    );

    // ── Occupancy-adjusted compute time ───────────────────────────────
    // If occupancy is less than maximum warps, compute throughput is
    // reduced proportionally (simplified model).
    const double occupancy_ratio = (target_.gpu.sm.max_warps > 0)
        ? static_cast<double>(est.occupancy_warps) /
          static_cast<double>(target_.gpu.sm.max_warps)
        : 1.0;

    // Effective compute time increases as occupancy drops
    double effective_compute_ns = (occupancy_ratio > 0.0)
                                  ? est.compute_ns / occupancy_ratio
                                  : est.compute_ns * 1e6;  // Very high if 0 occupancy

    // ── Total latency with overlap ────────────────────────────────────
    // Raw memory time (HBM load + HBM store)
    double total_memory_ns = est.hbm_load_ns + est.hbm_store_ns;

    // Overlap: memory that can be hidden behind compute
    double hidden_memory_ns = est.overlap_factor *
                             std::min(total_memory_ns, effective_compute_ns);

    // Exposed memory time after overlap
    double exposed_memory_ns = std::max(total_memory_ns - hidden_memory_ns, 0.0);

    // Total latency = max(effective compute, exposed memory) + SRAM overhead
    // SRAM overhead is typically hidden but we add a small fraction
    // (10%) to account for bank conflicts and synchronization.
    double sram_overhead_ns = (est.sram_load_ns + est.sram_store_ns) * 0.1;

    est.latency_ns = std::max(effective_compute_ns, exposed_memory_ns) +
                     sram_overhead_ns;

    return est;
}

// ---------------------------------------------------------------------------
// Conv2D estimation
// ---------------------------------------------------------------------------

AnalyticalEstimate AnalyticalCostModel::estimate_conv2d(
    int64_t batch, int64_t oc, int64_t ic,
    int64_t oh, int64_t ow, int64_t kh, int64_t kw,
    const schedule::TileConfig& tile
) const {
    AnalyticalEstimate est{};
    est.latency_ns          = 0.0;
    est.compute_ns          = 0.0;
    est.sram_load_ns        = 0.0;
    est.sram_store_ns       = 0.0;
    est.hbm_load_ns         = 0.0;
    est.hbm_store_ns        = 0.0;
    est.overlap_factor      = 0.0;
    est.sram_bytes_used     = 0;
    est.register_bytes_used = 0;
    est.occupancy_warps     = 0;

    // Conv2D can be mapped to a GEMM via im2col:
    //   output[batch * oc, oh * ow] += input[batch * ic, kh * kw * ow * oh] * weight[oc, ic * kh * kw]
    // Or more naturally, we treat it as a nested loop:
    //   for each (b, oc_tile, oh_tile, ow_tile):
    //     for each (ic_tile, kh, kw):
    //       compute
    //
    // The tile dimensions we expect:
    //   inner_tiles[0] = tile_oc   (output channels per tile)
    //   inner_tiles[1] = tile_oh   (output height per tile)
    //   inner_tiles[2] = tile_ow   (output width per tile)
    //   inner_tiles[3] = tile_ic   (input channels per tile)  [optional]
    //
    // If only 3 dimensions are provided, we use (tile_oc, tile_oh_ow, tile_ic)
    // where tile_oh_ow represents a combined spatial tile.

    if (tile.inner_tiles.empty()) return est;

    const int64_t bpe = target_.bytes_per_element;

    // Determine tile dimensions based on what's provided
    int64_t tile_oc, tile_spatial, tile_ic;
    if (tile.inner_tiles.size() >= 4) {
        tile_oc      = tile.inner_tiles[0];
        tile_spatial = tile.inner_tiles[1] * tile.inner_tiles[2];  // oh * ow
        tile_ic      = tile.inner_tiles[3];
    } else if (tile.inner_tiles.size() >= 3) {
        tile_oc      = tile.inner_tiles[0];
        tile_spatial = tile.inner_tiles[1];
        tile_ic      = tile.inner_tiles[2];
    } else if (tile.inner_tiles.size() >= 2) {
        tile_oc      = tile.inner_tiles[0];
        tile_spatial = tile.inner_tiles[1];
        tile_ic      = ic;  // Full IC dimension
    } else {
        tile_oc      = tile.inner_tiles[0];
        tile_spatial = oh * ow;
        tile_ic      = ic;
    }

    // Clamp to problem dimensions
    tile_oc      = std::min(tile_oc, oc);
    tile_spatial = std::min(tile_spatial, oh * ow);
    tile_ic      = std::min(tile_ic, ic);

    if (tile_oc <= 0 || tile_spatial <= 0 || tile_ic <= 0) return est;

    // ── SRAM footprint ────────────────────────────────────────────────
    // Input tile:  [batch_per_tile, tile_ic, ih_tile, iw_tile]
    //   where ih_tile = tile_spatial + kh - 1 (receptive field)
    //   simplified: tile_ic * (tile_spatial + kh*kw) * bpe
    // Weight tile: [tile_oc, tile_ic, kh, kw]
    // Output tile: [batch_per_tile, tile_oc, tile_spatial]
    const int64_t sram_input  = tile_ic * (tile_spatial + kh * kw) * bpe;
    const int64_t sram_weight = tile_oc * tile_ic * kh * kw * bpe;
    const int64_t sram_output = tile_oc * tile_spatial * bpe;
    const int64_t sram_per_stage = sram_input + sram_weight + sram_output;
    est.sram_bytes_used = sram_per_stage * target_.pipeline_stages;

    // ── Register footprint ────────────────────────────────────────────
    est.register_bytes_used = 3 * tile_oc * tile_spatial * bpe;

    // ── Total compute ops ─────────────────────────────────────────────
    // ops = 2 * batch * oc * oh * ow * ic * kh * kw
    const int64_t total_ops = 2 * batch * oc * oh * ow * ic * kh * kw;

    const double peak_flops = target_.peak_flops_fp16();
    double compute_time_s = (peak_flops > 0.0)
                            ? static_cast<double>(total_ops) / peak_flops
                            : 0.0;
    est.compute_ns = compute_time_s * 1e9;

    // ── Number of tiles ───────────────────────────────────────────────
    const int64_t num_tiles_oc      = ceil_div(oc, tile_oc);
    const int64_t num_tiles_spatial = ceil_div(oh * ow, tile_spatial);
    const int64_t num_tiles_ic      = ceil_div(ic, tile_ic);
    const int64_t total_tiles = batch * num_tiles_oc * num_tiles_spatial;

    // ── HBM traffic ───────────────────────────────────────────────────
    // For each (oc_tile, spatial_tile, batch) we iterate over IC tiles
    const int64_t hbm_load_per_tile_ic = sram_input + sram_weight;
    const int64_t hbm_load_bytes  = total_tiles * num_tiles_ic * hbm_load_per_tile_ic;
    const int64_t hbm_store_bytes = total_tiles * sram_output;

    const double hbm_bw_bps = target_.gpu.memory.global_bw_gbps * 1e9;

    double hbm_load_s = (hbm_bw_bps > 0.0)
                        ? static_cast<double>(hbm_load_bytes) / hbm_bw_bps
                        : 0.0;
    double hbm_store_s = (hbm_bw_bps > 0.0)
                         ? static_cast<double>(hbm_store_bytes) / hbm_bw_bps
                         : 0.0;

    est.hbm_load_ns  = hbm_load_s * 1e9;
    est.hbm_store_ns = hbm_store_s * 1e9;

    // ── SRAM traffic ──────────────────────────────────────────────────
    // SRAM load is the data movement WITHIN shared memory (from SRAM
    // buffers to register fragments), NOT the same as HBM load.
    // Per tile per IC iteration, the SRAM-to-register traffic is:
    //   Input tile:   kh * kw * tile_ic  * bpe   (receptive field read from SRAM)
    //   Kernel tile:  kh * kw * tile_oc  * bpe   (weight tile read from SRAM)
    //   Output partial: tile_spatial * tile_oc * bpe (accumulator read-back)
    const double sram_bw_bps = target_.gpu.memory.shared_bw_gbps * 1e9;

    const int64_t sram_load_input   = kh * kw * tile_ic * bpe;
    const int64_t sram_load_kernel  = kh * kw * tile_oc * bpe;
    const int64_t sram_load_output  = tile_spatial * tile_oc * bpe;
    const int64_t sram_load_per_tile_ic = sram_load_input + sram_load_kernel + sram_load_output;
    const int64_t total_sram_load = total_tiles * num_tiles_ic * sram_load_per_tile_ic;
    const int64_t total_sram_store = total_tiles * sram_output;

    double sram_load_s = (sram_bw_bps > 0.0)
                         ? static_cast<double>(total_sram_load) / sram_bw_bps
                         : 0.0;
    double sram_store_s = (sram_bw_bps > 0.0)
                          ? static_cast<double>(total_sram_store) / sram_bw_bps
                          : 0.0;

    est.sram_load_ns  = sram_load_s * 1e9;
    est.sram_store_ns = sram_store_s * 1e9;

    // ── Pipelining overlap ────────────────────────────────────────────
    const double pipeline_stages = static_cast<double>(target_.pipeline_stages);
    est.overlap_factor = (pipeline_stages > 1.0)
                         ? (pipeline_stages - 1.0) / pipeline_stages
                         : 0.0;

    // ── Occupancy ─────────────────────────────────────────────────────
    const int64_t threads_per_block = std::min(
        tile_oc * tile_spatial,
        target_.gpu.max_threads_per_block
    );
    const int64_t regs_per_thread = (threads_per_block > 0)
        ? std::max(est.register_bytes_used / (threads_per_block * 4), int64_t(1))
        : int64_t(32);

    est.occupancy_warps = target_.gpu.sm.compute_occupancy(
        regs_per_thread,
        est.sram_bytes_used
    );

    const double occupancy_ratio = (target_.gpu.sm.max_warps > 0)
        ? static_cast<double>(est.occupancy_warps) /
          static_cast<double>(target_.gpu.sm.max_warps)
        : 1.0;

    double effective_compute_ns = (occupancy_ratio > 0.0)
                                  ? est.compute_ns / occupancy_ratio
                                  : est.compute_ns * 1e6;

    // ── Total latency ─────────────────────────────────────────────────
    double total_memory_ns = est.hbm_load_ns + est.hbm_store_ns;
    double hidden_memory_ns = est.overlap_factor *
                             std::min(total_memory_ns, effective_compute_ns);
    double exposed_memory_ns = std::max(total_memory_ns - hidden_memory_ns, 0.0);
    double sram_overhead_ns = (est.sram_load_ns + est.sram_store_ns) * 0.1;

    est.latency_ns = std::max(effective_compute_ns, exposed_memory_ns) +
                     sram_overhead_ns;

    return est;
}

} // namespace symplex::costmodel
