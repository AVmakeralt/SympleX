// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// ═══════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE NOTE: This C++ implementation is now a LEGACY FALLBACK.
//
// The primary polyhedral engine is implemented in Rust (polyhedral.rs) which
// provides a comprehensive, production-quality optimizer including:
//   - N-dimensional TensorAccessRelation with matrix-vector affine mapping
//   - UTVPI exact integer solver (replaces pure Fourier-Motzkin)
//   - Hierarchical 3-tier tiling (L3/L2 → L1 → register micro-kernel)
//   - AMX/SME MatrixOuterProduct hardware targets
//   - Software pipelining with double buffering
//   - FlashAttention-style online softmax reductions
//   - Reverse-mode automatic differentiation (Polyhedral AD Engine)
//   - Parametric polyhedral boundaries (dynamic shapes for ML)
//   - Exact rational arithmetic (FieldFraction) for algebraic proofs
//   - Mixed-precision polyhedral spaces (FP16/BF16/INT8/FP4)
//   - Stochastic loop bounds for probabilistic programming
//   - Transcendental function fusion (VML/SVML)
//   - Ragged tensor & sparsity mapping (CSR)
//   - Time-space wavefront tiling (PDE/stencil)
//
// The Rust engine is wired into the JIT pipeline via:
//   - tracing_jit.rs: optimize_trace_polyhedral() and optimize_trace_polyhedral_specialized()
//   - phase3_jit.rs: SIMD dispatch + multi-stride address generation
//   - phase6_simd.rs: SIMD code generation for VectorPack/MatrixOuterProduct hints
//
// The C++ implementation below is retained for:
//   1. Standalone testing without the Rust runtime
//   2. Fallback when the Rust FFI is unavailable
//   3. Reference implementation for the Pluto/Feautrier scheduling algorithms
// ═══════════════════════════════════════════════════════════════════════════════

#include "symplex/polyhedral/polyhedral_optimizer.h"
#include "symplex/polyhedral/rust_bridge.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <limits>

namespace symplex::polyhedral {

// ── Constructor ──────────────────────────────────────────────────────────────

PolyhedralOptimizer::PolyhedralOptimizer()
    : config_{}, use_rust_engine_(true)
{
    // Try to initialize the Rust polyhedral engine
    rust::poly_engine_init();
}

PolyhedralOptimizer::PolyhedralOptimizer(Config config)
    : config_(std::move(config)), use_rust_engine_(true)
{
    rust::poly_engine_init();
}

// ── Main entry point ─────────────────────────────────────────────────────────

PolyhedralOptimizer::ScheduleResult PolyhedralOptimizer::optimize(
    const IterationSpace& ispace
) {
    ScheduleResult result;

    if (ispace.num_statements() == 0) {
        result.valid = false;
        return result;
    }

    // ── Try the Rust polyhedral engine first ──────────────────────────────
    // If the Rust engine is available and produces results, use its
    // superior scheduling (UTVPI solver, N-dim tensor access, etc.).
    // Fall back to the C++ legacy implementation if Rust returns empty.
    if (use_rust_engine_) {
        try {
            auto rust_result = optimize_via_rust(ispace);
            if (rust_result.valid) {
                return rust_result;
            }
        } catch (...) {
            // Rust engine failed — fall through to C++ implementation
            std::cerr << "[SympleX] Rust poly engine failed, falling back to C++\n";
        }
    }

    // ── Legacy C++ fallback ───────────────────────────────────────────────
    // Step 1: Ensure dependencies are computed.
    // The iteration space should already have deps; we rely on them.

    // Step 2: Compute schedule maps using Pluto-like algorithm.
    // Pluto gives multi-dimensional schedules that enable tiling.
    auto schedule_maps = pluto_schedule(ispace);

    // Fall back to Feautrier if Pluto produces nothing useful.
    if (schedule_maps.empty() ||
        std::all_of(schedule_maps.begin(), schedule_maps.end(),
            [](const AffineMap& m) { return m.n_out() == 0; }))
    {
        schedule_maps = feautrier_schedule(ispace);
    }

    if (schedule_maps.empty()) {
        result.valid = false;
        return result;
    }

    result.schedule_maps = schedule_maps;

    // Step 3: Determine which dimensions are parallelizable.
    size_t ndim = ispace.statement(0).domain.ndim();
    result.parallel_dims.resize(ndim, false);

    if (config_.enable_parallelism) {
        for (size_t d = 0; d < ndim; ++d) {
            result.parallel_dims[d] = ispace.is_parallelizable(d);
        }
    }

    // Step 4: Compute tile sizes if tiling is enabled.
    std::vector<size_t> band_dims;
    for (size_t d = 0; d < ndim; ++d) {
        band_dims.push_back(d);
    }

    if (config_.enable_tiling && !schedule_maps.empty()) {
        result.tile_sizes = compute_tile_sizes(ispace, schedule_maps, band_dims);
    } else {
        // No tiling: use full dimension sizes as tile sizes.
        auto bounds = ispace.statement(0).domain.bounds();
        for (const auto& [lo, hi] : bounds) {
            result.tile_sizes.push_back(hi - lo + 1);
        }
    }

    // Step 5: Build the schedule tree.
    result.schedule_tree = build_schedule_tree(ispace, schedule_maps);

    // Step 6: Estimate SRAM footprint.
    result.estimated_sram_bytes = static_cast<int64_t>(
        ispace.estimate_sram_footprint(result.tile_sizes, 2 /* FP16 */, true)
    );

    // Step 7: Estimate latency.
    result.estimated_latency_ns = estimate_latency(ispace, result.tile_sizes);

    result.valid = true;
    return result;
}

// ── Feautrier's 1-d scheduling algorithm ─────────────────────────────────────
//
// For each statement S_j, we solve the LP:
//   maximize  (parallelism: prefer zero coefficients)
//   subject to  theta . d >= 1  for all dependency vectors d
//
// We use a greedy heuristic:
//   1. Start with theta = 0
//   2. For each dependency d where theta . d < 1, increment
//      the coefficient of the first non-zero dimension of d
//      until theta . d >= 1.
//   3. Prefer making earlier dimensions zero (to enable parallelism
//      in later dimensions).

std::vector<AffineMap> PolyhedralOptimizer::feautrier_schedule(
    const IterationSpace& ispace
) {
    std::vector<AffineMap> schedules;
    auto all_deps = gather_all_dep_vectors(ispace);

    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        size_t ndim = ispace.statement(s).domain.ndim();
        auto sched = solve_feautrier_row(ndim, all_deps);
        schedules.push_back(std::move(sched));
    }

    return schedules;
}

AffineMap PolyhedralOptimizer::solve_feautrier_row(
    size_t ndim,
    const std::vector<DependencyVector>& deps
) {
    // The schedule is: theta(i) = sum_j theta_j * i_j
    // We want: theta . d >= 1 for all dependency vectors d.
    // Greedy strategy: process deps in lexicographic order.
    // For each dep, find the first non-zero component and set
    // the corresponding theta coefficient to satisfy the constraint.

    std::vector<int64_t> theta(ndim, 0);

    // If no deps, the identity schedule is trivially valid.
    if (deps.empty()) {
        // Return identity: theta(i) = i_0 (first dimension)
        if (ndim > 0) theta[0] = 1;
    } else {
        // Process each dependency
        for (const auto& dv : deps) {
            if (dv.components.size() < ndim) continue;

            // Compute current theta . d
            int64_t dot = 0;
            for (size_t j = 0; j < ndim; ++j) {
                dot += theta[j] * dv.components[j];
            }

            // If already satisfied, skip
            if (dot >= 1) continue;

            // Need to increase the dot product.
            // Strategy: find the last non-zero component of d (innermost dim)
            // and adjust its theta coefficient. This leaves outer dims free
            // for parallelism.
            // Feautrier's original algorithm prefers the outermost non-zero
            // component, but for parallelism we prefer the innermost.
            // We use a heuristic: find the dimension with the largest
            // absolute value in d and adjust theta there.

            int64_t deficit = 1 - dot;
            size_t best_dim = ndim;  // invalid
            int64_t best_abs = 0;

            for (size_t j = 0; j < ndim; ++j) {
                if (dv.components[j] != 0 && std::abs(dv.components[j]) > best_abs) {
                    best_abs = std::abs(dv.components[j]);
                    best_dim = j;
                }
            }

            if (best_dim < ndim) {
                int64_t d_comp = dv.components[best_dim];
                // We need: theta[best_dim] * d_comp >= deficit + theta[best_dim] * d_comp_current
                // Since we already computed dot with the current theta, we just need:
                // theta[best_dim] * d_comp >= deficit (approximate, since we can adjust more)
                int64_t needed = (deficit + std::abs(d_comp) - 1) / std::abs(d_comp);
                if (d_comp > 0) {
                    theta[best_dim] += needed;
                } else {
                    theta[best_dim] -= needed;
                }
            }
        }

        // If theta is still all zeros (e.g., deps have all zero components),
        // fall back to identity on first dimension.
        if (std::all_of(theta.begin(), theta.end(),
                        [](int64_t v) { return v == 0; })) {
            if (ndim > 0) theta[0] = 1;
        }
    }

    // Construct the 1-d AffineMap: theta(i) = theta . i
    AffineMap sched(ndim, 1);
    for (size_t j = 0; j < ndim; ++j) {
        sched.matrix_at(0, j) = theta[j];
    }
    sched.offset_at(0) = 0;

    return sched;
}

// ── Pluto-like multi-dimensional scheduling ──────────────────────────────────
//
// Iteratively finds scheduling rows (dimensions of the schedule).
// For each row, we solve:
//   find c such that  c . d >= 0  for all dependency vectors d
//   (with at least one c . d > 0, i.e., the row "carries" some dependency)
//   maximizing the number of zero coefficients (parallelism).
//
// After finding a row, the carried dependencies are removed and the
// process repeats to find the next row.

std::vector<AffineMap> PolyhedralOptimizer::pluto_schedule(
    const IterationSpace& ispace
) {
    if (ispace.num_statements() == 0) return {};

    size_t ndim = ispace.statement(0).domain.ndim();
    auto all_deps = gather_all_dep_vectors(ispace);

    // Collect all rows across all statements.
    // For simplicity, we compute one unified schedule for all statements.
    std::vector<std::vector<int64_t>> rows;

    auto remaining_deps = all_deps;

    // Find up to ndim scheduling rows.
    for (size_t row_idx = 0; row_idx < ndim; ++row_idx) {
        if (remaining_deps.empty()) {
            // No more deps to carry. Add identity rows for remaining dims.
            // This creates a permutable band.
            for (size_t d = row_idx; d < ndim; ++d) {
                std::vector<int64_t> identity_row(ndim, 0);
                identity_row[d] = 1;
                rows.push_back(std::move(identity_row));
            }
            break;
        }

        // Find the best scheduling row using a greedy search.
        // We try each basis vector e_j and also combinations.
        std::vector<int64_t> best_row(ndim, 0);
        int64_t best_cost = -1;

        // Try each canonical basis vector.
        for (size_t j = 0; j < ndim; ++j) {
            std::vector<int64_t> candidate(ndim, 0);
            candidate[j] = 1;

            if (row_is_legal(candidate, remaining_deps)) {
                int64_t cost = pluto_cost(candidate, remaining_deps);
                if (cost > best_cost) {
                    best_cost = cost;
                    best_row = candidate;
                }
            }

            // Also try -e_j
            candidate[j] = -1;
            if (row_is_legal(candidate, remaining_deps)) {
                int64_t cost = pluto_cost(candidate, remaining_deps);
                if (cost > best_cost) {
                    best_cost = cost;
                    best_row = candidate;
                }
            }
        }

        // Try combinations of two basis vectors (for skewing).
        for (size_t j1 = 0; j1 < ndim; ++j1) {
            for (size_t j2 = 0; j2 < ndim; ++j2) {
                if (j1 == j2) continue;

                // Try skewing: c = e_j1 + f * e_j2 for small f
                for (int64_t f = -2; f <= 2; ++f) {
                    if (f == 0) continue;
                    std::vector<int64_t> candidate(ndim, 0);
                    candidate[j1] = 1;
                    candidate[j2] = f;

                    if (row_is_legal(candidate, remaining_deps)) {
                        int64_t cost = pluto_cost(candidate, remaining_deps);
                        // Prefer simpler rows (fewer non-zero coefficients)
                        // for the same cost.
                        int64_t nnz = 0;
                        for (auto c : candidate) if (c != 0) ++nnz;
                        cost = cost * 10 - nnz;  // Simple weighting

                        if (cost > best_cost) {
                            best_cost = cost;
                            best_row = candidate;
                        }
                    }
                }
            }
        }

        if (best_cost < 0) {
            // Could not find a legal row. Use identity for remaining dims.
            for (size_t d = row_idx; d < ndim; ++d) {
                std::vector<int64_t> identity_row(ndim, 0);
                identity_row[d] = 1;
                rows.push_back(std::move(identity_row));
            }
            break;
        }

        rows.push_back(best_row);

        // Remove dependencies that are carried by this row
        // (i.e., row . d > 0).
        std::vector<DependencyVector> new_remaining;
        for (const auto& dv : remaining_deps) {
            int64_t dot = 0;
            for (size_t j = 0; j < ndim && j < dv.components.size(); ++j) {
                dot += best_row[j] * dv.components[j];
            }
            // If dot == 0, the dependency is NOT carried by this row;
            // it remains for subsequent rows to handle.
            if (dot == 0) {
                new_remaining.push_back(dv);
            }
        }
        remaining_deps = std::move(new_remaining);
    }

    // If we didn't get enough rows, pad with identity rows.
    while (rows.size() < ndim) {
        std::vector<int64_t> identity_row(ndim, 0);
        identity_row[rows.size()] = 1;
        rows.push_back(std::move(identity_row));
    }

    // Construct one AffineMap per statement (all statements share the
    // same schedule in this simplified implementation).
    std::vector<AffineMap> schedules;
    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        AffineMap sched(ndim, ndim);
        for (size_t r = 0; r < rows.size() && r < ndim; ++r) {
            for (size_t j = 0; j < ndim; ++j) {
                sched.matrix_at(r, j) = rows[r][j];
            }
            sched.offset_at(r) = 0;
        }
        schedules.push_back(std::move(sched));
    }

    return schedules;
}

// ── Pluto cost function ──────────────────────────────────────────────────────

int64_t PolyhedralOptimizer::pluto_cost(
    const std::vector<int64_t>& row,
    const std::vector<DependencyVector>& deps
) const {
    // Count the number of dependencies that are carried by this row
    // (i.e., row . d > 0).  A good row carries many dependencies.
    int64_t carried = 0;
    for (const auto& dv : deps) {
        int64_t dot = 0;
        for (size_t j = 0; j < row.size() && j < dv.components.size(); ++j) {
            dot += row[j] * dv.components[j];
        }
        if (dot > 0) ++carried;
    }

    // Bonus: prefer rows with more zero coefficients (more parallelism).
    int64_t nnz = 0;
    for (auto c : row) if (c != 0) ++nnz;
    int64_t parallelism_bonus = static_cast<int64_t>(row.size()) - nnz;

    return carried * 100 + parallelism_bonus;
}

bool PolyhedralOptimizer::row_is_legal(
    const std::vector<int64_t>& row,
    const std::vector<DependencyVector>& deps
) const {
    // A row is legal if for every dependency vector d:
    //   row . d >= 0
    // This ensures the transformation preserves dependency direction.
    for (const auto& dv : deps) {
        int64_t dot = 0;
        for (size_t j = 0; j < row.size() && j < dv.components.size(); ++j) {
            dot += row[j] * dv.components[j];
        }
        if (dot < 0) return false;
    }
    return true;
}

// ── Compute optimal tile sizes ───────────────────────────────────────────────
//
// Uses an analytical model to find tile sizes that:
// 1. Fit within the SRAM budget
// 2. Align to MMA fragment dimensions (for Tensor Core utilization)
// 3. Maximize compute intensity (FLOPS/byte)

std::vector<int64_t> PolyhedralOptimizer::compute_tile_sizes(
    const IterationSpace& ispace,
    const std::vector<AffineMap>& schedules,
    const std::vector<size_t>& band_dims
) {
    if (ispace.num_statements() == 0) return {};

    size_t ndim = band_dims.size();
    auto bounds = ispace.statement(0).domain.bounds();

    // Start with maximum tile sizes (bounded by iteration space).
    std::vector<int64_t> tile_sizes(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        auto [lo, hi] = bounds[d];
        tile_sizes[d] = std::min(hi - lo + 1, config_.max_tile_dim);
    }

    // Align tile sizes to MMA fragment dimensions.
    if (config_.target_tensor_cores) {
        if (ndim >= 3) {
            // Matmul-like: align inner 3 dims to (mma_m, mma_n, mma_k).
            // For higher-dimensional spaces (e.g., conv2d), align the
            // last three dims (the inner compute dimensions).
            size_t m_dim = ndim >= 3 ? ndim - 3 : 0;
            size_t n_dim = ndim >= 2 ? ndim - 2 : 1;
            size_t k_dim = ndim - 1;
            tile_sizes[m_dim] = align_to_mma(tile_sizes[m_dim], config_.mma_m);
            tile_sizes[n_dim] = align_to_mma(tile_sizes[n_dim], config_.mma_n);
            tile_sizes[k_dim] = align_to_mma(tile_sizes[k_dim], config_.mma_k);
        } else if (ndim >= 2) {
            tile_sizes[0] = align_to_mma(tile_sizes[0], config_.mma_m);
            tile_sizes[1] = align_to_mma(tile_sizes[1], config_.mma_n);
        } else if (ndim >= 1) {
            tile_sizes[0] = align_to_mma(tile_sizes[0], config_.mma_m);
        }
    }

    // Iteratively reduce tile sizes until they fit in SRAM.
    // We reduce the dimension with the largest tile first.
    for (int iteration = 0; iteration < 100; ++iteration) {
        size_t sram_footprint = ispace.estimate_sram_footprint(
            tile_sizes, 2 /* FP16 */, true /* double_buffer */
        );

        if (static_cast<int64_t>(sram_footprint) <= config_.sram_budget_bytes) {
            break;  // Fits in SRAM.
        }

        // Find the dimension with the largest tile to reduce.
        size_t max_dim = 0;
        int64_t max_size = tile_sizes[0];
        for (size_t d = 1; d < ndim; ++d) {
            if (tile_sizes[d] > max_size) {
                max_size = tile_sizes[d];
                max_dim = d;
            }
        }

        if (max_size <= 1) break;  // Cannot reduce further.

        // Reduce by the MMA alignment factor.
        int64_t mma_step = config_.mma_m;
        if (ndim >= 3 && max_dim == 1) mma_step = config_.mma_n;
        if (ndim >= 3 && max_dim == 2) mma_step = config_.mma_k;

        tile_sizes[max_dim] = std::max(int64_t(1), tile_sizes[max_dim] - mma_step);

        // Re-align.
        if (config_.target_tensor_cores) {
            int64_t mma_dim = config_.mma_m;
            if (ndim >= 3 && max_dim == 1) mma_dim = config_.mma_n;
            if (ndim >= 3 && max_dim == 2) mma_dim = config_.mma_k;
            tile_sizes[max_dim] = align_to_mma(tile_sizes[max_dim], mma_dim);
        }
    }

    return tile_sizes;
}

// ── Legality check ───────────────────────────────────────────────────────────

bool PolyhedralOptimizer::is_legal(
    const AffineMap& T,
    const IterationSpace& ispace
) {
    return ispace.is_valid_transformation(T);
}

// ── Build schedule tree ─────────────────────────────────────────────────────

schedule::ScheduleTreePtr PolyhedralOptimizer::build_schedule_tree(
    const IterationSpace& ispace,
    const std::vector<AffineMap>& schedules
) {
    using namespace schedule;

    // Create root: DOMAIN node
    auto root = ScheduleTree::create(ScheduleNodeType::DOMAIN);
    root->set_domain_name(ispace.name());

    if (ispace.num_statements() == 0) {
        auto leaf = root->add_child(ScheduleNodeType::LEAF);
        return root;
    }

    size_t ndim = ispace.statement(0).domain.ndim();

    // Create a band node for the schedule.
    auto band = root->add_child(ScheduleNodeType::BAND);

    // Populate band members from the schedule map.
    // The schedule is an AffineMap with ndim outputs.
    // Each output corresponds to one loop in the band.
    if (!schedules.empty()) {
        const auto& sched = schedules[0];  // Use first statement's schedule.
        size_t n_loops = sched.n_out();

        band->band_data().members.resize(n_loops);
        band->band_data().permutable = true;

        for (size_t r = 0; r < n_loops; ++r) {
            auto& member = band->band_data().members[r];
            member.coefficients.resize(ndim, 0);
            for (size_t j = 0; j < ndim; ++j) {
                member.coefficients[j] = sched.matrix()[r][j];
            }
            member.constant = sched.offset()[r];
        }
    } else {
        // No schedule: use identity.
        band->band_data().members.resize(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            band->band_data().members[d].coefficients.resize(ndim, 0);
            band->band_data().members[d].coefficients[d] = 1;
            band->band_data().members[d].constant = 0;
        }
        band->band_data().permutable = true;
    }

    // Mark parallel dimensions.
    if (config_.enable_parallelism) {
        for (size_t d = 0; d < band->band_data().members.size(); ++d) {
            if (d < ndim && ispace.is_parallelizable(d)) {
                band->mark_parallel(d);
            } else {
                band->mark_sequential(d);
            }
        }
    } else {
        // All sequential.
        for (size_t d = 0; d < band->band_data().members.size(); ++d) {
            band->mark_sequential(d);
        }
    }

    // For each statement, add a FILTER + LEAF subtree.
    if (ispace.num_statements() == 1) {
        auto leaf = band->add_child(ScheduleNodeType::LEAF);
    } else {
        // Multiple statements: use a SEQUENCE under the band.
        auto seq = band->add_child(ScheduleNodeType::SEQUENCE);
        for (size_t s = 0; s < ispace.num_statements(); ++s) {
            auto filter = seq->add_child(ScheduleNodeType::FILTER);
            filter->filter_data().statement_name = ispace.statement(s).name;

            // Set bounds from the statement's domain.
            auto bounds = ispace.statement(s).domain.bounds();
            filter->filter_data().lower_bounds.resize(bounds.size());
            filter->filter_data().upper_bounds.resize(bounds.size());
            for (size_t d = 0; d < bounds.size(); ++d) {
                filter->filter_data().lower_bounds[d] = bounds[d].first;
                filter->filter_data().upper_bounds[d] = bounds[d].second;
            }

            auto leaf = filter->add_child(ScheduleNodeType::LEAF);
        }
    }

    return root;
}

// ── Estimate latency ─────────────────────────────────────────────────────────
//
// Uses a simplified roofline model:
//   latency = max(compute_time, memory_time)
//   compute_time = total_flops / peak_flops
//   memory_time = bytes_moved / peak_bandwidth

double PolyhedralOptimizer::estimate_latency(
    const IterationSpace& ispace,
    const std::vector<int64_t>& tile_sizes
) {
    if (ispace.num_statements() == 0 || tile_sizes.empty()) return 0.0;

    // Estimate total FLOPS: product of iteration space dimensions * 2 (FMA)
    auto bounds = ispace.statement(0).domain.bounds();
    int64_t total_iters = 1;
    for (const auto& [lo, hi] : bounds) {
        total_iters *= (hi - lo + 1);
    }
    // Assume 2 FLOPS per iteration (multiply-add).
    double total_flops = static_cast<double>(total_iters) * 2.0;

    // Estimate bytes moved: number of tensor elements accessed * bytes_per_element.
    // For matmul: read A tile + read B tile + write C tile.
    int64_t tile_volume = 1;
    for (auto ts : tile_sizes) {
        tile_volume *= ts;
    }
    // Assume 3 tensors (A, B, C), 2 bytes each (FP16).
    double bytes_moved = static_cast<double>(tile_volume) * 3.0 * 2.0;

    // Roofline parameters (A100-like defaults).
    double peak_tflops = 312.0;       // A100 FP16 Tensor Core peak
    double peak_bw_gbps = 2039.0;     // A100 HBM bandwidth

    double compute_time_ns = total_flops / (peak_tflops * 1e12) * 1e9;
    double memory_time_ns = bytes_moved / (peak_bw_gbps * 1e9) * 1e9;

    // Latency is dominated by the bottleneck.
    return std::max(compute_time_ns, memory_time_ns);
}

// ── Helper: gather all dependency vectors ────────────────────────────────────

std::vector<DependencyVector> PolyhedralOptimizer::gather_all_dep_vectors(
    const IterationSpace& ispace
) const {
    std::vector<DependencyVector> all;

    for (const auto& dep : ispace.raw_deps()) {
        for (const auto& dv : dep.vectors()) {
            all.push_back(dv);
        }
    }
    for (const auto& dep : ispace.war_deps()) {
        for (const auto& dv : dep.vectors()) {
            all.push_back(dv);
        }
    }
    for (const auto& dep : ispace.waw_deps()) {
        for (const auto& dv : dep.vectors()) {
            all.push_back(dv);
        }
    }

    return all;
}

// ── Helper: align tile size to MMA dimension ─────────────────────────────────

int64_t PolyhedralOptimizer::align_to_mma(int64_t size, int64_t mma_dim) const {
    if (mma_dim <= 0) return size;
    if (size <= 0) return 1;  // Minimum tile size is 1.
    // Round down to nearest multiple of mma_dim.
    int64_t aligned = (size / mma_dim) * mma_dim;
    // If rounding produces 0, use the original size (don't align)
    // or the MMA dimension, whichever is smaller.
    if (aligned == 0) {
        return std::min(size, mma_dim);
    }
    return aligned;
}

// ── Rust engine integration ──────────────────────────────────────────────────

PolyhedralOptimizer::ScheduleResult PolyhedralOptimizer::optimize_via_rust(
    const IterationSpace& ispace
) {
    ScheduleResult result;

    // Serialize the iteration space into a simple instruction stream.
    // For now, we create a minimal serialized trace that represents the
    // loop nest structure.  The Rust engine will parse this and run its
    // full polyhedral optimization pipeline.
    std::vector<uint8_t> instr_data;

    // Serialize each statement's loop bounds as LoadI64 + BinOp pairs.
    // The Rust engine's `optimize_trace_polyhedral()` will extract the
    // SCoP structure from these instructions.
    //
    // Format per loop dimension i:
    //   LoadI64(slot=i*3+0, lower_bound)    → opcode 0x01
    //   LoadI64(slot=i*3+1, upper_bound)    → opcode 0x01
    //   BinOp(slot=i*3+2, Sub, i*3+1, i*3+0) → opcode 0x10

    if (ispace.num_statements() == 0) {
        result.valid = false;
        return result;
    }

    auto bounds = ispace.statement(0).domain.bounds();
    uint16_t slot = 0;
    for (size_t d = 0; d < bounds.size(); ++d) {
        auto [lo, hi] = bounds[d];

        // LoadI64(slot, lo)
        instr_data.push_back(0x01);
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&slot),
            reinterpret_cast<const uint8_t*>(&slot) + sizeof(slot));
        int64_t lo_val = lo;
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&lo_val),
            reinterpret_cast<const uint8_t*>(&lo_val) + sizeof(lo_val));
        slot++;

        // LoadI64(slot, hi)
        instr_data.push_back(0x01);
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&slot),
            reinterpret_cast<const uint8_t*>(&slot) + sizeof(slot));
        int64_t hi_val = hi;
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&hi_val),
            reinterpret_cast<const uint8_t*>(&hi_val) + sizeof(hi_val));
        slot++;

        // BinOp(slot, Sub, hi_slot, lo_slot)
        uint16_t hi_slot = slot - 1;
        uint16_t lo_slot = slot - 2;
        instr_data.push_back(0x10);
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&slot),
            reinterpret_cast<const uint8_t*>(&slot) + sizeof(slot));
        instr_data.push_back(1); // BinOpKind::Sub
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&hi_slot),
            reinterpret_cast<const uint8_t*>(&hi_slot) + sizeof(hi_slot));
        instr_data.insert(instr_data.end(),
            reinterpret_cast<const uint8_t*>(&lo_slot),
            reinterpret_cast<const uint8_t*>(&lo_slot) + sizeof(lo_slot));
        slot++;
    }

    // Build config
    rust::RustPolyConfig config;
    config.domain = rust::MathDomain::RealFloat;
    config.target = rust::HardwareTarget::ServerX86;
    config.compute_type = rust::ElementType::FP32;
    config.element_bytes = 4;
    config.enable_flash_attention = true;
    config.enable_transcendental_fusion = true;
    config.enable_double_buffering = true;
    config.enable_mixed_precision = false;
    config.enable_ad = false;

    // Call the Rust engine using the RAII wrapper
    auto rust_result = rust::poly_optimize_trace_wrapper(
        instr_data.data(), instr_data.size(), config);

    if (!rust_result.success) {
        result.valid = false;
        return result;
    }

    // Translate the Rust result into the C++ ScheduleResult.
    // Use the micro-kernel config from the Rust engine.
    result.valid = true;
    result.estimated_latency_ns = 0.0; // Rust estimates differently

    // Extract tile sizes from the Rust result
    size_t ndim = bounds.size();
    if (rust_result.tile_m > 0) {
        // The Rust engine returns a single 3D micro-kernel tile config.
        // Map these to the iteration space dimensions.
        result.tile_sizes.resize(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            if (d == 0) result.tile_sizes[d] = std::min(static_cast<int64_t>(rust_result.tile_m), bounds[d].second - bounds[d].first + 1);
            else if (d == 1) result.tile_sizes[d] = std::min(static_cast<int64_t>(rust_result.tile_n), bounds[d].second - bounds[d].first + 1);
            else if (d == 2) result.tile_sizes[d] = std::min(static_cast<int64_t>(rust_result.tile_k), bounds[d].second - bounds[d].first + 1);
            else result.tile_sizes[d] = bounds[d].second - bounds[d].first + 1;
        }
    } else {
        // No tiling from Rust engine — use full dimension sizes
        for (const auto& [lo, hi] : bounds) {
            result.tile_sizes.push_back(hi - lo + 1);
        }
    }

    // Parallel dimensions — assume outermost dimension is parallel
    result.parallel_dims.resize(ndim, false);
    if (config_.enable_parallelism && ndim > 0) {
        result.parallel_dims[0] = true;
    }

    // Build a simple schedule tree
    result.schedule_tree = build_schedule_tree(ispace, std::vector<AffineMap>());

    // Estimate SRAM footprint
    result.estimated_sram_bytes = static_cast<int64_t>(
        ispace.estimate_sram_footprint(result.tile_sizes, 2, true));

    // Store the estimated GFLOPS from the Rust roofline model
    // (used by downstream code generators for scheduling decisions)
    result.estimated_gflops = rust_result.estimated_gflops;

    // Store micro-kernel config from Rust
    result.micro_kernel_tile_m = rust_result.tile_m;
    result.micro_kernel_tile_n = rust_result.tile_n;
    result.micro_kernel_tile_k = rust_result.tile_k;
    result.accumulator_registers = rust_result.accumulator_registers;
    result.prefetch_distance = rust_result.prefetch_distance;
    result.simd_level = rust_result.simd_level;

    // Build identity schedule maps (the Rust engine already applied transforms)
    for (size_t s = 0; s < ispace.num_statements(); ++s) {
        result.schedule_maps.push_back(AffineMap::identity(ndim));
    }

    return result;
}

} // namespace symplex::polyhedral
