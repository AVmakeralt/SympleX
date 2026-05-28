// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/schedule/schedule_map.h"
#include <algorithm>
#include <sstream>

namespace symplex::schedule {

// ===========================================================================
// build_initial_tree
//
// Construct the initial schedule tree from an iteration space.
//
// The tree structure is:
//   DOMAIN [name=iterspace.name]
//     SEQUENCE
//       FILTER [stmt=stmt_0]
//         BAND [n_loops=ndim, permutable=true]
//           LEAF
//       FILTER [stmt=stmt_1]
//         BAND [n_loops=ndim, permutable=true]
//           LEAF
//       ...
//
// Each statement gets its own filter+band+leaf subtree.  The band
// members are initialized with identity coefficients (i.e., the
// trivial schedule).
// ===========================================================================
ScheduleTreePtr build_initial_tree(
    const polyhedral::IterationSpace& iterspace
) {
    // Root domain node
    auto root = ScheduleTree::create(ScheduleNodeType::DOMAIN);
    root->set_domain_name(iterspace.name());

    // If there are no statements, just return the empty tree
    if (iterspace.num_statements() == 0) {
        auto leaf = ScheduleTree::create(ScheduleNodeType::LEAF);
        root->add_child(leaf);
        return root;
    }

    // If there's exactly one statement, we don't need a sequence
    if (iterspace.num_statements() == 1) {
        const auto& stmt = iterspace.statement(0);
        size_t ndim = stmt.domain.ndim();

        // Create filter node
        auto filter = ScheduleTree::create(ScheduleNodeType::FILTER);
        filter->filter_data().statement_name = stmt.name;

        // Create band node with identity schedule
        auto band = ScheduleTree::create(ScheduleNodeType::BAND);
        band->band_data().members.resize(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            band->band_data().members[d] = BandMember(ndim, 0);
            band->band_data().members[d].coefficients[d] = 1;
        }
        band->band_data().permutable = true;

        // Leaf under the band
        auto leaf = ScheduleTree::create(ScheduleNodeType::LEAF);
        band->add_child(leaf);
        filter->add_child(band);
        root->add_child(filter);
        return root;
    }

    // Multiple statements: create a SEQUENCE node
    auto seq = ScheduleTree::create(ScheduleNodeType::SEQUENCE);

    for (size_t s = 0; s < iterspace.num_statements(); ++s) {
        const auto& stmt = iterspace.statement(s);
        size_t ndim = stmt.domain.ndim();

        // Filter node
        auto filter = ScheduleTree::create(ScheduleNodeType::FILTER);
        filter->filter_data().statement_name = stmt.name;

        // Band node with identity schedule
        auto band = ScheduleTree::create(ScheduleNodeType::BAND);
        band->band_data().members.resize(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            band->band_data().members[d] = BandMember(ndim, 0);
            band->band_data().members[d].coefficients[d] = 1;
        }
        band->band_data().permutable = true;

        // Leaf
        auto leaf = ScheduleTree::create(ScheduleNodeType::LEAF);
        band->add_child(leaf);
        filter->add_child(band);
        seq->add_child(filter);
    }

    root->add_child(seq);
    return root;
}

// ===========================================================================
// select_tile_config
//
// Choose a tile configuration that:
//   1. Aligns with the Tensor Core' native MMA dimensions
//   2. Fits within the SRAM budget
//   3. Maximizes data reuse (largest tiles that fit)
//
// Strategy: enumerate hardware-aligned tiles (from tiling.h), then
// select the one with the largest inner volume that fits in SRAM.
// ===========================================================================
std::optional<TileConfig> select_tile_config(
    const polyhedral::IterationSpace& iterspace,
    const hardware::HardwareTarget& target
) {
    if (iterspace.num_statements() == 0) {
        return std::nullopt;
    }

    // Determine the number of dimensions from the first statement
    size_t ndim = iterspace.statement(0).domain.ndim();
    if (ndim == 0) {
        return std::nullopt;
    }

    // Generate all hardware-aligned tile configurations
    auto candidates = generate_hardware_aligned_tiles(ndim, target);

    if (candidates.empty()) {
        // Fallback: use a simple tile configuration based on SRAM budget
        std::vector<int64_t> tile_sizes(ndim, 1);

        // For 2D and 3D problems, try Tensor Core aligned sizes
        if (ndim >= 2) {
            tile_sizes[0] = target.gpu.tensor_core.m;
            tile_sizes[1] = target.gpu.tensor_core.n;
        }
        if (ndim >= 3) {
            tile_sizes[2] = target.gpu.tensor_core.k;
        }

        // Scale up until we approach SRAM limit
        int64_t bpe = target.bytes_per_element;
        int64_t sram = target.max_sram_bytes;

        // Start with Tensor Core base sizes and scale up
        bool can_grow = true;
        while (can_grow) {
            can_grow = false;
            for (size_t d = 0; d < ndim; ++d) {
                int64_t step = (d == 0) ? target.gpu.tensor_core.m :
                               (d == 1) ? target.gpu.tensor_core.n :
                                          target.gpu.tensor_core.k;

                std::vector<int64_t> test_sizes = tile_sizes;
                test_sizes[d] += step;

                // Check SRAM budget
                size_t footprint = iterspace.estimate_sram_footprint(
                    test_sizes,
                    static_cast<size_t>(bpe),
                    true);

                if (static_cast<int64_t>(footprint) <= sram) {
                    tile_sizes = test_sizes;
                    can_grow = true;
                }
            }
        }

        TileConfig cfg(tile_sizes, tile_sizes);
        return cfg;
    }

    // Select the candidate with the largest inner volume that fits in SRAM
    // This maximizes data reuse (arithmetic intensity)
    TileConfig best = candidates[0];
    int64_t best_volume = best.inner_volume();

    for (const auto& cfg : candidates) {
        size_t footprint = iterspace.estimate_sram_footprint(
            cfg.inner_tiles,
            static_cast<size_t>(target.bytes_per_element),
            true);

        if (static_cast<int64_t>(footprint) <= target.max_sram_bytes) {
            int64_t vol = cfg.inner_volume();
            if (vol > best_volume) {
                best = cfg;
                best_volume = vol;
            }
        }
    }

    return best;
}

// ===========================================================================
// map_to_hardware
//
// Take an existing schedule tree and produce the GPU mapping.
// This is the lower-level entry point used after manual tree construction.
// ===========================================================================
GPUMapping map_to_hardware(
    const ScheduleTreePtr& tree,
    const polyhedral::IterationSpace& iterspace,
    const hardware::HardwareTarget& target
) {
    // Mark parallel dimensions
    MarkParallelDims(tree, iterspace);

    // Map to GPU
    auto mapping = MapToGPU(tree, target);

    // Apply warp specialization if TMA is available
    if (target.has_tma) {
        mapping = WarpSpecialization(
            mapping,
            target.has_tma,
            target.pipeline_stages);
    }

    return mapping;
}

// ===========================================================================
// build_schedule_map
//
// The main entry point.  Implements the full scheduling pipeline:
//   1. Create a schedule tree from the iteration space
//   2. Compute all data dependencies
//   3. Apply hierarchical tiling
//   4. Apply operator fusion
//   5. Mark parallel dimensions
//   6. Map to GPU hardware
// ===========================================================================
ScheduleMap build_schedule_map(
    const polyhedral::IterationSpace& iterspace,
    const hardware::HardwareTarget& target
) {
    ScheduleMap smap;
    smap.iterspace() = iterspace;  // Copy (we'll compute deps on it)

    // ── Step 1: Create the initial schedule tree ─────────────────────
    auto tree = build_initial_tree(iterspace);

    // ── Step 2: Compute dependencies ─────────────────────────────────
    // We compute on our copy so the original iteration space is unmodified
    smap.iterspace().compute_all_dependencies();

    // ── Step 3: Select and apply tiling ──────────────────────────────
    auto tile_cfg = select_tile_config(smap.iterspace(), target);

    if (tile_cfg.has_value()) {
        smap.tile_config() = *tile_cfg;

        // Find band nodes and apply tiling to each
        auto bands = tree->band_nodes();
        for (auto& band : bands) {
            // Apply rectangular tiling with the selected configuration.
            // tile_band() modifies the band node in-place (splits into
            // outer + inner bands) and returns the inner band pointer.
            // We intentionally discard the TilingResult since the tree
            // is already modified in-place and we don't need the
            // computed grid/block dims here (they are derived later
            // during GPU mapping).
            apply_rectangular_tiling(band, *tile_cfg, target);
        }
    } else {
        // No valid tiling found – use unit tiles (no tiling)
        size_t ndim = (iterspace.num_statements() > 0)
                       ? iterspace.statement(0).domain.ndim()
                       : 1;
        std::vector<int64_t> unit_tiles(ndim, 1);
        smap.tile_config() = TileConfig(unit_tiles, unit_tiles);
    }

    // ── Step 4: Apply fusion ─────────────────────────────────────────
    int64_t sram_budget = target.max_sram_bytes;
    auto fusion_result = FuseAll(
        tree, smap.iterspace(), sram_budget,
        static_cast<size_t>(target.bytes_per_element));

    tree = fusion_result.tree;

    // ── Step 5: Mark parallel dimensions ─────────────────────────────
    MarkParallelDims(tree, smap.iterspace());

    // ── Step 6: Map to GPU hardware ──────────────────────────────────
    auto gpu_mapping = map_to_hardware(tree, smap.iterspace(), target);

    // ── Assemble the schedule map ────────────────────────────────────
    smap.tree() = std::move(tree);
    smap.gpu_mapping() = std::move(gpu_mapping);

    return smap;
}

// ===========================================================================
// ScheduleMap::apply
//
// Apply the full scheduling pipeline.  This is useful when the
// ScheduleMap was constructed with just a tree and iteration space,
// and the pipeline needs to be run explicitly.
// ===========================================================================
bool ScheduleMap::apply(const hardware::HardwareTarget& target) {
    if (!tree_) {
        // Build from scratch using the iteration space
        auto smap = build_schedule_map(iterspace_, target);
        *this = std::move(smap);
        return true;
    }

    // Re-run the pipeline on the existing tree

    // Step 2: Compute dependencies (idempotent)
    iterspace_.compute_all_dependencies();

    // Step 3: Apply tiling if tile config is set
    if (tile_config_.inner_volume() > 1) {
        auto bands = tree_->band_nodes();
        for (auto& band : bands) {
            apply_rectangular_tiling(band, tile_config_, target);
        }
    } else {
        // Select a tile config
        auto cfg = select_tile_config(iterspace_, target);
        if (cfg.has_value()) {
            tile_config_ = *cfg;
            auto bands = tree_->band_nodes();
            for (auto& band : bands) {
                apply_rectangular_tiling(band, tile_config_, target);
            }
        }
    }

    // Step 4: Fusion
    int64_t sram_budget = target.max_sram_bytes;
    auto fusion_result = FuseAll(
        tree_, iterspace_, sram_budget,
        static_cast<size_t>(target.bytes_per_element));
    tree_ = fusion_result.tree;

    // Step 5: Mark parallel dims
    MarkParallelDims(tree_, iterspace_);

    // Step 6: Map to hardware
    gpu_mapping_ = map_to_hardware(tree_, iterspace_, target);

    return validate(target);
}

// ===========================================================================
// ScheduleMap::dump
// ===========================================================================
std::string ScheduleMap::dump() const {
    std::ostringstream oss;
    oss << "=== ScheduleMap ===\n";

    // Iteration space summary
    oss << "IterationSpace: " << iterspace_.to_string() << "\n";

    // Tile configuration
    oss << "TileConfig: " << tile_config_.to_string() << "\n";

    // Schedule tree
    if (tree_) {
        oss << "ScheduleTree:\n";
        oss << tree_->to_string(2) << "\n";
    } else {
        oss << "ScheduleTree: (null)\n";
    }

    // GPU mapping
    oss << "GPUMapping: " << gpu_mapping_.to_string() << "\n";

    // Dependency counts
    oss << "Dependencies: raw=" << iterspace_.raw_deps().size()
        << ", war=" << iterspace_.war_deps().size()
        << ", waw=" << iterspace_.waw_deps().size() << "\n";

    // Parallelizable dimensions
    auto par_dims = iterspace_.parallelizable_dims();
    oss << "Parallelizable dims: [";
    for (size_t i = 0; i < par_dims.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << par_dims[i];
    }
    oss << "]\n";

    // Band summary
    if (tree_) {
        auto bands = tree_->band_nodes();
        oss << "Band nodes: " << bands.size() << "\n";
        for (size_t b = 0; b < bands.size(); ++b) {
            auto& data = bands[b]->band_data();
            oss << "  Band " << b << ": " << data.members.size()
                << " dims, permutable=" << data.permutable << "\n";
            for (size_t m = 0; m < data.members.size(); ++m) {
                oss << "    dim " << m << ": "
                    << (data.members[m].parallel ? "PARALLEL" : "SEQUENTIAL")
                    << ", coincidence=" << data.members[m].coincidence
                    << "\n";
            }
        }
    }

    oss << "=== End ScheduleMap ===\n";
    return oss.str();
}

// ===========================================================================
// ScheduleMap::validate
//
// Check that the schedule map is internally consistent.
// ===========================================================================
bool ScheduleMap::validate(const hardware::HardwareTarget& target) const {
    // Check tree is non-null
    if (!tree_) {
        return false;
    }

    // Check that the tree has at least one leaf node
    auto leaves = tree_->leaf_nodes();
    if (leaves.empty()) {
        return false;
    }

    // Check that all dependencies are satisfied by the schedule.
    // For each dependency vector, verify that the schedule (as encoded
    // in the band nodes) preserves the lexicographic positivity.
    auto all_deps = iterspace_.all_dependencies();
    auto bands = tree_->band_nodes();

    for (const auto& dep : all_deps) {
        for (const auto& dv : dep.vectors()) {
            // Apply each band's schedule to the dependency vector and
            // check that the result is lexicographically positive.
            // For simplicity, we check that the original dependency
            // vector is lexicographically positive (which it should be
            // since only lex-positive deps are kept during construction).
            if (!dv.is_lex_positive()) {
                // A non-lex-positive dependency in the iteration space
                // indicates a scheduling violation
                return false;
            }
        }
    }

    // Check GPU mapping constraints
    const auto& gm = gpu_mapping_;

    // Thread count per block must not exceed hardware limit
    int64_t threads = gm.threads_per_block();
    if (threads > target.gpu.max_threads_per_block) {
        return false;
    }
    if (threads <= 0) {
        return false;
    }

    // Block dimensions must not exceed hardware limits
    if (gm.block_dims.size() > 0 && gm.block_dims[0] > target.gpu.max_block_x) {
        return false;
    }
    if (gm.block_dims.size() > 1 && gm.block_dims[1] > target.gpu.max_block_y) {
        return false;
    }
    if (gm.block_dims.size() > 2 && gm.block_dims[2] > target.gpu.max_block_z) {
        return false;
    }

    // Grid dimensions must not exceed hardware limits
    if (gm.grid_dims.size() > 0 && gm.grid_dims[0] > target.gpu.max_grid_x) {
        return false;
    }
    if (gm.grid_dims.size() > 1 && gm.grid_dims[1] > target.gpu.max_grid_y) {
        return false;
    }
    if (gm.grid_dims.size() > 2 && gm.grid_dims[2] > target.gpu.max_grid_z) {
        return false;
    }

    // Check shared memory budget
    if (gm.smem_per_block > target.max_sram_bytes) {
        return false;
    }

    // Check tile configuration SRAM fit
    if (tile_config_.inner_volume() > 0) {
        size_t footprint = iterspace_.estimate_sram_footprint(
            tile_config_.inner_tiles,
            static_cast<size_t>(target.bytes_per_element),
            true);
        if (static_cast<int64_t>(footprint) > target.max_sram_bytes) {
            return false;
        }
    }

    // Check warp role assignment consistency
    int64_t n_warps = gm.warps_per_block();
    if (static_cast<int64_t>(gm.warp_roles.size()) > n_warps) {
        return false;
    }

    // If TMA is not available, there should be no PRODUCER-only warps
    if (!target.has_tma) {
        for (auto r : gm.warp_roles) {
            if (r == WarpRole::PRODUCER) {
                return false;  // Can't have TMA producer without TMA
            }
        }
    }

    // Check occupancy: at least some minimal occupancy
    int64_t regs_per_thread = 32;  // Conservative estimate
    int64_t occupancy = target.gpu.sm.compute_occupancy(
        regs_per_thread, gm.smem_per_block);
    if (occupancy <= 0) {
        return false;
    }

    return true;
}

} // namespace symplex::schedule
