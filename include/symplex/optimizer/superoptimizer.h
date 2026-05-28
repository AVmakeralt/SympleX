// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/optimizer/egraph.h"
#include "symplex/optimizer/tile_config.h"
#include "symplex/optimizer/search_phase1.h"
#include "symplex/optimizer/search_phase2.h"
#include "symplex/optimizer/search_phase3.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/schedule/tiling.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>

namespace symplex::optimizer {

/// Result of the full two-level superoptimization search.
struct SuperoptimizerResult {
    // ── Level 1: Mathematical Superoptimizer results ─────────────────
    std::string original_expr;                 ///< Original program expression
    std::string optimized_expr;                ///< Optimized program expression
    std::vector<egraph::ENode> optimized_nodes;///< Extracted optimized nodes
    double egraph_cost = 0.0;                  ///< Cost of the extracted program
    int saturation_iters = 0;                  ///< How many saturation iterations ran
    int egraph_classes = 0;                    ///< E-classes in the saturated graph
    int egraph_nodes = 0;                      ///< E-nodes in the saturated graph
    int rewrites_applied = 0;                  ///< Total rewrites that fired
    bool polyhedral_valid = true;              ///< Did the extraction pass validation?
    egraph::ClassAnalysis root_analysis;       ///< Analysis data for the optimized expression

    // ── Level 2: Hardware-Mapping Search results ─────────────────────
    schedule::TileConfig best_tile;             ///< Best tile configuration found
    double estimated_latency_ns  = 0.0;         ///< Estimated kernel latency
    double speedup_vs_naive      = 0.0;         ///< Speedup vs naive single-MMA tile

    // ── Overall ──────────────────────────────────────────────────────
    std::string summary;                        ///< Human-readable summary

    /// Was a valid configuration found?
    [[nodiscard]] bool valid() const {
        return !best_tile.inner_tiles.empty() && !optimized_expr.empty();
    }
};

/// Superoptimizer: a true two-level search engine that combines
/// equality saturation (Level 1) with hardware mapping search (Level 2).
///
/// ┌──────────────────────────────────────────────────────────────┐
/// │ Level 1: Mathematical Superoptimizer                         │
/// │   - Builds an e-graph from the input expression              │
/// │   - Applies rewrite rules (algebraic, fusion, tiling)        │
/// │   - Priority-based rule scheduling with cost-guided filtering│
/// │   - Polyhedral guardrails prune invalid transformations      │
/// │   - Cost-guided extraction picks the best expression         │
/// │   - Per-class analysis: shape, dtype, layout, TC compat      │
/// └──────────────────────────────────────────────────────────────┘
///                         │
///                         ▼
/// ┌──────────────────────────────────────────────────────────────┐
/// │ Level 2: Hardware-Mapping Search                             │
/// │   - Takes the best expression from Level 1                   │
/// │   - Phase 1: Roofline pruning of tile configurations         │
/// │   - Phase 2: Compute-symmetry alignment (Tensor Core dims)   │
/// │   - Phase 3: Hardware occupancy sieve (analytical/empirical) │
/// └──────────────────────────────────────────────────────────────┘
///                         │
///                         ▼
///              [Optimal GPU Kernel]
///
class Superoptimizer {
public:
    /// Construct a superoptimizer for a specific hardware target.
    explicit Superoptimizer(hardware::HardwareTarget target);

    /// Run the full two-level superoptimization for the given
    /// iteration space.
    ///
    /// \param ispace              The polyhedral iteration space to optimize.
    /// \param max_tensor_dim      Upper bound on any single tile dimension.
    /// \param saturation_iters    Max iterations for equality saturation.
    /// \param max_egraph_nodes    Max e-graph nodes before stopping saturation.
    /// \return                    SuperoptimizerResult with the best program + tile.
    SuperoptimizerResult optimize(
        const polyhedral::IterationSpace& ispace,
        size_t max_tensor_dim = 1024,
        int saturation_iters = 20,
        int64_t max_egraph_nodes = 50000
    );

    /// Run Level 1 only: equality saturation on a tensor expression.
    /// Useful for exploring what algebraic optimizations are discovered.
    ///
    /// \param graph         The e-graph to saturate.
    /// \param root_class    The class ID of the expression root.
    /// \param rules         The rewrite rules to apply.
    /// \param cost_fn       The cost function for extraction.
    /// \param config        Saturation configuration.
    /// \return              The extraction result from the saturated e-graph.
    egraph::ExtractionResult run_level1(
        egraph::EGraph& graph,
        egraph::ClassId root_class,
        const std::vector<egraph::RewriteRule>& rules,
        std::function<double(egraph::OpId, const egraph::ENode&)> cost_fn,
        egraph::SaturationConfig config = egraph::SaturationConfig{}
    );

    /// Run Level 2 only: hardware mapping search on the extracted expression.
    /// Uses the 3-phase search (roofline -> symmetry -> occupancy).
    SuperoptimizerResult run_level2(
        const egraph::ExtractionResult& extracted,
        const polyhedral::IterationSpace& ispace,
        size_t max_tensor_dim = 1024
    );

    /// Access the hardware target.
    [[nodiscard]] const hardware::HardwareTarget& target() const;

    /// Get the last saturated e-graph (for debugging/inspection).
    const egraph::EGraph& last_egraph() const;

private:
    hardware::HardwareTarget target_;
    std::unique_ptr<egraph::EGraph> last_egraph_;

    /// Build an e-graph from a matmul iteration space.
    /// Returns the graph and the root class ID.
    std::pair<std::unique_ptr<egraph::EGraph>, egraph::ClassId>
    build_egraph_from_ispace(const polyhedral::IterationSpace& ispace);

    /// Collect all rewrite rules applicable to the given expression type.
    std::vector<egraph::RewriteRule> collect_rewrite_rules(
        const polyhedral::IterationSpace& ispace
    );

    /// Create a cost function tailored to the iteration space dimensions.
    std::function<double(egraph::OpId, const egraph::ENode&)>
    create_cost_function(const polyhedral::IterationSpace& ispace);
};

} // namespace symplex::optimizer
