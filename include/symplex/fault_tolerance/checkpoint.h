// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

namespace symplex::fault_tolerance {

/// CheckpointDecision: how to handle a layer's activation during training.
enum class CheckpointDecision {
    SAVE,       ///< Keep activation in GPU memory (fastest backward, uses most memory)
    RECOMPUTE,  ///< Discard activation, recompute during backward pass (saves memory, costs compute)
    OFFLOAD,    ///< Move activation to CPU memory (saves GPU memory, costs PCIe transfer)
};

/// CheckpointPlan: the result of a checkpoint planning decision for a model.
struct CheckpointPlan {
    struct LayerDecision {
        int64_t layer_id;              ///< Index of the layer in the model
        CheckpointDecision decision;   ///< Chosen strategy for this layer
        int64_t memory_bytes_if_save;  ///< Activation size if kept in GPU memory
        int64_t compute_ns_if_recompute; ///< Wall-clock recompute cost during backward pass
        double saving_ratio;           ///< memory_bytes / recompute_ns – higher = better recompute candidate
    };

    std::vector<LayerDecision> decisions;  ///< Per-layer decisions, in layer order
    int64_t total_memory_bytes = 0;        ///< Total GPU memory occupied by SAVE activations
    int64_t total_recompute_ns = 0;        ///< Total recompute cost from RECOMPUTE layers
    double memory_savings_percent = 0.0;   ///< Memory saved vs all-SAVE baseline, as a percentage
};

/// CheckpointPlanner: decides per-layer activation checkpointing strategy
/// to fit within a GPU memory budget while minimizing recompute overhead.
class CheckpointPlanner {
public:
    /// Construct a planner with the given GPU memory budget in bytes.
    explicit CheckpointPlanner(int64_t available_memory_bytes);

    /// Plan checkpointing decisions for a full model.
    /// @param layer_activation_bytes  Per-layer activation tensor size in bytes.
    /// @param layer_recompute_ns      Per-layer recompute wall-clock time in nanoseconds.
    /// @returns A CheckpointPlan with per-layer decisions and aggregate statistics.
    CheckpointPlan plan(
        const std::vector<int64_t>& layer_activation_bytes,
        const std::vector<int64_t>& layer_recompute_ns
    ) const;

    /// Decide the checkpoint strategy for a single activation.
    /// @param activation_bytes  Size of the activation tensor in bytes.
    /// @param recompute_ns      Wall-clock recompute time in nanoseconds.
    /// @param remaining_memory  Bytes of GPU memory still available.
    /// @returns SAVE if memory is available and recompute is more expensive than
    ///          the memory transfer cost; RECOMPUTE otherwise.
    CheckpointDecision decide(
        int64_t activation_bytes,
        int64_t recompute_ns,
        int64_t remaining_memory
    ) const;

    /// Return the configured GPU memory budget in bytes.
    int64_t available_memory() const;

private:
    int64_t available_memory_bytes_;
    double memory_cost_per_byte_ns_ = 0.003;   // ~3ps per byte at 3.3 TB/s HBM
    double compute_cost_per_flop_ns_ = 0.0001;  // ~0.1ps per FLOP at 1 PETAOPS
};

} // namespace symplex::fault_tolerance
