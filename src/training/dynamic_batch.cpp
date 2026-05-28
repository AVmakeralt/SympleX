// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/training/dynamic_batch.h"

namespace symplex::training {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DynamicBatchSizer::DynamicBatchSizer(
    int64_t global_batch_size,
    int64_t max_micro_batch,
    int64_t available_memory_bytes,
    int64_t bytes_per_element
)
    : global_batch_size_(global_batch_size)
    , max_micro_batch_(max_micro_batch)
    , available_memory_bytes_(available_memory_bytes)
    , bytes_per_element_(bytes_per_element)
{
    if (global_batch_size_ <= 0) global_batch_size_ = 1;
    if (max_micro_batch_ <= 0) max_micro_batch_ = 1;
    if (available_memory_bytes_ <= 0) available_memory_bytes_ = 0;
    if (bytes_per_element_ <= 0) bytes_per_element_ = 2;
}

// ---------------------------------------------------------------------------
// Transformer memory model
// ---------------------------------------------------------------------------

int64_t DynamicBatchSizer::activation_memory(
    int64_t mb,
    int64_t seq,
    int64_t hidden,
    int64_t layers
) const {
    // Per-layer activation tensor size in bytes.
    // Forward pass: one activation of shape [mb, seq, hidden]
    // Backward pass: we need the forward activations AND gradient activations,
    // so we multiply by 2 (forward + backward).
    int64_t per_element = mb * seq * hidden * bytes_per_element_;
    int64_t per_layer = 2 * per_element;  // forward + backward

    // Total activation memory across all layers
    int64_t total_activations = layers * per_layer;

    // Optimizer state (Adam): 2 additional tensors per parameter
    // (momentum + variance), each the same size as the weight.
    // For a transformer, the dominant weight per layer is the
    // attention + FFN weight matrices, which are roughly
    // 4 * hidden * hidden parameters (QKV + output + FFN up + FFN down).
    // We approximate optimizer state as 2x the per-layer activation.
    int64_t optimizer_state = 2 * per_layer;

    return total_activations + optimizer_state;
}

// ---------------------------------------------------------------------------
// Estimate memory needed for a given micro-batch size
// ---------------------------------------------------------------------------

int64_t DynamicBatchSizer::estimate_memory_needed(
    int64_t micro_batch,
    int64_t seq_len,
    int64_t hidden_dim,
    int64_t num_layers
) const {
    if (micro_batch <= 0 || seq_len <= 0 || hidden_dim <= 0 || num_layers <= 0) {
        return 0;
    }
    return activation_memory(micro_batch, seq_len, hidden_dim, num_layers);
}

// ---------------------------------------------------------------------------
// Compute optimal batch config
// ---------------------------------------------------------------------------

BatchConfig DynamicBatchSizer::compute_batch_config(
    int64_t sequence_length,
    int64_t hidden_dim,
    int64_t num_layers
) const {
    BatchConfig config;
    config.global_batch_size = global_batch_size_;
    config.micro_batch_size = max_micro_batch_;
    config.was_adjusted = false;

    // Try the requested max micro-batch size first, then halve until it fits.
    int64_t mb = max_micro_batch_;
    while (mb > 0) {
        int64_t needed = estimate_memory_needed(mb, sequence_length, hidden_dim, num_layers);
        if (needed <= available_memory_bytes_) {
            config.micro_batch_size = mb;
            break;
        }
        mb /= 2;
        config.was_adjusted = true;
    }

    // If even micro_batch=1 doesn't fit, still use 1 (we can't go lower).
    if (mb == 0) {
        config.micro_batch_size = 1;
        config.was_adjusted = true;
    }

    // Compute gradient accumulation steps.
    config.gradient_accumulation_steps =
        (global_batch_size_ + config.micro_batch_size - 1) / config.micro_batch_size;

    // Number of micro-batches in one global batch.
    config.num_micro_batches = config.gradient_accumulation_steps;

    return config;
}

// ---------------------------------------------------------------------------
// Adjust for memory (accounts for already-used memory)
// ---------------------------------------------------------------------------

BatchConfig DynamicBatchSizer::adjust_for_memory(
    int64_t current_memory_used,
    int64_t sequence_length,
    int64_t hidden_dim,
    int64_t num_layers
) const {
    // Effective available memory is reduced by what is already in use.
    int64_t effective_available =
        available_memory_bytes_ - current_memory_used;
    if (effective_available < 0) {
        effective_available = 0;
    }

    BatchConfig config;
    config.global_batch_size = global_batch_size_;
    config.micro_batch_size = max_micro_batch_;
    config.was_adjusted = false;

    // Try the requested max micro-batch size, then halve until it fits
    // within the effective (remaining) memory.
    int64_t mb = max_micro_batch_;
    while (mb > 0) {
        int64_t needed = estimate_memory_needed(mb, sequence_length, hidden_dim, num_layers);
        if (needed <= effective_available) {
            config.micro_batch_size = mb;
            break;
        }
        mb /= 2;
        config.was_adjusted = true;
    }

    if (mb == 0) {
        config.micro_batch_size = 1;
        config.was_adjusted = true;
    }

    config.gradient_accumulation_steps =
        (global_batch_size_ + config.micro_batch_size - 1) / config.micro_batch_size;
    config.num_micro_batches = config.gradient_accumulation_steps;

    return config;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int64_t DynamicBatchSizer::global_batch_size() const {
    return global_batch_size_;
}

int64_t DynamicBatchSizer::max_micro_batch() const {
    return max_micro_batch_;
}

} // namespace symplex::training
