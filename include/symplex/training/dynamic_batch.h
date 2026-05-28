// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

namespace symplex::training {

struct BatchConfig {
    int64_t global_batch_size;
    int64_t micro_batch_size;
    int64_t num_micro_batches;
    int64_t gradient_accumulation_steps;
    bool was_adjusted;  // True if we had to reduce from requested size
};

class DynamicBatchSizer {
public:
    DynamicBatchSizer(
        int64_t global_batch_size,
        int64_t max_micro_batch,
        int64_t available_memory_bytes,
        int64_t bytes_per_element = 2
    );

    // Compute the optimal micro-batch size given current memory state
    BatchConfig compute_batch_config(
        int64_t sequence_length,
        int64_t hidden_dim,
        int64_t num_layers
    ) const;

    // Dynamically adjust batch size based on real-time memory usage
    BatchConfig adjust_for_memory(
        int64_t current_memory_used,
        int64_t sequence_length,
        int64_t hidden_dim,
        int64_t num_layers
    ) const;

    // Estimate memory needed for a given micro-batch size
    int64_t estimate_memory_needed(
        int64_t micro_batch,
        int64_t seq_len,
        int64_t hidden_dim,
        int64_t num_layers
    ) const;

    int64_t global_batch_size() const;
    int64_t max_micro_batch() const;

private:
    int64_t global_batch_size_;
    int64_t max_micro_batch_;
    int64_t available_memory_bytes_;
    int64_t bytes_per_element_;

    // Transformer memory model: activations, gradients, optimizer states
    int64_t activation_memory(int64_t mb, int64_t seq, int64_t hidden, int64_t layers) const;
};

} // namespace symplex::training
