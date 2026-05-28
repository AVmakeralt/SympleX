// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

#include "symplex/distributed/mesh.h"
#include "symplex/distributed/nccl_bridge.h"

namespace symplex::distributed {

/// PipelineStage: describes a single stage in the pipeline schedule.
struct PipelineStage {
    int64_t stage_id = -1;
    int64_t micro_batch_start = 0;
    int64_t micro_batch_end = 0;
    int64_t forward_compute_ns = 0;
    int64_t backward_compute_ns = 0;
    int64_t comm_ns = 0;
    int64_t total_ns = 0;   // With overlap applied

    std::string to_string() const {
        std::ostringstream oss;
        oss << "PipelineStage{id=" << stage_id
            << ", micro_batches=[" << micro_batch_start
            << "," << micro_batch_end << ")"
            << ", fwd=" << forward_compute_ns << "ns"
            << ", bwd=" << backward_compute_ns << "ns"
            << ", comm=" << comm_ns << "ns"
            << ", total=" << total_ns << "ns}";
        return oss.str();
    }
};

/// PipelineSchedule: the complete schedule for a 1F1B pipeline
/// with communication-computation overlap.
struct PipelineSchedule {
    std::vector<PipelineStage> stages;
    int64_t total_iteration_ns = 0;
    int64_t bubble_ns = 0;              // Pipeline bubble overhead
    double efficiency = 0.0;            // 1.0 - bubble_fraction
    int64_t num_micro_batches = 0;

    std::string to_string() const {
        std::ostringstream oss;
        oss << "PipelineSchedule{stages=[";
        for (size_t i = 0; i < stages.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << stages[i].to_string();
        }
        oss << "], total_ns=" << total_iteration_ns
            << ", bubble_ns=" << bubble_ns
            << ", efficiency=" << efficiency
            << ", micro_batches=" << num_micro_batches << "}";
        return oss.str();
    }
};

/// PipelineOverlapper: computes optimal pipeline schedules with
/// communication-computation overlap using 1F1B scheduling.
///
/// 1F1B (one forward, one backward) pipeline scheduling:
///   - Warmup phase: forward passes fill the pipeline
///   - Steady state: 1 forward + 1 backward interleaved
///   - Cooldown phase: backward passes drain the pipeline
///
/// Communication-computation overlap:
///   - Communication for micro-batch N can start during
///     computation of micro-batch N+1 (async NCCL)
class PipelineOverlapper {
public:
    explicit PipelineOverlapper(const ClusterMesh& mesh, const NCCLBridge& nccl);

    /// Compute the optimal pipeline schedule for given compute/comm times.
    ///
    /// @param num_stages       Number of pipeline stages
    /// @param num_micro_batches  Number of micro-batches to schedule
    /// @param forward_compute_ns  Forward pass compute time per micro-batch (ns)
    /// @param backward_compute_ns Backward pass compute time per micro-batch (ns)
    /// @param comm_ns          Communication time per micro-batch between stages (ns)
    PipelineSchedule compute_schedule(
        int64_t num_stages,
        int64_t num_micro_batches,
        int64_t forward_compute_ns,
        int64_t backward_compute_ns,
        int64_t comm_ns
    ) const;

    /// Auto-tune the number of micro-batches for maximum efficiency.
    /// Searches from 1 to max_micro_batches, finding the count that
    /// minimizes the pipeline bubble fraction.
    int64_t optimal_micro_batches(
        int64_t num_stages,
        int64_t forward_ns,
        int64_t backward_ns,
        int64_t comm_ns,
        int64_t max_micro_batches = 256
    ) const;

    /// Generate the time steps at which async communication should be
    /// injected to overlap with computation.
    std::vector<int64_t> injection_points(const PipelineSchedule& schedule) const;

private:
    ClusterMesh mesh_;
    NCCLBridge nccl_;

    /// Compute the bubble time for a given configuration.
    int64_t compute_bubble_ns(
        int64_t num_stages,
        int64_t num_micro_batches,
        int64_t forward_ns,
        int64_t backward_ns,
        int64_t comm_ns
    ) const;
};

} // namespace symplex::distributed
