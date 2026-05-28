// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/hardware/hardware_target.h"
#include "symplex/optimizer/superoptimizer.h"
#include "symplex/codegen/code_generator.h"
#include "symplex/distributed/mesh.h"
#include "symplex/distributed/sharding.h"
#include "symplex/distributed/nccl_bridge.h"
#include "symplex/distributed/pipeline_overlap.h"
#include "symplex/fault_tolerance/health_monitor.h"
#include "symplex/fault_tolerance/forward_recovery.h"
#include "symplex/fault_tolerance/checkpoint.h"
#include "symplex/training/dynamic_batch.h"
#include "symplex/training/memory_watchdog.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

namespace symplex::training {

struct TrainingConfig {
    int64_t global_batch_size = 2048;
    int64_t max_micro_batch = 64;
    int64_t num_epochs = 1;
    int64_t max_steps = -1;        // -1 = unlimited
    double learning_rate = 1e-4;
    std::string optimizer_type = "adam";
    bool enable_fault_tolerance = true;
    bool enable_dynamic_batching = true;
    bool enable_checkpointing = true;
    int64_t checkpoint_every_n_steps = 1000;
};

struct StepResult {
    int64_t step;
    double loss;
    int64_t latency_ns;
    int64_t micro_batch_used;
    bool recovered_from_failure;
    std::string kernel_name;
};

class TrainingLoop {
public:
    TrainingLoop(
        TrainingConfig config,
        hardware::HardwareTarget target
    );

    // Initialize the training loop (compile kernels, setup distributed)
    void initialize(const polyhedral::IterationSpace& ispace);

    // Execute a single training step
    StepResult execute_step();

    // Execute a full epoch
    std::vector<StepResult> execute_epoch();

    // Handle a device failure
    bool handle_failure(int64_t failed_device_id);

    // Accessors
    const TrainingConfig& config() const;
    const DynamicBatchSizer& batch_sizer() const;
    const MemoryWatchdog& watchdog() const;

    // Callbacks for monitoring
    using StepCallback = std::function<void(const StepResult&)>;
    void on_step_complete(StepCallback cb);

private:
    TrainingConfig config_;
    hardware::HardwareTarget target_;

    // Core components
    std::unique_ptr<optimizer::Superoptimizer> superopt_;
    std::unique_ptr<codegen::CodeGenerator> codegen_;
    std::unique_ptr<DynamicBatchSizer> batch_sizer_;
    std::unique_ptr<MemoryWatchdog> watchdog_;

    // Distributed components (optional)
    std::unique_ptr<distributed::ClusterMesh> mesh_;
    std::unique_ptr<distributed::ShardingAnalyzer> sharding_;
    std::unique_ptr<distributed::NCCLBridge> nccl_;
    std::unique_ptr<distributed::PipelineOverlapper> pipeline_;

    // Fault tolerance components
    std::unique_ptr<fault_tolerance::HealthMonitor> health_;
    std::unique_ptr<fault_tolerance::ForwardRecovery> recovery_;
    std::unique_ptr<fault_tolerance::CheckpointPlanner> checkpoint_;

    // State
    int64_t current_step_ = 0;
    int64_t current_epoch_ = 0;
    std::vector<double> loss_history_;
    BatchConfig current_batch_;
    optimizer::SuperoptimizerResult optimized_result_;
    codegen::GeneratedKernel current_kernel_;
    bool initialized_ = false;
    std::vector<StepCallback> step_callbacks_;

    // Internal methods
    void compile_optimal_kernel(const polyhedral::IterationSpace& ispace);
    void setup_distributed();
    void notify_step(const StepResult& result);
};

} // namespace symplex::training
