// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/training/training_loop.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace symplex::training {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TrainingLoop::TrainingLoop(
    TrainingConfig config,
    hardware::HardwareTarget target
)
    : config_(std::move(config))
    , target_(std::move(target))
{
    // Compute the GPU's total memory in bytes for the batch sizer and watchdog.
    int64_t total_gpu_bytes =
        target_.gpu.memory.global_capacity_gb * 1024LL * 1024LL * 1024LL;

    // Reserve 80 % of GPU memory for training tensors; the rest is for
    // framework overhead, CUDA context, etc.
    int64_t available_bytes = static_cast<int64_t>(total_gpu_bytes * 0.80);

    batch_sizer_ = std::make_unique<DynamicBatchSizer>(
        config_.global_batch_size,
        config_.max_micro_batch,
        available_bytes,
        target_.bytes_per_element
    );

    watchdog_ = std::make_unique<MemoryWatchdog>(target_);

    // Initial (default) batch config — will be refined during initialize().
    current_batch_ = BatchConfig{
        config_.global_batch_size,
        config_.max_micro_batch,
        /*num_micro_batches=*/1,
        /*gradient_accumulation_steps=*/1,
        /*was_adjusted=*/false
    };
}

// ---------------------------------------------------------------------------
// initialize()
// ---------------------------------------------------------------------------

void TrainingLoop::initialize(const polyhedral::IterationSpace& ispace) {
    // 1. Run the superoptimizer to find the optimal tile configuration.
    compile_optimal_kernel(ispace);

    // 2. Set up the distributed training topology.
    setup_distributed();

    // 3. Initialise fault-tolerance components.
    if (config_.enable_fault_tolerance && mesh_) {
        health_ = std::make_unique<fault_tolerance::HealthMonitor>(*mesh_);
        recovery_ = std::make_unique<fault_tolerance::ForwardRecovery>(*mesh_);
    }

    // 4. Initialise the checkpoint planner.
    if (config_.enable_checkpointing) {
        int64_t total_gpu_bytes =
            target_.gpu.memory.global_capacity_gb * 1024LL * 1024LL * 1024LL;
        int64_t ckpt_budget = static_cast<int64_t>(total_gpu_bytes * 0.50);
        checkpoint_ = std::make_unique<fault_tolerance::CheckpointPlanner>(ckpt_budget);
    }

    // 5. Compute initial batch configuration with a sensible default
    //    transformer shape (will be overridden on the first real step).
    current_batch_ = batch_sizer_->compute_batch_config(
        /*sequence_length=*/2048,
        /*hidden_dim=*/4096,
        /*num_layers=*/32
    );

    initialized_ = true;
}

// ---------------------------------------------------------------------------
// compile_optimal_kernel()
// ---------------------------------------------------------------------------

void TrainingLoop::compile_optimal_kernel(
    const polyhedral::IterationSpace& ispace
) {
    superopt_ = std::make_unique<optimizer::Superoptimizer>(target_);
    optimized_result_ = superopt_->optimize(ispace);

    if (!optimized_result_.valid()) {
        // No valid tile configuration found — leave current_kernel_ invalid.
        current_kernel_ = codegen::GeneratedKernel{};
        return;
    }

    codegen_ = std::make_unique<codegen::CodeGenerator>(target_);

    // Determine matrix dimensions from the iteration space (3-D matmul case).
    // If the ispace has a 3-D domain, extract M, N, K for codegen.
    // Otherwise fall back to a reasonable default.
    int64_t M = 4096, N = 4096, K = 4096;
    if (ispace.num_statements() > 0) {
        const auto& domain = ispace.statement(0).domain;
        if (domain.ndim() == 3) {
            auto bounds = domain.bounds();
            if (bounds.size() >= 3) {
                M = bounds[0].second - bounds[0].first;
                N = bounds[1].second - bounds[1].first;
                K = bounds[2].second - bounds[2].first;
            }
        }
    }

    current_kernel_ = codegen_->generate_matmul(M, N, K, optimized_result_.best_tile);
}

// ---------------------------------------------------------------------------
// setup_distributed()
// ---------------------------------------------------------------------------

void TrainingLoop::setup_distributed() {
    // Create a standard 2-D mesh: tensor-parallel × pipeline-parallel × data-parallel.
    // For single-GPU training, use 1×1×1.
    int64_t tp_size = 1;
    int64_t pp_size = 1;
    int64_t dp_size = 1;

    // If the hardware target suggests a multi-device topology, expand.
    // (In a real system this would be configured; here we default to single-GPU.)
    mesh_ = std::make_unique<distributed::ClusterMesh>(
        distributed::ClusterMesh::create_2d_mesh(tp_size, pp_size, dp_size, target_)
    );

    sharding_ = std::make_unique<distributed::ShardingAnalyzer>(*mesh_);
    nccl_ = std::make_unique<distributed::NCCLBridge>(*mesh_);
    pipeline_ = std::make_unique<distributed::PipelineOverlapper>(*mesh_, *nccl_);
}

// ---------------------------------------------------------------------------
// execute_step()
// ---------------------------------------------------------------------------

StepResult TrainingLoop::execute_step() {
    StepResult result;
    result.step = current_step_;
    result.recovered_from_failure = false;

    auto step_start = std::chrono::steady_clock::now();

    // ── Health check (fault tolerance) ──────────────────────────────
    if (config_.enable_fault_tolerance && health_) {
        auto events = health_->check_health();
        for (const auto& ev : events) {
            if (ev.status == fault_tolerance::DeviceHealth::DEAD) {
                bool ok = handle_failure(ev.device_id);
                result.recovered_from_failure = ok;
                if (!ok) {
                    // Cannot recover — bail out with a sentinel result.
                    result.loss = std::numeric_limits<double>::quiet_NaN();
                    result.latency_ns = 0;
                    result.micro_batch_used = 0;
                    result.kernel_name = "<recovery_failed>";
                    return result;
                }
            }
        }
    }

    // ── Dynamic batch adjustment ────────────────────────────────────
    if (config_.enable_dynamic_batching) {
        // Check memory pressure and adjust micro-batch size if needed.
        double reduction = watchdog_->recommended_reduction_factor();
        if (reduction < 1.0) {
            // Apply the recommended reduction by capping the micro-batch.
            int64_t reduced_mb = static_cast<int64_t>(
                static_cast<double>(current_batch_.micro_batch_size) * reduction);
            if (reduced_mb < 1) reduced_mb = 1;

            BatchConfig adjusted;
            adjusted.global_batch_size = config_.global_batch_size;
            adjusted.micro_batch_size = reduced_mb;
            adjusted.gradient_accumulation_steps =
                (config_.global_batch_size + reduced_mb - 1) / reduced_mb;
            adjusted.num_micro_batches = adjusted.gradient_accumulation_steps;
            adjusted.was_adjusted = true;
            current_batch_ = adjusted;
        }

        // Further adjust based on what the watchdog sees as actually allocated.
        MemorySnapshot snap = watchdog_->snapshot();
        current_batch_ = batch_sizer_->adjust_for_memory(
            snap.used_bytes,
            /*sequence_length=*/2048,
            /*hidden_dim=*/4096,
            /*num_layers=*/32
        );
    }

    // ── Launch the compiled kernel (simulated) ──────────────────────
    // In a real system this would enqueue the kernel on the GPU.
    // We simulate execution by recording the kernel metadata.
    result.kernel_name = current_kernel_.valid
        ? current_kernel_.kernel_name
        : "<invalid_kernel>";
    result.micro_batch_used = current_batch_.micro_batch_size;

    // Track the memory allocation for this step's activations.
    int64_t activation_bytes = batch_sizer_->estimate_memory_needed(
        current_batch_.micro_batch_size,
        /*seq_len=*/2048,
        /*hidden_dim=*/4096,
        /*num_layers=*/32
    );
    watchdog_->allocate(activation_bytes);

    // Simulate loss: a decreasing curve with noise.
    double base_loss = 10.0 / (1.0 + static_cast<double>(current_step_) * 0.01);
    double noise = std::sin(static_cast<double>(current_step_) * 0.3) * 0.05;
    result.loss = base_loss + noise;
    if (result.loss < 0.0) result.loss = 0.0;

    // Record loss history for checkpointing
    loss_history_.push_back(result.loss);

    // ── Checkpoint ──────────────────────────────────────────────────
    if (config_.enable_checkpointing &&
        config_.checkpoint_every_n_steps > 0 &&
        current_step_ > 0 &&
        current_step_ % config_.checkpoint_every_n_steps == 0)
    {
        // Serialize checkpoint to a JSON-like file.
        // Format:
        //   { "step": N, "loss": X, "tile": { "inner": [...], "outer": [...] } }
        std::ostringstream fname;
        fname << "symplex_ckpt_step_" << std::setw(8) << std::setfill('0')
              << current_step_ << ".json";

        std::ofstream ofs(fname.str());
        if (ofs.is_open()) {
            ofs << "{\n";
            ofs << "  \"step\": " << current_step_ << ",\n";
            ofs << "  \"epoch\": " << current_epoch_ << ",\n";
            ofs << std::scientific << std::setprecision(10);
            ofs << "  \"loss\": " << result.loss << ",\n";

            // Tile configuration
            ofs << "  \"tile\": {\n";
            ofs << "    \"inner\": [";
            const auto& inner = optimized_result_.best_tile.inner_tiles;
            for (size_t i = 0; i < inner.size(); ++i) {
                if (i > 0) ofs << ", ";
                ofs << inner[i];
            }
            ofs << "],\n";
            ofs << "    \"outer\": [";
            const auto& outer = optimized_result_.best_tile.outer_tiles;
            for (size_t i = 0; i < outer.size(); ++i) {
                if (i > 0) ofs << ", ";
                ofs << outer[i];
            }
            ofs << "]\n";
            ofs << "  },\n";

            // Batch config
            ofs << "  \"batch\": {\n";
            ofs << "    \"micro_batch_size\": " << current_batch_.micro_batch_size << ",\n";
            ofs << "    \"num_micro_batches\": " << current_batch_.num_micro_batches << ",\n";
            ofs << "    \"grad_accum_steps\": " << current_batch_.gradient_accumulation_steps << "\n";
            ofs << "  },\n";

            // Loss history (last 100 entries)
            ofs << "  \"loss_history\": [";
            size_t hist_start = (loss_history_.size() > 100)
                ? loss_history_.size() - 100 : 0;
            for (size_t i = hist_start; i < loss_history_.size(); ++i) {
                if (i > hist_start) ofs << ", ";
                ofs << loss_history_[i];
            }
            ofs << "]\n";

            ofs << "}\n";
        }
    }

    // ── Deallocate activation memory ────────────────────────────────
    watchdog_->deallocate(activation_bytes);

    auto step_end = std::chrono::steady_clock::now();
    result.latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        step_end - step_start).count();

    // Use the superoptimizer's latency estimate if available for a more
    // realistic number.
    if (optimized_result_.valid() && optimized_result_.estimated_latency_ns > 0) {
        result.latency_ns = static_cast<int64_t>(optimized_result_.estimated_latency_ns);
    }

    ++current_step_;

    notify_step(result);
    return result;
}

// ---------------------------------------------------------------------------
// execute_epoch()
// ---------------------------------------------------------------------------

std::vector<StepResult> TrainingLoop::execute_epoch() {
    std::vector<StepResult> results;

    // Determine how many steps constitute one epoch.
    // If max_steps is set, cap the number of steps.
    int64_t steps_in_epoch = config_.max_steps >= 0
        ? config_.max_steps
        : current_batch_.num_micro_batches;

    for (int64_t s = 0; s < steps_in_epoch; ++s) {
        StepResult r = execute_step();

        // If the step failed (NaN loss due to unrecoverable failure),
        // stop the epoch early.
        if (std::isnan(r.loss)) {
            results.push_back(std::move(r));
            break;
        }

        results.push_back(std::move(r));
    }

    ++current_epoch_;
    return results;
}

// ---------------------------------------------------------------------------
// handle_failure()
// ---------------------------------------------------------------------------

bool TrainingLoop::handle_failure(int64_t failed_device_id) {
    if (!config_.enable_fault_tolerance) {
        return false;
    }

    // Mark the device as dead in the mesh.
    if (mesh_) {
        mesh_->mark_device_dead(failed_device_id);
    }

    // Attempt forward recovery.
    if (recovery_) {
        auto plan = recovery_->recover({failed_device_id});
        if (!plan.success) {
            return false;
        }

        // If the mesh needs rebuilding, re-initialise distributed components.
        if (plan.needs_remesh) {
            // Recreate the sharding analyzer, NCCL bridge, and pipeline
            // overlapper with the updated mesh.
            sharding_ = std::make_unique<distributed::ShardingAnalyzer>(*mesh_);
            nccl_ = std::make_unique<distributed::NCCLBridge>(*mesh_);
            pipeline_ = std::make_unique<distributed::PipelineOverlapper>(
                *mesh_, *nccl_);
        }
    }

    // Recompile the kernel if the surviving device count changed
    // (the tile configuration may need adjustment for the new topology).
    if (mesh_ && mesh_->num_alive_devices() < mesh_->total_devices()) {
        // Re-run the superoptimizer with a reconstructed iteration space
        // scaled for the new device count, then regenerate the kernel.
        if (superopt_ && optimized_result_.valid()) {
            // Reconstruct a matmul iteration space from the current
            // best tile's dimensions, scaled to the reduced device count.
            const auto& inner = optimized_result_.best_tile.inner_tiles;
            int64_t M = (inner.size() > 0) ? inner[0] * 8 : 4096;
            int64_t N = (inner.size() > 1) ? inner[1] * 8 : 4096;
            int64_t K = (inner.size() > 2) ? inner[2] * 8 : 4096;

            // Scale dimensions proportionally to the reduced device count
            int64_t total = mesh_->total_devices();
            int64_t alive = mesh_->num_alive_devices();
            if (total > 0 && alive > 0) {
                double scale = static_cast<double>(alive) / static_cast<double>(total);
                M = std::max<int64_t>(static_cast<int64_t>(M * scale), 16);
                N = std::max<int64_t>(static_cast<int64_t>(N * scale), 16);
                K = std::max<int64_t>(static_cast<int64_t>(K * scale), 16);
            }

            auto ispace = polyhedral::make_matmul_iteration_space(M, N, K);
            optimized_result_ = superopt_->optimize(ispace);

            // Regenerate the kernel with the new tile configuration
            if (optimized_result_.valid() && codegen_) {
                current_kernel_ = codegen_->generate_matmul(
                    M, N, K, optimized_result_.best_tile);
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const TrainingConfig& TrainingLoop::config() const {
    return config_;
}

const DynamicBatchSizer& TrainingLoop::batch_sizer() const {
    return *batch_sizer_;
}

const MemoryWatchdog& TrainingLoop::watchdog() const {
    return *watchdog_;
}

// ---------------------------------------------------------------------------
// Callback registration
// ---------------------------------------------------------------------------

void TrainingLoop::on_step_complete(StepCallback cb) {
    step_callbacks_.push_back(std::move(cb));
}

// ---------------------------------------------------------------------------
// Internal: notify all registered step callbacks
// ---------------------------------------------------------------------------

void TrainingLoop::notify_step(const StepResult& result) {
    for (const auto& cb : step_callbacks_) {
        cb(result);
    }
}

} // namespace symplex::training
