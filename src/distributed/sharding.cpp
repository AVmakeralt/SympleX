// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/distributed/sharding.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace symplex::distributed {

// ── Constructor ────────────────────────────────────────────────────────

ShardingAnalyzer::ShardingAnalyzer(const ClusterMesh& mesh)
    : mesh_(mesh)
{}

// ── Helpers ────────────────────────────────────────────────────────────

int64_t ShardingAnalyzer::tensor_bytes(const std::vector<int64_t>& shape, int64_t bpe) {
    if (shape.empty()) return 0;
    int64_t elements = 1;
    for (auto s : shape) {
        elements *= s;
    }
    return elements * bpe;
}

int64_t ShardingAnalyzer::find_mesh_dim(const std::string& name) const {
    const auto& dims = mesh_.dimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i].name == name) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

// ── Matmul analysis ────────────────────────────────────────────────────

ShardPlan ShardingAnalyzer::analyze_matmul(
    int64_t M, int64_t N, int64_t K,
    const std::string& tensor_a,
    const std::string& tensor_b,
    const std::string& tensor_c
) const {
    ShardPlan plan;

    // Original tensor shapes
    // A: [M, K], B: [K, N], C: [M, N]
    std::vector<int64_t> shape_a = {M, K};
    std::vector<int64_t> shape_b = {K, N};
    std::vector<int64_t> shape_c = {M, N};

    // Look for tensor_parallel dimension
    int64_t tp_dim = find_mesh_dim("tensor_parallel");
    int64_t tp_size = (tp_dim >= 0) ? mesh_.size(static_cast<size_t>(tp_dim)) : 1;

    // Look for pipeline_parallel dimension
    int64_t pp_dim = find_mesh_dim("pipeline_parallel");
    int64_t pp_size = (pp_dim >= 0) ? mesh_.size(static_cast<size_t>(pp_dim)) : 1;

    // Look for data_parallel dimension
    int64_t dp_dim = find_mesh_dim("data_parallel");
    int64_t dp_size = (dp_dim >= 0) ? mesh_.size(static_cast<size_t>(dp_dim)) : 1;

    // ── Tensor parallelism sharding ────────────────────────────────
    //
    // Megatron-LM column-parallel style:
    //   - A[M,K] is split along K -> local A is [M, K/tp]
    //   - B[K,N] is split along K -> local B is [K/tp, N]
    //   - Each device computes a partial C[M,N]
    //   - All-reduce C across TP dimension to get the full result
    //
    // For 2D sharding (SHARD_2D):
    //   - A[M,K] split along M and K -> local A is [M/tp, K/tp]
    //   - B[K,N] split along K and N -> local B is [K/tp, N/tp]
    //   - C[M,N] split along M and N -> local C is [M/tp, N/tp]
    //   - Requires all-reduce + all-gather

    if (tp_size > 1 && tp_dim >= 0) {
        // Use 1D column-parallel sharding along K dimension
        ShardingSpec spec_a;
        spec_a.tensor_name = tensor_a;
        spec_a.strategy = ShardingStrategy::SHARD_DIM_1;
        spec_a.shard_dims = {tp_dim};
        spec_a.original_shape = shape_a;
        spec_a.local_shape = {M, (K + tp_size - 1) / tp_size};

        ShardingSpec spec_b;
        spec_b.tensor_name = tensor_b;
        spec_b.strategy = ShardingStrategy::SHARD_DIM_0;
        spec_b.shard_dims = {tp_dim};
        spec_b.original_shape = shape_b;
        spec_b.local_shape = {(K + tp_size - 1) / tp_size, N};

        ShardingSpec spec_c;
        spec_c.tensor_name = tensor_c;
        spec_c.strategy = ShardingStrategy::REPLICATE;
        spec_c.shard_dims = {};
        spec_c.original_shape = shape_c;
        spec_c.local_shape = {M, N};

        plan.tensor_shards.push_back(std::move(spec_a));
        plan.tensor_shards.push_back(std::move(spec_b));
        plan.tensor_shards.push_back(std::move(spec_c));

        // After local matmul, each device has a partial C.
        // An all-reduce is needed across the TP mesh dimension.
        plan.all_reduce_ops.push_back(
            "all_reduce(" + tensor_c + ", mesh_dim=tensor_parallel, size=" +
            std::to_string(tp_size) + ")");

        // Communication volume: each device sends/receives its local
        // C fragment to all TP peers. In an all-reduce with ring algorithm,
        // each device sends 2*(tp_size-1)/tp_size of the data volume.
        int64_t c_local_bytes = tensor_bytes({M, N});
        int64_t allreduce_bytes = c_local_bytes * 2 * (tp_size - 1) / tp_size;
        plan.total_communication_bytes += allreduce_bytes;
    } else {
        // No tensor parallelism: replicate everything
        ShardingSpec spec_a;
        spec_a.tensor_name = tensor_a;
        spec_a.strategy = ShardingStrategy::REPLICATE;
        spec_a.shard_dims = {};
        spec_a.original_shape = shape_a;
        spec_a.local_shape = shape_a;

        ShardingSpec spec_b;
        spec_b.tensor_name = tensor_b;
        spec_b.strategy = ShardingStrategy::REPLICATE;
        spec_b.shard_dims = {};
        spec_b.original_shape = shape_b;
        spec_b.local_shape = shape_b;

        ShardingSpec spec_c;
        spec_c.tensor_name = tensor_c;
        spec_c.strategy = ShardingStrategy::REPLICATE;
        spec_c.shard_dims = {};
        spec_c.original_shape = shape_c;
        spec_c.local_shape = shape_c;

        plan.tensor_shards.push_back(std::move(spec_a));
        plan.tensor_shards.push_back(std::move(spec_b));
        plan.tensor_shards.push_back(std::move(spec_c));
    }

    // ── Pipeline parallelism ───────────────────────────────────────
    // With pipeline parallelism, the K dimension is effectively
    // partitioned across PP stages. Each stage processes a slice
    // of the K dimension, and activations are sent between stages.
    if (pp_size > 1 && pp_dim >= 0) {
        // In pipeline parallelism for matmul, we split K across
        // pipeline stages. Each stage computes a partial matmul
        // with K/pp_size of the columns of A and rows of B.
        // The result is sent to the next stage for accumulation.

        // Adjust local shapes to account for pipeline splitting
        for (auto& spec : plan.tensor_shards) {
            if (spec.tensor_name == tensor_a) {
                // A is further split along K (dim 1) by pp_size
                int64_t local_k = spec.local_shape[1];
                // If already sharded by TP, the effective local K
                // doesn't change further for PP; instead PP handles
                // layer-level pipelining. For a single matmul, PP
                // splits K across stages.
                spec.local_shape[1] = (local_k + pp_size - 1) / pp_size;
            } else if (spec.tensor_name == tensor_b) {
                int64_t local_k = spec.local_shape[0];
                spec.local_shape[0] = (local_k + pp_size - 1) / pp_size;
            }
        }

        // Pipeline communication: send activations between stages
        // Each stage sends its partial result (shape [M, N]) to the next
        int64_t activation_bytes = tensor_bytes({M, N});
        // pp_size - 1 point-to-point transfers
        plan.total_communication_bytes += activation_bytes * (pp_size - 1);

        plan.all_reduce_ops.push_back(
            "pipeline_sendrecv(activation, mesh_dim=pipeline_parallel, stages=" +
            std::to_string(pp_size) + ")");
    }

    // ── Data parallelism ──────────────────────────────────────────
    // With data parallelism, the entire computation is replicated
    // across DP replicas, and gradients are all-reduced.
    if (dp_size > 1 && dp_dim >= 0) {
        // All tensors are replicated across DP dimension
        // Gradient all-reduce is needed during backward pass
        int64_t grad_bytes_a = tensor_bytes(shape_a);
        int64_t grad_bytes_b = tensor_bytes(shape_b);
        int64_t total_grad_bytes = grad_bytes_a + grad_bytes_b;

        // Ring all-reduce: 2*(dp-1)/dp of the gradient data
        int64_t dp_allreduce_bytes = total_grad_bytes * 2 * (dp_size - 1) / dp_size;
        plan.total_communication_bytes += dp_allreduce_bytes;

        plan.all_reduce_ops.push_back(
            "all_reduce(gradients, mesh_dim=data_parallel, size=" +
            std::to_string(dp_size) + ")");
    }

    // ── Peak memory per device ────────────────────────────────────
    int64_t peak_mem = 0;
    for (const auto& spec : plan.tensor_shards) {
        peak_mem += tensor_bytes(spec.local_shape);
    }
    // Account for activation memory and gradient buffers (2x for grad)
    peak_mem *= 3;  // Forward activations + backward gradients + optimizer states
    plan.peak_memory_per_device = peak_mem;

    return plan;
}

// ── Transformer layer analysis ─────────────────────────────────────────

ShardPlan ShardingAnalyzer::analyze_transformer_layer(
    int64_t hidden_dim, int64_t num_heads, int64_t seq_len,
    int64_t ff_dim
) const {
    ShardPlan plan;

    int64_t tp_dim = find_mesh_dim("tensor_parallel");
    int64_t tp_size = (tp_dim >= 0) ? mesh_.size(static_cast<size_t>(tp_dim)) : 1;

    int64_t pp_dim = find_mesh_dim("pipeline_parallel");
    int64_t pp_size = (pp_dim >= 0) ? mesh_.size(static_cast<size_t>(pp_dim)) : 1;

    int64_t dp_dim = find_mesh_dim("data_parallel");
    int64_t dp_size = (dp_dim >= 0) ? mesh_.size(static_cast<size_t>(dp_dim)) : 1;

    // Head dimension for multi-head attention
    [[maybe_unused]] int64_t head_dim = hidden_dim / num_heads;

    // ── QKV projection: [seq_len, hidden_dim] * [hidden_dim, 3*hidden_dim] ─
    // Sharded across TP: Q, K, V projections are split per head group
    {
        ShardingSpec spec_qkv_weight;
        spec_qkv_weight.tensor_name = "qkv_weight";
        spec_qkv_weight.original_shape = {hidden_dim, 3 * hidden_dim};
        if (tp_size > 1 && tp_dim >= 0) {
            // Column-parallel: split output dimension across TP
            spec_qkv_weight.strategy = ShardingStrategy::SHARD_DIM_1;
            spec_qkv_weight.shard_dims = {tp_dim};
            spec_qkv_weight.local_shape = {hidden_dim, (3 * hidden_dim + tp_size - 1) / tp_size};
        } else {
            spec_qkv_weight.strategy = ShardingStrategy::REPLICATE;
            spec_qkv_weight.shard_dims = {};
            spec_qkv_weight.local_shape = spec_qkv_weight.original_shape;
        }
        plan.tensor_shards.push_back(std::move(spec_qkv_weight));
    }

    // ── QKV bias: [3 * hidden_dim] ────────────────────────────────────
    {
        ShardingSpec spec_qkv_bias;
        spec_qkv_bias.tensor_name = "qkv_bias";
        spec_qkv_bias.original_shape = {3 * hidden_dim};
        if (tp_size > 1 && tp_dim >= 0) {
            spec_qkv_bias.strategy = ShardingStrategy::SHARD_DIM_0;
            spec_qkv_bias.shard_dims = {tp_dim};
            spec_qkv_bias.local_shape = {(3 * hidden_dim + tp_size - 1) / tp_size};
        } else {
            spec_qkv_bias.strategy = ShardingStrategy::REPLICATE;
            spec_qkv_bias.shard_dims = {};
            spec_qkv_bias.local_shape = spec_qkv_bias.original_shape;
        }
        plan.tensor_shards.push_back(std::move(spec_qkv_bias));
    }

    // ── Attention output projection: [hidden_dim, hidden_dim] ──────────
    {
        ShardingSpec spec_attn_out_weight;
        spec_attn_out_weight.tensor_name = "attn_output_weight";
        spec_attn_out_weight.original_shape = {hidden_dim, hidden_dim};
        if (tp_size > 1 && tp_dim >= 0) {
            // Row-parallel: split input dimension across TP
            spec_attn_out_weight.strategy = ShardingStrategy::SHARD_DIM_0;
            spec_attn_out_weight.shard_dims = {tp_dim};
            spec_attn_out_weight.local_shape = {(hidden_dim + tp_size - 1) / tp_size, hidden_dim};
        } else {
            spec_attn_out_weight.strategy = ShardingStrategy::REPLICATE;
            spec_attn_out_weight.shard_dims = {};
            spec_attn_out_weight.local_shape = spec_attn_out_weight.original_shape;
        }
        plan.tensor_shards.push_back(std::move(spec_attn_out_weight));
    }

    // ── Attention output bias: [hidden_dim] ────────────────────────────
    {
        ShardingSpec spec_attn_out_bias;
        spec_attn_out_bias.tensor_name = "attn_output_bias";
        spec_attn_out_bias.original_shape = {hidden_dim};
        spec_attn_out_bias.strategy = ShardingStrategy::REPLICATE;
        spec_attn_out_bias.shard_dims = {};
        spec_attn_out_bias.local_shape = spec_attn_out_bias.original_shape;
        plan.tensor_shards.push_back(std::move(spec_attn_out_bias));
    }

    // ── FFN up projection: [hidden_dim, ff_dim] ───────────────────────
    {
        ShardingSpec spec_ffn_up;
        spec_ffn_up.tensor_name = "ffn_up_weight";
        spec_ffn_up.original_shape = {hidden_dim, ff_dim};
        if (tp_size > 1 && tp_dim >= 0) {
            // Column-parallel: split ff_dim across TP
            spec_ffn_up.strategy = ShardingStrategy::SHARD_DIM_1;
            spec_ffn_up.shard_dims = {tp_dim};
            spec_ffn_up.local_shape = {hidden_dim, (ff_dim + tp_size - 1) / tp_size};
        } else {
            spec_ffn_up.strategy = ShardingStrategy::REPLICATE;
            spec_ffn_up.shard_dims = {};
            spec_ffn_up.local_shape = spec_ffn_up.original_shape;
        }
        plan.tensor_shards.push_back(std::move(spec_ffn_up));
    }

    // ── FFN down projection: [ff_dim, hidden_dim] ─────────────────────
    {
        ShardingSpec spec_ffn_down;
        spec_ffn_down.tensor_name = "ffn_down_weight";
        spec_ffn_down.original_shape = {ff_dim, hidden_dim};
        if (tp_size > 1 && tp_dim >= 0) {
            // Row-parallel: split ff_dim (input) across TP
            spec_ffn_down.strategy = ShardingStrategy::SHARD_DIM_0;
            spec_ffn_down.shard_dims = {tp_dim};
            spec_ffn_down.local_shape = {(ff_dim + tp_size - 1) / tp_size, hidden_dim};
        } else {
            spec_ffn_down.strategy = ShardingStrategy::REPLICATE;
            spec_ffn_down.shard_dims = {};
            spec_ffn_down.local_shape = spec_ffn_down.original_shape;
        }
        plan.tensor_shards.push_back(std::move(spec_ffn_down));
    }

    // ── Layer norm parameters: [hidden_dim] (replicated) ──────────────
    {
        ShardingSpec spec_ln1;
        spec_ln1.tensor_name = "layer_norm_1";
        spec_ln1.original_shape = {hidden_dim};
        spec_ln1.strategy = ShardingStrategy::REPLICATE;
        spec_ln1.shard_dims = {};
        spec_ln1.local_shape = spec_ln1.original_shape;
        plan.tensor_shards.push_back(std::move(spec_ln1));
    }
    {
        ShardingSpec spec_ln2;
        spec_ln2.tensor_name = "layer_norm_2";
        spec_ln2.original_shape = {hidden_dim};
        spec_ln2.strategy = ShardingStrategy::REPLICATE;
        spec_ln2.shard_dims = {};
        spec_ln2.local_shape = spec_ln2.original_shape;
        plan.tensor_shards.push_back(std::move(spec_ln2));
    }

    // ── Communication operations ──────────────────────────────────────
    if (tp_size > 1 && tp_dim >= 0) {
        // After QKV projection (column-parallel): no all-reduce needed
        // Each TP shard has different heads

        // After attention output projection (row-parallel):
        // All-reduce across TP dimension
        int64_t attn_out_bytes = tensor_bytes({seq_len, hidden_dim});
        int64_t ar_bytes = attn_out_bytes * 2 * (tp_size - 1) / tp_size;
        plan.total_communication_bytes += ar_bytes;
        plan.all_reduce_ops.push_back(
            "all_reduce(attn_output, mesh_dim=tensor_parallel, size=" +
            std::to_string(tp_size) + ")");

        // After FFN down projection (row-parallel):
        // All-reduce across TP dimension
        int64_t ffn_out_bytes = tensor_bytes({seq_len, hidden_dim});
        int64_t ffn_ar_bytes = ffn_out_bytes * 2 * (tp_size - 1) / tp_size;
        plan.total_communication_bytes += ffn_ar_bytes;
        plan.all_reduce_ops.push_back(
            "all_reduce(ffn_output, mesh_dim=tensor_parallel, size=" +
            std::to_string(tp_size) + ")");
    }

    if (pp_size > 1 && pp_dim >= 0) {
        // Pipeline: send activations between stages
        int64_t activation_bytes = tensor_bytes({seq_len, hidden_dim});
        plan.total_communication_bytes += activation_bytes * (pp_size - 1) * 2;
        // *2 for forward + backward activation transfers
        plan.all_reduce_ops.push_back(
            "pipeline_sendrecv(activations, mesh_dim=pipeline_parallel, stages=" +
            std::to_string(pp_size) + ")");
    }

    if (dp_size > 1 && dp_dim >= 0) {
        // Data parallel gradient all-reduce
        int64_t total_param_bytes = 0;
        for (const auto& spec : plan.tensor_shards) {
            total_param_bytes += tensor_bytes(spec.local_shape);
        }
        int64_t dp_ar_bytes = total_param_bytes * 2 * (dp_size - 1) / dp_size;
        plan.total_communication_bytes += dp_ar_bytes;
        plan.all_reduce_ops.push_back(
            "all_reduce(gradients, mesh_dim=data_parallel, size=" +
            std::to_string(dp_size) + ")");
    }

    // ── Peak memory per device ────────────────────────────────────────
    int64_t param_mem = 0;
    for (const auto& spec : plan.tensor_shards) {
        param_mem += tensor_bytes(spec.local_shape);
    }
    // Activation memory: input [seq_len, hidden_dim] + QKV output + attention output
    int64_t activation_mem = tensor_bytes({seq_len, hidden_dim}) * 4;
    // Gradient + optimizer states: roughly 3x parameter memory (Adam: param + momentum + variance)
    int64_t optimizer_mem = param_mem * 2;
    plan.peak_memory_per_device = param_mem + activation_mem + optimizer_mem;

    return plan;
}

// ── Local dimension computation ────────────────────────────────────────

std::vector<int64_t> ShardingAnalyzer::compute_local_dims(
    const std::vector<int64_t>& global_dims,
    const ShardingSpec& spec,
    const MeshCoord& device_coord
) const {
    if (global_dims.empty()) return {};

    std::vector<int64_t> local = global_dims;

    switch (spec.strategy) {
        case ShardingStrategy::REPLICATE:
            // Full replica: local shape equals global shape
            break;

        case ShardingStrategy::SHARD_DIM_0: {
            // Shard dimension 0 across the specified mesh dimension
            if (spec.shard_dims.empty() || global_dims.empty()) break;
            int64_t mesh_dim_idx = spec.shard_dims[0];
            if (mesh_dim_idx < 0 ||
                mesh_dim_idx >= static_cast<int64_t>(mesh_.ndim())) {
                break;
            }
            int64_t shard_size = mesh_.size(static_cast<size_t>(mesh_dim_idx));
            if (shard_size <= 0) break;
            // Determine which shard this device owns
            int64_t shard_idx = (device_coord.size() > static_cast<size_t>(mesh_dim_idx))
                                ? device_coord[mesh_dim_idx] : 0;
            int64_t base = global_dims[0] / shard_size;
            int64_t remainder = global_dims[0] % shard_size;
            local[0] = base + (shard_idx < remainder ? 1 : 0);
            break;
        }

        case ShardingStrategy::SHARD_DIM_1: {
            // Shard dimension 1 across the specified mesh dimension
            if (spec.shard_dims.empty() || global_dims.size() < 2) break;
            int64_t mesh_dim_idx = spec.shard_dims[0];
            if (mesh_dim_idx < 0 ||
                mesh_dim_idx >= static_cast<int64_t>(mesh_.ndim())) {
                break;
            }
            int64_t shard_size = mesh_.size(static_cast<size_t>(mesh_dim_idx));
            if (shard_size <= 0) break;
            int64_t shard_idx = (device_coord.size() > static_cast<size_t>(mesh_dim_idx))
                                ? device_coord[mesh_dim_idx] : 0;
            int64_t base = global_dims[1] / shard_size;
            int64_t remainder = global_dims[1] % shard_size;
            local[1] = base + (shard_idx < remainder ? 1 : 0);
            break;
        }

        case ShardingStrategy::SHARD_2D: {
            // Shard two dimensions across two mesh dimensions
            if (spec.shard_dims.size() < 2 || global_dims.size() < 2) break;
            int64_t mesh_dim0 = spec.shard_dims[0];
            int64_t mesh_dim1 = spec.shard_dims[1];

            // Dimension 0 sharded along mesh_dim0
            if (mesh_dim0 >= 0 && mesh_dim0 < static_cast<int64_t>(mesh_.ndim())) {
                int64_t shard_size0 = mesh_.size(static_cast<size_t>(mesh_dim0));
                if (shard_size0 > 0) {
                    int64_t shard_idx0 = (device_coord.size() > static_cast<size_t>(mesh_dim0))
                                         ? device_coord[mesh_dim0] : 0;
                    int64_t base0 = global_dims[0] / shard_size0;
                    int64_t rem0 = global_dims[0] % shard_size0;
                    local[0] = base0 + (shard_idx0 < rem0 ? 1 : 0);
                }
            }

            // Dimension 1 sharded along mesh_dim1
            if (mesh_dim1 >= 0 && mesh_dim1 < static_cast<int64_t>(mesh_.ndim())) {
                int64_t shard_size1 = mesh_.size(static_cast<size_t>(mesh_dim1));
                if (shard_size1 > 0) {
                    int64_t shard_idx1 = (device_coord.size() > static_cast<size_t>(mesh_dim1))
                                         ? device_coord[mesh_dim1] : 0;
                    int64_t base1 = global_dims[1] / shard_size1;
                    int64_t rem1 = global_dims[1] % shard_size1;
                    local[1] = base1 + (shard_idx1 < rem1 ? 1 : 0);
                }
            }
            break;
        }

        case ShardingStrategy::PIPELINE: {
            // Pipeline: the entire computation is replicated but
            // processes different micro-batches; tensor shapes remain
            // the same as global (layer-level pipelining doesn't
            // change per-layer tensor shapes).
            break;
        }
    }

    return local;
}

} // namespace symplex::distributed
