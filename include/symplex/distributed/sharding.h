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

namespace symplex::distributed {

/// ShardingStrategy: how a tensor is distributed across the mesh.
enum class ShardingStrategy {
    REPLICATE,            // Full copy on each device
    SHARD_DIM_0,         // Split along dimension 0
    SHARD_DIM_1,         // Split along dimension 1
    SHARD_2D,            // Split along two dimensions (e.g., Megatron-LM style)
    PIPELINE,            // Pipeline parallelism along layer dimension
};

/// ShardingSpec: complete specification of how one tensor is sharded.
struct ShardingSpec {
    std::string tensor_name;
    ShardingStrategy strategy = ShardingStrategy::REPLICATE;
    std::vector<int64_t> shard_dims;   // Which mesh dims to shard along
    std::vector<int64_t> original_shape;
    std::vector<int64_t> local_shape;  // Shape on each device after sharding

    std::string to_string() const {
        std::ostringstream oss;
        oss << "ShardingSpec{name='" << tensor_name << "', strategy=";
        switch (strategy) {
            case ShardingStrategy::REPLICATE:  oss << "REPLICATE"; break;
            case ShardingStrategy::SHARD_DIM_0: oss << "SHARD_DIM_0"; break;
            case ShardingStrategy::SHARD_DIM_1: oss << "SHARD_DIM_1"; break;
            case ShardingStrategy::SHARD_2D:   oss << "SHARD_2D"; break;
            case ShardingStrategy::PIPELINE:   oss << "PIPELINE"; break;
        }
        oss << ", shard_dims=[";
        for (size_t i = 0; i < shard_dims.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shard_dims[i];
        }
        oss << "], original=[";
        for (size_t i = 0; i < original_shape.size(); ++i) {
            if (i > 0) oss << "x";
            oss << original_shape[i];
        }
        oss << "], local=[";
        for (size_t i = 0; i < local_shape.size(); ++i) {
            if (i > 0) oss << "x";
            oss << local_shape[i];
        }
        oss << "]}";
        return oss.str();
    }
};

/// ShardPlan: the complete sharding plan for all tensors in a computation.
struct ShardPlan {
    std::vector<ShardingSpec> tensor_shards;
    std::vector<std::string> all_reduce_ops;       // Required collective operations
    std::vector<std::string> all_to_all_ops;
    int64_t total_communication_bytes = 0;
    int64_t peak_memory_per_device = 0;

    std::string to_string() const {
        std::ostringstream oss;
        oss << "ShardPlan{tensors=[";
        for (size_t i = 0; i < tensor_shards.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << tensor_shards[i].to_string();
        }
        oss << "], all_reduce_ops=[";
        for (size_t i = 0; i < all_reduce_ops.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << all_reduce_ops[i];
        }
        oss << "], all_to_all_ops=[";
        for (size_t i = 0; i < all_to_all_ops.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << all_to_all_ops[i];
        }
        oss << "], comm_bytes=" << total_communication_bytes
            << ", peak_mem=" << peak_memory_per_device << "}";
        return oss.str();
    }
};

/// ShardingAnalyzer: analyzes polyhedral iteration spaces and determines
/// optimal sharding strategies for distributing tensors across the mesh.
class ShardingAnalyzer {
public:
    explicit ShardingAnalyzer(const ClusterMesh& mesh);

    /// Analyze a matrix multiplication (C = A * B) and determine
    /// the optimal sharding strategy across the mesh dimensions.
    ///
    /// If TP mesh dim exists: shard A along K, B along K,
    ///   all-reduce C along TP dim (Megatron-LM column-parallel style).
    /// If PP mesh dim exists: pipeline layers along PP dim.
    ShardPlan analyze_matmul(
        int64_t M, int64_t N, int64_t K,
        const std::string& tensor_a = "A",
        const std::string& tensor_b = "B",
        const std::string& tensor_c = "C"
    ) const;

    /// Analyze a transformer layer and determine sharding for
    /// attention and feed-forward components.
    ShardPlan analyze_transformer_layer(
        int64_t hidden_dim, int64_t num_heads, int64_t seq_len,
        int64_t ff_dim
    ) const;

    /// Compute the local tile dimensions for a given device
    /// after applying the sharding spec.
    std::vector<int64_t> compute_local_dims(
        const std::vector<int64_t>& global_dims,
        const ShardingSpec& spec,
        const MeshCoord& device_coord
    ) const;

private:
    ClusterMesh mesh_;

    /// Compute the number of bytes for a tensor of the given shape.
    /// @param shape  Dimension sizes of the tensor.
    /// @param bpe    Bytes per element (default 2 for FP16).
    static int64_t tensor_bytes(const std::vector<int64_t>& shape, int64_t bpe = 2);

    /// Find a mesh dimension by name; returns its index or -1.
    int64_t find_mesh_dim(const std::string& name) const;
};

} // namespace symplex::distributed
