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
#include "symplex/distributed/sharding.h"

namespace symplex::distributed {

/// CollectiveOp: types of collective communication operations.
enum class CollectiveOp {
    ALL_REDUCE,
    ALL_GATHER,
    REDUCE_SCATTER,
    ALL_TO_ALL,
    BROADCAST,
    SEND_RECV,
};

/// CommunicationSchedule: the full schedule of collective operations
/// required to implement a ShardPlan.
struct CommunicationSchedule {
    struct CommOp {
        CollectiveOp op;
        int64_t bytes;
        std::vector<int64_t> src_devices;
        std::vector<int64_t> dst_devices;
        std::string tensor_name;
        bool async;     // Can this overlap with computation?

        std::string to_string() const {
            std::ostringstream oss;
            oss << "CommOp{op=";
            switch (op) {
                case CollectiveOp::ALL_REDUCE:    oss << "ALL_REDUCE"; break;
                case CollectiveOp::ALL_GATHER:    oss << "ALL_GATHER"; break;
                case CollectiveOp::REDUCE_SCATTER: oss << "REDUCE_SCATTER"; break;
                case CollectiveOp::ALL_TO_ALL:    oss << "ALL_TO_ALL"; break;
                case CollectiveOp::BROADCAST:     oss << "BROADCAST"; break;
                case CollectiveOp::SEND_RECV:     oss << "SEND_RECV"; break;
            }
            oss << ", bytes=" << bytes
                << ", tensor='" << tensor_name << "'"
                << ", src=[";
            for (size_t i = 0; i < src_devices.size(); ++i) {
                if (i > 0) oss << ",";
                oss << src_devices[i];
            }
            oss << "], dst=[";
            for (size_t i = 0; i < dst_devices.size(); ++i) {
                if (i > 0) oss << ",";
                oss << dst_devices[i];
            }
            oss << "], async=" << (async ? "yes" : "no") << "}";
            return oss.str();
        }
    };

    std::vector<CommOp> operations;
    int64_t total_sync_bytes = 0;      // Bytes that must be synchronized
    int64_t total_async_bytes = 0;     // Bytes that can overlap

    std::string to_string() const {
        std::ostringstream oss;
        oss << "CommunicationSchedule{ops=[";
        for (size_t i = 0; i < operations.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << operations[i].to_string();
        }
        oss << "], sync_bytes=" << total_sync_bytes
            << ", async_bytes=" << total_async_bytes << "}";
        return oss.str();
    }
};

/// NCCLBridge: generates NCCL-style communication schedules from
/// ShardPlans and estimates communication latency.
class NCCLBridge {
public:
    explicit NCCLBridge(const ClusterMesh& mesh);

    /// Generate the communication schedule for a sharding plan.
    CommunicationSchedule generate_schedule(const ShardPlan& plan) const;

    /// Generate NCCL-style collective operation code strings
    /// (ncclAllReduce, ncclAllGather, etc.).
    std::string emit_nccl_ops(const CommunicationSchedule& schedule) const;

    /// Estimate communication latency in nanoseconds for the schedule.
    double estimate_communication_latency_ns(const CommunicationSchedule& schedule) const;

    /// Check if NCCL is available at runtime.
    bool is_available() const;

private:
    ClusterMesh mesh_;
    double intra_node_bw_gbps_ = 600.0;    // NVLink bandwidth
    double inter_node_bw_gbps_ = 100.0;    // InfiniBand bandwidth
    double intra_node_latency_ns_ = 500.0;
    double inter_node_latency_ns_ = 5000.0;

    /// Check if two devices are on the same physical node.
    bool is_same_node(int64_t dev1, int64_t dev2) const;

    /// Determine the effective bandwidth for a set of devices.
    double effective_bandwidth_gbps(
        const std::vector<int64_t>& devices) const;

    /// Convert a CollectiveOp to its NCCL function name.
    static std::string nccl_op_name(CollectiveOp op);
};

} // namespace symplex::distributed
