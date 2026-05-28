// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/distributed/nccl_bridge.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <cmath>

namespace symplex::distributed {

// ── Constructor ────────────────────────────────────────────────────────

NCCLBridge::NCCLBridge(const ClusterMesh& mesh)
    : mesh_(mesh)
{}

// ── Private helpers ────────────────────────────────────────────────────

bool NCCLBridge::is_same_node(int64_t dev1, int64_t dev2) const {
    const auto& devices = mesh_.devices();
    if (dev1 < 0 || dev2 < 0 ||
        dev1 >= static_cast<int64_t>(devices.size()) ||
        dev2 >= static_cast<int64_t>(devices.size())) {
        return false;
    }
    return devices[static_cast<size_t>(dev1)].node_id ==
           devices[static_cast<size_t>(dev2)].node_id;
}

double NCCLBridge::effective_bandwidth_gbps(
    const std::vector<int64_t>& devices
) const {
    if (devices.size() <= 1) return intra_node_bw_gbps_;

    // Check if all devices are on the same node
    bool all_same_node = true;
    for (size_t i = 1; i < devices.size(); ++i) {
        if (!is_same_node(devices[0], devices[i])) {
            all_same_node = false;
            break;
        }
    }

    if (all_same_node) {
        // Intra-node: NVLink bandwidth, divided across concurrent links
        // With NVLink, all GPUs on the same node can communicate at full bandwidth
        return intra_node_bw_gbps_;
    }

    // Inter-node: bottleneck is the InfiniBand link
    // The effective bandwidth is limited by the slowest link
    return inter_node_bw_gbps_;
}

std::string NCCLBridge::nccl_op_name(CollectiveOp op) {
    switch (op) {
        case CollectiveOp::ALL_REDUCE:     return "ncclAllReduce";
        case CollectiveOp::ALL_GATHER:     return "ncclAllGather";
        case CollectiveOp::REDUCE_SCATTER: return "ncclReduceScatter";
        case CollectiveOp::ALL_TO_ALL:     return "ncclAllToAll";
        case CollectiveOp::BROADCAST:      return "ncclBroadcast";
        case CollectiveOp::SEND_RECV:      return "ncclSendRecv";
    }
    return "ncclUnknown";
}

// ── Schedule generation ────────────────────────────────────────────────

CommunicationSchedule NCCLBridge::generate_schedule(const ShardPlan& plan) const {
    CommunicationSchedule schedule;
    int64_t total_sync = 0;
    int64_t total_async = 0;

    const auto& devices = mesh_.devices();

    // Process all-reduce operations from the shard plan
    for (const auto& op_str : plan.all_reduce_ops) {
        // Parse the operation string to determine which mesh dimension
        // and how many devices are involved.
        // Format: "all_reduce(tensor_name, mesh_dim=X, size=N)"

        // Determine the mesh dimension and participating devices
        int64_t tp_dim = -1;
        int64_t dp_dim = -1;
        int64_t pp_dim = -1;
        const auto& mesh_dims = mesh_.dimensions();
        for (size_t i = 0; i < mesh_dims.size(); ++i) {
            if (mesh_dims[i].name == "tensor_parallel") tp_dim = static_cast<int64_t>(i);
            if (mesh_dims[i].name == "data_parallel")   dp_dim = static_cast<int64_t>(i);
            if (mesh_dims[i].name == "pipeline_parallel") pp_dim = static_cast<int64_t>(i);
        }

        // For each all-reduce, determine the device groups
        // An all-reduce operates across one mesh dimension.
        // We need to enumerate all groups of devices that share
        // the same coordinates in all other dimensions.

        int64_t target_dim = -1;
        int64_t group_size = 1;
        std::string tensor_name;

        // Determine which mesh dimension this all-reduce targets
        if (op_str.find("tensor_parallel") != std::string::npos) {
            target_dim = tp_dim;
            tensor_name = op_str.substr(
                op_str.find('(') + 1,
                op_str.find(',') - op_str.find('(') - 1);
        } else if (op_str.find("data_parallel") != std::string::npos) {
            target_dim = dp_dim;
            tensor_name = op_str.substr(
                op_str.find('(') + 1,
                op_str.find(',') - op_str.find('(') - 1);
        } else if (op_str.find("pipeline_parallel") != std::string::npos) {
            target_dim = pp_dim;
            tensor_name = "activations";
        } else {
            tensor_name = "unknown";
        }

        if (target_dim >= 0 && target_dim < static_cast<int64_t>(mesh_.ndim())) {
            group_size = mesh_.size(static_cast<size_t>(target_dim));
        }

        if (group_size <= 1) continue;  // No communication needed

        // Generate all-reduce operations for each device group
        // along the target mesh dimension.
        //
        // For a 3D mesh [dp, pp, tp], an all-reduce along tp
        // groups devices that share the same (dp, pp) coordinates.
        // Each group has tp_size devices.

        int64_t num_groups = mesh_.total_devices() / group_size;

        for (int64_t g = 0; g < num_groups; ++g) {
            CommunicationSchedule::CommOp comm;
            comm.op = CollectiveOp::ALL_REDUCE;

            // Compute bytes: find the tensor in the shard plan
            int64_t op_bytes = 0;
            for (const auto& spec : plan.tensor_shards) {
                if (spec.tensor_name == tensor_name ||
                    tensor_name == "activations" ||
                    tensor_name == "gradients") {
                    int64_t elements = 1;
                    for (auto s : spec.local_shape) elements *= s;
                    op_bytes += elements * 2;  // 2 bytes per element (FP16)
                }
            }
            if (op_bytes == 0) {
                // Fallback: estimate from total communication bytes
                op_bytes = plan.total_communication_bytes /
                          (num_groups > 0 ? num_groups : 1);
            }
            comm.bytes = op_bytes;
            comm.tensor_name = tensor_name;

            // Determine source and destination devices for this group
            // Build the base coordinate for this group
            // Group index g maps to a specific set of non-target-dimension
            // coordinates. The simpler enumeration below avoids complex
            // index decomposition.

            // Simpler approach: enumerate all devices and group them
            // by their coordinates in non-target dimensions
            std::vector<int64_t> group_devices;
            for (int64_t dev_id = 0; dev_id < static_cast<int64_t>(devices.size()); ++dev_id) {
                const auto& dev = devices[static_cast<size_t>(dev_id)];
                // Check if this device belongs to group g
                // Group by matching all non-target coordinates
                if (target_dim >= 0 &&
                    dev.mesh_coord.size() > static_cast<size_t>(target_dim)) {
                    // Build a group key from non-target coordinates
                    int64_t group_key = 0;
                    int64_t stride = 1;
                    for (size_t d = 0; d < mesh_.ndim(); ++d) {
                        if (static_cast<int64_t>(d) == target_dim) continue;
                        group_key += dev.mesh_coord[d] * stride;
                        stride *= mesh_.size(d);
                    }
                    if (group_key == g) {
                        group_devices.push_back(dev_id);
                    }
                }
            }

            comm.src_devices = group_devices;
            comm.dst_devices = group_devices;  // All-reduce: all devices are both src and dst

            // All-reduces can potentially overlap with compute
            // if using NCCL async operations
            comm.async = (op_str.find("gradients") != std::string::npos);

            if (comm.async) {
                total_async += comm.bytes;
            } else {
                total_sync += comm.bytes;
            }

            schedule.operations.push_back(std::move(comm));
        }
    }

    // Process all-to-all operations from the shard plan
    for (const auto& op_str : plan.all_to_all_ops) {
        CommunicationSchedule::CommOp comm;
        comm.op = CollectiveOp::ALL_TO_ALL;

        // All-to-all: every device sends a different slice to every other device
        int64_t total_dev = mesh_.total_devices();
        comm.bytes = plan.total_communication_bytes /
                    (total_dev > 0 ? total_dev : 1);
        comm.tensor_name = op_str;
        comm.async = false;

        for (int64_t i = 0; i < total_dev; ++i) {
            comm.src_devices.push_back(i);
            comm.dst_devices.push_back(i);
        }

        total_sync += comm.bytes * total_dev;
        schedule.operations.push_back(std::move(comm));
    }

    // Generate pipeline send/recv operations if pipeline parallelism is used
    int64_t pp_dim = -1;
    const auto& mesh_dims = mesh_.dimensions();
    for (size_t i = 0; i < mesh_dims.size(); ++i) {
        if (mesh_dims[i].name == "pipeline_parallel") {
            pp_dim = static_cast<int64_t>(i);
        }
    }

    if (pp_dim >= 0) {
        int64_t pp_size = mesh_.size(static_cast<size_t>(pp_dim));
        if (pp_size > 1) {
            // Generate point-to-point send/recv between adjacent pipeline stages
            // For each (dp, tp) combination, create a chain of PP stages
            int64_t dp_dim = -1;
            int64_t tp_dim_idx = -1;
            for (size_t i = 0; i < mesh_dims.size(); ++i) {
                if (mesh_dims[i].name == "data_parallel") dp_dim = static_cast<int64_t>(i);
                if (mesh_dims[i].name == "tensor_parallel") tp_dim_idx = static_cast<int64_t>(i);
            }

            // Enumerate all (dp, tp) pairs
            int64_t dp_size = (dp_dim >= 0) ? mesh_.size(static_cast<size_t>(dp_dim)) : 1;
            int64_t tp_size_val = (tp_dim_idx >= 0) ? mesh_.size(static_cast<size_t>(tp_dim_idx)) : 1;

            for (int64_t dp = 0; dp < dp_size; ++dp) {
                for (int64_t tp = 0; tp < tp_size_val; ++tp) {
                    for (int64_t pp = 0; pp < pp_size - 1; ++pp) {
                        // Send from stage pp to stage pp+1
                        MeshCoord src_coord(std::vector<int64_t>{});
                        if (mesh_.ndim() == 3) {
                            src_coord = MeshCoord({dp, pp, tp});
                        }
                        MeshCoord dst_coord(std::vector<int64_t>{});
                        if (mesh_.ndim() == 3) {
                            dst_coord = MeshCoord({dp, pp + 1, tp});
                        }

                        if (src_coord.empty() || dst_coord.empty()) continue;

                        int64_t src_linear = mesh_.coord_to_linear(src_coord);
                        int64_t dst_linear = mesh_.coord_to_linear(dst_coord);

                        CommunicationSchedule::CommOp comm;
                        comm.op = CollectiveOp::SEND_RECV;
                        comm.bytes = plan.total_communication_bytes /
                                   (pp_size > 0 ? pp_size : 1);
                        comm.src_devices = {src_linear};
                        comm.dst_devices = {dst_linear};
                        comm.tensor_name = "pipeline_activation_" +
                                          std::to_string(pp) + "_to_" +
                                          std::to_string(pp + 1);
                        comm.async = true;  // Pipeline comms can overlap with compute

                        total_async += comm.bytes;
                        schedule.operations.push_back(std::move(comm));
                    }
                }
            }
        }
    }

    schedule.total_sync_bytes = total_sync;
    schedule.total_async_bytes = total_async;

    return schedule;
}

// ── NCCL operation code emission ───────────────────────────────────────

std::string NCCLBridge::emit_nccl_ops(const CommunicationSchedule& schedule) const {
    std::ostringstream oss;
    oss << "// Auto-generated NCCL communication schedule\n";
    oss << "// SympleX distributed backend\n\n";

    oss << "#include <nccl.h>\n";
    oss << "#include <cuda_runtime.h>\n\n";

    oss << "// Initialize NCCL communicators\n";
    oss << "ncclComm_t nccl_comms[" << mesh_.total_devices() << "];\n";
    oss << "cudaStream_t nccl_streams[" << mesh_.total_devices() << "];\n\n";

    // Emit initialization code
    oss << "void init_nccl() {\n";
    oss << "  int ndev = " << mesh_.total_devices() << ";\n";
    oss << "  ncclUniqueId id;\n";
    oss << "  ncclGetUniqueId(&id);\n";
    oss << "  // Broadcast unique ID to all ranks\n";
    oss << "  ncclGroupStart();\n";
    oss << "  for (int i = 0; i < ndev; ++i) {\n";
    oss << "    cudaSetDevice(i);\n";
    oss << "    cudaStreamCreate(&nccl_streams[i]);\n";
    oss << "    ncclCommInitRank(&nccl_comms[i], ndev, id, i);\n";
    oss << "  }\n";
    oss << "  ncclGroupEnd();\n";
    oss << "}\n\n";

    // Emit each communication operation
    int op_idx = 0;
    for (const auto& op : schedule.operations) {
        oss << "// Op " << op_idx << ": " << op.to_string() << "\n";

        switch (op.op) {
            case CollectiveOp::ALL_REDUCE: {
                oss << "void all_reduce_" << op_idx << "(void* sendbuf, void* recvbuf, int rank) {\n";
                oss << "  ncclAllReduce(sendbuf, recvbuf, "
                    << (op.bytes / 2) << ", ncclFloat16, ncclSum, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  cudaStreamSynchronize(nccl_streams[rank]);\n";
                oss << "}\n\n";
                break;
            }

            case CollectiveOp::ALL_GATHER: {
                oss << "void all_gather_" << op_idx << "(void* sendbuf, void* recvbuf, int rank) {\n";
                oss << "  ncclAllGather(sendbuf, recvbuf, "
                    << (op.bytes / 2) << ", ncclFloat16, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  cudaStreamSynchronize(nccl_streams[rank]);\n";
                oss << "}\n\n";
                break;
            }

            case CollectiveOp::REDUCE_SCATTER: {
                oss << "void reduce_scatter_" << op_idx << "(void* sendbuf, void* recvbuf, int rank) {\n";
                oss << "  ncclReduceScatter(sendbuf, recvbuf, "
                    << (op.bytes / 2) << ", ncclFloat16, ncclSum, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  cudaStreamSynchronize(nccl_streams[rank]);\n";
                oss << "}\n\n";
                break;
            }

            case CollectiveOp::ALL_TO_ALL: {
                oss << "void all_to_all_" << op_idx << "(void* sendbuf, void* recvbuf, int rank) {\n";
                oss << "  // NCCL does not have native all-to-all; emulate with send/recv pairs\n";
                oss << "  int ndev = " << op.src_devices.size() << ";\n";
                oss << "  ncclGroupStart();\n";
                oss << "  for (int peer = 0; peer < ndev; ++peer) {\n";
                oss << "    if (peer == rank) continue;\n";
                oss << "    ncclSend(static_cast<char*>(sendbuf) + peer * "
                    << (op.bytes / 2) << " * sizeof(ncclFloat16), "
                    << (op.bytes / 2) << ", ncclFloat16, peer, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "    ncclRecv(static_cast<char*>(recvbuf) + peer * "
                    << (op.bytes / 2) << " * sizeof(ncclFloat16), "
                    << (op.bytes / 2) << ", ncclFloat16, peer, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  }\n";
                oss << "  ncclGroupEnd();\n";
                oss << "  cudaStreamSynchronize(nccl_streams[rank]);\n";
                oss << "}\n\n";
                break;
            }

            case CollectiveOp::BROADCAST: {
                oss << "void broadcast_" << op_idx << "(void* buf, int root, int rank) {\n";
                oss << "  ncclBroadcast(buf, buf, "
                    << (op.bytes / 2) << ", ncclFloat16, root, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  cudaStreamSynchronize(nccl_streams[rank]);\n";
                oss << "}\n\n";
                break;
            }

            case CollectiveOp::SEND_RECV: {
                oss << "void send_recv_" << op_idx << "(void* sendbuf, void* recvbuf, int rank) {\n";
                if (!op.src_devices.empty() && !op.dst_devices.empty()) {
                    oss << "  // Pipeline stage " << op.src_devices[0]
                        << " -> " << op.dst_devices[0] << "\n";
                }
                oss << "  ncclGroupStart();\n";
                oss << "  ncclSend(sendbuf, " << (op.bytes / 2)
                    << ", ncclFloat16, /*dst=*/rank + 1, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  ncclRecv(recvbuf, " << (op.bytes / 2)
                    << ", ncclFloat16, /*src=*/rank - 1, "
                    << "nccl_comms[rank], nccl_streams[rank]);\n";
                oss << "  ncclGroupEnd();\n";
                if (!op.async) {
                    oss << "  cudaStreamSynchronize(nccl_streams[rank]);\n";
                }
                oss << "}\n\n";
                break;
            }
        }

        ++op_idx;
    }

    // Emit the main execution function
    oss << "// Execute the full communication schedule\n";
    oss << "void execute_schedule(int rank) {\n";
    for (int i = 0; i < op_idx; ++i) {
        const auto& op = schedule.operations[static_cast<size_t>(i)];
        switch (op.op) {
            case CollectiveOp::ALL_REDUCE:
                oss << "  all_reduce_" << i << "(send_buf_" << i
                    << ", recv_buf_" << i << ", rank);\n";
                break;
            case CollectiveOp::ALL_GATHER:
                oss << "  all_gather_" << i << "(send_buf_" << i
                    << ", recv_buf_" << i << ", rank);\n";
                break;
            case CollectiveOp::REDUCE_SCATTER:
                oss << "  reduce_scatter_" << i << "(send_buf_" << i
                    << ", recv_buf_" << i << ", rank);\n";
                break;
            case CollectiveOp::ALL_TO_ALL:
                oss << "  all_to_all_" << i << "(send_buf_" << i
                    << ", recv_buf_" << i << ", rank);\n";
                break;
            case CollectiveOp::BROADCAST:
                oss << "  broadcast_" << i << "(buf_" << i
                    << ", /*root=*/0, rank);\n";
                break;
            case CollectiveOp::SEND_RECV:
                oss << "  send_recv_" << i << "(send_buf_" << i
                    << ", recv_buf_" << i << ", rank);\n";
                break;
        }
    }
    oss << "}\n";

    return oss.str();
}

// ── Latency estimation ─────────────────────────────────────────────────

double NCCLBridge::estimate_communication_latency_ns(
    const CommunicationSchedule& schedule
) const {
    double total_latency_ns = 0.0;

    for (const auto& op : schedule.operations) {
        double latency_ns = 0.0;

        if (op.op == CollectiveOp::SEND_RECV) {
            // Point-to-point: check if same node or cross-node
            double bw_gbps = intra_node_bw_gbps_;
            double base_latency_ns = intra_node_latency_ns_;

            if (!op.src_devices.empty() && !op.dst_devices.empty() &&
                !is_same_node(op.src_devices[0], op.dst_devices[0])) {
                bw_gbps = inter_node_bw_gbps_;
                base_latency_ns = inter_node_latency_ns_;
            }

            // Transfer time = latency + bytes / bandwidth
            double transfer_ns = static_cast<double>(op.bytes) /
                                (bw_gbps * 1e9 / 1e9);  // GB/s to bytes/ns
            // 1 GB/s = 1e9 bytes / 1e9 ns = 1 byte/ns
            transfer_ns = static_cast<double>(op.bytes) / bw_gbps;
            latency_ns = base_latency_ns + transfer_ns;

        } else if (op.op == CollectiveOp::ALL_REDUCE) {
            // Ring all-reduce latency:
            // For ring algorithm with N devices and total data M bytes:
            //   latency = 2 * (N-1) * (alpha + M / (N * BW))
            // where alpha is the per-step latency and BW is the bandwidth.
            int64_t num_dev = static_cast<int64_t>(op.src_devices.size());
            if (num_dev <= 1) continue;

            double bw = effective_bandwidth_gbps(op.src_devices);
            double alpha = intra_node_latency_ns_;

            // Check if any cross-node communication is involved
            bool has_cross_node = false;
            for (size_t i = 0; i < op.src_devices.size() && !has_cross_node; ++i) {
                for (size_t j = i + 1; j < op.src_devices.size(); ++j) {
                    if (!is_same_node(op.src_devices[i], op.src_devices[j])) {
                        has_cross_node = true;
                        break;
                    }
                }
            }
            if (has_cross_node) {
                bw = inter_node_bw_gbps_;
                alpha = inter_node_latency_ns_;
            }

            double data_per_step = static_cast<double>(op.bytes) /
                                  static_cast<double>(num_dev);
            double step_time_ns = alpha + data_per_step / bw;
            latency_ns = 2.0 * (num_dev - 1) * step_time_ns;

        } else if (op.op == CollectiveOp::ALL_GATHER) {
            // Ring all-gather: (N-1) steps
            int64_t num_dev = static_cast<int64_t>(op.src_devices.size());
            if (num_dev <= 1) continue;

            double bw = effective_bandwidth_gbps(op.src_devices);
            double alpha = intra_node_latency_ns_;

            bool has_cross_node = false;
            for (size_t i = 0; i < op.src_devices.size() && !has_cross_node; ++i) {
                for (size_t j = i + 1; j < op.src_devices.size(); ++j) {
                    if (!is_same_node(op.src_devices[i], op.src_devices[j])) {
                        has_cross_node = true;
                        break;
                    }
                }
            }
            if (has_cross_node) {
                bw = inter_node_bw_gbps_;
                alpha = inter_node_latency_ns_;
            }

            double data_per_step = static_cast<double>(op.bytes) /
                                  static_cast<double>(num_dev);
            double step_time_ns = alpha + data_per_step / bw;
            latency_ns = (num_dev - 1) * step_time_ns;

        } else if (op.op == CollectiveOp::REDUCE_SCATTER) {
            // Ring reduce-scatter: (N-1) steps
            int64_t num_dev = static_cast<int64_t>(op.src_devices.size());
            if (num_dev <= 1) continue;

            double bw = effective_bandwidth_gbps(op.src_devices);
            double alpha = intra_node_latency_ns_;

            bool has_cross_node = false;
            for (size_t i = 0; i < op.src_devices.size() && !has_cross_node; ++i) {
                for (size_t j = i + 1; j < op.src_devices.size(); ++j) {
                    if (!is_same_node(op.src_devices[i], op.src_devices[j])) {
                        has_cross_node = true;
                        break;
                    }
                }
            }
            if (has_cross_node) {
                bw = inter_node_bw_gbps_;
                alpha = inter_node_latency_ns_;
            }

            double data_per_step = static_cast<double>(op.bytes) /
                                  static_cast<double>(num_dev);
            double step_time_ns = alpha + data_per_step / bw;
            latency_ns = (num_dev - 1) * step_time_ns;

        } else if (op.op == CollectiveOp::ALL_TO_ALL) {
            // All-to-all: each device sends to all others
            int64_t num_dev = static_cast<int64_t>(op.src_devices.size());
            if (num_dev <= 1) continue;

            double bw = effective_bandwidth_gbps(op.src_devices);
            double alpha = inter_node_latency_ns_;  // Assume cross-node for all-to-all

            double data_per_pair = static_cast<double>(op.bytes) /
                                  static_cast<double>(num_dev);
            // With NCCL's optimized implementation, it's approximately
            // max over all pairs of (alpha + data_per_pair / bw)
            latency_ns = alpha + data_per_pair / bw;

        } else if (op.op == CollectiveOp::BROADCAST) {
            // Broadcast: tree-based, O(log N) steps
            int64_t num_dev = static_cast<int64_t>(op.src_devices.size());
            if (num_dev <= 1) continue;

            double bw = effective_bandwidth_gbps(op.src_devices);
            double alpha = intra_node_latency_ns_;

            int64_t steps = static_cast<int64_t>(std::ceil(std::log2(num_dev)));
            latency_ns = steps * (alpha + static_cast<double>(op.bytes) / bw);
        }

        total_latency_ns += latency_ns;
    }

    return total_latency_ns;
}

// ── NCCL availability check ────────────────────────────────────────────

bool NCCLBridge::is_available() const {
#ifdef SYMPLEX_ENABLE_NCCL
    return true;
#else
    return false;
#endif
}

} // namespace symplex::distributed
