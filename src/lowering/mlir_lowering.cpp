// SympleX – Polyhedral Tensor Superoptimizer
// MLIR Lowering Bridge Implementation
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// Generates valid MLIR assembly text from optimized SympleX IR.
// The output is parseable by mlir-opt and can be lowered to LLVM IR.

#include "symplex/lowering/mlir_lowering.h"

namespace symplex::lowering {

// ─────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────

MLIRLowering::MLIRLowering(MLIRLoweringConfig config)
    : config_(std::move(config)) {}

// ─────────────────────────────────────────────────────────────────────────
// Main Lowering Entry Points
// ─────────────────────────────────────────────────────────────────────────

MLIRLoweringResult MLIRLowering::lower(const ir::SympleXIR& ir_module) {
    MLIRLoweringResult result;

    if (ir_module.num_ops() == 0) {
        result.valid = false;
        result.error_message = "Empty IR module — nothing to lower";
        return result;
    }

    if (!ir_module.validate()) {
        result.valid = false;
        result.error_message = "IR module failed validation";
        return result;
    }

    // Collect input symbols and output
    std::vector<std::string> input_names;
    std::vector<ir::IRShape> input_shapes;
    std::vector<ir::IRDType> input_dtypes;

    for (const auto& [name, id] : ir_module.symbols()) {
        const auto& sym_op = ir_module.op(id);
        input_names.push_back(name);
        input_shapes.push_back(sym_op.shape);
        input_dtypes.push_back(sym_op.dtype);
    }

    int64_t root_id = ir_module.root_id();
    const auto& root_op = ir_module.op(root_id);
    ir::IRShape output_shape = root_op.shape;
    ir::IRDType output_dtype = root_op.dtype;

    result.inputs = input_names;
    result.input_shapes = input_shapes;
    result.output_shape = output_shape;
    result.output_name = ssa_name(root_id);

    // Generate kernel name
    std::string kernel_name = config_.kernel_name_prefix + "kernel";
    if (is_matmul_op(root_op.kind)) {
        kernel_name = config_.kernel_name_prefix + "matmul_kernel";
    } else if (is_norm_op(root_op.kind)) {
        kernel_name = config_.kernel_name_prefix + "norm_kernel";
    }
    result.kernel_name = kernel_name;

    // ── Emit MLIR ─────────────────────────────────────────────────

    std::ostringstream mlir;

    // Module header
    mlir << emit_module_header(kernel_name);

    // Function header
    mlir << emit_function_header(
        kernel_name,
        input_names, input_shapes, input_dtypes,
        output_shape, output_dtype,
        config_.emit_gpu_kernel
    );

    // Build a name map: op_id → SSA name
    // For symbols, use their name directly
    // For other ops, use %op_N
    std::unordered_map<int64_t, std::string> name_map;
    for (const auto& [name, id] : ir_module.symbols()) {
        name_map[id] = "%" + name;
    }

    // Emit operations in topological order
    auto topo_order = ir_module.topological_order();

    // If GPU kernel, wrap body in gpu.launch
    std::ostringstream body;

    for (int64_t op_id : topo_order) {
        const auto& ir_op = ir_module.op(op_id);

        // Skip symbols (already declared as function args)
        if (ir_op.kind == ir::IROp::Kind::SYMBOL) {
            continue;
        }

        // Skip constants (emit inline or as arith constants)
        if (ir_op.kind == ir::IROp::Kind::CONSTANT) {
            std::string const_name = ssa_name(op_id);
            name_map[op_id] = const_name;

            if (ir_op.float_value != 0.0 && ir_op.int_value == 0) {
                body << emit_constant(const_name, ir_op.float_value, ir_op.dtype);
            } else {
                body << emit_constant(const_name, static_cast<double>(ir_op.int_value), ir_op.dtype);
            }
            continue;
        }

        // Resolve operand names
        std::vector<std::string> operand_names;
        operand_names.reserve(ir_op.operands.size());
        for (auto oid : ir_op.operands) {
            auto it = name_map.find(oid);
            if (it != name_map.end()) {
                operand_names.push_back(it->second);
            } else {
                operand_names.push_back(ssa_name(oid));
            }
        }

        std::string result_name = ssa_name(op_id);
        name_map[op_id] = result_name;

        // Emit op-specific MLIR
        if (is_matmul_op(ir_op.kind)) {
            // Matmul and fused matmul variants
            body << emit_linalg_op(ir_op.kind, result_name, operand_names,
                                   ir_op.shape, ir_op.dtype);
            result.num_linalg_ops++;
        } else if (is_elementwise_op(ir_op.kind)) {
            body << emit_elementwise_op(ir_op.kind, result_name, operand_names,
                                        ir_op.shape, ir_op.dtype);
        } else if (is_reduction_op(ir_op.kind)) {
            body << emit_reduction_op(ir_op.kind, result_name, operand_names[0],
                                      ir_module.op(ir_op.operands[0]).shape,
                                      ir_op.shape, ir_op.dtype, ir_op.axis);
        } else if (is_norm_op(ir_op.kind)) {
            // Normalization ops are decomposed
            if (ir_op.kind == ir::IROp::Kind::SOFTMAX) {
                body << "// Softmax decomposition\n";
                // Softmax = exp(x) / reduce_sum(exp(x))
                std::string exp_name = unique_name("exp");
                body << emit_elementwise_op(ir::IROp::Kind::EXP, exp_name,
                                            operand_names, ir_op.shape, ir_op.dtype);
                std::string sum_name = unique_name("sum");
                body << emit_reduction_op(ir::IROp::Kind::REDUCE_SUM, sum_name, exp_name,
                                          ir_op.shape, ir::IRShape(), ir_op.dtype, ir_op.axis);
                std::string bcast_name = unique_name("bcast");
                // Broadcast the reduction result back to the original shape.
                // The reduction removed the axis dimension; we need to re-insert it.
                {
                    int64_t ndim = static_cast<int64_t>(ir_op.shape.ndim());
                    int64_t red_axis = ir_op.axis;
                    // Compute the reduction result shape (input shape minus the axis dim)
                    ir::IRShape red_shape = ir::reduce_shape(ir_op.shape, red_axis);
                    std::ostringstream all_dims, bcast_in_map, bcast_out_map, bcast_iters;
                    for (int64_t i = 0; i < ndim; ++i) {
                        if (i > 0) { all_dims << ", "; bcast_iters << ", "; }
                        all_dims << "d" << i;
                        bcast_iters << "\"parallel\"";
                    }
                    // Input map: skip the reduced axis dimension
                    bcast_in_map << "affine_map<(" << all_dims.str() << ") -> (";
                    bool first = true;
                    for (int64_t i = 0; i < ndim; ++i) {
                        if (i == red_axis) continue;
                        if (!first) bcast_in_map << ", ";
                        bcast_in_map << "d" << i;
                        first = false;
                    }
                    bcast_in_map << ")>";
                    bcast_out_map << "affine_map<(" << all_dims.str() << ") -> (" << all_dims.str() << ")>";

                    auto bcast_init = unique_name("bcast_init");
                    body << indent_str(4) << bcast_init
                         << " = linalg.init_tensor " << ir_op.shape.to_mlir_dims()
                         << " : " << emit_tensor_type(ir_op.shape, ir_op.dtype) << "\n";
                    body << indent_str(4) << bcast_name
                         << " = linalg.generic {\n"
                         << indent_str(6) << "indexing_maps = [" << bcast_in_map.str() << ", "
                         << bcast_out_map.str() << "],\n"
                         << indent_str(6) << "iterator_types = [" << bcast_iters.str() << "]\n"
                         << indent_str(4) << "} ins(" << sum_name << " : "
                         << emit_tensor_type(red_shape, ir_op.dtype) << ")"
                         << " outs(" << bcast_init << " : "
                         << emit_tensor_type(ir_op.shape, ir_op.dtype) << ") {\n"
                         << indent_str(6) << "^bb0(%in: " << irdtype_to_mlir(ir_op.dtype)
                         << ", %out: " << irdtype_to_mlir(ir_op.dtype) << "):\n"
                         << indent_str(8) << "linalg.yield %in : " << irdtype_to_mlir(ir_op.dtype) << "\n"
                         << indent_str(4) << "} -> " << emit_tensor_type(ir_op.shape, ir_op.dtype) << "\n";
                }
                body << emit_elementwise_op(ir::IROp::Kind::DIV, result_name,
                                            {exp_name, bcast_name}, ir_op.shape, ir_op.dtype);
            } else if (ir_op.kind == ir::IROp::Kind::LAYERNORM ||
                       ir_op.kind == ir::IROp::Kind::FUSED_LAYERNORM) {
                body << "// LayerNorm decomposition\n";
                // LN = (x - mean(x)) / sqrt(var(x) + eps)
                std::string mean_name = unique_name("mean");
                body << emit_reduction_op(ir::IROp::Kind::REDUCE_MEAN, mean_name,
                                          operand_names[0], ir_op.shape, ir::IRShape(), ir_op.dtype, -1);
                std::string sub_name = unique_name("sub");
                body << emit_elementwise_op(ir::IROp::Kind::SUB, sub_name,
                                            {operand_names[0], mean_name}, ir_op.shape, ir_op.dtype);
                std::string sq_name = unique_name("sq");
                body << emit_elementwise_op(ir::IROp::Kind::MUL, sq_name,
                                            {sub_name, sub_name}, ir_op.shape, ir_op.dtype);
                std::string var_name = unique_name("var");
                body << emit_reduction_op(ir::IROp::Kind::REDUCE_MEAN, var_name,
                                          sq_name, ir_op.shape, ir::IRShape(), ir_op.dtype, -1);
                std::string eps_name = unique_name("eps");
                body << emit_constant(eps_name, 1e-5, ir_op.dtype);
                std::string var_eps_name = unique_name("var_eps");
                body << emit_elementwise_op(ir::IROp::Kind::ADD, var_eps_name,
                                            {var_name, eps_name}, ir_op.shape, ir_op.dtype);
                std::string sqrt_name = unique_name("sqrt");
                body << emit_elementwise_op(ir::IROp::Kind::SQRT, sqrt_name,
                                            {var_eps_name}, ir_op.shape, ir_op.dtype);
                body << emit_elementwise_op(ir::IROp::Kind::DIV, result_name,
                                            {sub_name, sqrt_name}, ir_op.shape, ir_op.dtype);
            } else if (ir_op.kind == ir::IROp::Kind::RMSNORM ||
                       ir_op.kind == ir::IROp::Kind::FUSED_RMSNORM) {
                body << "// RMSNorm decomposition\n";
                // RMSNorm = x / sqrt(mean(x^2) + eps)
                std::string sq_name = unique_name("sq");
                body << emit_elementwise_op(ir::IROp::Kind::MUL, sq_name,
                                            {operand_names[0], operand_names[0]}, ir_op.shape, ir_op.dtype);
                std::string ms_name = unique_name("ms");
                body << emit_reduction_op(ir::IROp::Kind::REDUCE_MEAN, ms_name,
                                          sq_name, ir_op.shape, ir::IRShape(), ir_op.dtype, -1);
                std::string eps_name = unique_name("eps");
                body << emit_constant(eps_name, 1e-5, ir_op.dtype);
                std::string ms_eps_name = unique_name("ms_eps");
                body << emit_elementwise_op(ir::IROp::Kind::ADD, ms_eps_name,
                                            {ms_name, eps_name}, ir_op.shape, ir_op.dtype);
                std::string rms_name = unique_name("rms");
                body << emit_elementwise_op(ir::IROp::Kind::SQRT, rms_name,
                                            {ms_eps_name}, ir_op.shape, ir_op.dtype);
                body << emit_elementwise_op(ir::IROp::Kind::DIV, result_name,
                                            {operand_names[0], rms_name}, ir_op.shape, ir_op.dtype);
            } else if (ir_op.kind == ir::IROp::Kind::FUSED_ADD_LN) {
                body << "// FusedAddLN decomposition\n";
                // FusedAddLN = LayerNorm(a + b)
                std::string add_name = unique_name("add");
                body << emit_elementwise_op(ir::IROp::Kind::ADD, add_name,
                                            {operand_names[0], operand_names[1]},
                                            ir_op.shape, ir_op.dtype);
                // Now apply LN to add_name
                std::string mean_name = unique_name("mean");
                body << emit_reduction_op(ir::IROp::Kind::REDUCE_MEAN, mean_name,
                                          add_name, ir_op.shape, ir::IRShape(), ir_op.dtype, -1);
                std::string sub_name = unique_name("sub");
                body << emit_elementwise_op(ir::IROp::Kind::SUB, sub_name,
                                            {add_name, mean_name}, ir_op.shape, ir_op.dtype);
                std::string sq_name = unique_name("sq");
                body << emit_elementwise_op(ir::IROp::Kind::MUL, sq_name,
                                            {sub_name, sub_name}, ir_op.shape, ir_op.dtype);
                std::string var_name = unique_name("var");
                body << emit_reduction_op(ir::IROp::Kind::REDUCE_MEAN, var_name,
                                          sq_name, ir_op.shape, ir::IRShape(), ir_op.dtype, -1);
                std::string eps_name = unique_name("eps");
                body << emit_constant(eps_name, 1e-5, ir_op.dtype);
                std::string var_eps_name = unique_name("var_eps");
                body << emit_elementwise_op(ir::IROp::Kind::ADD, var_eps_name,
                                            {var_name, eps_name}, ir_op.shape, ir_op.dtype);
                std::string sqrt_name = unique_name("sqrt");
                body << emit_elementwise_op(ir::IROp::Kind::SQRT, sqrt_name,
                                            {var_eps_name}, ir_op.shape, ir_op.dtype);
                body << emit_elementwise_op(ir::IROp::Kind::DIV, result_name,
                                            {sub_name, sqrt_name}, ir_op.shape, ir_op.dtype);
            }
        } else if (ir_op.kind == ir::IROp::Kind::FUSED_MHA) {
            body << "// Fused Multi-Head Attention decomposition\n";
            // MHA(Q, K, V, bias) = softmax(Q @ K^T / sqrt(d)) @ V
            // Q = operand[0], K = operand[1], V = operand[2]
            std::string kT_name = unique_name("kT");
            {
                auto k_shape = ir_module.op(ir_op.operands[1]).shape;
                // Transposed K has swapped last two dims
                ir::IRShape kT_shape = k_shape;
                if (k_shape.ndim() >= 2) {
                    kT_shape.dims[k_shape.ndim() - 2] = k_shape[k_shape.ndim() - 1];
                    kT_shape.dims[k_shape.ndim() - 1] = k_shape[k_shape.ndim() - 2];
                }
                auto kT_init = unique_name("kT_init");
                body << indent_str(4) << kT_init
                     << " = linalg.init_tensor " << kT_shape.to_mlir_dims()
                     << " : " << emit_tensor_type(kT_shape, ir_op.dtype) << "\n";
                body << indent_str(4) << kT_name
                     << " = linalg.transpose ins(" << operand_names[1]
                     << " : " << emit_tensor_type(k_shape, ir_op.dtype) << ")"
                     << " outs(" << kT_init
                     << " : " << emit_tensor_type(kT_shape, ir_op.dtype) << ")"
                     << " permutation = [1, 0]\n";
            }

            // Q @ K^T
            auto q_shape = ir_module.op(ir_op.operands[0]).shape;
            auto k_shape = ir_module.op(ir_op.operands[1]).shape;
            ir::IRShape qk_shape({q_shape[0], k_shape[0]});
            std::string qk_name = unique_name("qk");
            body << emit_linalg_op(ir::IROp::Kind::MATMUL, qk_name,
                                   {operand_names[0], kT_name}, qk_shape, ir_op.dtype);

            // Scale by 1/sqrt(d)
            std::string scale_name = unique_name("scale");
            double d_val = (q_shape.ndim() >= 1 && q_shape[q_shape.ndim() - 1] > 0)
                           ? static_cast<double>(q_shape[q_shape.ndim() - 1])
                           : 64.0;
            body << emit_constant(scale_name, 1.0 / std::sqrt(d_val), ir_op.dtype);
            std::string scaled_name = unique_name("scaled");
            body << emit_elementwise_op(ir::IROp::Kind::MUL, scaled_name,
                                        {qk_name, scale_name}, qk_shape, ir_op.dtype);

            // Softmax
            std::string attn_name = unique_name("attn");
            body << "// Softmax over attention scores\n";
            std::string exp_name = unique_name("exp");
            body << emit_elementwise_op(ir::IROp::Kind::EXP, exp_name,
                                        {scaled_name}, qk_shape, ir_op.dtype);
            std::string sum_name = unique_name("sum");
            body << emit_reduction_op(ir::IROp::Kind::REDUCE_SUM, sum_name, exp_name,
                                      qk_shape, ir::IRShape(), ir_op.dtype, -1);
            body << emit_elementwise_op(ir::IROp::Kind::DIV, attn_name,
                                        {exp_name, sum_name}, qk_shape, ir_op.dtype);

            // attn @ V
            auto v_shape = ir_module.op(ir_op.operands[2]).shape;
            ir::IRShape out_shape({q_shape[0], v_shape.ndim() >= 2 ? v_shape[v_shape.ndim() - 1] : -1});
            body << emit_linalg_op(ir::IROp::Kind::MATMUL, result_name,
                                   {attn_name, operand_names[2]}, out_shape, ir_op.dtype);
            result.num_linalg_ops++;
        } else if (ir_op.kind == ir::IROp::Kind::TRANSPOSE) {
            // linalg.transpose requires ins/outs and permutation syntax:
            //   %r = linalg.transpose ins(%in : tensor<MxNxf32>)
            //                          outs(%init : tensor<NxMxf32>)
            //                          permutation = [1, 0]
            {
                auto in_shape = ir_module.op(ir_op.operands[0]).shape;
                auto out_shape = ir_op.shape;
                auto trans_init = unique_name("trans_init");
                body << indent_str(4) << trans_init
                     << " = linalg.init_tensor " << out_shape.to_mlir_dims()
                     << " : " << emit_tensor_type(out_shape, ir_op.dtype) << "\n";
                body << indent_str(4) << result_name
                     << " = linalg.transpose ins(" << operand_names[0]
                     << " : " << emit_tensor_type(in_shape, ir_op.dtype) << ")"
                     << " outs(" << trans_init
                     << " : " << emit_tensor_type(out_shape, ir_op.dtype) << ")"
                     << " permutation = [1, 0]\n";
            }
        } else if (ir_op.kind == ir::IROp::Kind::RESHAPE) {
            // tensor.reshape with a `to` clause does not exist in standard MLIR.
            // Use tensor.expand_shape (rank increases) or tensor.collapse_shape
            // (rank decreases) with reassociation indices.
            {
                auto in_shape = ir_module.op(ir_op.operands[0]).shape;
                auto out_shape = ir_op.shape;
                int64_t in_rank = static_cast<int64_t>(in_shape.ndim());
                int64_t out_rank = static_cast<int64_t>(out_shape.ndim());

                if (out_rank > in_rank) {
                    // tensor.expand_shape: each output dim group maps to one input dim
                    std::ostringstream reassoc;
                    reassoc << "[";
                    // Simple strategy: expand the last input dimension into
                    // (out_rank - in_rank + 1) output dimensions; all preceding
                    // input dims map 1:1.
                    int64_t one_to_one = in_rank - 1;
                    for (int64_t i = 0; i < one_to_one; ++i) {
                        if (i > 0) reassoc << ", ";
                        reassoc << "[" << i << "]";
                    }
                    if (one_to_one > 0) reassoc << ", ";
                    reassoc << "[";
                    for (int64_t i = one_to_one; i < out_rank; ++i) {
                        if (i > one_to_one) reassoc << ", ";
                        reassoc << i;
                    }
                    reassoc << "]";
                    reassoc << "]";

                    body << indent_str(4) << result_name
                         << " = tensor.expand_shape " << operand_names[0]
                         << " " << reassoc.str()
                         << " : " << emit_tensor_type(in_shape, ir_op.dtype)
                         << " into " << emit_tensor_type(out_shape, ir_op.dtype) << "\n";
                } else {
                    // tensor.collapse_shape: each input dim group maps to one output dim
                    std::ostringstream reassoc;
                    reassoc << "[";
                    // Simple strategy: collapse the trailing input dimensions into
                    // one output dimension; all preceding dims map 1:1.
                    int64_t one_to_one = out_rank - 1;
                    for (int64_t i = 0; i < one_to_one; ++i) {
                        if (i > 0) reassoc << ", ";
                        reassoc << "[" << i << "]";
                    }
                    if (one_to_one > 0) reassoc << ", ";
                    reassoc << "[";
                    for (int64_t i = one_to_one; i < in_rank; ++i) {
                        if (i > one_to_one) reassoc << ", ";
                        reassoc << i;
                    }
                    reassoc << "]";
                    reassoc << "]";

                    body << indent_str(4) << result_name
                         << " = tensor.collapse_shape " << operand_names[0]
                         << " " << reassoc.str()
                         << " : " << emit_tensor_type(in_shape, ir_op.dtype)
                         << " into " << emit_tensor_type(out_shape, ir_op.dtype) << "\n";
                }
            }
        } else if (ir_op.kind == ir::IROp::Kind::BROADCAST) {
            body << indent_str(4) << result_name
                 << " = linalg.generic {\n"
                 << indent_str(6) << "indexing_maps = [affine_map<() -> ()>],\n"
                 << indent_str(6) << "iterator_types = [\"parallel\"]\n"
                 << indent_str(4) << "} ins(" << operand_names[0]
                 << " : " << emit_tensor_type(ir_module.op(ir_op.operands[0]).shape, ir_op.dtype)
                 << ") outs(" << result_name
                 << " : " << emit_tensor_type(ir_op.shape, ir_op.dtype) << ") {\n"
                 << indent_str(6) << "^bb0(%in: " << irdtype_to_mlir(ir_op.dtype)
                 << ", %out: " << irdtype_to_mlir(ir_op.dtype) << "):\n"
                 << indent_str(8) << "linalg.yield %in : " << irdtype_to_mlir(ir_op.dtype) << "\n"
                 << indent_str(4) << "} -> " << emit_tensor_type(ir_op.shape, ir_op.dtype) << "\n";
        } else if (ir_op.kind == ir::IROp::Kind::SELECT) {
            body << indent_str(4) << result_name
                 << " = arith.select " << operand_names[0] << ", "
                 << operand_names[1] << ", " << operand_names[2]
                 << " : " << emit_tensor_type(ir_op.shape, ir_op.dtype) << "\n";
        } else if (ir_op.kind == ir::IROp::Kind::IDENTITY) {
            name_map[op_id] = operand_names[0];  // Just alias
        } else if (ir_op.kind == ir::IROp::Kind::TILE ||
                   ir_op.kind == ir::IROp::Kind::UNTILE) {
            name_map[op_id] = operand_names[0];  // Alias for now
        } else {
            // Generic fallback
            body << indent_str(4) << "// Unknown op: "
                 << ir::IROp::kind_to_string(ir_op.kind) << "\n";
        }

        // Handle polyhedral annotation: emit tiled loop nest
        if (ir_op.poly_annotation.has_value() && is_matmul_op(ir_op.kind)) {
            const auto& pa = *ir_op.poly_annotation;
            if (pa.schedule_matrix.has_value()) {
                body << "// Schedule applied: " << pa.to_string() << "\n";
            }
        }
    }

    // Return statement
    auto root_it = name_map.find(root_id);
    std::string root_name = (root_it != name_map.end()) ? root_it->second : ssa_name(root_id);
    body << emit_return(root_name);

    // Assemble the final MLIR
    if (config_.emit_gpu_kernel && is_matmul_op(root_op.kind)) {
        // Wrap in gpu.launch
        std::vector<int64_t> grid_dims = {1, 1, 1};
        std::vector<int64_t> block_dims = {256, 1, 1};

        // Compute grid/block from polyhedral annotation
        if (root_op.poly_annotation.has_value()) {
            const auto& pa = *root_op.poly_annotation;
            for (size_t i = 0; i < pa.num_dims() && i < 3; ++i) {
                int64_t extent = pa.loop_extent(i);
                if (extent > 0) {
                    grid_dims[i] = (extent + 15) / 16;
                    block_dims[i] = 16;
                }
            }
        }

        mlir << emit_gpu_launch(body.str(), grid_dims, block_dims);
    } else {
        mlir << body.str();
    }

    mlir << "  }\n";
    mlir << "}\n";

    result.mlir_text = mlir.str();
    result.valid = true;
    return result;
}

MLIRLoweringResult MLIRLowering::lower_with_schedule(
    const ir::SympleXIR& ir_module,
    const std::vector<polyhedral::AffineMap>& schedules,
    const std::vector<int64_t>& tile_sizes
) {
    // Apply schedules as polyhedral annotations to the IR ops,
    // then lower with the standard pipeline.
    // For now, we delegate to the standard lower() and attach
    // tiling information to the output.

    MLIRLoweringResult result = lower(ir_module);

    if (!result.valid) return result;

    // If we have tile sizes, re-emit with tiled loops
    if (!tile_sizes.empty() && config_.emit_affine_loops) {
        // Append tiled loop annotations to the MLIR
        std::ostringstream tiled_mlir;
        tiled_mlir << "// Tiled with sizes: [";
        for (size_t i = 0; i < tile_sizes.size(); ++i) {
            if (i > 0) tiled_mlir << ", ";
            tiled_mlir << tile_sizes[i];
        }
        tiled_mlir << "]\n";

        // Add schedule map annotations
        for (size_t i = 0; i < schedules.size(); ++i) {
            tiled_mlir << "// Schedule " << i << ": "
                       << schedules[i].to_string() << "\n";
        }

        tiled_mlir << result.mlir_text;
        result.mlir_text = tiled_mlir.str();
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────
// Emit Helpers
// ─────────────────────────────────────────────────────────────────────────

std::string MLIRLowering::emit_module_header(const std::string& /*name*/) {
    std::ostringstream oss;
    // Emit dialect availability as comments (standard MLIR text doesn't require
    // explicit imports, but documenting used dialects aids readability and
    // ensures downstream tools know which dialects to register).
    oss << "// Dialects used: arith, func, linalg, math, tensor, affine, gpu, nvgpu, scf\n";
    oss << "module {\n";
    return oss.str();
}

std::string MLIRLowering::emit_tensor_type(const ir::IRShape& shape,
                                             ir::IRDType dtype) {
    if (shape.is_unknown()) {
        return "tensor<?x" + std::string(irdtype_to_mlir(dtype)) + ">";
    }
    std::ostringstream oss;
    oss << "tensor<";
    oss << shape.to_mlir_dims();
    oss << "x" << irdtype_to_mlir(dtype);
    oss << ">";
    return oss.str();
}

std::string MLIRLowering::emit_function_header(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<ir::IRShape>& input_shapes,
    const std::vector<ir::IRDType>& input_dtypes,
    const ir::IRShape& output_shape,
    ir::IRDType output_dtype,
    bool is_gpu_kernel
) {
    std::ostringstream oss;
    oss << indent_str(2) << "func.func @" << name << "(";

    for (size_t i = 0; i < input_names.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "%" << input_names[i] << ": "
            << emit_tensor_type(input_shapes[i], input_dtypes[i]);
    }

    oss << ") -> " << emit_tensor_type(output_shape, output_dtype);

    if (is_gpu_kernel) {
        oss << " attributes {gpu.kernel}";
    }

    oss << " {\n";
    return oss.str();
}

std::string MLIRLowering::emit_linalg_op(
    ir::IROp::Kind kind,
    const std::string& result_name,
    const std::vector<std::string>& operand_names,
    const ir::IRShape& result_shape,
    ir::IRDType dtype
) {
    std::ostringstream oss;
    std::string mlir_dtype = irdtype_to_mlir(dtype);

    if (kind == ir::IROp::Kind::MATMUL) {
        // Standard linalg.matmul
        auto init_name = unique_name("init");
        oss << indent_str(4) << init_name
            << " = linalg.init_tensor " << result_shape.to_mlir_dims()
            << " : " << emit_tensor_type(result_shape, dtype) << "\n";

        oss << indent_str(4) << result_name
            << " = linalg.matmul"
            << " ins(" << operand_names[0] << ", " << operand_names[1]
            << " : " << emit_tensor_type(ir::IRShape(), dtype)  // Will be inferred
            << ", " << emit_tensor_type(ir::IRShape(), dtype)
            << ")"
            << " outs(" << init_name
            << " : " << emit_tensor_type(result_shape, dtype) << ")"
            << " -> " << emit_tensor_type(result_shape, dtype) << "\n";
    } else if (kind == ir::IROp::Kind::FUSED_MATMUL_RELU) {
        // Fused MatMul + ReLU: matmul then elementwise max(0, x)
        auto init_name = unique_name("init");
        oss << indent_str(4) << init_name
            << " = linalg.init_tensor " << result_shape.to_mlir_dims()
            << " : " << emit_tensor_type(result_shape, dtype) << "\n";

        auto mm_name = unique_name("mm");
        oss << indent_str(4) << mm_name
            << " = linalg.matmul"
            << " ins(" << operand_names[0] << ", " << operand_names[1]
            << " : " << emit_tensor_type(ir::IRShape(), dtype)
            << ", " << emit_tensor_type(ir::IRShape(), dtype)
            << ")"
            << " outs(" << init_name
            << " : " << emit_tensor_type(result_shape, dtype) << ")"
            << " -> " << emit_tensor_type(result_shape, dtype) << "\n";

        // ReLU as linalg.generic
        auto zero_name = unique_name("zero");
        oss << emit_constant(zero_name, 0.0, dtype);

        oss << indent_str(4) << result_name
            << " = linalg.generic {\n"
            << indent_str(6) << "indexing_maps = ["
            << "affine_map<(d0, d1) -> (d0, d1)>, "
            << "affine_map<(d0, d1) -> ()>],\n"
            << indent_str(6) << "iterator_types = [\"parallel\", \"parallel\"]\n"
            << indent_str(4) << "} ins(" << mm_name << ", " << zero_name
            << " : " << emit_tensor_type(result_shape, dtype)
            << ", " << mlir_dtype << ")"
            << " outs(" << result_name
            << " : " << emit_tensor_type(result_shape, dtype) << ") {\n"
            << indent_str(6) << "^bb0(%in: " << mlir_dtype
            << ", %zero: " << mlir_dtype
            << ", %out: " << mlir_dtype << "):\n"
            << indent_str(8) << "%max = arith.maxf %in, %zero : " << mlir_dtype << "\n"
            << indent_str(8) << "linalg.yield %max : " << mlir_dtype << "\n"
            << indent_str(4) << "} -> " << emit_tensor_type(result_shape, dtype) << "\n";

    } else if (kind == ir::IROp::Kind::FUSED_MATMUL_ADD) {
        // Fused MatMul + Add: matmul then add bias
        auto init_name = unique_name("init");
        oss << indent_str(4) << init_name
            << " = linalg.init_tensor " << result_shape.to_mlir_dims()
            << " : " << emit_tensor_type(result_shape, dtype) << "\n";

        auto mm_name = unique_name("mm");
        oss << indent_str(4) << mm_name
            << " = linalg.matmul"
            << " ins(" << operand_names[0] << ", " << operand_names[1]
            << " : " << emit_tensor_type(ir::IRShape(), dtype)
            << ", " << emit_tensor_type(ir::IRShape(), dtype)
            << ")"
            << " outs(" << init_name
            << " : " << emit_tensor_type(result_shape, dtype) << ")"
            << " -> " << emit_tensor_type(result_shape, dtype) << "\n";

        // Add bias
        oss << emit_elementwise_op(ir::IROp::Kind::ADD, result_name,
                                    {mm_name, operand_names[2]},
                                    result_shape, dtype);

    } else if (kind == ir::IROp::Kind::FUSED_MATMUL_ADD_RELU ||
               kind == ir::IROp::Kind::FUSED_GEMM) {
        // Fused GEMM: alpha*A*B + beta*C, then optionally ReLU
        auto init_name = unique_name("init");
        oss << indent_str(4) << init_name
            << " = linalg.init_tensor " << result_shape.to_mlir_dims()
            << " : " << emit_tensor_type(result_shape, dtype) << "\n";

        // Start with C (beta*C part)
        auto mm_name = unique_name("mm");
        oss << indent_str(4) << mm_name
            << " = linalg.matmul"
            << " ins(" << operand_names[0] << ", " << operand_names[1]
            << " : " << emit_tensor_type(ir::IRShape(), dtype)
            << ", " << emit_tensor_type(ir::IRShape(), dtype)
            << ")"
            << " outs(" << init_name
            << " : " << emit_tensor_type(result_shape, dtype) << ")"
            << " -> " << emit_tensor_type(result_shape, dtype) << "\n";

        // Add C
        auto add_name = unique_name("add");
        oss << emit_elementwise_op(ir::IROp::Kind::ADD, add_name,
                                    {mm_name, operand_names[2]},
                                    result_shape, dtype);

        if (kind == ir::IROp::Kind::FUSED_MATMUL_ADD_RELU) {
            // Apply ReLU
            auto zero_name = unique_name("zero");
            oss << emit_constant(zero_name, 0.0, dtype);
            oss << indent_str(4) << result_name
                << " = linalg.generic {\n"
                << indent_str(6) << "indexing_maps = ["
                << "affine_map<(d0, d1) -> (d0, d1)>, "
                << "affine_map<(d0, d1) -> ()>],\n"
                << indent_str(6) << "iterator_types = [\"parallel\", \"parallel\"]\n"
                << indent_str(4) << "} ins(" << add_name << ", " << zero_name
                << " : " << emit_tensor_type(result_shape, dtype)
                << ", " << mlir_dtype << ")"
                << " outs(" << result_name
                << " : " << emit_tensor_type(result_shape, dtype) << ") {\n"
                << indent_str(6) << "^bb0(%in: " << mlir_dtype
                << ", %zero: " << mlir_dtype
                << ", %out: " << mlir_dtype << "):\n"
                << indent_str(8) << "%max = arith.maxf %in, %zero : " << mlir_dtype << "\n"
                << indent_str(8) << "linalg.yield %max : " << mlir_dtype << "\n"
                << indent_str(4) << "} -> " << emit_tensor_type(result_shape, dtype) << "\n";
        } else {
            // GEMM without ReLU: result is just the add.
            // Bare SSA assignment like `%0 = %1` is invalid in MLIR, so emit a
            // linalg.generic identity/copy operation instead.
            {
                int64_t ndim = static_cast<int64_t>(result_shape.ndim());
                if (ndim == 0) ndim = 1;
                std::ostringstream dims_str, maps;
                for (int64_t i = 0; i < ndim; ++i) {
                    if (i > 0) dims_str << ", ";
                    dims_str << "d" << i;
                }
                std::string iters;
                {
                    std::ostringstream it;
                    for (int64_t i = 0; i < ndim; ++i) {
                        if (i > 0) it << ", ";
                        it << "\"parallel\"";
                    }
                    iters = it.str();
                }
                maps << "affine_map<(" << dims_str.str() << ") -> (" << dims_str.str() << ")>, "
                     << "affine_map<(" << dims_str.str() << ") -> (" << dims_str.str() << ")>";
                auto copy_init = unique_name("copy_init");
                oss << indent_str(4) << copy_init
                    << " = linalg.init_tensor " << result_shape.to_mlir_dims()
                    << " : " << emit_tensor_type(result_shape, dtype) << "\n";
                oss << indent_str(4) << result_name
                    << " = linalg.generic {\n"
                    << indent_str(6) << "indexing_maps = [" << maps.str() << "],\n"
                    << indent_str(6) << "iterator_types = [" << iters << "]\n"
                    << indent_str(4) << "} ins(" << add_name << " : " << emit_tensor_type(result_shape, dtype) << ")"
                    << " outs(" << copy_init << " : " << emit_tensor_type(result_shape, dtype) << ") {\n"
                    << indent_str(6) << "^bb0(%in: " << mlir_dtype << ", %out: " << mlir_dtype << "):\n"
                    << indent_str(8) << "linalg.yield %in : " << mlir_dtype << "\n"
                    << indent_str(4) << "} -> " << emit_tensor_type(result_shape, dtype) << "\n";
            }
        }
    } else {
        // Generic linalg op fallback
        oss << indent_str(4) << result_name
            << " = linalg.generic {\n"
            << indent_str(6) << "// Generic op for " << ir::IROp::kind_to_string(kind) << "\n"
            << indent_str(4) << "} -> " << emit_tensor_type(result_shape, dtype) << "\n";
    }

    return oss.str();
}

std::string MLIRLowering::emit_elementwise_op(
    ir::IROp::Kind kind,
    const std::string& result_name,
    const std::vector<std::string>& operand_names,
    const ir::IRShape& result_shape,
    ir::IRDType dtype
) {
    std::ostringstream oss;
    std::string mlir_dtype = irdtype_to_mlir(dtype);
    std::string tensor_type = emit_tensor_type(result_shape, dtype);

    if (result_shape.is_unknown()) {
        // Fallback for unknown shapes
        oss << indent_str(4) << result_name
            << " = linalg.generic // elementwise "
            << ir::IROp::kind_to_string(kind) << "\n";
        return oss.str();
    }

    // Determine the number of parallel dimensions
    int64_t ndim = static_cast<int64_t>(result_shape.ndim());
    if (ndim == 0) ndim = 1;  // Scalar

    // Build indexing maps string
    std::ostringstream maps;
    std::string dims_str;
    {
        std::ostringstream dim_oss;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i > 0) dim_oss << ", ";
            dim_oss << "d" << i;
        }
        dims_str = dim_oss.str();
    }

    // Build affine maps for each operand
    for (size_t i = 0; i < operand_names.size(); ++i) {
        if (i > 0) maps << ", ";
        maps << "affine_map<(" << dims_str << ") -> (" << dims_str << ")>";
    }
    // Output map
    maps << ", affine_map<(" << dims_str << ") -> (" << dims_str << ")>";

    // Build iterator types
    std::string iter_types;
    {
        std::ostringstream iter_oss;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i > 0) iter_oss << ", ";
            iter_oss << "\"parallel\"";
        }
        iter_types = iter_oss.str();
    }

    // Build the body
    std::string body_op;
    if (kind == ir::IROp::Kind::ADD) {
        body_op = "arith.addf";
    } else if (kind == ir::IROp::Kind::MUL) {
        body_op = "arith.mulf";
    } else if (kind == ir::IROp::Kind::SUB) {
        body_op = "arith.subf";
    } else if (kind == ir::IROp::Kind::DIV) {
        body_op = "arith.divf";
    } else if (kind == ir::IROp::Kind::NEG) {
        body_op = "arith.negf";
    } else if (kind == ir::IROp::Kind::RELU) {
        // ReLU = max(x, 0)
        body_op = "arith.maxf";  // Special case handled below
    } else if (kind == ir::IROp::Kind::EXP) {
        body_op = "math.exp";
    } else if (kind == ir::IROp::Kind::LOG) {
        body_op = "math.log";
    } else if (kind == ir::IROp::Kind::SQRT) {
        body_op = "math.sqrt";
    } else if (kind == ir::IROp::Kind::GELU) {
        body_op = "";  // Handled specially below via decomposition
    } else if (kind == ir::IROp::Kind::SIGMOID) {
        body_op = "";  // Handled specially below via decomposition
    } else if (kind == ir::IROp::Kind::RECIPROCAL) {
        // 1/x
        body_op = "arith.divf";  // Special case: 1/x
    } else {
        body_op = "// unknown_elementwise";
    }

    // Input types
    std::ostringstream ins_str;
    for (size_t i = 0; i < operand_names.size(); ++i) {
        if (i > 0) ins_str << ", ";
        ins_str << operand_names[i] << " : " << tensor_type;
    }

    // BB args
    std::ostringstream bb_args;
    for (size_t i = 0; i < operand_names.size(); ++i) {
        if (i > 0) bb_args << ", ";
        bb_args << "%in" << i << ": " << mlir_dtype;
    }
    bb_args << ", %out: " << mlir_dtype;

    oss << indent_str(4) << result_name
        << " = linalg.generic {\n"
        << indent_str(6) << "indexing_maps = [" << maps.str() << "],\n"
        << indent_str(6) << "iterator_types = [" << iter_types << "]\n"
        << indent_str(4) << "} ins(" << ins_str.str() << ")"
        << " outs(" << result_name << " : " << tensor_type << ") {\n"
        << indent_str(6) << "^bb0(" << bb_args.str() << "):\n";

    // Emit the body operation
    if (kind == ir::IROp::Kind::RELU) {
        // ReLU: max(x, 0)
        auto zero_name = unique_name("zero");
        oss << indent_str(8) << zero_name << " = arith.constant 0.0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "%result = arith.maxf %in0, " << zero_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else if (kind == ir::IROp::Kind::GELU) {
        // GELU decomposition: 0.5 * x * (1.0 + math.erf(x / sqrt(2.0)))
        auto half_name = unique_name("half");
        auto sqrt2_name = unique_name("sqrt2");
        auto one_name = unique_name("gelu_one");
        auto div_name = unique_name("div");
        auto erf_name = unique_name("erf");
        auto add_name = unique_name("add");
        auto mul1_name = unique_name("mul1");
        oss << indent_str(8) << half_name << " = arith.constant 0.5 : " << mlir_dtype << "\n";
        oss << indent_str(8) << sqrt2_name << " = arith.constant 1.4142135623730951 : " << mlir_dtype << "\n";
        oss << indent_str(8) << one_name << " = arith.constant 1.0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << div_name << " = arith.divf %in0, " << sqrt2_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << erf_name << " = math.erf " << div_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << add_name << " = arith.addf " << one_name << ", " << erf_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << mul1_name << " = arith.mulf " << half_name << ", %in0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "%result = arith.mulf " << mul1_name << ", " << add_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else if (kind == ir::IROp::Kind::SIGMOID) {
        // Sigmoid decomposition: 1.0 / (1.0 + math.exp(-x))
        auto one_name = unique_name("sig_one");
        auto neg_name = unique_name("neg");
        auto exp_name = unique_name("exp");
        auto add_name = unique_name("add");
        oss << indent_str(8) << one_name << " = arith.constant 1.0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << neg_name << " = arith.negf %in0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << exp_name << " = math.exp " << neg_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << add_name << " = arith.addf " << one_name << ", " << exp_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << "%result = arith.divf " << one_name << ", " << add_name << " : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else if (kind == ir::IROp::Kind::RECIPROCAL) {
        // 1/x
        auto one_name = unique_name("one");
        oss << indent_str(8) << one_name << " = arith.constant 1.0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "%result = arith.divf " << one_name << ", %in0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else if (kind == ir::IROp::Kind::NEG) {
        oss << indent_str(8) << "%result = arith.negf %in0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else if (operand_names.size() == 2) {
        // Binary elementwise
        oss << indent_str(8) << "%result = " << body_op << " %in0, %in1 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else if (operand_names.size() == 1) {
        // Unary elementwise
        oss << indent_str(8) << "%result = " << body_op << " %in0 : " << mlir_dtype << "\n";
        oss << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n";
    } else {
        oss << indent_str(8) << "linalg.yield %in0 : " << mlir_dtype << "\n";
    }

    oss << indent_str(4) << "} -> " << tensor_type << "\n";

    return oss.str();
}

std::string MLIRLowering::emit_reduction_op(
    ir::IROp::Kind kind,
    const std::string& result_name,
    const std::string& operand_name,
    const ir::IRShape& input_shape,
    const ir::IRShape& result_shape,
    ir::IRDType dtype,
    int64_t axis
) {
    std::ostringstream oss;
    std::string mlir_dtype = irdtype_to_mlir(dtype);
    std::string input_type = emit_tensor_type(input_shape, dtype);

    // For REDUCE_MEAN, we compute the sum first into a temporary, then
    // divide by the product of reduced dimensions.
    std::string sum_result_name = result_name;
    if (kind == ir::IROp::Kind::REDUCE_MEAN) {
        sum_result_name = unique_name("sum");
    }

    std::string reduce_op;
    if (kind == ir::IROp::Kind::REDUCE_SUM || kind == ir::IROp::Kind::REDUCE_MEAN) {
        reduce_op = "arith.addf";
    } else if (kind == ir::IROp::Kind::REDUCE_MAX) {
        reduce_op = "arith.maxf";
    } else {
        reduce_op = "arith.addf";
    }

    // Determine output shape and indexing maps
    if (result_shape.is_unknown() || result_shape.ndim() == 0) {
        // Scalar reduction result — reduce all dims of a 1-D (or flattened) input
        oss << indent_str(4) << sum_result_name
            << " = linalg.generic {\n"
            << indent_str(6) << "indexing_maps = ["
            << "affine_map<(d0) -> (d0)>, "
            << "affine_map<(d0) -> ()>"
            << "],\n"
            << indent_str(6) << "iterator_types = [\"reduction\"]\n"
            << indent_str(4) << "} ins(" << operand_name << " : "
            << input_type << ")"
            << " outs(" << sum_result_name << " : " << mlir_dtype << ") {\n"
            << indent_str(6) << "^bb0(%in: " << mlir_dtype
            << ", %out: " << mlir_dtype << "):\n"
            << indent_str(8) << "%result = " << reduce_op
            << " %in, %out : " << mlir_dtype << "\n"
            << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n"
            << indent_str(4) << "} -> " << mlir_dtype << "\n";
    } else {
        // Reduction with axis specified
        int64_t in_ndim = static_cast<int64_t>(input_shape.ndim());
        std::ostringstream in_dims, out_dims, in_map, out_map, iters;

        for (int64_t i = 0; i < in_ndim; ++i) {
            if (i > 0) { in_dims << ", "; iters << ", "; }
            in_dims << "d" << i;
            int64_t eff_axis = (axis < 0) ? (in_ndim - 1) : axis;
            if (i == eff_axis) {
                iters << "\"reduction\"";
            } else {
                iters << "\"parallel\"";
            }
        }

        for (int64_t i = 0, j = 0; i < in_ndim; ++i) {
            int64_t eff_axis = (axis < 0) ? (in_ndim - 1) : axis;
            if (i == eff_axis) continue;
            if (j > 0) out_dims << ", ";
            out_dims << "d" << i;
            j++;
        }

        in_map << "affine_map<(" << in_dims.str() << ") -> ("
               << in_dims.str() << ")>";
        out_map << "affine_map<(" << in_dims.str() << ") -> ("
                << out_dims.str() << ")>";

        std::string output_type = emit_tensor_type(result_shape, dtype);

        oss << indent_str(4) << sum_result_name
            << " = linalg.generic {\n"
            << indent_str(6) << "indexing_maps = [" << in_map.str() << ", "
            << out_map.str() << "],\n"
            << indent_str(6) << "iterator_types = [" << iters.str() << "]\n"
            << indent_str(4) << "} ins(" << operand_name << " : "
            << input_type << ")"
            << " outs(" << sum_result_name << " : " << output_type << ") {\n"
            << indent_str(6) << "^bb0(%in: " << mlir_dtype
            << ", %out: " << mlir_dtype << "):\n"
            << indent_str(8) << "%result = " << reduce_op
            << " %in, %out : " << mlir_dtype << "\n"
            << indent_str(8) << "linalg.yield %result : " << mlir_dtype << "\n"
            << indent_str(4) << "} -> " << output_type << "\n";
    }

    // For REDUCE_MEAN, divide by the product of the reduced dimensions
    if (kind == ir::IROp::Kind::REDUCE_MEAN) {
        // Compute the count (product of reduced dimensions)
        double count = 1.0;
        if (axis < 0) {
            // Reduce all dims — product of all input dimensions
            for (size_t i = 0; i < input_shape.ndim(); ++i) {
                int64_t dim = input_shape[i];
                count *= (dim > 0) ? static_cast<double>(dim) : 1.0;
            }
        } else {
            int64_t dim = input_shape[static_cast<size_t>(axis)];
            count = (dim > 0) ? static_cast<double>(dim) : 1.0;
        }

        auto count_name = unique_name("count");
        oss << emit_constant(count_name, count, dtype);

        if (result_shape.is_unknown() || result_shape.ndim() == 0) {
            // Scalar result: direct arith.divf
            oss << indent_str(4) << result_name << " = arith.divf "
                << sum_result_name << ", " << count_name << " : " << mlir_dtype << "\n";
        } else {
            // Tensor result: elementwise division by scalar count
            // Emit a linalg.generic that divides each element by the count
            int64_t out_ndim = static_cast<int64_t>(result_shape.ndim());
            std::ostringstream out_dims, data_map, scalar_map, iters;
            for (int64_t i = 0; i < out_ndim; ++i) {
                if (i > 0) { out_dims << ", "; iters << ", "; }
                out_dims << "d" << i;
                iters << "\"parallel\"";
            }
            data_map << "affine_map<(" << out_dims.str() << ") -> (" << out_dims.str() << ")>";
            scalar_map << "affine_map<(" << out_dims.str() << ") -> ()>";

            std::string output_type = emit_tensor_type(result_shape, dtype);
            auto mean_init = unique_name("mean_init");
            oss << indent_str(4) << mean_init
                << " = linalg.init_tensor " << result_shape.to_mlir_dims()
                << " : " << output_type << "\n";
            oss << indent_str(4) << result_name
                << " = linalg.generic {\n"
                << indent_str(6) << "indexing_maps = [" << data_map.str() << ", "
                << scalar_map.str() << ", " << data_map.str() << "],\n"
                << indent_str(6) << "iterator_types = [" << iters.str() << "]\n"
                << indent_str(4) << "} ins(" << sum_result_name << " : " << output_type
                << ", " << count_name << " : " << mlir_dtype << ")"
                << " outs(" << mean_init << " : " << output_type << ") {\n"
                << indent_str(6) << "^bb0(%in: " << mlir_dtype
                << ", %cnt: " << mlir_dtype
                << ", %out: " << mlir_dtype << "):\n"
                << indent_str(8) << "%div = arith.divf %in, %cnt : " << mlir_dtype << "\n"
                << indent_str(8) << "linalg.yield %div : " << mlir_dtype << "\n"
                << indent_str(4) << "} -> " << output_type << "\n";
        }
    }

    return oss.str();
}

std::string MLIRLowering::emit_affine_loop(
    const std::string& var, int64_t lo, int64_t hi,
    const std::string& body, int indent
) {
    std::ostringstream oss;
    oss << indent_str(indent) << "affine.for %" << var
        << " = " << lo << " to " << hi << " {\n"
        << body
        << indent_str(indent) << "}\n";
    return oss.str();
}

std::string MLIRLowering::emit_tiled_loop(
    const std::vector<int64_t>& tile_sizes,
    const std::vector<std::pair<int64_t, int64_t>>& bounds,
    const std::string& body,
    int indent
) {
    std::ostringstream oss;

    // Emit outer tiled loops
    for (size_t i = 0; i < tile_sizes.size() && i < bounds.size(); ++i) {
        int64_t lo = bounds[i].first;
        int64_t hi = bounds[i].second;
        int64_t tile = tile_sizes[i];
        int64_t outer_hi = (hi - lo + tile - 1) / tile;

        std::string outer_var = "tile_" + std::to_string(i);
        std::string inner_var = "local_" + std::to_string(i);

        oss << indent_str(indent) << "affine.for %" << outer_var
            << " = 0 to " << outer_hi << " {\n";
        oss << indent_str(indent + 2) << "affine.for %" << inner_var
            << " = 0 to " << tile << " {\n";
    }

    oss << body;

    // Close loops
    for (size_t i = tile_sizes.size(); i > 0; --i) {
        oss << indent_str(indent + 2 * static_cast<int>(i) - 2) << "}\n";
        oss << indent_str(indent + 2 * static_cast<int>(i) - 4) << "}\n";
    }

    return oss.str();
}

std::string MLIRLowering::emit_gpu_launch(
    const std::string& kernel_body,
    const std::vector<int64_t>& grid_dims,
    const std::vector<int64_t>& block_dims,
    int indent
) {
    std::ostringstream oss;

    // Ensure 3 dimensions
    int64_t gx = (grid_dims.size() > 0) ? grid_dims[0] : 1;
    int64_t gy = (grid_dims.size() > 1) ? grid_dims[1] : 1;
    int64_t gz = (grid_dims.size() > 2) ? grid_dims[2] : 1;
    int64_t bx = (block_dims.size() > 0) ? block_dims[0] : 1;
    int64_t by = (block_dims.size() > 1) ? block_dims[1] : 1;
    int64_t bz = (block_dims.size() > 2) ? block_dims[2] : 1;

    // Use unique names for grid/block dimension constants to avoid
    // SSA name collisions when multiple dimensions share the same value.
    std::string grid_x = unique_name("grid_x");
    std::string grid_y = unique_name("grid_y");
    std::string grid_z = unique_name("grid_z");
    std::string blk_x = unique_name("blk_x");
    std::string blk_y = unique_name("blk_y");
    std::string blk_z = unique_name("blk_z");

    oss << indent_str(indent) << grid_x << " = arith.constant " << gx << " : index\n";
    oss << indent_str(indent) << grid_y << " = arith.constant " << gy << " : index\n";
    oss << indent_str(indent) << grid_z << " = arith.constant " << gz << " : index\n";
    oss << indent_str(indent) << blk_x << " = arith.constant " << bx << " : index\n";
    oss << indent_str(indent) << blk_y << " = arith.constant " << by << " : index\n";
    oss << indent_str(indent) << blk_z << " = arith.constant " << bz << " : index\n";

    // gpu.launch defines block and thread IDs as new SSA values.
    //   blocks(%bX, %bY, %bZ) in (%gX = %sizeX, ...) 
    //   threads(%tX, %tY, %tZ) in (%sX = %sizeX, ...)
    // The %bX, %bY, %bZ are the block IDs (available in the body).
    // The %tX, %tY, %tZ are the thread IDs (available in the body).
    std::string block_idx_x = unique_name("block_idx_x");
    std::string block_idx_y = unique_name("block_idx_y");
    std::string block_idx_z = unique_name("block_idx_z");
    std::string thread_idx_x = unique_name("thread_idx_x");
    std::string thread_idx_y = unique_name("thread_idx_y");
    std::string thread_idx_z = unique_name("thread_idx_z");

    oss << indent_str(indent)
        << "gpu.launch blocks(" << block_idx_x << ", " << block_idx_y << ", " << block_idx_z
        << ") in (%grid_x = " << grid_x
        << ", %grid_y = " << grid_y << ", %grid_z = " << grid_z << ")"
        << " threads(" << thread_idx_x << ", " << thread_idx_y << ", " << thread_idx_z
        << ") in (%block_x = " << blk_x
        << ", %block_y = " << blk_y << ", %block_z = " << blk_z << ") {\n";

    // Emit tensor core intrinsics if configured
    if (config_.emit_tensor_core_intrinsics) {
        oss << indent_str(indent + 2)
            << "// Tensor Core MMA body\n";
        // The actual MMA body would be inserted here based on the
        // specific tiling and the op being lowered.
        // For now, emit a placeholder.
        oss << indent_str(indent + 2)
            << "// nvgpu.mma_sync would be emitted here for "
            << "m16n8k16 or m16n8k32 fragment operations\n";
    }

    oss << kernel_body;

    oss << indent_str(indent + 2) << "gpu.terminator\n";
    oss << indent_str(indent) << "}\n";

    return oss.str();
}

std::string MLIRLowering::emit_tensor_core_mma(
    const std::string& A, const std::string& B,
    const std::string& C, int64_t m, int64_t n, int64_t k,
    int indent
) {
    std::ostringstream oss;

    // Emit nvgpu.mma_sync pattern for Tensor Core operations
    // This follows the nvgpu dialect patterns in MLIR

    std::string frag_a = unique_name("fragA");
    std::string frag_b = unique_name("fragB");
    std::string frag_c = unique_name("fragC");

    oss << indent_str(indent) << "// Tensor Core MMA: " << m << "x" << n << "x" << k << "\n";

    // Load A fragment into registers
    oss << indent_str(indent) << frag_a
        << " = nvgpu.ldmatrix " << A
        << " : !nvgpu.warpgroup.fragment<m" << m << "x" << k
        << ", a, f16>\n";

    // Load B fragment into registers
    oss << indent_str(indent) << frag_b
        << " = nvgpu.ldmatrix " << B
        << " : !nvgpu.warpgroup.fragment<k" << k << "x" << n
        << ", b, f16>\n";

    // Load C accumulator
    oss << indent_str(indent) << frag_c
        << " = nvgpu.warpgroup.mma_init"
        << " : !nvgpu.warpgroup.accumulator<m" << m << "x" << n
        << ", f32>\n";

    // MMA sync operation
    oss << indent_str(indent) << C
        << " = nvgpu.warpgroup.mma_sync(" << frag_a << ", " << frag_b
        << ", " << frag_c << ")"
        << " : (!nvgpu.warpgroup.fragment<m" << m << "x" << k << ", a, f16>,"
        << " !nvgpu.warpgroup.fragment<k" << k << "x" << n << ", b, f16>,"
        << " !nvgpu.warpgroup.accumulator<m" << m << "x" << n << ", f32>)"
        << " -> !nvgpu.warpgroup.accumulator<m" << m << "x" << n << ", f32>\n";

    return oss.str();
}

std::string MLIRLowering::emit_return(const std::string& result, int indent) {
    std::ostringstream oss;
    oss << indent_str(indent) << "return " << result << "\n";
    return oss.str();
}

std::string MLIRLowering::emit_constant(
    const std::string& name, double value,
    ir::IRDType dtype, int indent
) {
    std::ostringstream oss;
    std::string mlir_dtype = irdtype_to_mlir(dtype);

    if (dtype == ir::IRDType::INT8 || dtype == ir::IRDType::INT4) {
        oss << indent_str(indent) << name
            << " = arith.constant " << static_cast<int64_t>(value)
            << " : " << mlir_dtype << "\n";
    } else {
        oss << indent_str(indent) << name
            << " = arith.constant " << value
            << " : " << mlir_dtype << "\n";
    }
    return oss.str();
}

std::string MLIRLowering::emit_shared_memory_alloc(
    const std::string& name,
    const ir::IRShape& shape,
    ir::IRDType dtype,
    int indent
) {
    std::ostringstream oss;
    std::string tensor_type = emit_tensor_type(shape, dtype);

    // In MLIR GPU dialect, shared memory is allocated via gpu.dynamic_shared_memory
    oss << indent_str(indent) << name
        << " = gpu.dynamic_shared_memory"
        << " : " << tensor_type << "\n";

    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────
// Op Classification
// ─────────────────────────────────────────────────────────────────────────

bool MLIRLowering::is_elementwise_op(ir::IROp::Kind kind) {
    switch (kind) {
        case ir::IROp::Kind::ADD:
        case ir::IROp::Kind::MUL:
        case ir::IROp::Kind::SUB:
        case ir::IROp::Kind::DIV:
        case ir::IROp::Kind::NEG:
        case ir::IROp::Kind::RELU:
        case ir::IROp::Kind::GELU:
        case ir::IROp::Kind::SIGMOID:
        case ir::IROp::Kind::EXP:
        case ir::IROp::Kind::LOG:
        case ir::IROp::Kind::SQRT:
        case ir::IROp::Kind::RECIPROCAL:
            return true;
        default:
            return false;
    }
}

bool MLIRLowering::is_reduction_op(ir::IROp::Kind kind) {
    switch (kind) {
        case ir::IROp::Kind::REDUCE_SUM:
        case ir::IROp::Kind::REDUCE_MAX:
        case ir::IROp::Kind::REDUCE_MEAN:
            return true;
        default:
            return false;
    }
}

bool MLIRLowering::is_matmul_op(ir::IROp::Kind kind) {
    switch (kind) {
        case ir::IROp::Kind::MATMUL:
        case ir::IROp::Kind::FUSED_MATMUL_RELU:
        case ir::IROp::Kind::FUSED_MATMUL_ADD:
        case ir::IROp::Kind::FUSED_MATMUL_ADD_RELU:
        case ir::IROp::Kind::FUSED_GEMM:
        case ir::IROp::Kind::FUSED_MHA:
            return true;
        default:
            return false;
    }
}

bool MLIRLowering::is_norm_op(ir::IROp::Kind kind) {
    switch (kind) {
        case ir::IROp::Kind::SOFTMAX:
        case ir::IROp::Kind::LAYERNORM:
        case ir::IROp::Kind::RMSNORM:
        case ir::IROp::Kind::FUSED_SOFTMAX:
        case ir::IROp::Kind::FUSED_LAYERNORM:
        case ir::IROp::Kind::FUSED_RMSNORM:
        case ir::IROp::Kind::FUSED_ADD_LN:
            return true;
        default:
            return false;
    }
}

std::string MLIRLowering::arith_op_string(ir::IROp::Kind kind) {
    switch (kind) {
        case ir::IROp::Kind::ADD: return "arith.addf";
        case ir::IROp::Kind::MUL: return "arith.mulf";
        case ir::IROp::Kind::SUB: return "arith.subf";
        case ir::IROp::Kind::DIV: return "arith.divf";
        case ir::IROp::Kind::NEG: return "arith.negf";
        case ir::IROp::Kind::EXP: return "math.exp";
        case ir::IROp::Kind::LOG: return "math.log";
        case ir::IROp::Kind::SQRT: return "math.sqrt";
        default: return "arith.addf";  // Fallback
    }
}

std::string MLIRLowering::linalg_op_name(ir::IROp::Kind kind) {
    switch (kind) {
        case ir::IROp::Kind::MATMUL: return "linalg.matmul";
        case ir::IROp::Kind::ADD: return "linalg.add";
        case ir::IROp::Kind::MUL: return "linalg.mul";
        case ir::IROp::Kind::SUB: return "linalg.sub";
        case ir::IROp::Kind::DIV: return "linalg.div";
        default: return "linalg.generic";
    }
}

// ─────────────────────────────────────────────────────────────────────────
// ID Management
// ─────────────────────────────────────────────────────────────────────────

std::string MLIRLowering::ssa_name(int64_t op_id) const {
    return "%op_" + std::to_string(op_id);
}

std::string MLIRLowering::unique_name(const std::string& prefix) {
    return "%" + prefix + "_" + std::to_string(unique_counter_++);
}

} // namespace symplex::lowering
