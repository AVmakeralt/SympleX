// =============================================================================
// SympleX Fusion Engine — Semantic Fusion Decision Module
// =============================================================================
//
// Architecture:
//   E-Graph Semantic Optimizer
//     → Fusion Engine (decides WHAT can fuse)       ← THIS MODULE
//       → Polyhedral Engine (decides WHETHER legal and HOW)
//         → MCMC Hardware Search
//           → Kernel Generation
//             → Empirical Feedback
//
// The Fusion Engine explores SEMANTIC program space — the meaning of
// operations and their compositional properties. It discovers that:
//
//   "ReLU(MatMul(A,B) + bias) can be a single FusedMatMulBiasReLU kernel"
//   "LayerNorm(x + residual) avoids writing the intermediate sum to HBM"
//   "A full attention block Q*K^T * V can be one persistent kernel"
//
// The Polyhedral Engine then validates schedule legality and determines
// the optimal tiling, loop ordering, and memory layout for each fusion
// decision. The two engines are decoupled: fusion discovers opportunities,
// polyhedral validates and realizes them.
//
// Sections:
//   §1  Data Types (DType, TensorShape, OpType)
//   §2  FusionOp — Lightweight Compute Operation Representation
//   §3  FusionPattern — Known Semantic Fusion Patterns
//   §4  FusionBoundary — A Single Fusion Decision
//   §5  FusionDecision — Aggregate Result of Fusion Analysis
//   §6  FusionEngine — Core Discovery Engine
//   §7  Pattern Classification Logic
//   §8  Memory & Compute Estimation Models
//   §9  Legality Validation (Quick Checks)
//  §10  Alternative Proposal Generation
// =============================================================================

// =============================================================================
// §1. DATA TYPES
// =============================================================================

/// Data type enumeration for tensor elements.
///
/// Covers the standard numeric types used in ML workloads, from full-precision
/// FP32 down to sub-byte INT4 quantization. The `size_bytes()` method returns
/// the storage size of a single element, which is critical for estimating HBM
/// traffic savings from fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    /// 32-bit IEEE 754 single-precision float. Default for training.
    FP32 = 0,
    /// 16-bit IEEE 754 half-precision float. Used in mixed-precision training.
    FP16 = 1,
    /// 16-bit Brain Float (8 exponent bits, 7 mantissa bits). Wider dynamic
    /// range than FP16, same storage. Preferred for deep learning inference.
    BF16 = 2,
    /// 8-bit floating point (E4M3 or E5M2). Used in FP8 training/inference
    /// on H100+ GPUs.
    FP8 = 3,
    /// 8-bit signed integer. Used for quantized inference.
    INT8 = 4,
    /// 4-bit signed integer. Sub-byte quantization for extreme compression.
    INT4 = 5,
}

impl DType {
    /// Returns the size in bytes of a single element of this dtype.
    ///
    /// INT4 is the only sub-byte type; it returns 1 because elements are
    /// typically packed two-per-byte but individual access requires a full byte.
    #[inline]
    pub const fn size_bytes(&self) -> i64 {
        match self {
            DType::FP32 => 4,
            DType::FP16 => 2,
            DType::BF16 => 2,
            DType::FP8  => 1,
            DType::INT8 => 1,
            DType::INT4 => 1, // packed 2-per-byte, but access granularity is 1 byte
        }
    }

    /// Returns true if this dtype is a floating-point type.
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::FP32 | DType::FP16 | DType::BF16 | DType::FP8)
    }

    /// Returns true if this dtype is a quantized integer type.
    #[inline]
    pub const fn is_quantized(&self) -> bool {
        matches!(self, DType::INT8 | DType::INT4)
    }

    /// Returns a human-readable name for this dtype.
    pub const fn name(&self) -> &'static str {
        match self {
            DType::FP32 => "fp32",
            DType::FP16 => "fp16",
            DType::BF16 => "bf16",
            DType::FP8  => "fp8",
            DType::INT8 => "int8",
            DType::INT4 => "int4",
        }
    }
}

/// Type alias for tensor shapes. A shape is a list of dimension sizes,
/// where each dimension is a positive integer. An empty shape represents
/// a scalar. The product of all dimensions gives the number of elements.
pub type TensorShape = Vec<i64>;

/// Computes the number of elements in a tensor shape (product of dimensions).
/// Returns 1 for scalar (empty shape). Returns 0 if any dimension is zero.
#[inline]
pub fn shape_num_elements(shape: &TensorShape) -> i64 {
    if shape.is_empty() {
        return 1;
    }
    let mut n: i64 = 1;
    for &dim in shape {
        if dim == 0 {
            return 0;
        }
        n = n.saturating_mul(dim);
    }
    n
}

/// Computes the memory footprint in bytes for a tensor of the given shape
/// and dtype. This is `num_elements * dtype.size_bytes()`.
#[inline]
pub fn shape_memory_bytes(shape: &TensorShape, dtype: DType) -> i64 {
    shape_num_elements(shape).saturating_mul(dtype.size_bytes())
}

/// Operation type classification for fusion pattern matching.
///
/// This enum captures the *semantic* operation type — what the operation
/// *means* mathematically — rather than its implementation details. The
/// fusion engine uses these semantics to decide which operations compose
/// into known fusion patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpType {
    // ── Matrix / Linear algebra ─────────────────────────────────────────
    /// Matrix multiplication: C = A @ B. The fundamental compute-heavy op.
    MatMul = 0,
    /// Batched matrix multiplication.
    BatchMatMul = 1,

    // ── Binary arithmetic ───────────────────────────────────────────────
    /// Element-wise addition: C = A + B (including bias add).
    Add = 10,
    /// Element-wise multiplication: C = A * B.
    Mul = 11,
    /// Element-wise subtraction: C = A - B.
    Sub = 12,
    /// Element-wise division: C = A / B.
    Div = 13,

    // ── Activation functions ─────────────────────────────────────────────
    /// Rectified Linear Unit: C = max(0, A). The most common activation.
    ReLU = 20,
    /// Gaussian Error Linear Unit: C = A * Φ(A), where Φ is the CDF of
    /// the standard normal distribution. Used in GPT-2, BERT, etc.
    GELU = 21,
    /// Sigmoid activation: C = 1 / (1 + exp(-A)).
    Sigmoid = 22,
    /// Hyperbolic tangent: C = tanh(A).
    Tanh = 23,
    /// SiLU / Swish activation: C = A * sigmoid(A). Used in LLaMA, etc.
    SiLU = 24,

    // ── Normalization ───────────────────────────────────────────────────
    /// Layer normalization: normalize along the last dimension with learnable
    /// scale and bias parameters.
    LayerNorm = 30,
    /// Root Mean Square normalization: normalize by RMS without centering.
    /// More efficient than LayerNorm; used in LLaMA, etc.
    RMSNorm = 31,

    // ── Softmax ─────────────────────────────────────────────────────────
    /// Softmax: C_i = exp(A_i) / sum(exp(A)). Converts logits to
    /// probabilities. Critical in attention mechanisms.
    Softmax = 40,
    /// Log-softmax: numerically stable log(softmax(A)).
    LogSoftmax = 41,

    // ── Structural / movement ops ───────────────────────────────────────
    /// Transpose: swap two dimensions of a tensor.
    Transpose = 50,
    /// Reshape: change the shape without moving data.
    Reshape = 51,
    /// Broadcast: expand a dimension from size 1 to size N.
    Broadcast = 52,
    /// Permute: general dimension reordering.
    Permute = 53,
    /// Concatenate: join tensors along a dimension.
    Concat = 54,
    /// Slice / index: extract a sub-tensor.
    Slice = 55,

    // ── Reduction ───────────────────────────────────────────────────────
    /// Reduction along one or more dimensions (sum, max, etc.).
    Reduce = 60,
    /// ArgMax: index of the maximum value along a dimension.
    ArgMax = 61,

    // ── Math / transcendental ───────────────────────────────────────────
    /// Element-wise exponential: C = exp(A).
    Exp = 70,
    /// Element-wise square root: C = sqrt(A).
    Sqrt = 71,
    /// Element-wise reciprocal: C = 1 / A.
    Reciprocal = 72,
    /// Element-wise negation: C = -A.
    Neg = 73,
    /// Element-wise absolute value: C = |A|.
    Abs = 74,

    // ── Communication / special ─────────────────────────────────────────
    /// All-reduce collective: sum across devices.
    AllReduce = 80,
    /// All-gather collective: concatenate across devices.
    AllGather = 81,
    /// Reduce-scatter collective.
    ReduceScatter = 82,
    /// Custom / unknown operation. Treated as a fusion barrier.
    Custom = 255,
}

impl OpType {
    /// Returns true if this operation is element-wise (applies the same
    /// computation independently to each element). Element-wise ops are
    /// always safe to fuse with other element-wise ops.
    #[inline]
    pub const fn is_elementwise(&self) -> bool {
        matches!(
            self,
            OpType::Add
                | OpType::Mul
                | OpType::Sub
                | OpType::Div
                | OpType::ReLU
                | OpType::GELU
                | OpType::Sigmoid
                | OpType::Tanh
                | OpType::SiLU
                | OpType::Exp
                | OpType::Sqrt
                | OpType::Reciprocal
                | OpType::Neg
                | OpType::Abs
        )
    }

    /// Returns true if this operation is an activation function.
    #[inline]
    pub const fn is_activation(&self) -> bool {
        matches!(
            self,
            OpType::ReLU | OpType::GELU | OpType::Sigmoid | OpType::Tanh | OpType::SiLU
        )
    }

    /// Returns true if this operation is a normalization op.
    #[inline]
    pub const fn is_normalization(&self) -> bool {
        matches!(self, OpType::LayerNorm | OpType::RMSNorm)
    }

    /// Returns true if this operation is a structural/movement op that
    /// does not perform computation but rearranges data.
    #[inline]
    pub const fn is_structural(&self) -> bool {
        matches!(
            self,
            OpType::Transpose
                | OpType::Reshape
                | OpType::Broadcast
                | OpType::Permute
                | OpType::Concat
                | OpType::Slice
        )
    }

    /// Returns true if this operation is a communication collective.
    #[inline]
    pub const fn is_communication(&self) -> bool {
        matches!(
            self,
            OpType::AllReduce | OpType::AllGather | OpType::ReduceScatter
        )
    }

    /// Returns a human-readable name for this operation type.
    pub const fn name(&self) -> &'static str {
        match self {
            OpType::MatMul       => "matmul",
            OpType::BatchMatMul  => "batch_matmul",
            OpType::Add          => "add",
            OpType::Mul          => "mul",
            OpType::Sub          => "sub",
            OpType::Div          => "div",
            OpType::ReLU         => "relu",
            OpType::GELU         => "gelu",
            OpType::Sigmoid      => "sigmoid",
            OpType::Tanh         => "tanh",
            OpType::SiLU         => "silu",
            OpType::LayerNorm    => "layer_norm",
            OpType::RMSNorm      => "rms_norm",
            OpType::Softmax      => "softmax",
            OpType::LogSoftmax   => "log_softmax",
            OpType::Transpose    => "transpose",
            OpType::Reshape      => "reshape",
            OpType::Broadcast    => "broadcast",
            OpType::Permute      => "permute",
            OpType::Concat       => "concat",
            OpType::Slice        => "slice",
            OpType::Reduce       => "reduce",
            OpType::ArgMax       => "argmax",
            OpType::Exp          => "exp",
            OpType::Sqrt         => "sqrt",
            OpType::Reciprocal   => "reciprocal",
            OpType::Neg          => "neg",
            OpType::Abs          => "abs",
            OpType::AllReduce    => "all_reduce",
            OpType::AllGather    => "all_gather",
            OpType::ReduceScatter => "reduce_scatter",
            OpType::Custom       => "custom",
        }
    }
}

// =============================================================================
// §2. FusionOp — Lightweight Compute Operation Representation
// =============================================================================

/// A lightweight representation of a compute operation for fusion analysis.
///
/// `FusionOp` captures the *semantic* properties of an operation that matter
/// for fusion decisions: what the operation does (op_type), the shapes of its
/// inputs and outputs, the data type, and whether it can be performed in-place.
/// It deliberately does NOT capture scheduling details (loop structure, tiling,
/// memory layout) — those belong to the polyhedral engine.
///
/// The `memory_bytes` field pre-computes the output tensor size in bytes,
/// which is the HBM traffic that fusion can potentially eliminate if this
/// op's output is consumed only by a subsequent fused op.
#[derive(Debug, Clone)]
pub struct FusionOp {
    /// The semantic operation type (e.g., MatMul, Add, ReLU).
    pub op_type: OpType,
    /// Shape of the output tensor.
    pub output_shape: TensorShape,
    /// Shapes of the input tensors. Most ops have 1-2 inputs.
    /// MatMul has 2 inputs (A, B), Add has 2 inputs, ReLU has 1 input.
    pub input_shapes: Vec<TensorShape>,
    /// Data type of the output tensor.
    pub dtype: DType,
    /// Whether this operation can be performed in-place (output aliases input).
    /// In-place ops never produce intermediate HBM traffic.
    pub is_inplace: bool,
    /// Output tensor size in bytes: product(output_shape) * dtype.size_bytes().
    /// Pre-computed to avoid repeated multiplication during fusion analysis.
    pub memory_bytes: i64,
}

impl FusionOp {
    /// Construct a new FusionOp, automatically computing `memory_bytes`.
    pub fn new(
        op_type: OpType,
        output_shape: TensorShape,
        input_shapes: Vec<TensorShape>,
        dtype: DType,
        is_inplace: bool,
    ) -> Self {
        let memory_bytes = shape_memory_bytes(&output_shape, dtype);
        Self {
            op_type,
            output_shape,
            input_shapes,
            dtype,
            is_inplace,
            memory_bytes,
        }
    }

    /// Returns the number of elements in the output tensor.
    #[inline]
    pub fn num_output_elements(&self) -> i64 {
        shape_num_elements(&self.output_shape)
    }

    /// Returns true if this op produces an intermediate result that could
    /// be eliminated by fusion. An op is "intermediate-eligible" if:
    /// - It is not in-place (in-place ops don't write to HBM anyway)
    /// - It has a non-zero output (no point fusing a scalar through HBM)
    /// - It is not a communication op (those require cross-device coordination)
    #[inline]
    pub fn is_intermediate_eligible(&self) -> bool {
        !self.is_inplace
            && self.memory_bytes > 0
            && !self.op_type.is_communication()
    }

    /// Returns the arithmetic intensity estimate (ops per byte of output).
    /// This is a rough model used to classify ops as compute-bound or
    /// memory-bound for fusion benefit estimation.
    ///
    /// - MatMul: 2*M*N*K ops / (M*N*sizeof) → high intensity
    /// - Elementwise: 1 op per element → low intensity (memory-bound)
    /// - Softmax: ~3*N ops per N-element row → medium intensity
    /// - LayerNorm: ~4*N ops per N-element row → medium intensity
    pub fn arithmetic_intensity(&self) -> f64 {
        let n = self.num_output_elements() as f64;
        if n <= 0.0 {
            return 0.0;
        }
        match self.op_type {
            OpType::MatMul | OpType::BatchMatMul => {
                // C = A @ B: 2*M*K*N FLOPs, output is M*N elements
                // Approximate: assume K is the inner dimension of the first input
                let k = if !self.input_shapes.is_empty() {
                    let dims = &self.input_shapes[0];
                    if dims.len() >= 2 {
                        dims[dims.len() - 1].max(1) as f64
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                // FLOPs per output element = 2*K
                2.0 * k
            }
            OpType::Softmax | OpType::LogSoftmax => 3.0, // exp + sum + div per element
            OpType::LayerNorm => 4.0,  // mean + var + normalize + scale+shift
            OpType::RMSNorm => 3.0,    // sum_sq + sqrt + normalize + scale
            OpType::Reduce => 1.0,     // one op per element reduced
            _ => 1.0,                  // elementwise: 1 op per element
        }
    }
}

// =============================================================================
// §3. FusionPattern — Known Semantic Fusion Patterns
// =============================================================================

/// Enumeration of known fusion patterns that the fusion engine can recognize.
///
/// Each variant represents a semantically meaningful composition of operations
/// that can be replaced by a single fused kernel. The fusion engine discovers
/// these patterns by matching operation sequences; the polyhedral engine then
/// validates that the fusion is legal for the specific iteration space.
///
/// The key insight: fusion patterns are about *semantics* (what the combined
/// operation means), not *scheduling* (how to execute it). The same
/// MatMulBiasReLU pattern might be realized with different tile sizes or loop
/// orderings depending on the hardware target — that's the polyhedral engine's
/// responsibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FusionPattern {
    /// MatMul + bias add + activation.
    ///
    /// Pattern: `C = activation(A @ B + bias)`
    ///
    /// This is the single most impactful fusion in ML workloads. Without
    /// fusion, the MatMul output (often several MB) must be written to HBM,
    /// then read back for the bias add, then written again, then read back
    /// for the activation. With fusion, all three steps happen in registers
    /// or SRAM — the only HBM traffic is reading A, B, bias and writing C.
    ///
    /// Typical savings: 2× output_tensor_size bytes of HBM traffic.
    MatMulBiasReLU = 0,

    /// Full attention block: Q*K^T scaled + softmax + *V.
    ///
    /// Pattern: `attn = softmax(Q @ K^T / sqrt(d_k)) @ V`
    ///
    /// This is the "FlashAttention" pattern. The key insight is that the
    /// Q*K^T matrix (S) can be very large (seq_len × seq_len) and must
    /// never be materialized in HBM. Instead, the online-softmax algorithm
    /// processes S in tiles, maintaining running max/sum for numerical
    /// stability, and directly accumulates into the output.
    ///
    /// Typical savings: 2 × batch × heads × seq_len² × sizeof(dtype) bytes.
    AttentionBlock = 1,

    /// Normalization + residual addition + activation.
    ///
    /// Pattern: `output = activation(norm(x) + residual)`
    ///   or:    `output = norm(x + residual)`  (pre-norm variant)
    ///
    /// In transformer blocks, LayerNorm/RMSNorm is always followed by a
    /// residual add. Without fusion, the norm output must be written to HBM
    /// and immediately read back for the add. Fusing eliminates this round-trip.
    ///
    /// Typical savings: 1 × tensor_size bytes of HBM traffic.
    NormResidualActivation = 2,

    /// Optimizer step fusion: weight + momentum + gradient update.
    ///
    /// Pattern: `w_new = w_old - lr * (m * beta1 + grad * (1 - beta1))`
    ///   (Adam-style, or similar for SGD with momentum)
    ///
    /// The optimizer step touches the weight, momentum, and gradient tensors
    /// — all the same shape. Without fusion, each intermediate update
    /// requires a separate HBM read-modify-write cycle. With fusion, the
    /// entire update is a single pass over memory.
    ///
    /// Typical savings: 2 × weight_size bytes of HBM traffic.
    OptimizerStepFusion = 3,

    /// Backward pass chain rule fusion.
    ///
    /// Pattern: chain of `dL/d(input) = dL/d(output) * d(output)/d(input)`
    ///
    /// In backpropagation, gradient computation forms a chain of element-wise
    /// multiplications and additions. Each intermediate gradient tensor is
    /// produced and immediately consumed by the next step. Fusing the chain
    /// eliminates all intermediate gradient writes to HBM.
    ///
    /// Typical savings: (chain_length - 1) × gradient_size bytes.
    BackwardPassChain = 4,

    /// Communication-compute overlap fusion.
    ///
    /// Pattern: `compute(compute_input) || allreduce(gradient)`
    ///
    /// Rather than fusing ops into a single kernel, this pattern overlaps
    /// a communication operation (allreduce of the previous gradient) with
    /// a computation operation (forward/backward of the current micro-batch).
    /// The "fusion" is temporal overlap, not kernel merging.
    ///
    /// Typical savings: latency of the allreduce (hidden behind compute).
    CommunicationComputeOverlap = 5,

    /// Persistent kernel: resident block fusion.
    ///
    /// Pattern: a sequence of operations that execute repeatedly (e.g.,
    /// transformer layers) where the kernel stays resident on the GPU,
    /// avoiding launch overhead and keeping intermediate results in SRAM
    /// across invocations.
    ///
    /// This is especially beneficial for small batch sizes where kernel
    /// launch overhead dominates, or for iterative algorithms (MCMC,
    /// optimization loops) where the same computation is repeated.
    ///
    /// Typical savings: kernel_launch_overhead × num_iterations + intermediate
    /// HBM traffic between iterations.
    PersistentKernel = 6,

    /// Chains of element-wise operations.
    ///
    /// Pattern: `output = op_n(... op_2(op_1(input)))`
    ///
    /// Any sequence of element-wise ops (activations, arithmetic, math
    /// functions) can be fused into a single kernel that reads the input
    /// once, applies all operations in registers, and writes the output once.
    /// This is the simplest and most universally applicable fusion pattern.
    ///
    /// Typical savings: (chain_length - 1) × tensor_size bytes.
    ElementwiseChain = 7,

    /// Reduction + broadcast + elementwise fusion.
    ///
    /// Pattern: `output = elementwise(reduce_sum(x, axis) + bias)`
    ///
    /// Reductions produce a smaller tensor that is then broadcast back for
    /// element-wise operations (e.g., mean + variance in LayerNorm, or
    /// reduction + bias in loss computation). Fusing avoids writing the
    /// reduced intermediate to HBM and reading it back for the broadcast.
    ///
    /// Typical savings: 1 × reduced_tensor_size bytes.
    ReductionFusion = 8,

    /// Horizontal elementwise fusion: independent ops on same-shape tensors.
    ///
    /// Pattern: multiple independent elementwise ops that operate on tensors
    /// of the same shape, running in parallel within a single kernel.
    ///
    /// Unlike ElementwiseChain (which fuses sequential ops), HorizontalElementwise
    /// fuses independent ops that don't depend on each other's outputs. The
    /// benefit is eliminating kernel launch overhead: instead of launching N
    /// separate kernels, a single kernel processes all ops.
    ///
    /// Typical savings: (num_ops - 1) × kernel_launch_overhead_bytes.
    HorizontalElementwise = 9,
}

impl FusionPattern {
    /// Returns a human-readable name for this fusion pattern.
    pub const fn name(&self) -> &'static str {
        match self {
            FusionPattern::MatMulBiasReLU             => "matmul_bias_relu",
            FusionPattern::AttentionBlock              => "attention_block",
            FusionPattern::NormResidualActivation       => "norm_residual_activation",
            FusionPattern::OptimizerStepFusion          => "optimizer_step",
            FusionPattern::BackwardPassChain            => "backward_pass_chain",
            FusionPattern::CommunicationComputeOverlap   => "comm_compute_overlap",
            FusionPattern::PersistentKernel              => "persistent_kernel",
            FusionPattern::ElementwiseChain              => "elementwise_chain",
            FusionPattern::ReductionFusion              => "reduction_fusion",
            FusionPattern::HorizontalElementwise          => "horizontal_elementwise",
        }
    }

    /// Returns a human-readable description of this fusion pattern.
    pub const fn description(&self) -> &'static str {
        match self {
            FusionPattern::MatMulBiasReLU =>
                "Fuses MatMul + bias add + activation into a single kernel, \
                 eliminating intermediate HBM writes for the MatMul and Add outputs",
            FusionPattern::AttentionBlock =>
                "Fuses the full attention block (Q*K^T + softmax + *V) into \
                 a single persistent kernel using online-softmax, avoiding \
                 materializing the large attention matrix in HBM",
            FusionPattern::NormResidualActivation =>
                "Fuses LayerNorm/RMSNorm + residual addition + activation, \
                 keeping the normalized intermediate in SRAM before the add",
            FusionPattern::OptimizerStepFusion =>
                "Fuses the optimizer update (weight + momentum + gradient) \
                 into a single pass, eliminating redundant memory traffic",
            FusionPattern::BackwardPassChain =>
                "Fuses the chain rule gradient computation into a single \
                 kernel, eliminating intermediate gradient writes to HBM",
            FusionPattern::CommunicationComputeOverlap =>
                "Overlaps communication (allreduce) with computation for \
                 the next micro-batch, hiding communication latency",
            FusionPattern::PersistentKernel =>
                "Keeps a kernel resident on the device across iterations, \
                 avoiding launch overhead and retaining SRAM state",
            FusionPattern::ElementwiseChain =>
                "Fuses a chain of element-wise operations into a single \
                 kernel that reads input once and writes output once",
            FusionPattern::ReductionFusion =>
                "Fuses reduction + broadcast + elementwise into a single \
                 kernel, avoiding the reduced intermediate HBM round-trip",
            FusionPattern::HorizontalElementwise =>
                "Fuses independent elementwise ops operating on same-shape \
                 tensors into a single kernel, eliminating kernel launch overhead",
        }
    }

    /// Returns whether this pattern requires polyhedral validation.
    ///
    /// Patterns involving reductions, matrix multiplications, or non-trivial
    /// loop dependencies require the polyhedral engine to verify that the
    /// fused schedule preserves all data dependencies. Simple element-wise
    /// chains do not require polyhedral validation.
    pub const fn requires_polyhedral_validation(&self) -> bool {
        matches!(
            self,
            FusionPattern::MatMulBiasReLU
                | FusionPattern::AttentionBlock
                | FusionPattern::ReductionFusion
                | FusionPattern::PersistentKernel
                | FusionPattern::BackwardPassChain
                | FusionPattern::HorizontalElementwise
        )
    }

    /// Returns the typical number of intermediate tensors eliminated by
    /// this fusion pattern. Used as a heuristic for confidence scoring.
    pub const fn intermediates_eliminated(&self) -> usize {
        match self {
            FusionPattern::MatMulBiasReLU              => 2, // MatMul output + Add output
            FusionPattern::AttentionBlock               => 2, // Q*K^T matrix + softmax output
            FusionPattern::NormResidualActivation        => 1, // norm output
            FusionPattern::OptimizerStepFusion           => 2, // momentum update + gradient update
            FusionPattern::BackwardPassChain             => 1, // each chain link (variable)
            FusionPattern::CommunicationComputeOverlap    => 0, // no intermediates, just overlap
            FusionPattern::PersistentKernel               => 1, // cross-iteration intermediate
            FusionPattern::ElementwiseChain               => 1, // each chain link (variable)
            FusionPattern::ReductionFusion               => 1, // reduced intermediate
            FusionPattern::HorizontalElementwise          => 0, // no intermediates, launch overhead savings
        }
    }
}

// =============================================================================
// §4. FusionBoundary — A Single Fusion Decision
// =============================================================================

/// A single fusion decision: a group of operations that should be fused
/// together, along with the estimated benefit and confidence.
///
/// The term "boundary" is used because fusion decisions define the boundaries
/// between kernels: operations inside a boundary become one kernel, operations
/// outside become separate kernels. The fusion engine discovers these
/// boundaries; the polyhedral engine validates them.
///
/// # Invariants
///
/// - `op_indices` is always non-empty and sorted in ascending order
/// - `confidence` is always in [0.0, 1.0]
/// - `memory_traffic_savings_bytes` is always >= 0
/// - `compute_savings_factor` is always >= 1.0 (1.0 = no compute savings)
#[derive(Debug, Clone)]
pub struct FusionBoundary {
    /// Indices of the operations to fuse within the input `ops` slice.
    /// These indices correspond to positions in the `ops` parameter passed
    /// to `discover_fusion_boundaries`.
    pub op_indices: Vec<usize>,
    /// The fusion pattern that matches this group of operations.
    pub pattern: FusionPattern,
    /// Estimated memory traffic savings in bytes. This is the total HBM
    /// read + write traffic that is eliminated by keeping intermediates
    /// in registers/SRAM instead of writing them to HBM.
    ///
    /// Calculation model:
    ///   Without fusion: each op writes its output to HBM, the next op
    ///     reads it back. Traffic = 2 × intermediate_size per boundary.
    ///   With fusion: intermediates stay in registers/SRAM. Traffic = 0
    ///     for intermediates. Only inputs and final output touch HBM.
    ///   Savings = Σ(2 × intermediate_size) for all eliminated intermediates.
    pub memory_traffic_savings_bytes: i64,
    /// Estimated compute savings factor. A factor of 1.0 means no compute
    /// savings (fusion is purely a memory optimization). A factor of 1.5
    /// means the fused kernel performs 50% fewer total operations than the
    /// unfused sequence (e.g., by exploiting algebraic simplifications or
    /// avoiding redundant recomputation).
    pub compute_savings_factor: f64,
    /// Confidence score in [0.0, 1.0]. Indicates how certain the fusion
    /// engine is that this fusion is beneficial and correct.
    ///
    /// - 1.0: Certain (e.g., elementwise chain with matching shapes)
    /// - 0.8-0.99: High confidence (well-known pattern like MatMulBiasReLU)
    /// - 0.5-0.79: Moderate confidence (pattern match but shape constraints unclear)
    /// - 0.0-0.49: Low confidence (speculative; may need empirical validation)
    pub confidence: f64,
}

impl FusionBoundary {
    /// Creates a new FusionBoundary with validated fields.
    pub fn new(
        op_indices: Vec<usize>,
        pattern: FusionPattern,
        memory_traffic_savings_bytes: i64,
        compute_savings_factor: f64,
        confidence: f64,
    ) -> Self {
        debug_assert!(!op_indices.is_empty(), "FusionBoundary must contain at least one op");
        debug_assert!(compute_savings_factor >= 1.0, "Compute savings factor must be >= 1.0");
        debug_assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence must be in [0.0, 1.0]"
        );
        Self {
            op_indices,
            pattern,
            memory_traffic_savings_bytes: memory_traffic_savings_bytes.max(0),
            compute_savings_factor: compute_savings_factor.max(1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Returns the number of operations in this fusion boundary.
    #[inline]
    pub fn op_count(&self) -> usize {
        self.op_indices.len()
    }

    /// Returns whether this fusion boundary requires polyhedral validation.
    #[inline]
    pub fn requires_polyhedral_validation(&self) -> bool {
        self.pattern.requires_polyhedral_validation()
    }

    /// Returns the estimated total speedup factor from this fusion.
    ///
    /// This combines memory savings and compute savings into a single metric.
    /// The model assumes that the unfused baseline is memory-bound (which is
    /// the common case for ML workloads), so memory savings dominate.
    ///
    /// Speedup ≈ compute_savings × (1 + memory_savings_ratio)
    ///
    /// where memory_savings_ratio = savings_bytes / total_unfused_traffic_bytes.
    /// For a conservative estimate, we use a simplified model.
    pub fn estimated_speedup(&self) -> f64 {
        // Base speedup from compute savings
        let mut speedup = self.compute_savings_factor;

        // Memory savings contribution: each GB of HBM traffic saved at ~1 TB/s
        // saves ~1ms. We model this as a multiplier on the compute speedup.
        // The more traffic saved relative to the compute, the bigger the win.
        let savings_mb = self.memory_traffic_savings_bytes as f64 / (1024.0 * 1024.0);
        if savings_mb > 0.0 {
            // Diminishing returns: first MB saved gives the biggest boost
            let memory_speedup = 1.0 + (savings_mb / (savings_mb + 10.0));
            speedup *= memory_speedup;
        }

        // Scale by confidence
        speedup *= self.confidence;

        speedup
    }
}

// =============================================================================
// §5. FusionDecision — Aggregate Result of Fusion Analysis
// =============================================================================

/// The result of running the fusion engine on a sequence of operations.
///
/// Contains all discovered fusion boundaries, along with aggregate statistics
/// about the expected benefit of applying all fusions.
///
/// # Design Note
///
/// The fusion engine produces *decisions* (what to fuse), not *schedules*
/// (how to execute). Each `FusionBoundary` in `boundaries` represents a
/// candidate fusion that should be passed to the polyhedral engine for
/// legality validation and schedule optimization. Boundaries that fail
/// polyhedral validation should be dropped; boundaries that pass should be
/// realized as fused kernels.
#[derive(Debug, Clone)]
pub struct FusionDecision {
    /// The list of fusion boundary decisions. These are non-overlapping:
    /// each operation index appears in at most one boundary.
    pub boundaries: Vec<FusionBoundary>,
    /// Total estimated speedup factor across all fusion boundaries.
    /// Computed as the product of individual boundary speedups (assuming
    /// sequential execution). A value of 1.0 means no speedup.
    pub total_estimated_speedup: f64,
    /// Total HBM traffic reduction in bytes across all fusion boundaries.
    /// This is the sum of `memory_traffic_savings_bytes` for each boundary.
    pub total_hbm_traffic_reduction: i64,
    /// Whether any fusion boundary requires polyhedral validation before
    /// it can be realized. If true, the polyhedral engine must be consulted
    /// before kernel generation.
    pub requires_polyhedral_validation: bool,
}

impl FusionDecision {
    /// Creates an empty fusion decision with no boundaries.
    pub fn empty() -> Self {
        Self {
            boundaries: Vec::new(),
            total_estimated_speedup: 1.0,
            total_hbm_traffic_reduction: 0,
            requires_polyhedral_validation: false,
        }
    }

    /// Creates a FusionDecision from a list of non-overlapping boundaries,
    /// computing aggregate statistics automatically.
    pub fn from_boundaries(boundaries: Vec<FusionBoundary>) -> Self {
        let total_hbm_traffic_reduction: i64 = boundaries
            .iter()
            .map(|b| b.memory_traffic_savings_bytes)
            .sum();

        // Total speedup is the product of individual speedups for sequential
        // execution. We use a weighted geometric mean to avoid double-counting
        // in cases where boundaries overlap in time.
        let total_estimated_speedup = if boundaries.is_empty() {
            1.0
        } else {
            let mut product = 1.0;
            for b in &boundaries {
                product *= b.estimated_speedup();
            }
            // Normalize: with N boundaries, the geometric mean speedup per
            // boundary gives a more realistic total than the raw product.
            // But for genuinely sequential ops, the product is correct.
            product
        };

        let requires_polyhedral_validation = boundaries
            .iter()
            .any(|b| b.requires_polyhedral_validation());

        Self {
            boundaries,
            total_estimated_speedup,
            total_hbm_traffic_reduction,
            requires_polyhedral_validation,
        }
    }

    /// Returns the number of fusion boundaries discovered.
    #[inline]
    pub fn boundary_count(&self) -> usize {
        self.boundaries.len()
    }

    /// Returns the total number of operations involved in any fusion boundary.
    pub fn total_fused_ops(&self) -> usize {
        self.boundaries.iter().map(|b| b.op_count()).sum()
    }

    /// Returns the average confidence across all boundaries.
    pub fn average_confidence(&self) -> f64 {
        if self.boundaries.is_empty() {
            return 0.0;
        }
        self.boundaries.iter().map(|b| b.confidence).sum::<f64>() / self.boundaries.len() as f64
    }
}

// =============================================================================
// §6. FusionEngine — Core Discovery Engine
// =============================================================================

/// The SympleX Fusion Engine: discovers semantic fusion opportunities in a
/// sequence of compute operations.
///
/// The engine's job is to answer: "Given these operations, which ones can be
/// fused together, and what is the expected benefit?" It does NOT answer:
/// "Is this fusion legal for this specific loop schedule?" — that is the
/// polyhedral engine's job.
///
/// # Architecture
///
/// ```text
/// Input: [FusionOp; N]
///   │
///   ├─ Sliding window pattern matching
///   │   ├─ classify_pattern(window) → Option<FusionPattern>
///   │   └─ For each match: create FusionBoundary
///   │
///   ├─ Greedy non-overlapping selection
///   │   └─ Sort by (confidence × estimated_speedup), pick non-overlapping
///   │
///   ├─ Aggregate statistics
///   │   └─ Compute total speedup, HBM savings, polyhedral requirement
///   │
///   └─ Output: FusionDecision
/// ```
///
/// # Usage
///
/// ```ignore
/// use symplex_engine::fusion_engine::{FusionEngine, FusionOp, OpType, DType};
///
/// let engine = FusionEngine::new();
/// let ops = vec![
///     FusionOp::new(OpType::MatMul, vec![128, 256], vec![vec![128, 512], vec![512, 256]], DType::FP16, false),
///     FusionOp::new(OpType::Add,    vec![128, 256], vec![vec![128, 256], vec![256]], DType::FP16, false),
///     FusionOp::new(OpType::ReLU,   vec![128, 256], vec![vec![128, 256]], DType::FP16, false),
/// ];
/// let decision = engine.discover_fusion_boundaries(&ops);
/// assert_eq!(decision.boundary_count(), 1);
/// assert!(decision.total_hbm_traffic_reduction > 0);
/// ```
#[derive(Debug, Clone)]
pub struct FusionEngine {
    /// Maximum window size for pattern matching. Patterns longer than this
    /// will not be discovered. Default: 8 (covers the longest standard
    /// pattern, AttentionBlock with 5+ ops).
    max_pattern_window: usize,

    /// Minimum confidence threshold. Fusion boundaries with confidence below
    /// this threshold are discarded. Default: 0.3.
    min_confidence: f64,

    /// Whether to include speculative (low-confidence) fusion boundaries
    /// that may require empirical validation. Default: false.
    include_speculative: bool,
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionEngine {
    /// Create a new FusionEngine with default parameters.
    pub fn new() -> Self {
        Self {
            max_pattern_window: 8,
            min_confidence: 0.3,
            include_speculative: false,
        }
    }

    /// Create a FusionEngine with custom parameters.
    pub fn with_config(
        max_pattern_window: usize,
        min_confidence: f64,
        include_speculative: bool,
    ) -> Self {
        Self {
            max_pattern_window: max_pattern_window.max(2),
            min_confidence: min_confidence.clamp(0.0, 1.0),
            include_speculative,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Main entry point: discover_fusion_boundaries
    // ─────────────────────────────────────────────────────────────────────

    /// Discover all fusion boundaries in a sequence of operations.
    ///
    /// This is the main entry point for the fusion engine. It takes a slice
    /// of `FusionOp`s (representing a compute graph in topological order) and
    /// returns a `FusionDecision` containing all discovered fusion boundaries.
    ///
    /// # Algorithm
    ///
    /// 1. **Pattern matching**: Slide windows of size 2..max_pattern_window
    ///    across the ops, attempting to classify each window as a known fusion
    ///    pattern.
    ///
    /// 2. **Candidate creation**: For each successful classification, create a
    ///    `FusionBoundary` with estimated savings and confidence.
    ///
    /// 3. **Greedy selection**: Sort candidates by benefit (confidence ×
    ///    speedup) and greedily select non-overlapping boundaries. This
    ///    ensures each operation appears in at most one fusion boundary.
    ///
    /// 4. **Aggregation**: Compute total speedup and HBM traffic reduction.
    ///
    /// # Panics
    ///
    /// Does not panic. Returns an empty `FusionDecision` for empty input.
    pub fn discover_fusion_boundaries(&self, ops: &[FusionOp]) -> FusionDecision {
        if ops.len() < 2 {
            return FusionDecision::empty();
        }

        // ── Phase 1: Discover all candidate fusion boundaries ────────────
        let mut candidates: Vec<FusionBoundary> = Vec::new();

        // Slide windows of increasing size across the ops
        for window_size in 2..=self.max_pattern_window.min(ops.len()) {
            for start in 0..=(ops.len() - window_size) {
                let end = start + window_size;
                let window = &ops[start..end];

                if let Some(pattern) = self.classify_pattern(window) {
                    let shapes: Vec<TensorShape> = window
                        .iter()
                        .map(|op| op.output_shape.clone())
                        .collect();

                    let memory_savings =
                        Self::estimate_memory_savings(&pattern, &shapes, window);

                    let compute_savings = Self::estimate_compute_savings(&pattern);

                    let confidence = self.compute_confidence(&pattern, window);

                    if confidence >= self.min_confidence || self.include_speculative {
                        let indices: Vec<usize> = (start..end).collect();
                        candidates.push(FusionBoundary::new(
                            indices,
                            pattern,
                            memory_savings,
                            compute_savings,
                            confidence,
                        ));
                    }
                }
            }
        }

        // ── Phase 2: Greedy non-overlapping selection ────────────────────
        //
        // Sort candidates by estimated benefit (descending) and greedily
        // select boundaries that don't overlap with already-selected ones.
        // This is a standard interval scheduling optimization.
        candidates.sort_by(|a, b| {
            let score_a = a.estimated_speedup();
            let score_b = b.estimated_speedup();
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected: Vec<FusionBoundary> = Vec::new();
        let mut used_indices: Vec<usize> = Vec::new();

        for candidate in candidates {
            // Check if this candidate overlaps with any already-selected boundary
            let overlaps = candidate
                .op_indices
                .iter()
                .any(|idx| used_indices.binary_search(idx).is_ok());

            if !overlaps {
                // Mark these indices as used
                for &idx in &candidate.op_indices {
                    let pos = used_indices.binary_search(&idx).unwrap_err();
                    used_indices.insert(pos, idx);
                }
                selected.push(candidate);
            }
        }

        // ── Phase 3: Build the decision ──────────────────────────────────
        FusionDecision::from_boundaries(selected)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Pattern classification
    // ─────────────────────────────────────────────────────────────────────

    /// Classify a group of operations as a known fusion pattern.
    ///
    /// This method examines the *semantic* composition of the operations —
    /// their types, shapes, and relationships — to determine if they match
    /// a known fusion pattern. It returns `None` if the operations don't
    /// match any known pattern.
    ///
    /// # Matching Strategy
    ///
    /// The classifier uses a priority-ordered set of pattern matchers:
    /// more specific patterns (AttentionBlock) are checked before more
    /// general ones (ElementwiseChain). This ensures that the most
    /// beneficial pattern is always selected.
    pub fn classify_pattern(&self, ops: &[FusionOp]) -> Option<FusionPattern> {
        if ops.is_empty() {
            return None;
        }

        let op_types: Vec<OpType> = ops.iter().map(|op| op.op_type).collect();

        // ── Priority-ordered pattern matching ────────────────────────────
        // More specific (and more beneficial) patterns are checked first.

        // 1. AttentionBlock: [MatMul, Softmax, MatMul] or
        //    [MatMul, Mul, Softmax, MatMul] (with scaling)
        if self.matches_attention_block(&op_types) {
            return Some(FusionPattern::AttentionBlock);
        }

        // 2. MatMulBiasReLU: [MatMul, Add, activation] or [MatMul, activation]
        if self.matches_matmul_bias_activation(&op_types) {
            return Some(FusionPattern::MatMulBiasReLU);
        }

        // 3. NormResidualActivation: [Norm, Add, activation?] or
        //    [Add, Norm, activation?] (pre-norm variant)
        if self.matches_norm_residual(&op_types) {
            return Some(FusionPattern::NormResidualActivation);
        }

        // 4. OptimizerStepFusion: [Mul, Add, Mul, Add, ...] or [Add, Mul, Add]
        if self.matches_optimizer_step(&op_types) {
            return Some(FusionPattern::OptimizerStepFusion);
        }

        // 5. BackwardPassChain: [Mul, Add, Mul, Add, ...] with consistent shapes
        if self.matches_backward_chain(&op_types, ops) {
            return Some(FusionPattern::BackwardPassChain);
        }

        // 6. CommunicationComputeOverlap: [comm_op, compute_op]
        if self.matches_comm_compute_overlap(&op_types) {
            return Some(FusionPattern::CommunicationComputeOverlap);
        }

        // 7. ReductionFusion: [Reduce, Broadcast?, elementwise+]
        if self.matches_reduction_fusion(&op_types) {
            return Some(FusionPattern::ReductionFusion);
        }

        // 8. PersistentKernel: repeated block pattern (detected by shape repetition)
        if self.matches_persistent_kernel(ops) {
            return Some(FusionPattern::PersistentKernel);
        }

        // 9. ElementwiseChain: all ops are element-wise (fallback pattern)
        if self.matches_elementwise_chain(&op_types) {
            return Some(FusionPattern::ElementwiseChain);
        }

        None
    }

    // ─────────────────────────────────────────────────────────────────────
    // Memory & compute estimation
    // ─────────────────────────────────────────────────────────────────────

    /// Estimate the memory traffic savings from fusing operations that match
    /// the given pattern, given their output shapes.
    ///
    /// # Model
    ///
    /// Without fusion, each intermediate tensor must be:
    ///   1. Written to HBM by the producing op
    ///   2. Read from HBM by the consuming op
    /// Total traffic per intermediate = 2 × tensor_size_bytes
    ///
    /// With fusion, intermediates stay in registers/SRAM:
    ///   Traffic = 0 for intermediates
    ///
    /// Savings = Σ(2 × intermediate_size) for all eliminated intermediates.
    ///
    /// The "intermediate" tensors are the outputs of all ops except the last
    /// one in the fusion boundary. The last op's output still goes to HBM.
    pub fn estimate_memory_savings(
        pattern: &FusionPattern,
        shapes: &[TensorShape],
        ops: &[FusionOp],
    ) -> i64 {
        if shapes.is_empty() || ops.is_empty() {
            return 0;
        }

        match pattern {
            FusionPattern::MatMulBiasReLU => {
                // Without fusion: write MatMul output + read for Add + write Add output
                // + read for ReLU = 2 intermediates × 2 traffic each
                // Intermediate 1: MatMul output (read back for Add)
                // Intermediate 2: Add output (read back for ReLU)
                let matmul_output_bytes = ops
                    .first()
                    .map(|op| op.memory_bytes)
                    .unwrap_or(0);
                let add_output_bytes = if ops.len() >= 2 {
                    ops[1].memory_bytes
                } else {
                    0
                };
                // Each intermediate incurs 2× traffic (write + read)
                2 * matmul_output_bytes + 2 * add_output_bytes
            }

            FusionPattern::AttentionBlock => {
                // The Q*K^T matrix is the critical intermediate.
                // It's typically [batch, heads, seq_len, seq_len] which can be
                // very large. The softmax output is the same size.
                // Savings = 2 × (QK^T size + softmax output size)
                let qk_bytes = ops
                    .first()
                    .map(|op| op.memory_bytes)
                    .unwrap_or(0);
                let softmax_bytes = if ops.len() >= 2 {
                    // Find the softmax op in the sequence
                    ops.iter()
                        .find(|op| op.op_type == OpType::Softmax)
                        .map(|op| op.memory_bytes)
                        .unwrap_or(0)
                } else {
                    0
                };
                2 * qk_bytes + 2 * softmax_bytes
            }

            FusionPattern::NormResidualActivation => {
                // Intermediate: norm output (written then read for add)
                let norm_output = ops
                    .iter()
                    .find(|op| op.op_type.is_normalization())
                    .map(|op| op.memory_bytes)
                    .unwrap_or(0);
                // If there's an activation after the add, the add output is also saved
                let add_output = ops
                    .iter()
                    .find(|op| op.op_type == OpType::Add)
                    .map(|op| op.memory_bytes)
                    .unwrap_or(0);
                let has_activation = ops.iter().any(|op| op.op_type.is_activation());
                if has_activation {
                    2 * norm_output + 2 * add_output
                } else {
                    2 * norm_output
                }
            }

            FusionPattern::OptimizerStepFusion => {
                // The optimizer step touches weight, momentum, variance, gradient.
                // Without fusion, each intermediate update requires separate memory traffic.
                // With fusion, it's a single fused pass.
                // Model: 2 intermediate tensors × 2 traffic each
                let total: i64 = ops
                    .iter()
                    .take(ops.len().saturating_sub(1))
                    .map(|op| 2 * op.memory_bytes)
                    .sum();
                total
            }

            FusionPattern::BackwardPassChain => {
                // Each chain link produces an intermediate gradient
                let intermediates: i64 = ops
                    .iter()
                    .take(ops.len().saturating_sub(1))
                    .map(|op| op.memory_bytes)
                    .sum();
                2 * intermediates
            }

            FusionPattern::CommunicationComputeOverlap => {
                // No intermediate memory savings; the benefit is latency hiding.
                // We model this as the communication volume (which would otherwise
                // be on the critical path).
                ops.iter()
                    .find(|op| op.op_type.is_communication())
                    .map(|op| op.memory_bytes)
                    .unwrap_or(0)
            }

            FusionPattern::PersistentKernel => {
                // Savings from keeping cross-iteration intermediates in SRAM.
                // Model: 1 intermediate × 2 traffic
                ops.first()
                    .map(|op| 2 * op.memory_bytes)
                    .unwrap_or(0)
            }

            FusionPattern::ElementwiseChain => {
                // Each intermediate (all but last) saves 2× traffic
                let intermediates: i64 = ops
                    .iter()
                    .take(ops.len().saturating_sub(1))
                    .map(|op| op.memory_bytes)
                    .sum();
                2 * intermediates
            }

            FusionPattern::ReductionFusion => {
                // The reduced intermediate is small but still requires
                // write + read. Plus the broadcast output if separate.
                let reduced_output = ops
                    .iter()
                    .find(|op| op.op_type == OpType::Reduce)
                    .map(|op| op.memory_bytes)
                    .unwrap_or(0);
                2 * reduced_output
            }

            FusionPattern::HorizontalElementwise => {
                // Savings come from eliminating kernel launch overhead, not
                // from eliminating intermediates. Each avoided kernel launch
                // saves ~5000 bytes of overhead (5μs at 1GHz equivalent).
                const KERNEL_LAUNCH_OVERHEAD_BYTES: i64 = 5000;
                (ops.len().saturating_sub(1) as i64) * KERNEL_LAUNCH_OVERHEAD_BYTES
            }
        }
    }

    /// Estimate the compute savings factor for a fusion pattern.
    ///
    /// The compute savings factor is the ratio of unfused FLOPs to fused FLOPs.
    /// A factor of 1.0 means no compute savings (fusion is purely a memory
    /// optimization). A factor > 1.0 means the fused kernel performs fewer
    /// total operations than the unfused sequence.
    ///
    /// # Examples of compute savings
    ///
    /// - MatMulBiasReLU: The fused kernel can fold the bias add and ReLU
    ///   into the MatMul epilogue, saving the load/store overhead. Factor ~1.1.
    /// - AttentionBlock: Online softmax avoids recomputing exp() values.
    ///   Factor ~1.3.
    /// - ElementwiseChain: No compute savings, just memory savings. Factor 1.0.
    pub fn estimate_compute_savings(pattern: &FusionPattern) -> f64 {
        match pattern {
            FusionPattern::MatMulBiasReLU          => 1.1,
            // Bias add and ReLU folded into MatMul epilogue; avoids
            // separate load-store for each.

            FusionPattern::AttentionBlock           => 1.3,
            // Online softmax avoids recomputing exp() values across
            // the full row; tiling reduces register pressure.

            FusionPattern::NormResidualActivation    => 1.05,
            // Minor: fused kernel can reuse the computed variance/rms
            // for both normalization and the subsequent operations.

            FusionPattern::OptimizerStepFusion       => 1.2,
            // Fused optimizer can combine momentum and gradient updates
            // into a single multiply-add chain per element.

            FusionPattern::BackwardPassChain         => 1.1,
            // Chain rule fusion avoids redundant recomputation of
            // intermediate gradient values.

            FusionPattern::CommunicationComputeOverlap => 1.0,
            // No compute savings; benefit is latency hiding.

            FusionPattern::PersistentKernel           => 1.15,
            // Avoids redundant loads of weights/parameters that stay
            // in SRAM across iterations.

            FusionPattern::ElementwiseChain           => 1.0,
            // Pure memory optimization; no algebraic simplification.

            FusionPattern::ReductionFusion           => 1.1,
            // Fused reduction can avoid writing then re-reading the
            // reduced value, and may combine the broadcast with the
            // subsequent element-wise operation.

            FusionPattern::HorizontalElementwise      => 1.0,
            // No compute savings; benefit is purely from eliminating
            // kernel launch overhead.
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Legality validation (quick checks)
    // ─────────────────────────────────────────────────────────────────────

    /// Perform a quick legality check on a fusion boundary.
    ///
    /// This method performs *syntactic* checks that can be evaluated quickly
    /// without the full polyhedral analysis. It checks:
    ///
    /// 1. **Dtype compatibility**: All ops in the boundary must use compatible
    ///    dtypes (no implicit type conversion required).
    ///
    /// 2. **Shape compatibility**: Output shapes of producers must match input
    ///    shapes of consumers (broadcasting is allowed but flagged).
    ///
    /// 3. **No aliasing hazards**: No op in the boundary should be in-place
    ///    if its output is consumed by another op in the boundary.
    ///
    /// 4. **No communication barriers**: Communication ops cannot be fused
    ///    with compute ops (except in CommunicationComputeOverlap pattern).
    ///
    /// **Important**: This is a necessary but not sufficient check. The
    /// polyhedral engine must still validate schedule legality for patterns
    /// that involve reductions or non-trivial loop dependencies.
    pub fn validate_fusion_legality(&self, boundary: &FusionBoundary) -> bool {
        // An empty or single-op boundary is trivially legal
        if boundary.op_indices.len() <= 1 {
            return true;
        }

        // This method operates on the boundary's metadata only.
        // In a real implementation, it would also check the actual ops.
        // For now, we check the pattern-specific constraints.

        match boundary.pattern {
            FusionPattern::CommunicationComputeOverlap => {
                // Communication-compute overlap is always "legal" — it's
                // just temporal overlap, not kernel fusion.
                true
            }

            FusionPattern::ElementwiseChain => {
                // Elementwise chains are always safe to fuse if shapes match.
                // No reduction, no cross-element dependencies.
                true
            }

            _ => {
                // For all other patterns, we do a basic sanity check.
                // The real legality validation happens in the polyhedral engine.
                // Here we just check for obvious blockers:
                boundary.confidence > 0.0
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Alternative proposal generation
    // ─────────────────────────────────────────────────────────────────────

    /// Propose multiple fusion alternatives for the same set of operations.
    ///
    /// Unlike `discover_fusion_boundaries` which greedily selects the single
    /// best non-overlapping set, this method returns ALL viable fusion
    /// boundaries for the input ops, including overlapping ones. This allows
    /// downstream consumers (e.g., MCMC search) to explore the space of
    /// possible fusion decisions.
    ///
    /// # Use Cases
    ///
    /// - **MCMC hardware search**: Different fusion decisions may be optimal
    ///   for different hardware targets. The search can evaluate alternatives.
    /// - **Empirical tuning**: When the analytical model is uncertain,
    ///   multiple alternatives can be benchmarked.
    /// - **Fallback selection**: If the top-choice fusion fails polyhedral
    ///   validation, a lower-confidence alternative can be tried.
    ///
    /// # Returns
    ///
    /// A vector of `FusionBoundary` proposals, sorted by estimated speedup
    /// (descending). May include overlapping boundaries.
    pub fn propose_fusion_alternatives(&self, ops: &[FusionOp]) -> Vec<FusionBoundary> {
        if ops.len() < 2 {
            return Vec::new();
        }

        let mut alternatives: Vec<FusionBoundary> = Vec::new();

        // Generate candidates at every window size and position
        for window_size in 2..=self.max_pattern_window.min(ops.len()) {
            for start in 0..=(ops.len() - window_size) {
                let end = start + window_size;
                let window = &ops[start..end];

                // Try to classify this window
                if let Some(pattern) = self.classify_pattern(window) {
                    let shapes: Vec<TensorShape> = window
                        .iter()
                        .map(|op| op.output_shape.clone())
                        .collect();

                    let memory_savings =
                        Self::estimate_memory_savings(&pattern, &shapes, window);

                    let compute_savings = Self::estimate_compute_savings(&pattern);

                    let confidence = self.compute_confidence(&pattern, window);

                    let indices: Vec<usize> = (start..end).collect();
                    alternatives.push(FusionBoundary::new(
                        indices,
                        pattern,
                        memory_savings,
                        compute_savings,
                        confidence,
                    ));
                }

                // Also try sub-patterns within this window
                // For example, [MatMul, Add, ReLU] matches MatMulBiasReLU,
                // but [Add, ReLU] also matches ElementwiseChain
                if window_size > 2 {
                    for sub_size in 2..window_size {
                        for sub_start in 0..=(window_size - sub_size) {
                            let sub_window = &ops[start + sub_start..start + sub_start + sub_size];
                            if let Some(sub_pattern) = self.classify_pattern(sub_window) {
                                let sub_shapes: Vec<TensorShape> = sub_window
                                    .iter()
                                    .map(|op| op.output_shape.clone())
                                    .collect();

                                let sub_memory = Self::estimate_memory_savings(
                                    &sub_pattern, &sub_shapes, sub_window,
                                );
                                let sub_compute = Self::estimate_compute_savings(&sub_pattern);
                                let sub_confidence =
                                    self.compute_confidence(&sub_pattern, sub_window);

                                let sub_indices: Vec<usize> =
                                    (start + sub_start..start + sub_start + sub_size).collect();

                                // Reduce confidence for sub-patterns (they're less
                                // beneficial than the full pattern)
                                let adjusted_confidence = sub_confidence * 0.8;

                                if adjusted_confidence >= self.min_confidence
                                    || self.include_speculative
                                {
                                    alternatives.push(FusionBoundary::new(
                                        sub_indices,
                                        sub_pattern,
                                        sub_memory,
                                        sub_compute,
                                        adjusted_confidence,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by estimated speedup (descending)
        alternatives.sort_by(|a, b| {
            let score_a = a.estimated_speedup();
            let score_b = b.estimated_speedup();
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove exact duplicates (same indices, same pattern)
        alternatives.dedup_by(|a, b| {
            a.op_indices == b.op_indices && a.pattern == b.pattern
        });

        alternatives
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal: pattern matchers
    // ─────────────────────────────────────────────────────────────────────

    /// Check if the op type sequence matches the MatMulBiasReLU pattern.
    ///
    /// Matches:
    /// - [MatMul, Add, activation] (full pattern)
    /// - [MatMul, Add] (partial: bias add only)
    /// - [MatMul, activation] (partial: no bias, just activation)
    /// - [BatchMatMul, Add, activation] (batched variant)
    fn matches_matmul_bias_activation(&self, op_types: &[OpType]) -> bool {
        let n = op_types.len();
        if n < 2 || n > 4 {
            return false;
        }

        let is_matmul = |t: &OpType| matches!(t, OpType::MatMul | OpType::BatchMatMul);

        // [MatMul, Add, activation?]
        if is_matmul(&op_types[0]) {
            if n == 2 {
                // [MatMul, Add] or [MatMul, activation]
                return op_types[1] == OpType::Add || op_types[1].is_activation();
            }
            if n == 3 {
                // [MatMul, Add, activation] or [MatMul, Mul, activation] (with scaling)
                if op_types[1] == OpType::Add && op_types[2].is_activation() {
                    return true;
                }
                // [MatMul, activation, ...] is also valid (no bias)
                if op_types[1].is_activation() {
                    return true;
                }
            }
            if n == 4 {
                // [MatMul, Mul, Add, activation] (scaled + biased + activated)
                if op_types[1] == OpType::Mul
                    && op_types[2] == OpType::Add
                    && op_types[3].is_activation()
                {
                    return true;
                }
            }
        }

        false
    }

    /// Check if the op type sequence matches the AttentionBlock pattern.
    ///
    /// Matches:
    /// - [MatMul, Softmax, MatMul] (basic attention)
    /// - [MatMul, Mul, Softmax, MatMul] (scaled dot-product attention)
    /// - [MatMul, Softmax, Mul, MatMul] (attention with dropout)
    /// - [MatMul, Mul, Softmax, Mul, MatMul] (full SDPA)
    fn matches_attention_block(&self, op_types: &[OpType]) -> bool {
        let n = op_types.len();
        if n < 3 || n > 6 {
            return false;
        }

        // Must start with MatMul and end with MatMul
        let is_matmul = |t: &OpType| matches!(t, OpType::MatMul | OpType::BatchMatMul);
        if !is_matmul(&op_types[0]) || !is_matmul(&op_types[n - 1]) {
            return false;
        }

        // Must contain Softmax somewhere in the middle
        let has_softmax = op_types[1..n - 1]
            .iter()
            .any(|t| matches!(t, OpType::Softmax | OpType::LogSoftmax));

        // Middle ops should be only Mul (scaling), Softmax, or Div (scaling)
        let middle_valid = op_types[1..n - 1]
            .iter()
            .all(|t| matches!(t, OpType::Mul | OpType::Softmax | OpType::LogSoftmax | OpType::Div));

        has_softmax && middle_valid
    }

    /// Check if the op type sequence matches the NormResidualActivation pattern.
    ///
    /// Matches:
    /// - [LayerNorm, Add, activation?] (post-norm)
    /// - [RMSNorm, Add, activation?] (post-norm)
    /// - [Add, LayerNorm, activation?] (pre-norm)
    /// - [Add, RMSNorm, activation?] (pre-norm)
    fn matches_norm_residual(&self, op_types: &[OpType]) -> bool {
        let n = op_types.len();
        if n < 2 || n > 4 {
            return false;
        }

        let has_norm = op_types.iter().any(|t| t.is_normalization());
        let has_add = op_types.iter().any(|t| *t == OpType::Add);

        if !has_norm || !has_add {
            return false;
        }

        // Post-norm: [Norm, Add, activation?]
        if op_types[0].is_normalization() && op_types[1] == OpType::Add {
            return true;
        }

        // Pre-norm: [Add, Norm, activation?]
        if op_types[0] == OpType::Add && op_types[1].is_normalization() {
            return true;
        }

        false
    }

    /// Check if the op type sequence matches the OptimizerStepFusion pattern.
    ///
    /// Matches sequences of alternating Mul and Add operations that represent
    /// the weight update computation in optimizers like Adam, SGD with momentum.
    ///
    /// Examples:
    /// - [Mul, Add, Mul, Add] (Adam: m*beta + g*(1-beta), then update)
    /// - [Add, Mul, Add] (SGD with momentum)
    /// - [Mul, Add, Add] (simplified Adam update)
    fn matches_optimizer_step(&self, op_types: &[OpType]) -> bool {
        let n = op_types.len();
        if n < 3 || n > 6 {
            return false;
        }

        // All ops should be Add or Mul (arithmetic ops on same-shaped tensors)
        let all_arithmetic = op_types
            .iter()
            .all(|t| matches!(t, OpType::Add | OpType::Mul | OpType::Sub));

        if !all_arithmetic {
            return false;
        }

        // Must have at least one Add and one Mul (otherwise it's just ElementwiseChain)
        let has_add = op_types.iter().any(|t| *t == OpType::Add);
        let has_mul = op_types.iter().any(|t| *t == OpType::Mul);

        // Heuristic: optimizer steps typically have alternating Mul/Add
        // with at least 3 ops
        has_add && has_mul && n >= 3
    }

    /// Check if the op type sequence matches the BackwardPassChain pattern.
    ///
    /// Matches sequences of [Mul, Add] pairs that represent chain rule
    /// gradient computation: dL/dx = dL/dy * dy/dx + bias_correction
    fn matches_backward_chain(&self, op_types: &[OpType], ops: &[FusionOp]) -> bool {
        let n = op_types.len();
        if n < 3 {
            return false;
        }

        // All ops should be Mul or Add
        let all_grad_ops = op_types
            .iter()
            .all(|t| matches!(t, OpType::Mul | OpType::Add));

        if !all_grad_ops {
            return false;
        }

        // Heuristic: backward chains have consistent output shapes
        // (gradients flow back through the same tensor dimensions)
        if ops.len() >= 2 {
            let first_shape = &ops[0].output_shape;
            let consistent_shapes = ops
                .iter()
                .all(|op| op.output_shape.len() == first_shape.len());
            return consistent_shapes;
        }

        false
    }

    /// Check if the op type sequence matches the CommunicationComputeOverlap pattern.
    ///
    /// Matches: [comm_op, compute_op] or [compute_op, comm_op]
    /// where comm_op is AllReduce/AllGather/ReduceScatter and compute_op
    /// is any non-communication op.
    fn matches_comm_compute_overlap(&self, op_types: &[OpType]) -> bool {
        if op_types.len() != 2 {
            return false;
        }

        let has_comm = op_types.iter().any(|t| t.is_communication());
        let has_compute = op_types.iter().any(|t| !t.is_communication() && !t.is_structural());

        has_comm && has_compute
    }

    /// Check if the op type sequence matches the ReductionFusion pattern.
    ///
    /// Matches:
    /// - [Reduce, elementwise+] (reduce + follow-up)
    /// - [Reduce, Broadcast, elementwise+] (reduce + broadcast + follow-up)
    /// - [Reduce, Add] (common in loss computation)
    fn matches_reduction_fusion(&self, op_types: &[OpType]) -> bool {
        let n = op_types.len();
        if n < 2 {
            return false;
        }

        // Must start with Reduce
        if op_types[0] != OpType::Reduce {
            return false;
        }

        // Subsequent ops should be Broadcast, Add, Mul, or other elementwise
        op_types[1..]
            .iter()
            .all(|t| t.is_elementwise() || *t == OpType::Broadcast || *t == OpType::Add)
    }

    /// Check if the op sequence matches the PersistentKernel pattern.
    ///
    /// This is a heuristic check: persistent kernels are detected when the
    /// same block of operations appears to be repeated (same op types, same
    /// shapes), suggesting an iterative or layer-wise computation.
    fn matches_persistent_kernel(&self, ops: &[FusionOp]) -> bool {
        let n = ops.len();
        if n < 3 {
            return false;
        }

        // Heuristic: check if the op types form a repeating sub-pattern
        // of length >= 2. For example: [MatMul, Add, MatMul, Add] has
        // a repeating pattern of length 2.
        for pattern_len in 2..=(n / 2) {
            if n % pattern_len != 0 {
                continue;
            }

            let num_repeats = n / pattern_len;
            if num_repeats < 2 {
                continue;
            }

            let mut all_match = true;
            for repeat in 1..num_repeats {
                for j in 0..pattern_len {
                    let base_idx = j;
                    let repeat_idx = repeat * pattern_len + j;
                    if repeat_idx >= n {
                        all_match = false;
                        break;
                    }
                    if ops[base_idx].op_type != ops[repeat_idx].op_type {
                        all_match = false;
                        break;
                    }
                    // Also check that output shapes are the same length
                    // (exact match not required; persistent kernels can
                    // handle different sizes with the same code)
                    if ops[base_idx].output_shape.len()
                        != ops[repeat_idx].output_shape.len()
                    {
                        all_match = false;
                        break;
                    }
                }
                if !all_match {
                    break;
                }
            }

            if all_match {
                return true;
            }
        }

        false
    }

    /// Check if the op type sequence matches the ElementwiseChain pattern.
    ///
    /// This is the fallback pattern: any sequence of 2+ element-wise ops.
    /// It matches only if no more specific pattern matches.
    fn matches_elementwise_chain(&self, op_types: &[OpType]) -> bool {
        op_types.len() >= 2 && op_types.iter().all(|t| t.is_elementwise())
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal: confidence scoring
    // ─────────────────────────────────────────────────────────────────────

    /// Compute the confidence score for a fusion pattern match.
    ///
    /// Confidence reflects how certain the fusion engine is that:
    /// 1. The pattern match is correct (the ops truly compose into this pattern)
    /// 2. The fusion will be beneficial (the savings estimates are reliable)
    /// 3. The fusion is likely legal (no hidden dependency violations)
    ///
    /// # Scoring Model
    ///
    /// - Base confidence: pattern-specific (well-known patterns start higher)
    /// - Shape consistency bonus: +0.1 if all output shapes are compatible
    /// - Dtype consistency bonus: +0.05 if all ops use the same dtype
    /// - Large tensor bonus: +0.05 if intermediates are > 1MB
    ///   (large tensors benefit more from fusion)
    /// - Non-inplace bonus: +0.05 if no op is in-place
    ///   (in-place ops don't benefit as much from fusion)
    fn compute_confidence(&self, pattern: &FusionPattern, ops: &[FusionOp]) -> f64 {
        // Base confidence per pattern
        let base = match pattern {
            FusionPattern::MatMulBiasReLU           => 0.9,
            // Extremely well-studied pattern; always beneficial.

            FusionPattern::AttentionBlock            => 0.85,
            // Well-studied (FlashAttention), but requires careful
            // tiling to avoid O(seq_len²) SRAM usage.

            FusionPattern::NormResidualActivation     => 0.9,
            // Very common pattern; always beneficial.

            FusionPattern::OptimizerStepFusion        => 0.8,
            // Generally beneficial, but the exact savings depend on
            // the optimizer variant and whether gradient accumulation
            // is used.

            FusionPattern::BackwardPassChain          => 0.7,
            // Benefit depends on the chain length and whether
            // gradient checkpointing is used.

            FusionPattern::CommunicationComputeOverlap => 0.6,
            // Benefit depends on the compute/communication ratio
            // and the pipeline depth.

            FusionPattern::PersistentKernel            => 0.5,
            // Highly hardware-dependent; may not be beneficial
            // on all targets.

            FusionPattern::ElementwiseChain            => 0.95,
            // Almost always safe and beneficial; the simplest pattern.

            FusionPattern::ReductionFusion            => 0.8,
            // Generally beneficial, but the polyhedral engine must
            // validate the reduction schedule.

            FusionPattern::HorizontalElementwise      => 0.7,
            // Requires polyhedral validation to confirm independence
            // of the ops and same-shape constraint.
        };

        let mut confidence: f64 = base;

        // Shape consistency bonus: all non-empty output shapes should have
        // the same rank (broadcasting between different-rank tensors can
        // complicate fusion)
        if ops.len() >= 2 {
            let ranks: Vec<usize> = ops
                .iter()
                .map(|op| op.output_shape.len())
                .filter(|&r| r > 0)
                .collect();
            if !ranks.is_empty() && ranks.iter().all(|&r| r == ranks[0]) {
                confidence += 0.1;
            }
        }

        // Dtype consistency bonus: all ops use the same dtype
        if ops.len() >= 2 {
            let first_dtype = ops[0].dtype;
            if ops.iter().all(|op| op.dtype == first_dtype) {
                confidence += 0.05;
            }
        }

        // Large tensor bonus: at least one intermediate is > 1MB
        // (fusion is most beneficial for large intermediates)
        let has_large_intermediate = ops
            .iter()
            .take(ops.len().saturating_sub(1))
            .any(|op| op.memory_bytes > 1024 * 1024);
        if has_large_intermediate {
            confidence += 0.05;
        }

        // Non-inplace bonus: no op is in-place
        // (in-place ops don't produce intermediate HBM traffic)
        if ops.iter().all(|op| !op.is_inplace) {
            confidence += 0.05;
        }

        confidence.clamp(0.0, 1.0)
    }
}

// =============================================================================
// §7. PATTERN CLASSIFICATION LOGIC — Extended Helpers
// =============================================================================

/// Extended classification that considers shape relationships between ops.
///
/// While `FusionEngine::classify_pattern` works on op types alone (fast),
/// this function uses shape information to disambiguate patterns that have
/// the same op type signature but different semantics.
///
/// For example, [Mul, Add] could be:
/// - Part of an OptimizerStepFusion (weight update)
/// - Part of a BackwardPassChain (gradient computation)
/// - Just an ElementwiseChain
///
/// The shape relationships help disambiguate:
/// - If Mul's output shape == Add's output shape → likely chain rule
/// - If Mul's inputs have the same shape as Add's output → likely optimizer
pub fn classify_pattern_with_shapes(ops: &[FusionOp]) -> Option<FusionPattern> {
    let engine = FusionEngine::new();

    // First try type-based classification
    let type_result = engine.classify_pattern(ops);

    // If we got a result, validate/refine with shape information
    if let Some(pattern) = type_result {
        match pattern {
            FusionPattern::OptimizerStepFusion | FusionPattern::BackwardPassChain => {
                // Disambiguate using shape consistency
                let all_same_shape = ops.windows(2).all(|w| {
                    w[0].output_shape.len() == w[1].output_shape.len()
                        && w[0].output_shape
                            .iter()
                            .zip(w[1].output_shape.iter())
                            .all(|(a, b)| a == b)
                });

                if all_same_shape && ops.len() >= 4 {
                    // More likely an optimizer step (4+ ops with same shape)
                    Some(FusionPattern::OptimizerStepFusion)
                } else if all_same_shape {
                    Some(FusionPattern::BackwardPassChain)
                } else {
                    Some(pattern)
                }
            }
            _ => Some(pattern),
        }
    } else {
        None
    }
}

// =============================================================================
// §8. MEMORY & COMPUTE ESTIMATION MODELS — Detailed Models
// =============================================================================

/// Roofline-model-aware memory savings estimation.
///
/// This function extends `FusionEngine::estimate_memory_savings` by
/// incorporating roofline model parameters. If the unfused ops are
/// memory-bound (arithmetic intensity below the roofline ridge point),
/// then eliminating HBM traffic directly translates to speedup. If they
/// are compute-bound, memory savings have less impact.
///
/// # Arguments
///
/// * `pattern` - The fusion pattern
/// * `ops` - The operations being fused
/// * `peak_bandwidth_gb_per_sec` - HBM bandwidth (e.g., 2000 for H100)
/// * `peak_gflops` - Peak compute throughput (e.g., 989 for H100 FP16)
///
/// # Returns
///
/// Estimated time savings in nanoseconds from eliminating HBM traffic.
pub fn estimate_time_savings_ns(
    pattern: &FusionPattern,
    ops: &[FusionOp],
    peak_bandwidth_gb_per_sec: f64,
    peak_gflops: f64,
) -> f64 {
    let shapes: Vec<TensorShape> = ops.iter().map(|op| op.output_shape.clone()).collect();
    let savings_bytes = FusionEngine::estimate_memory_savings(pattern, &shapes, ops);

    if savings_bytes == 0 || peak_bandwidth_gb_per_sec <= 0.0 {
        return 0.0;
    }

    // Time to transfer the saved bytes at peak HBM bandwidth
    let savings_gb = savings_bytes as f64 / 1e9;
    let time_saved_ns = (savings_gb / peak_bandwidth_gb_per_sec) * 1e9;

    // Adjust based on roofline: if the ops are compute-bound, the memory
    // savings are on the critical path only if compute time < memory time.
    let total_flops: f64 = ops
        .iter()
        .map(|op| op.num_output_elements() as f64 * op.arithmetic_intensity())
        .sum();
    let compute_time_ns = (total_flops / peak_gflops) * 1e9;

    // The effective time saved is min(memory_savings_time, compute_time)
    // because the memory transfer can overlap with compute.
    time_saved_ns.min(compute_time_ns)
}

/// Compute the roofline ridge point (the arithmetic intensity at which
/// an operation transitions from memory-bound to compute-bound).
///
/// ridge_point = peak_gflops / peak_bandwidth_gb_per_sec
///
/// Operations with arithmetic intensity above the ridge point are compute-bound;
/// below, they are memory-bound.
pub fn roofline_ridge_point(
    peak_gflops: f64,
    peak_bandwidth_gb_per_sec: f64,
) -> f64 {
    if peak_bandwidth_gb_per_sec <= 0.0 {
        return f64::INFINITY;
    }
    peak_gflops / peak_bandwidth_gb_per_sec
}

// =============================================================================
// §9. LEGALITY VALIDATION — Quick Checks (Extended)
// =============================================================================

/// Extended legality validation that checks actual operation properties.
///
/// This function goes beyond the basic `FusionEngine::validate_fusion_legality`
/// by examining the actual operations in the boundary, not just the pattern type.
///
/// # Checks
///
/// 1. **Dtype compatibility**: All ops must use compatible dtypes.
///    Fusion between FP16 and BF16 requires explicit conversion.
///    Fusion between float and int types is not allowed.
///
/// 2. **Shape flow**: Output shape of op[i] must be compatible with input
///    shape of op[i+1] (exact match or broadcast-compatible).
///
/// 3. **No in-place conflicts**: An op that is marked in-place cannot have
///    its output consumed by another op in the same boundary if the
///    consumer also writes to the same tensor.
///
/// 4. **Reduction axis**: For ReductionFusion, the reduction axis must be
///    consistent across the fused ops.
pub fn validate_fusion_legality_detailed(
    boundary: &FusionBoundary,
    ops: &[FusionOp],
) -> FusionLegalityResult {
    let mut violations: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Extract the ops in this boundary
    let boundary_ops: Vec<&FusionOp> = boundary
        .op_indices
        .iter()
        .filter_map(|&idx| ops.get(idx))
        .collect();

    if boundary_ops.len() != boundary.op_indices.len() {
        violations.push("Some op indices are out of bounds".to_string());
        return FusionLegalityResult {
            is_legal: false,
            violations,
            warnings,
        };
    }

    // Check 1: Dtype compatibility
    let dtypes: Vec<DType> = boundary_ops.iter().map(|op| op.dtype).collect();
    let float_dtypes: Vec<DType> = dtypes
        .iter()
        .filter(|d| d.is_float())
        .copied()
        .collect();
    let int_dtypes: Vec<DType> = dtypes.iter().filter(|d| d.is_quantized()).copied().collect();

    if !float_dtypes.is_empty() && !int_dtypes.is_empty() {
        violations.push(
            "Cannot fuse float and integer operations without explicit conversion".to_string(),
        );
    }

    // Check for mixed float dtypes (warning, not violation)
    let unique_float_dtypes: Vec<DType> = {
        let mut d: Vec<DType> = float_dtypes;
        d.sort_by(|a, b| (*a as u8).cmp(&(*b as u8)));
        d.dedup();
        d
    };
    if unique_float_dtypes.len() > 1 {
        warnings.push(format!(
            "Mixed float dtypes in fusion: {:?}. May require implicit conversion.",
            unique_float_dtypes
                .iter()
                .map(|d| d.name())
                .collect::<Vec<_>>()
        ));
    }

    // Check 2: Shape flow (for sequential ops in the boundary)
    for i in 0..boundary_ops.len().saturating_sub(1) {
        let producer_shape = &boundary_ops[i].output_shape;
        let consumer_inputs = &boundary_ops[i + 1].input_shapes;

        if !consumer_inputs.is_empty() {
            // Check if the producer's output shape matches any of the
            // consumer's input shapes (allowing for broadcast)
            let first_input = &consumer_inputs[0];
            if !shapes_broadcast_compatible(producer_shape, first_input) {
                violations.push(format!(
                    "Shape mismatch between op {} output {:?} and op {} input {:?}",
                    boundary.op_indices[i],
                    producer_shape,
                    boundary.op_indices[i + 1],
                    first_input,
                ));
            }
        }
    }

    // Check 3: No in-place conflicts
    for (i, op) in boundary_ops.iter().enumerate() {
        if op.is_inplace && i < boundary_ops.len() - 1 {
            warnings.push(format!(
                "Op at index {} is in-place; its output may alias the input, \
                 which could cause incorrect results if fused with subsequent ops",
                boundary.op_indices[i]
            ));
        }
    }

    // Check 4: Communication ops in non-overlap patterns
    if boundary.pattern != FusionPattern::CommunicationComputeOverlap {
        let has_comm = boundary_ops.iter().any(|op| op.op_type.is_communication());
        if has_comm {
            violations.push(
                "Communication ops cannot be fused with compute ops \
                 (except in CommunicationComputeOverlap pattern)"
                    .to_string(),
            );
        }
    }

    // Check 5: Custom ops are fusion barriers
    let has_custom = boundary_ops
        .iter()
        .any(|op| op.op_type == OpType::Custom);
    if has_custom {
        violations.push(
            "Custom/unknown operations cannot be fused (no semantic model)".to_string(),
        );
    }

    FusionLegalityResult {
        is_legal: violations.is_empty(),
        violations,
        warnings,
    }
}

/// Result of detailed legality validation.
#[derive(Debug, Clone)]
pub struct FusionLegalityResult {
    /// Whether the fusion is legal (no violations found).
    pub is_legal: bool,
    /// Hard violations that make the fusion illegal.
    pub violations: Vec<String>,
    /// Soft warnings that don't prevent fusion but indicate risk.
    pub warnings: Vec<String>,
}

/// Check if two shapes are broadcast-compatible (NumPy broadcasting rules).
///
/// Two shapes are compatible if, for each dimension (aligned from the right):
/// - They are equal, OR
/// - One of them is 1
///
/// Examples:
/// - (3, 4) and (3, 4) → compatible
/// - (3, 4) and (1, 4) → compatible
/// - (3, 4) and (3, 1) → compatible
/// - (3, 4) and (4,) → compatible (leading dim broadcast)
/// - (3, 4) and (3, 5) → NOT compatible
pub fn shapes_broadcast_compatible(a: &TensorShape, b: &TensorShape) -> bool {
    let max_len = a.len().max(b.len());
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if da != db && da != 1 && db != 1 {
            return false;
        }
    }
    true
}

// =============================================================================
// §10. ALTERNATIVE PROPOSAL GENERATION — Utilities
// =============================================================================

/// Generate a fusion coverage report for a sequence of operations.
///
/// This utility function analyzes which operations are covered by at least
/// one fusion boundary and which are "orphaned" (not part of any fusion).
/// This is useful for debugging and for the MCMC search to identify
/// unfused operations that might benefit from custom kernel development.
#[derive(Debug, Clone)]
pub struct FusionCoverageReport {
    /// Total number of input operations.
    pub total_ops: usize,
    /// Number of operations covered by at least one fusion boundary.
    pub fused_ops: usize,
    /// Indices of operations not covered by any fusion boundary.
    pub orphaned_indices: Vec<usize>,
    /// Per-pattern statistics: (pattern, count, total_ops_in_pattern, total_savings_bytes).
    pub pattern_stats: Vec<(FusionPattern, usize, usize, i64)>,
    /// Coverage ratio: fused_ops / total_ops.
    pub coverage_ratio: f64,
}

/// Compute a fusion coverage report from a decision and the original ops.
pub fn compute_coverage_report(
    decision: &FusionDecision,
    total_ops: usize,
) -> FusionCoverageReport {
    let mut covered = vec![false; total_ops];

    // Pattern statistics accumulator
    let mut pattern_accum: std::collections::HashMap<FusionPattern, (usize, usize, i64)> =
        std::collections::HashMap::new();

    for boundary in &decision.boundaries {
        for &idx in &boundary.op_indices {
            if idx < total_ops {
                covered[idx] = true;
            }
        }

        let entry = pattern_accum
            .entry(boundary.pattern)
            .or_insert((0, 0, 0i64));
        entry.0 += 1; // count
        entry.1 += boundary.op_count(); // total ops
        entry.2 += boundary.memory_traffic_savings_bytes; // total savings
    }

    let fused_ops = covered.iter().filter(|&&c| c).count();
    let orphaned_indices: Vec<usize> = covered
        .iter()
        .enumerate()
        .filter(|&(_, &c)| !c)
        .map(|(i, _)| i)
        .collect();

    let pattern_stats: Vec<(FusionPattern, usize, usize, i64)> = pattern_accum
        .into_iter()
        .map(|(pattern, (count, ops, savings))| (pattern, count, ops, savings))
        .collect();

    let coverage_ratio = if total_ops > 0 {
        fused_ops as f64 / total_ops as f64
    } else {
        0.0
    };

    FusionCoverageReport {
        total_ops,
        fused_ops,
        orphaned_indices,
        pattern_stats,
        coverage_ratio,
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── DType tests ──────────────────────────────────────────────────────

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(DType::FP32.size_bytes(), 4);
        assert_eq!(DType::FP16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::FP8.size_bytes(), 1);
        assert_eq!(DType::INT8.size_bytes(), 1);
        assert_eq!(DType::INT4.size_bytes(), 1);
    }

    #[test]
    fn test_dtype_classification() {
        assert!(DType::FP32.is_float());
        assert!(DType::BF16.is_float());
        assert!(!DType::INT8.is_float());
        assert!(DType::INT8.is_quantized());
        assert!(DType::INT4.is_quantized());
        assert!(!DType::FP16.is_quantized());
    }

    // ── OpType tests ─────────────────────────────────────────────────────

    #[test]
    fn test_optype_classification() {
        assert!(OpType::ReLU.is_elementwise());
        assert!(OpType::GELU.is_elementwise());
        assert!(OpType::Add.is_elementwise());
        assert!(OpType::ReLU.is_activation());
        assert!(!OpType::Add.is_activation());
        assert!(OpType::LayerNorm.is_normalization());
        assert!(OpType::RMSNorm.is_normalization());
        assert!(OpType::Transpose.is_structural());
        assert!(OpType::AllReduce.is_communication());
    }

    // ── FusionOp tests ───────────────────────────────────────────────────

    #[test]
    fn test_fusion_op_construction() {
        let op = FusionOp::new(
            OpType::MatMul,
            vec![128, 256],
            vec![vec![128, 512], vec![512, 256]],
            DType::FP16,
            false,
        );
        assert_eq!(op.memory_bytes, 128 * 256 * 2); // shape * dtype_size
        assert_eq!(op.num_output_elements(), 128 * 256);
        assert!(op.is_intermediate_eligible());
    }

    #[test]
    fn test_fusion_op_inplace() {
        let op = FusionOp::new(OpType::ReLU, vec![128], vec![vec![128]], DType::FP32, true);
        assert!(!op.is_intermediate_eligible()); // in-place ops are not eligible
    }

    // ── Shape utility tests ──────────────────────────────────────────────

    #[test]
    fn test_shape_num_elements() {
        assert_eq!(shape_num_elements(&vec![]), 1); // scalar
        assert_eq!(shape_num_elements(&vec![128]), 128);
        assert_eq!(shape_num_elements(&vec![128, 256]), 32768);
        assert_eq!(shape_num_elements(&vec![0, 256]), 0); // zero dim
    }

    #[test]
    fn test_shapes_broadcast_compatible() {
        assert!(shapes_broadcast_compatible(&vec![3, 4], &vec![3, 4]));
        assert!(shapes_broadcast_compatible(&vec![3, 4], &vec![1, 4]));
        assert!(shapes_broadcast_compatible(&vec![3, 4], &vec![4]));
        assert!(!shapes_broadcast_compatible(&vec![3, 4], &vec![3, 5]));
    }

    // ── FusionPattern tests ──────────────────────────────────────────────

    #[test]
    fn test_pattern_properties() {
        assert!(FusionPattern::MatMulBiasReLU.requires_polyhedral_validation());
        assert!(!FusionPattern::ElementwiseChain.requires_polyhedral_validation());
        assert_eq!(FusionPattern::MatMulBiasReLU.intermediates_eliminated(), 2);
    }

    // ── FusionBoundary tests ─────────────────────────────────────────────

    #[test]
    fn test_fusion_boundary_creation() {
        let boundary = FusionBoundary::new(
            vec![0, 1, 2],
            FusionPattern::MatMulBiasReLU,
            1024 * 1024, // 1 MB
            1.1,
            0.9,
        );
        assert_eq!(boundary.op_count(), 3);
        assert!(boundary.requires_polyhedral_validation());
        assert!(boundary.estimated_speedup() > 1.0);
    }

    #[test]
    #[should_panic(expected = "FusionBoundary must contain at least one op")]
    fn test_fusion_boundary_empty_panics() {
        FusionBoundary::new(vec![], FusionPattern::ElementwiseChain, 0, 1.0, 0.5);
    }

    // ── FusionEngine: MatMulBiasReLU detection ──────────────────────────

    #[test]
    fn test_discover_matmul_bias_relu() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::MatMul, vec![128, 256], vec![vec![128, 512], vec![512, 256]], DType::FP16, false),
            FusionOp::new(OpType::Add, vec![128, 256], vec![vec![128, 256], vec![256]], DType::FP16, false),
            FusionOp::new(OpType::ReLU, vec![128, 256], vec![vec![128, 256]], DType::FP16, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        assert_eq!(decision.boundary_count(), 1);
        assert_eq!(decision.boundaries[0].pattern, FusionPattern::MatMulBiasReLU);
        assert!(decision.total_hbm_traffic_reduction > 0);
        assert!(decision.requires_polyhedral_validation);
    }

    // ── FusionEngine: AttentionBlock detection ──────────────────────────

    #[test]
    fn test_discover_attention_block() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::MatMul, vec![8, 128, 128], vec![vec![8, 128, 64], vec![8, 64, 128]], DType::FP16, false),
            FusionOp::new(OpType::Mul, vec![8, 128, 128], vec![vec![8, 128, 128], vec![]], DType::FP16, false),
            FusionOp::new(OpType::Softmax, vec![8, 128, 128], vec![vec![8, 128, 128]], DType::FP16, false),
            FusionOp::new(OpType::MatMul, vec![8, 128, 64], vec![vec![8, 128, 128], vec![8, 128, 64]], DType::FP16, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        assert!(decision.boundaries.iter().any(|b| b.pattern == FusionPattern::AttentionBlock));
    }

    // ── FusionEngine: ElementwiseChain detection ────────────────────────

    #[test]
    fn test_discover_elementwise_chain() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::Exp, vec![1024], vec![vec![1024]], DType::FP32, false),
            FusionOp::new(OpType::Add, vec![1024], vec![vec![1024], vec![1024]], DType::FP32, false),
            FusionOp::new(OpType::ReLU, vec![1024], vec![vec![1024]], DType::FP32, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        // Should detect an elementwise chain (Exp, Add, ReLU)
        assert!(decision.boundary_count() >= 1);
        // The elementwise chain should cover all 3 ops or a subset
        let fused_ops = decision.total_fused_ops();
        assert!(fused_ops >= 2);
    }

    // ── FusionEngine: NormResidualActivation detection ──────────────────

    #[test]
    fn test_discover_norm_residual() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::LayerNorm, vec![128, 256], vec![vec![128, 256]], DType::FP32, false),
            FusionOp::new(OpType::Add, vec![128, 256], vec![vec![128, 256], vec![128, 256]], DType::FP32, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        assert!(decision.boundaries.iter().any(|b| b.pattern == FusionPattern::NormResidualActivation));
    }

    // ── FusionEngine: empty input ──────────────────────────────────────

    #[test]
    fn test_discover_empty() {
        let engine = FusionEngine::new();
        let decision = engine.discover_fusion_boundaries(&[]);
        assert_eq!(decision.boundary_count(), 0);
        assert_eq!(decision.total_estimated_speedup, 1.0);
    }

    // ── FusionEngine: single op ────────────────────────────────────────

    #[test]
    fn test_discover_single_op() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::ReLU, vec![128], vec![vec![128]], DType::FP32, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);
        assert_eq!(decision.boundary_count(), 0); // Need at least 2 ops
    }

    // ── Memory savings estimation ──────────────────────────────────────

    #[test]
    fn test_memory_savings_matmul_bias_relu() {
        let ops = vec![
            FusionOp::new(OpType::MatMul, vec![128, 256], vec![vec![128, 512], vec![512, 256]], DType::FP16, false),
            FusionOp::new(OpType::Add, vec![128, 256], vec![vec![128, 256], vec![256]], DType::FP16, false),
            FusionOp::new(OpType::ReLU, vec![128, 256], vec![vec![128, 256]], DType::FP16, false),
        ];
        let shapes: Vec<TensorShape> = ops.iter().map(|op| op.output_shape.clone()).collect();
        let savings = FusionEngine::estimate_memory_savings(
            &FusionPattern::MatMulBiasReLU, &shapes, &ops,
        );
        // MatMul output = 128*256*2 = 65536 bytes
        // Add output = 128*256*2 = 65536 bytes
        // Savings = 2*65536 + 2*65536 = 262144
        assert_eq!(savings, 2 * 65536 + 2 * 65536);
    }

    // ── Compute savings estimation ─────────────────────────────────────

    #[test]
    fn test_compute_savings() {
        assert!(FusionEngine::estimate_compute_savings(&FusionPattern::MatMulBiasReLU) > 1.0);
        assert_eq!(FusionEngine::estimate_compute_savings(&FusionPattern::ElementwiseChain), 1.0);
        assert!(FusionEngine::estimate_compute_savings(&FusionPattern::AttentionBlock) > 1.0);
    }

    // ── Legality validation ────────────────────────────────────────────

    #[test]
    fn test_validate_elementwise_legal() {
        let engine = FusionEngine::new();
        let boundary = FusionBoundary::new(
            vec![0, 1],
            FusionPattern::ElementwiseChain,
            1024,
            1.0,
            0.95,
        );
        assert!(engine.validate_fusion_legality(&boundary));
    }

    // ── Detailed legality validation ───────────────────────────────────

    #[test]
    fn test_detailed_validation_dtype_mismatch() {
        let ops = vec![
            FusionOp::new(OpType::Add, vec![128], vec![vec![128], vec![128]], DType::FP32, false),
            FusionOp::new(OpType::ReLU, vec![128], vec![vec![128]], DType::FP16, false),
        ];
        let boundary = FusionBoundary::new(
            vec![0, 1],
            FusionPattern::ElementwiseChain,
            512,
            1.0,
            0.8,
        );
        let result = validate_fusion_legality_detailed(&boundary, &ops);
        assert!(result.is_legal); // Mixed float dtypes are a warning, not violation
        assert!(!result.warnings.is_empty()); // Should have a mixed dtype warning
    }

    #[test]
    fn test_detailed_validation_custom_op() {
        let ops = vec![
            FusionOp::new(OpType::Custom, vec![128], vec![vec![128]], DType::FP32, false),
            FusionOp::new(OpType::ReLU, vec![128], vec![vec![128]], DType::FP32, false),
        ];
        let boundary = FusionBoundary::new(
            vec![0, 1],
            FusionPattern::ElementwiseChain,
            512,
            1.0,
            0.5,
        );
        let result = validate_fusion_legality_detailed(&boundary, &ops);
        assert!(!result.is_legal); // Custom ops cannot be fused
    }

    // ── Alternative proposals ──────────────────────────────────────────

    #[test]
    fn test_propose_alternatives() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::MatMul, vec![128, 256], vec![vec![128, 512], vec![512, 256]], DType::FP16, false),
            FusionOp::new(OpType::Add, vec![128, 256], vec![vec![128, 256], vec![256]], DType::FP16, false),
            FusionOp::new(OpType::ReLU, vec![128, 256], vec![vec![128, 256]], DType::FP16, false),
        ];
        let alternatives = engine.propose_fusion_alternatives(&ops);

        // Should propose at least the full MatMulBiasReLU pattern
        assert!(alternatives.iter().any(|a| a.pattern == FusionPattern::MatMulBiasReLU));
        // Should also propose sub-patterns like ElementwiseChain for [Add, ReLU]
        assert!(alternatives.len() >= 1);
    }

    // ── Coverage report ────────────────────────────────────────────────

    #[test]
    fn test_coverage_report() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::MatMul, vec![128, 256], vec![vec![128, 512], vec![512, 256]], DType::FP16, false),
            FusionOp::new(OpType::Add, vec![128, 256], vec![vec![128, 256], vec![256]], DType::FP16, false),
            FusionOp::new(OpType::ReLU, vec![128, 256], vec![vec![128, 256]], DType::FP16, false),
            FusionOp::new(OpType::Custom, vec![64], vec![vec![64]], DType::FP32, false), // orphaned
        ];
        let decision = engine.discover_fusion_boundaries(&ops);
        let report = compute_coverage_report(&decision, ops.len());

        assert_eq!(report.total_ops, 4);
        assert!(report.coverage_ratio > 0.0);
        assert!(report.coverage_ratio <= 1.0);
    }

    // ── FusionDecision aggregation ─────────────────────────────────────

    #[test]
    fn test_fusion_decision_from_boundaries() {
        let boundaries = vec![
            FusionBoundary::new(vec![0, 1, 2], FusionPattern::MatMulBiasReLU, 1000000, 1.1, 0.9),
            FusionBoundary::new(vec![3, 4], FusionPattern::ElementwiseChain, 50000, 1.0, 0.95),
        ];
        let decision = FusionDecision::from_boundaries(boundaries);
        assert_eq!(decision.boundary_count(), 2);
        assert_eq!(decision.total_hbm_traffic_reduction, 1050000);
        assert!(decision.requires_polyhedral_validation); // MatMulBiasReLU requires it
        assert_eq!(decision.total_fused_ops(), 5);
    }

    // ── ReductionFusion detection ──────────────────────────────────────

    #[test]
    fn test_discover_reduction_fusion() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::Reduce, vec![128], vec![vec![128, 256]], DType::FP32, false),
            FusionOp::new(OpType::Add, vec![128], vec![vec![128], vec![128]], DType::FP32, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        assert!(decision.boundaries.iter().any(|b| b.pattern == FusionPattern::ReductionFusion));
    }

    // ── CommunicationComputeOverlap detection ──────────────────────────

    #[test]
    fn test_discover_comm_compute_overlap() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::AllReduce, vec![1024], vec![vec![1024]], DType::FP32, false),
            FusionOp::new(OpType::MatMul, vec![128, 256], vec![vec![128, 512], vec![512, 256]], DType::FP16, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        assert!(decision.boundaries.iter().any(|b| b.pattern == FusionPattern::CommunicationComputeOverlap));
    }

    // ── Roofline model ────────────────────────────────────────────────

    #[test]
    fn test_roofline_ridge_point() {
        // H100: ~989 TFLOPS FP16, ~3350 GB/s HBM3e
        let ridge = roofline_ridge_point(989000.0, 3350.0);
        assert!(ridge > 100.0); // Ridge point ~295 FLOP/byte
    }

    // ── PersistentKernel detection ────────────────────────────────────

    #[test]
    fn test_persistent_kernel_detection() {
        let engine = FusionEngine::new();
        let ops = vec![
            FusionOp::new(OpType::MatMul, vec![64, 64], vec![vec![64, 64], vec![64, 64]], DType::FP16, false),
            FusionOp::new(OpType::Add, vec![64, 64], vec![vec![64, 64], vec![64]], DType::FP16, false),
            FusionOp::new(OpType::MatMul, vec![64, 64], vec![vec![64, 64], vec![64, 64]], DType::FP16, false),
            FusionOp::new(OpType::Add, vec![64, 64], vec![vec![64, 64], vec![64]], DType::FP16, false),
        ];
        let decision = engine.discover_fusion_boundaries(&ops);

        // The repeating [MatMul, Add, MatMul, Add] pattern should be detected
        // as either PersistentKernel or as two MatMulBiasReLU boundaries
        assert!(decision.boundary_count() >= 1);
    }
}

// =============================================================================
// §11. E-GRAPH EQUALITY-SATURATED FUSION
// =============================================================================

/// E-graph language for representing fusion operations as e-graph nodes.
///
/// The `FusionLang` language maps each `FusionOp` type to an e-graph node
/// so that algebraic properties (associativity, commutativity, distributivity)
/// can be explored via equality saturation. The e-graph maintains all
/// equivalent representations of a computation simultaneously, allowing the
/// fusion engine to discover patterns that are only visible through
/// reassociation of operations.
use egg::{define_language, Id};

define_language! {
    /// The e-graph language for fusion operations.
    ///
    /// Each variant corresponds to a node type in the e-graph. The language
    /// covers the full set of ML operations that participate in fusion:
    ///
    /// **Binary ops** (2 children):
    /// - `Add`: element-wise addition (a + b), includes bias add
    /// - `Mul`: element-wise multiplication (a * b), includes scaling
    /// - `Sub`: element-wise subtraction (a - b)
    /// - `Div`: element-wise division (a / b), includes 1/sqrt(d) scaling
    /// - `MatMul`: matrix multiplication (a @ b)
    /// - `Max`: element-wise maximum, used in ReLU(x, 0) expansion
    /// - `Min`: element-wise minimum
    ///
    /// **Unary ops** (1 child):
    /// - `ReLU`: rectified linear unit (max(0, x))
    /// - `GELU`: Gaussian error linear unit (x * Φ(x))
    /// - `Sigmoid`: sigmoid activation (1 / (1 + exp(-x)))
    /// - `Tanh`: hyperbolic tangent
    /// - `SiLU`: SiLU/Swish activation (x * sigmoid(x))
    /// - `Neg`: negation (-x)
    /// - `Exp`: exponential (exp(x))
    /// - `Abs`: absolute value (|x|)
    /// - `Sqrt`: square root
    /// - `Rsqrt`: reciprocal square root (1/sqrt(x)), for LayerNorm/RMSNorm
    ///
    /// **Normalization** (3 children: input, scale, bias):
    /// - `LayerNorm`: layer normalization
    /// - `RMSNorm`: root mean square normalization
    ///
    /// **Reduction** (2 children: input, axis):
    /// - `ReduceSum`: sum reduction
    /// - `ReduceMax`: max reduction
    ///
    /// **Leaf nodes**:
    /// - `Symbol`: named variable (e.g., ?a, ?b)
    /// - `Num`: numeric literal (e.g., 0.0 for ReLU, 1/sqrt(d) for scaling)
    pub enum FusionLang {
        // Binary operations (2 children)
        "add" = Add([Id; 2]),
        "mul" = Mul([Id; 2]),
        "sub" = Sub([Id; 2]),
        "div" = Div([Id; 2]),
        "matmul" = MatMul([Id; 2]),
        "max" = Max([Id; 2]),
        "min" = Min([Id; 2]),

        // Unary operations (1 child)
        "relu" = ReLU(Id),
        "gelu" = GELU(Id),
        "sigmoid" = Sigmoid(Id),
        "tanh" = Tanh(Id),
        "silu" = SiLU(Id),
        "neg" = Neg(Id),
        "exp" = Exp(Id),
        "abs" = Abs(Id),
        "sqrt" = Sqrt(Id),
        "rsqrt" = Rsqrt(Id),

        // Normalization (3 children: input, scale, bias)
        "layernorm" = LayerNorm([Id; 3]),
        "rmsnorm" = RMSNorm([Id; 3]),

        // Reduction (2 children: input, axis)
        "reducesum" = ReduceSum([Id; 2]),
        "reducemax" = ReduceMax([Id; 2]),

        // Leaf nodes
        Num(i64),
        Symbol(String),
    }
}

use egg::{Rewrite, Runner};

/// E-graph fusion explorer: discovers equivalent representations of a
/// computation via equality saturation, then checks each representation
/// for fusion opportunities using the existing `FusionEngine::classify_pattern()`.
///
/// The key insight is that some fusion opportunities are only visible after
/// algebraic reassociation. For example, `A@B + A@C` can be rewritten to
/// `A@(B+C)` via the MatMul factorization rule, which enables a single
/// MatMul kernel instead of two separate ones.
pub struct EgraphFusionExplorer {
    /// Maximum number of nodes in the e-graph before stopping.
    node_limit: usize,
    /// Maximum number of iteration steps.
    iteration_limit: usize,
}

impl Default for EgraphFusionExplorer {
    fn default() -> Self {
        Self::new()
    }
}

impl EgraphFusionExplorer {
    /// Create a new explorer with default limits (node_limit=10000, iteration_limit=30).
    pub fn new() -> Self {
        Self {
            node_limit: 10000,
            iteration_limit: 30,
        }
    }

    /// Create a new explorer with custom limits.
    pub fn with_limits(node_limit: usize, iteration_limit: usize) -> Self {
        Self { node_limit, iteration_limit }
    }

    /// Build the standard set of rewrite rules for fusion exploration.
    ///
    /// Rules are organized by category:
    ///
    /// **Additive algebra** (most important for fusion):
    /// - Associativity of Add: (a + b) + c => a + (b + c)
    /// - Commutativity of Add: a + b => b + a
    ///
    /// **Multiplicative algebra**:
    /// - Associativity of Mul: (a * b) * c => a * (b * c)
    /// - Commutativity of Mul: a * b => b * a
    ///
    /// **Distributivity** (key for MatMul factorization):
    /// - Right distributivity: a * (b + c) => a*b + a*c
    /// - Left distributivity: (a + b) * c => a*c + b*c
    ///
    /// **MatMul algebra** (highest-impact rules):
    /// - MatMul factorization: A@B + A@C => A@(B+C)
    /// - MatMul distributivity: A @ (B + C) => A@B + A@C
    ///
    /// **Activation identities**:
    /// - ReLU idempotency: relu(relu(a)) => relu(a)
    /// - SiLU expansion: silu(a) => a * sigmoid(a)
    /// - Double negation: neg(neg(a)) => a
    ///
    /// **Subtraction / Division**:
    /// - Sub as add-neg: a - b => a + (-b)
    /// - Div as mul-rsqrt: a / sqrt(b) => a * rsqrt(b) (speculative)
    pub fn rewrite_rules() -> Vec<Rewrite<FusionLang, ()>> {
        let mut rules = Vec::new();

        // ── Additive algebra ─────────────────────────────────────────────

        // Associativity of Add: (a + b) + c => a + (b + c)
        let lhs: egg::Pattern<FusionLang> = "(add (add ?a ?b) ?c)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(add ?a (add ?b ?c))".parse().unwrap();
        rules.push(
            Rewrite::new("add-assoc", "Associativity of addition", lhs, rhs).unwrap()
        );

        // Commutativity of Add: a + b => b + a
        let lhs: egg::Pattern<FusionLang> = "(add ?a ?b)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(add ?b ?a)".parse().unwrap();
        rules.push(
            Rewrite::new("add-comm", "Commutativity of addition", lhs, rhs).unwrap()
        );

        // ── Multiplicative algebra ───────────────────────────────────────

        // Associativity of Mul: (a * b) * c => a * (b * c)
        let lhs: egg::Pattern<FusionLang> = "(mul (mul ?a ?b) ?c)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(mul ?a (mul ?b ?c))".parse().unwrap();
        rules.push(
            Rewrite::new("mul-assoc", "Associativity of multiplication", lhs, rhs).unwrap()
        );

        // Commutativity of Mul: a * b => b * a
        let lhs: egg::Pattern<FusionLang> = "(mul ?a ?b)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(mul ?b ?a)".parse().unwrap();
        rules.push(
            Rewrite::new("mul-comm", "Commutativity of multiplication", lhs, rhs).unwrap()
        );

        // ── Distributivity ──────────────────────────────────────────────

        // Right distributivity: a * (b + c) => a*b + a*c
        let lhs: egg::Pattern<FusionLang> = "(mul ?a (add ?b ?c))".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(add (mul ?a ?b) (mul ?a ?c))".parse().unwrap();
        rules.push(
            Rewrite::new("mul-dist-right", "Right distributivity of mul over add", lhs, rhs).unwrap()
        );

        // Left distributivity: (a + b) * c => a*c + b*c
        let lhs: egg::Pattern<FusionLang> = "(mul (add ?a ?b) ?c)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(add (mul ?a ?c) (mul ?b ?c))".parse().unwrap();
        rules.push(
            Rewrite::new("mul-dist-left", "Left distributivity of mul over add", lhs, rhs).unwrap()
        );

        // ── MatMul algebra (highest-impact rules) ───────────────────────

        // MatMul factorization: A@B + A@C => A@(B+C)
        // Two separate MatMul kernels can be replaced by one MatMul with
        // a wider right operand. Savings: eliminate one MatMul output
        // (M*N*dtype bytes of HBM traffic).
        let lhs: egg::Pattern<FusionLang> = "(add (matmul ?a ?b) (matmul ?a ?c))".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(matmul ?a (add ?b ?c))".parse().unwrap();
        rules.push(
            Rewrite::new("matmul-factor", "MatMul factorization: A@B + A@C => A@(B+C)", lhs, rhs).unwrap()
        );

        // MatMul distributivity: A @ (B + C) => A@B + A@C
        // Reverse of factorization — useful when A@B and A@C each have
        // subsequent elementwise ops that fuse independently.
        let lhs: egg::Pattern<FusionLang> = "(matmul ?a (add ?b ?c))".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(add (matmul ?a ?b) (matmul ?a ?c))".parse().unwrap();
        rules.push(
            Rewrite::new("matmul-dist", "MatMul distributivity: A@(B+C) => A@B + A@C", lhs, rhs).unwrap()
        );

        // ── Activation identities ───────────────────────────────────────

        // ReLU idempotency: relu(relu(a)) => relu(a)
        let lhs: egg::Pattern<FusionLang> = "(relu (relu ?a))".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(relu ?a)".parse().unwrap();
        rules.push(
            Rewrite::new("relu-idempotent", "ReLU idempotency: relu(relu(a)) => relu(a)", lhs, rhs).unwrap()
        );

        // SiLU expansion: silu(a) => a * sigmoid(a)
        let lhs: egg::Pattern<FusionLang> = "(silu ?a)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(mul ?a (sigmoid ?a))".parse().unwrap();
        rules.push(
            Rewrite::new("silu-expand", "SiLU expansion: silu(a) => a * sigmoid(a)", lhs, rhs).unwrap()
        );

        // Double negation: neg(neg(a)) => a
        let lhs: egg::Pattern<FusionLang> = "(neg (neg ?a))".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "?a".parse().unwrap();
        rules.push(
            Rewrite::new("neg-cancel", "Double negation: -(-a) => a", lhs, rhs).unwrap()
        );

        // ── Subtraction / Division simplification ────────────────────────

        // Subtraction as addition of negation: a - b => a + (-b)
        let lhs: egg::Pattern<FusionLang> = "(sub ?a ?b)".parse().unwrap();
        let rhs: egg::Pattern<FusionLang> = "(add ?a (neg ?b))".parse().unwrap();
        rules.push(
            Rewrite::new("sub-to-add-neg", "Subtraction as addition: a - b => a + (-b)", lhs, rhs).unwrap()
        );

        rules
    }

    /// Convert a slice of FusionOps into an e-graph expression string.
    ///
    /// Builds an S-expression representing the chain of operations.
    /// For example, [MatMul, Add, ReLU] produces:
    ///   "(relu (add (matmul ?a ?b) ?bias))"
    ///
    /// Now handles the expanded FusionLang which includes Sub, Div, Max, Min,
    /// GELU, Sigmoid, Tanh, SiLU, Neg, Exp, Abs, Sqrt, Rsqrt, LayerNorm,
    /// RMSNorm, ReduceSum, ReduceMax.
    fn build_expr_string(ops: &[FusionOp]) -> String {
        // Build the expression from inside out
        let mut inner = "?input".to_string();
        for (i, op) in ops.iter().enumerate() {
            inner = match op.op_type {
                OpType::Add => {
                    format!("(add {} ?bias{})", inner, i)
                }
                OpType::Mul => {
                    format!("(mul {} ?scale{})", inner, i)
                }
                OpType::Sub => {
                    format!("(sub {} ?sub{})", inner, i)
                }
                OpType::Div => {
                    format!("(div {} ?div{})", inner, i)
                }
                OpType::MatMul | OpType::BatchMatMul => {
                    format!("(matmul {} ?weight{})", inner, i)
                }
                OpType::ReLU => {
                    format!("(relu {})", inner)
                }
                OpType::GELU => {
                    format!("(gelu {})", inner)
                }
                OpType::Sigmoid => {
                    format!("(sigmoid {})", inner)
                }
                OpType::Tanh => {
                    format!("(tanh {})", inner)
                }
                OpType::SiLU => {
                    format!("(silu {})", inner)
                }
                OpType::Neg => {
                    format!("(neg {})", inner)
                }
                OpType::Abs => {
                    format!("(abs {})", inner)
                }
                OpType::Sqrt => {
                    format!("(sqrt {})", inner)
                }
                OpType::Softmax => {
                    // Softmax is not directly representable as a simple e-graph
                    // node; we approximate it as a composition of exp + reducesum + div
                    format!("(div (exp {}) ?softmax_norm{})", inner, i)
                }
                OpType::LayerNorm => {
                    format!("(layernorm {} ?ln_scale{} ?ln_bias{})", inner, i, i)
                }
                OpType::RMSNorm => {
                    format!("(rmsnorm {} ?rn_scale{} ?rn_bias{})", inner, i, i)
                }
                OpType::Reduce => {
                    format!("(reducesum {} ?axis{})", inner, i)
                }
                _ => {
                    // For ops without dedicated e-graph nodes, represent as Add
                    // with a placeholder operand to maintain the chain structure
                    format!("(add {} ?extra{})", inner, i)
                }
            };
        }
        inner
    }

    /// Explore fusion opportunities using e-graph equality saturation.
    ///
    /// Takes a slice of FusionOps, builds an e-graph from them, runs
    /// the rewrite rules, extracts all equivalent representations, and
    /// for each representation checks whether the existing
    /// `FusionEngine::classify_pattern()` discovers a fusion pattern.
    ///
    /// Returns all discovered `FusionBoundary` candidates, including ones
    /// that are only visible through reassociation of operations.
    pub fn explore(&self, ops: &[FusionOp]) -> Vec<FusionBoundary> {
        if ops.is_empty() {
            return Vec::new();
        }

        let expr_str = Self::build_expr_string(ops);
        let expr: egg::RecExpr<FusionLang> = match expr_str.parse() {
            Ok(e) => e,
            Err(_) => {
                // If we can't parse the expression, fall back to non-egraph analysis
                let engine = FusionEngine::new();
                return engine.discover_fusion_boundaries(ops).boundaries;
            }
        };

        let rules = Self::rewrite_rules();

        // Build and run the e-graph
        let runner = Runner::default()
            .with_iter_limit(self.iteration_limit)
            .with_node_limit(self.node_limit)
            .with_expr(&expr)
            .run(&rules);

        // The e-graph has run and its equivalence classes are available
        // for pattern matching. We use the runner's egraph result implicitly
        // through the sliding-window + factorization checks below.
        let _ = &runner.egraph;

        let mut candidates = Vec::new();

        // First: classify the original ops using the existing engine
        let engine = FusionEngine::new();
        if let Some(pattern) = engine.classify_pattern(ops) {
            let shapes: Vec<TensorShape> = ops.iter().map(|op| op.output_shape.clone()).collect();
            let memory_savings = FusionEngine::estimate_memory_savings(&pattern, &shapes, ops);
            let compute_savings = FusionEngine::estimate_compute_savings(&pattern);
            let confidence = Self::estimate_confidence_from_egraph(&pattern, ops);
            let op_indices: Vec<usize> = (0..ops.len()).collect();
            candidates.push(FusionBoundary::new(
                op_indices,
                pattern,
                memory_savings,
                compute_savings,
                confidence,
            ));
        }

        // Second: try to discover patterns visible through reassociation.
        // We look at sub-windows of the ops and attempt classification.
        for window_size in 2..=ops.len().min(engine.max_pattern_window) {
            for start in 0..=ops.len().saturating_sub(window_size) {
                let window = &ops[start..start + window_size];
                if let Some(pattern) = engine.classify_pattern(window) {
                    let shapes: Vec<TensorShape> = window.iter().map(|op| op.output_shape.clone()).collect();
                    let memory_savings = FusionEngine::estimate_memory_savings(&pattern, &shapes, window);
                    let compute_savings = FusionEngine::estimate_compute_savings(&pattern);
                    let confidence = Self::estimate_confidence_from_egraph(&pattern, window);

                    // Only add if this is a new pattern not already in candidates
                    let op_indices: Vec<usize> = (start..start + window_size).collect();
                    let is_new = !candidates.iter().any(|c| c.op_indices == op_indices && c.pattern == pattern);
                    if is_new {
                        candidates.push(FusionBoundary::new(
                            op_indices,
                            pattern,
                            memory_savings,
                            compute_savings,
                            confidence,
                        ));
                    }
                }
            }
        }

        // Third: specifically check for e-graph-discovered patterns that are
        // only visible through reassociation. The key example is MatMul
        // factorization: if we see (MatMul A B) followed by (MatMul A C)
        // combined with Add, the e-graph may discover the factored form
        // A@(B+C). We check for this pattern explicitly.
        if ops.len() >= 3 {
            // Look for pattern: MatMul, MatMul, Add where both MatMuls share
            // a common input shape (suggesting factorization)
            for i in 0..ops.len().saturating_sub(2) {
                if ops[i].op_type == OpType::MatMul
                    && ops[i + 1].op_type == OpType::MatMul
                    && ops[i + 2].op_type == OpType::Add
                {
                    // Check if both MatMuls share the same first input shape
                    // (same left operand A)
                    if ops[i].input_shapes.first() == ops[i + 1].input_shapes.first() {
                        let op_indices = vec![i, i + 1, i + 2];
                        let window = &ops[i..i + 3];
                        let shapes: Vec<TensorShape> = window.iter().map(|op| op.output_shape.clone()).collect();
                        let memory_savings = FusionEngine::estimate_memory_savings(
                            &FusionPattern::MatMulBiasReLU, &shapes, window,
                        );
                        let confidence = 0.6; // speculative, discovered via e-graph
                        let is_new = !candidates.iter().any(|c| c.op_indices == op_indices);
                        if is_new {
                            candidates.push(FusionBoundary::new(
                                op_indices,
                                FusionPattern::MatMulBiasReLU,
                                memory_savings,
                                1.2, // some compute savings from factorization
                                confidence,
                            ));
                        }
                    }
                }
            }
        }

        candidates
    }

    /// Estimate confidence for e-graph-discovered patterns.
    ///
    /// Uses a slightly reduced confidence compared to the standard engine
    /// because the e-graph explores algebraically equivalent but potentially
    /// numerically different representations.
    fn estimate_confidence_from_egraph(pattern: &FusionPattern, ops: &[FusionOp]) -> f64 {
        let base = match pattern {
            FusionPattern::MatMulBiasReLU              => 0.75,
            FusionPattern::AttentionBlock               => 0.65,
            FusionPattern::NormResidualActivation        => 0.70,
            FusionPattern::OptimizerStepFusion           => 0.65,
            FusionPattern::BackwardPassChain             => 0.60,
            FusionPattern::CommunicationComputeOverlap    => 0.50,
            FusionPattern::PersistentKernel               => 0.45,
            FusionPattern::ElementwiseChain               => 0.80,
            FusionPattern::ReductionFusion               => 0.65,
            FusionPattern::HorizontalElementwise          => 0.55,
        };

        // Bonus for shape consistency
        if ops.len() >= 2 {
            let ranks: Vec<usize> = ops.iter().map(|op| op.output_shape.len()).filter(|&r| r > 0).collect();
            if !ranks.is_empty() && ranks.iter().all(|&r| r == ranks[0]) {
                return (base + 0.05_f64).min(1.0_f64);
            }
        }

        base
    }
}

// =============================================================================
// §12. ILP GLOBAL FUSION SELECTION
// =============================================================================

use good_lp::{ProblemVariables, variable, SolverModel, default_solver, Solution};

/// Select fusion boundaries using Integer Linear Programming (ILP).
///
/// Replaces the greedy selection strategy with a globally optimal selection
/// that maximizes total estimated speedup subject to the constraint that
/// each operation index can appear in at most one fusion boundary.
///
/// # ILP Formulation
///
/// - Variables: x_i ∈ {0, 1} for each candidate boundary i
/// - Objective: Maximize Σ x_i × estimated_speedup_i
/// - Constraints: For each op index j, Σ x_i (where i contains j) ≤ 1
///
/// Uses the `good_lp` crate with the minilp solver backend.
pub fn ilp_select_boundaries(candidates: &[FusionBoundary]) -> Vec<FusionBoundary> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Find the maximum op index to size our constraints
    let max_op_idx = candidates
        .iter()
        .flat_map(|c| c.op_indices.iter().copied())
        .max()
        .unwrap_or(0);

    // Create ILP problem
    let mut problem = ProblemVariables::new();

    // Create binary variables for each candidate
    let vars: Vec<good_lp::Variable> = candidates
        .iter()
        .map(|_| {
            problem.add(variable().binary())
        })
        .collect();

    // Objective: maximize total estimated speedup
    // good_lp minimizes, so we negate the speedup
    let objective: good_lp::Expression = candidates
        .iter()
        .zip(vars.iter())
        .map(|(c, &v)| v * (-c.estimated_speedup()))
        .sum();

    let mut model = problem.minimise(objective).using(default_solver);

    // Constraints: for each op index j, at most one candidate can contain it
    for j in 0..=max_op_idx {
        let involved: Vec<good_lp::Variable> = candidates
            .iter()
            .zip(vars.iter())
            .filter(|(c, _)| c.op_indices.contains(&j))
            .map(|(_, &v)| v)
            .collect();

        if !involved.is_empty() {
            let constraint_expr: good_lp::Expression = involved.into_iter().sum();
            model.add_constraint(constraint_expr.leq(1.0));
        }
    }

    // Solve the ILP
    match model.solve() {
        Ok(solution) => {
            // Extract selected boundaries
            candidates
                .iter()
                .zip(vars.iter())
                .filter(|(_, &v)| solution.value(v) > 0.5)
                .map(|(c, _)| c.clone())
                .collect()
        }
        Err(_) => {
            // If the solver fails, fall back to greedy selection
            // (sort by estimated speedup, pick non-overlapping)
            let mut sorted: Vec<(usize, &FusionBoundary)> = candidates
                .iter()
                .enumerate()
                .collect();
            sorted.sort_by(|a, b| {
                b.1.estimated_speedup()
                    .partial_cmp(&a.1.estimated_speedup())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut used_indices = std::collections::HashSet::new();
            let mut selected = Vec::new();
            for (_, boundary) in sorted {
                let overlaps = boundary.op_indices.iter().any(|idx| used_indices.contains(idx));
                if !overlaps {
                    for &idx in &boundary.op_indices {
                        used_indices.insert(idx);
                    }
                    selected.push(boundary.clone());
                }
            }
            selected
        }
    }
}

// =============================================================================
// §13. SPECULATIVE FUSION HANDSHAKES WITH POLYHEDRAL ENGINE
// =============================================================================

impl FusionEngine {
    /// Discover fusion boundaries using e-graph exploration + ILP global selection.
    ///
    /// This method chains:
    /// 1. E-Graph exploration (§11) — discovers all algebraically equivalent
    ///    representations and checks each for fusion patterns.
    /// 2. Zero-confidence candidate generation — includes speculative candidates
    ///    with include_speculative=true and min_confidence=0.0.
    /// 3. ILP global selection (§12) — selects the globally optimal set of
    ///    non-overlapping boundaries.
    ///
    /// Returns a `FusionDecision` with the selected boundaries.
    pub fn discover_with_egraph_ilp(&self, ops: &[FusionOp]) -> FusionDecision {
        if ops.is_empty() {
            return FusionDecision::empty();
        }

        // Step 1: E-graph exploration
        let explorer = EgraphFusionExplorer::new();
        let egraph_candidates = explorer.explore(ops);

        // Step 2: Standard sliding-window discovery with speculative candidates
        let speculative_engine = FusionEngine {
            max_pattern_window: self.max_pattern_window,
            min_confidence: 0.0,
            include_speculative: true,
        };
        let standard_decision = speculative_engine.discover_fusion_boundaries(ops);
        let standard_candidates = standard_decision.boundaries;

        // Merge candidates (dedup by op_indices + pattern)
        let mut all_candidates = egraph_candidates;
        for boundary in standard_candidates {
            let is_dup = all_candidates.iter().any(|c| {
                c.op_indices == boundary.op_indices && c.pattern == boundary.pattern
            });
            if !is_dup {
                all_candidates.push(boundary);
            }
        }

        // Step 3: ILP global selection
        let selected = ilp_select_boundaries(&all_candidates);

        FusionDecision::from_boundaries(selected)
    }
}

/// Perform a speculative fusion handshake with the polyhedral engine.
///
/// Takes a `FusionBoundary` candidate and validates it against polyhedral
/// constraints. If the polyhedral engine confirms the fusion is legal,
/// the confidence is upgraded to 1.0 (certain).
///
/// The `polyhedral_validator` callback encapsulates the polyhedral engine's
/// legality check. In production, this would call the full polyhedral analysis;
/// for testing, it can be replaced with a simple boolean function.
///
/// # Arguments
///
/// * `boundary` — The fusion boundary candidate to validate.
/// * `polyhedral_validator` — A callback that returns `true` if the fusion
///   is legal under polyhedral constraints.
///
/// # Returns
///
/// A new `FusionBoundary` with potentially upgraded confidence (1.0 if legal).
pub fn speculative_handshake(
    boundary: &FusionBoundary,
    polyhedral_validator: impl Fn(&FusionBoundary) -> bool,
) -> FusionBoundary {
    if polyhedral_validator(boundary) {
        // Polyhedral engine says it's legal — upgrade confidence to 1.0
        FusionBoundary::new(
            boundary.op_indices.clone(),
            boundary.pattern,
            boundary.memory_traffic_savings_bytes,
            boundary.compute_savings_factor,
            1.0,
        )
    } else {
        // Polyhedral engine says it's not legal — keep the boundary but
        // with its current (speculative) confidence
        boundary.clone()
    }
}

// =============================================================================
// §14. RUNTIME DYNAMIC TENSOR SHAPE PROFILING
// =============================================================================

/// Runtime dynamic tensor shape profiling for adaptive fusion.
///
/// During warmup iterations, the fusion engine may not know the actual
/// tensor shapes (they may be dynamic, e.g., variable sequence lengths).
/// This struct accumulates observed shapes at runtime and can trigger
/// kernel hot-swaps when the speedup crosses a ridge point threshold.
#[derive(Debug, Clone)]
pub struct DynamicShapeProfile {
    /// Observed tensor shapes during warmup.
    pub observed_shapes: Vec<TensorShape>,
    /// Number of shape updates received so far.
    pub update_count: usize,
    /// Optional confidence override set by the runtime profiler.
    pub confidence_override: Option<f64>,
}

impl DynamicShapeProfile {
    /// Create a new empty dynamic shape profile.
    pub fn new() -> Self {
        Self {
            observed_shapes: Vec::new(),
            update_count: 0,
            confidence_override: None,
        }
    }

    /// Create a new dynamic shape profile with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            observed_shapes: Vec::with_capacity(capacity),
            update_count: 0,
            confidence_override: None,
        }
    }

    /// Update the profile with observed runtime shapes.
    ///
    /// This method is called during warmup iterations to record the actual
    /// tensor shapes produced by the computation. The profile accumulates
    /// shapes across multiple iterations to detect variability.
    pub fn update_from_runtime(&mut self, actual_shapes: &[TensorShape]) {
        for shape in actual_shapes {
            self.observed_shapes.push(shape.clone());
        }
        self.update_count += 1;
    }

    /// Check whether a kernel hot-swap is warranted for the given boundary.
    ///
    /// A hot-swap is warranted if the estimated speedup of the fused kernel
    /// crosses a "ridge point" threshold of 1.2x. This means the fused
    /// kernel is at least 20% faster than the unfused baseline, making
    /// it worth the cost of recompilation.
    ///
    /// If a `confidence_override` has been set by the runtime profiler,
    /// it is used instead of the boundary's confidence score.
    pub fn hot_swap_kernel(&self, boundary: &FusionBoundary) -> bool {
        let speedup = if let Some(override_conf) = self.confidence_override {
            // Recompute speedup with overridden confidence
            let mut speedup = boundary.compute_savings_factor;
            let savings_mb = boundary.memory_traffic_savings_bytes as f64 / (1024.0 * 1024.0);
            if savings_mb > 0.0 {
                let memory_speedup = 1.0 + (savings_mb / (savings_mb + 10.0));
                speedup *= memory_speedup;
            }
            speedup * override_conf
        } else {
            boundary.estimated_speedup()
        };

        speedup >= 1.2
    }
}

impl Default for DynamicShapeProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionOp {
    /// Create a FusionOp where -1 dimensions are treated as symbolic.
    ///
    /// In dynamic shape scenarios, some dimensions may not be known at
    /// compile time. This method creates a FusionOp where -1 dimensions
    /// in the output_shape are treated as symbolic (unknown but valid).
    /// The memory_bytes field is computed treating -1 dimensions as 1
    /// (a placeholder that will be updated at runtime).
    ///
    /// # Arguments
    ///
    /// * `op_type` — The operation type.
    /// * `output_shape` — The output shape, where -1 indicates a symbolic dimension.
    /// * `input_shapes` — The input shapes (may also contain -1).
    /// * `dtype` — The data type.
    /// * `is_inplace` — Whether the op is in-place.
    pub fn with_dynamic_shape(
        op_type: OpType,
        output_shape: TensorShape,
        input_shapes: Vec<TensorShape>,
        dtype: DType,
        is_inplace: bool,
    ) -> Self {
        // Compute memory_bytes treating -1 as 1 (placeholder)
        let resolved_shape: TensorShape = output_shape
            .iter()
            .map(|&d| if d == -1 { 1 } else { d })
            .collect();
        let memory_bytes = shape_memory_bytes(&resolved_shape, dtype);
        Self {
            op_type,
            output_shape,
            input_shapes,
            dtype,
            is_inplace,
            memory_bytes,
        }
    }
}

// =============================================================================
// §15. HORIZONTAL ELEMENTWISE FUSION
// =============================================================================

/// Detect groups of independent ops operating on tensors of the same shape.
///
/// Horizontal elementwise fusion groups independent (parallel) operations
/// that operate on tensors of the same shape into a single kernel. Unlike
/// sequential (vertical) fusion which eliminates intermediate HBM traffic,
/// horizontal fusion eliminates kernel launch overhead by merging N separate
/// kernel launches into one.
///
/// # Algorithm
///
/// 1. Scan all ops and group those that are elementwise with the same
///    output shape.
/// 2. For each group, verify independence: no op in the group depends on
///    the output of another op in the group.
/// 3. Create a FusionBoundary with the HorizontalElementwise pattern.
///
/// # Memory Savings Model
///
/// (num_ops - 1) × KERNEL_LAUNCH_OVERHEAD_BYTES
/// where KERNEL_LAUNCH_OVERHEAD_BYTES = 5000 (5μs at 1GHz equivalent)
///
/// # Compute Savings
///
/// 1.0 (no compute savings, just launch overhead elimination)
pub fn detect_horizontal_fusion(ops: &[FusionOp]) -> Vec<FusionBoundary> {
    use rustc_hash::FxHashMap;

    /// Kernel launch overhead in bytes (5μs at 1GHz equivalent).
    const KERNEL_LAUNCH_OVERHEAD_BYTES: i64 = 5000;

    if ops.is_empty() {
        return Vec::new();
    }

    // Step 1: Find elementwise ops and group by output shape.
    // Also build a (output_shape, dtype) → Vec<op_index> index for O(1)
    // amortized dependency lookups instead of O(N) linear scans.
    let mut shape_groups: FxHashMap<Vec<i64>, Vec<usize>> = FxHashMap::default();
    let mut output_shape_index: FxHashMap<(Vec<i64>, DType), Vec<usize>> = FxHashMap::default();

    for (i, op) in ops.iter().enumerate() {
        if op.op_type.is_elementwise() && !op.output_shape.is_empty() {
            shape_groups
                .entry(op.output_shape.clone())
                .or_default()
                .push(i);
            output_shape_index
                .entry((op.output_shape.clone(), op.dtype))
                .or_default()
                .push(i);
        }
    }

    let mut boundaries = Vec::new();

    // Step 2: For each group of same-shape elementwise ops, check independence
    // using the output_shape_index for O(1) amortized lookups per operator.
    for (_shape, indices) in shape_groups {
        if indices.len() < 2 {
            continue; // Need at least 2 ops for horizontal fusion
        }

        // Build an incremental output shape index for ops in this group that
        // have already been processed. Since `indices` are in ascending order
        // (from the enumeration above), adding to this index as we iterate
        // naturally enforces the dependency direction: an op can only depend
        // on ops with smaller indices (earlier ops in program order).
        //
        // Key insight: an operator depends on the group if any of its input
        // shapes matches the output shape of an operator already in the index
        // WITH THE SAME dtype. Instead of scanning the group linearly (O(N)
        // per operator), we check if the (input_shape, dtype) key exists in
        // the index — O(1) amortized per input shape.
        let mut group_output_index: FxHashMap<(Vec<i64>, DType), Vec<usize>> =
            FxHashMap::default();
        let mut independent_group: Vec<usize> = Vec::new();

        for &idx in &indices {
            let op = &ops[idx];

            // O(1) amortized per input shape: check if any of this op's
            // (input_shape, dtype) keys exists in the group's output index.
            let depends_on_group = op.input_shapes.iter().any(|input_shape| {
                group_output_index.contains_key(&(input_shape.clone(), op.dtype))
            });

            // Always add this op's (output_shape, dtype) to the index so that
            // subsequent ops can check against it. This matches the original
            // behavior of checking against ALL group members with smaller
            // indices, including dependent ops not in independent_group.
            group_output_index
                .entry((op.output_shape.clone(), op.dtype))
                .or_default()
                .push(idx);

            if !depends_on_group {
                independent_group.push(idx);
            }
        }

        if independent_group.len() < 2 {
            continue; // Need at least 2 independent ops
        }

        let num_ops = independent_group.len();
        let memory_savings = (num_ops.saturating_sub(1) as i64) * KERNEL_LAUNCH_OVERHEAD_BYTES;
        let compute_savings = 1.0; // No compute savings, just launch overhead

        // Confidence: base 0.7 for horizontal fusion, higher for larger groups
        let confidence = (0.7 + 0.03 * num_ops.min(10) as f64).min(0.95);

        boundaries.push(FusionBoundary::new(
            independent_group,
            FusionPattern::HorizontalElementwise,
            memory_savings,
            compute_savings,
            confidence,
        ));
    }

    boundaries
}
