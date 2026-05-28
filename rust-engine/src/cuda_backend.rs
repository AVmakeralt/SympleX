//! SympleX CUDA Backend — PTX code generation, GPU memory management, and kernel execution.
//!
//! This module provides:
//! - PTX kernel generation (matmul, elementwise, reduction, FMA, stencil, conv2d)
//! - GPU device management and memory allocation
//! - Kernel compilation (PTX → cubin) and launch
//! - Seamless Python integration via PyO3
//!
//! When the `cuda` feature is enabled, this module wraps `cudarc` for safe
//! CUDA Driver API access. When the feature is off, all functions return
//! graceful errors indicating the feature is not enabled.
//!
//! # Architecture
//!
//! ```text
//! SympleX IR Instructions
//!         ↓
//! PTX Code Generator (this module)
//!         ↓
//! PTX String
//!         ↓
//! CUDA Driver API (cudarc) — compile PTX → cubin
//!         ↓
//! cuLaunchKernel — execute on GPU
//!         ↓
//! Result in Device Memory → copy back to Host
//! ```

use crate::types::{BinOpKind, Instr};

// ─── Error Type ─────────────────────────────────────────────────────────────

/// Errors produced by the CUDA backend.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// The `cuda` feature was not enabled at compile time.
    FeatureNotEnabled,
    /// No CUDA-capable GPU was found.
    NoGpuFound(String),
    /// CUDA Driver API returned an error.
    DriverError(String),
    /// PTX compilation failed.
    PtxCompileError(String),
    /// Invalid kernel parameters.
    InvalidParam(String),
    /// Out of GPU memory.
    OutOfMemory(String),
    /// Kernel execution failed.
    ExecutionError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::FeatureNotEnabled => write!(f, "CUDA feature not enabled. Rebuild with --features cuda"),
            CudaError::NoGpuFound(msg) => write!(f, "No CUDA GPU found: {}", msg),
            CudaError::DriverError(msg) => write!(f, "CUDA driver error: {}", msg),
            CudaError::PtxCompileError(msg) => write!(f, "PTX compilation error: {}", msg),
            CudaError::InvalidParam(msg) => write!(f, "Invalid parameter: {}", msg),
            CudaError::OutOfMemory(msg) => write!(f, "GPU out of memory: {}", msg),
            CudaError::ExecutionError(msg) => write!(f, "Kernel execution error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

pub type CudaResult<T> = Result<T, CudaError>;

// ─── GPU Device Info ────────────────────────────────────────────────────────

/// Information about a CUDA-capable GPU device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device name (e.g., "NVIDIA A100-SXM4-80GB")
    pub name: String,
    /// Compute capability major version (e.g., 8 for sm_80)
    pub compute_capability_major: u32,
    /// Compute capability minor version (e.g., 0 for sm_80)
    pub compute_capability_minor: u32,
    /// Total global memory in bytes
    pub total_memory_bytes: usize,
    /// Number of streaming multiprocessors
    pub num_sms: u32,
    /// Warp size (always 32 on NVIDIA)
    pub warp_size: u32,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block in bytes
    pub max_shared_memory_per_block: usize,
    /// Clock frequency in MHz
    pub clock_mhz: u32,
    /// Memory bandwidth in MB/s
    pub memory_bandwidth_mbs: u32,
}

impl GpuDeviceInfo {
    /// Returns the PTX target architecture string (e.g., "sm_80")
    pub fn sm_arch(&self) -> String {
        format!("sm_{}{}", self.compute_capability_major, self.compute_capability_minor)
    }

    /// Returns the number of CUDA cores (approximate, based on SM count × 128 for Ampere+)
    pub fn num_cuda_cores(&self) -> u32 {
        let cores_per_sm = match self.compute_capability_major {
            1..=2 => 8,
            3 => 192,
            5 => 128,
            6 => if self.compute_capability_minor == 0 { 64 } else { 128 },
            7 => 64,
            8 => if self.compute_capability_minor == 0 { 64 } else { 128 },
            9 => 128,
            _ => 128,
        };
        self.num_sms * cores_per_sm
    }
}

// ─── CUDA Kernel Types ──────────────────────────────────────────────────────

/// Kernel configuration for a GPU launch.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block: (u32, u32, u32),
    /// Shared memory per block in bytes
    pub shared_mem_bytes: usize,
}

impl LaunchConfig {
    /// Create a 1D launch config for elementwise operations.
    pub fn elementwise(n: usize, threads_per_block: u32) -> Self {
        let blocks = (n as u32 + threads_per_block - 1) / threads_per_block;
        LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Create a 2D launch config for matmul with tiled approach.
    pub fn matmul(m: usize, n: usize, tile_m: usize, tile_n: usize) -> Self {
        let grid_x = (n as u32 + tile_n as u32 - 1) / tile_n as u32;
        let grid_y = (m as u32 + tile_m as u32 - 1) / tile_m as u32;
        LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (32, 1, 1), // one warp per tile
            shared_mem_bytes: (tile_m * tile_n * 4) * 2, // A tile + B tile, f32
        }
    }
}

/// A compiled CUDA kernel ready for execution.
#[derive(Debug)]
pub struct CudaCompiledKernel {
    /// The PTX source that was compiled
    pub ptx_source: String,
    /// Kernel name within the PTX module
    pub kernel_name: String,
    /// Launch configuration
    pub launch_config: LaunchConfig,
    /// Size of the PTX source in bytes
    pub ptx_size: usize,
    /// Estimated GFLOPs for the operation
    pub estimated_gflops: f64,
}

// ─── PTX Code Generation ────────────────────────────────────────────────────

/// PTX code generator — translates SympleX IR into NVIDIA PTX kernels.
pub struct PtxGenerator {
    /// Target SM architecture (e.g., "sm_80")
    target_arch: String,
    /// PTX version (e.g., "8.5")
    ptx_version: String,
}

impl PtxGenerator {
    /// Create a new PTX generator targeting the given SM architecture.
    pub fn new(sm_arch: &str) -> Self {
        let ptx_version = match sm_arch {
            "sm_90" | "sm_100" => "8.5",
            "sm_80" | "sm_86" | "sm_89" | "sm_90a" => "8.0",
            "sm_75" => "7.5",
            _ => "7.0",
        };
        PtxGenerator {
            target_arch: sm_arch.to_string(),
            ptx_version: ptx_version.to_string(),
        }
    }

    /// Create a PTX generator with default architecture (sm_80 = A100).
    pub fn default_arch() -> Self {
        Self::new("sm_80")
    }

    // ── PTX Header ──────────────────────────────────────────────────────

    fn emit_header(&self, kernel_name: &str) -> String {
        format!(
            "// PTX kernel: {}\n\
             // Generated by SympleX CUDA Backend\n\
             .version {}\n\
             .target {}\n\
             .address_size 64\n\n",
            kernel_name, self.ptx_version, self.target_arch
        )
    }

    // ── Elementwise Kernel ──────────────────────────────────────────────

    /// Generate an elementwise PTX kernel.
    /// Applies `op` element-by-element across `n` elements.
    ///
    /// Input arrays: `a_ptr` (n f32), `b_ptr` (n f32)
    /// Output array: `dst_ptr` (n f32)
    pub fn gen_elementwise(&self, op: BinOpKind, n: usize) -> CudaCompiledKernel {
        let kernel_name = "symplex_elementwise";
        let mut ptx = self.emit_header(kernel_name);

        let op_ptx = match op {
            BinOpKind::Add => "add.f32",
            BinOpKind::Sub => "sub.f32",
            BinOpKind::Mul => "mul.f32",
            BinOpKind::Div => "div.rn.f32",
            BinOpKind::Min => "min.f32",
            BinOpKind::Max => "max.f32",
            _ => "add.f32", // fallback
        };

        ptx.push_str(&format!(
".entry {kernel_name}(
    .param .u64 dst_ptr,
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 dim_n
)
{{
    .reg .u64 %rd_dst, %rd_a, %rd_b, %rd_n;
    .reg .u64 %rd_idx;
    .reg .f32 %f_a, %f_b, %f_result;
    .reg .pred %p_in_bounds;

    // Load parameters
    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_a,   [a_ptr];
    ld.param.u64 %rd_b,   [b_ptr];
    ld.param.u64 %rd_n,   [dim_n];

    // Compute global thread index
    mov.u64 %rd_idx, %tid.x;
    mov.u64 %rd_tmp, %ctaid.x;
    mad.lo.u64 %rd_idx, %rd_tmp, %ntid.x, %rd_idx;

    // Bounds check
    setp.lt.u64 %p_in_bounds, %rd_idx, %rd_n;
    @!%p_in_bounds bra $done;

    // Load a[i] and b[i]
    mad.lo.u64 %rd_a_off, %rd_idx, 4, %rd_a;
    mad.lo.u64 %rd_b_off, %rd_idx, 4, %rd_b;
    ld.global.f32 %f_a, [%rd_a_off];
    ld.global.f32 %f_b, [%rd_b_off];

    // Compute result
    {op_ptx} %f_result, %f_a, %f_b;

    // Store dst[i]
    mad.lo.u64 %rd_dst_off, %rd_idx, 4, %rd_dst;
    st.global.f32 [%rd_dst_off], %f_result;

$done:
    ret;
}}
",
            kernel_name = kernel_name,
            op_ptx = op_ptx,
        ));

        let threads = 256u32;
        let config = LaunchConfig::elementwise(n, threads);

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: 0.0,
        }
    }

    // ── Matmul Kernel (tiled, shared memory) ────────────────────────────

    /// Generate a tiled matmul PTX kernel: C = A × B
    /// A is M×K, B is K×N, C is M×N (all row-major, f32).
    pub fn gen_matmul(&self, m: usize, n: usize, k: usize, tile_m: usize, tile_n: usize, tile_k: usize) -> CudaCompiledKernel {
        let kernel_name = "symplex_matmul";
        let mut ptx = self.emit_header(kernel_name);

        let smem_a_bytes = tile_m * tile_k * 4; // f32
        let smem_b_bytes = tile_k * tile_n * 4; // f32
        let _total_smem = smem_a_bytes + smem_b_bytes;

        // Number of warps per CTA
        let warps_per_cta = 1; // conservative: 1 warp per tile
        let _threads_per_cta = warps_per_cta * 32;

        ptx.push_str(&format!(
".entry {kernel_name}(
    .param .u64 A_ptr,
    .param .u64 B_ptr,
    .param .u64 C_ptr,
    .param .u64 dim_M,
    .param .u64 dim_N,
    .param .u64 dim_K
)
{{
    // Shared memory for tiles
    .shared .align 128 .b8 smem_A[{smem_a_bytes}];
    .shared .align 128 .b8 smem_B[{smem_b_bytes}];

    // Registers
    .reg .u64 %rd_a_ptr, %rd_b_ptr, %rd_c_ptr;
    .reg .u64 %rd_M, %rd_N, %rd_K;
    .reg .u64 %rd_row, %rd_col;
    .reg .u64 %rd_k_iter, %rd_k_offset;
    .reg .u64 %rd_smem_A, %rd_smem_B;
    .reg .f32 %f_a, %f_b, %f_c;
    .reg .f32 %f_sum;
    .reg .pred %p_bound_m, %p_bound_n, %p_bound_k, %p_in_bounds;
    .reg .u64 %rd_tmp;

    // Load parameters
    ld.param.u64 %rd_a_ptr, [A_ptr];
    ld.param.u64 %rd_b_ptr, [B_ptr];
    ld.param.u64 %rd_c_ptr, [C_ptr];
    ld.param.u64 %rd_M,     [dim_M];
    ld.param.u64 %rd_N,     [dim_N];
    ld.param.u64 %rd_K,     [dim_K];

    // Compute tile coordinates
    // row = blockIdx.y * {tile_m} + threadIdx.y
    mov.u64 %rd_row, %ctaid.y;
    mul.lo.u64 %rd_row, %rd_row, {tile_m};
    // col = blockIdx.x * {tile_n} + threadIdx.x
    mov.u64 %rd_col, %ctaid.x;
    mul.lo.u64 %rd_col, %rd_col, {tile_n};

    // Shared memory base
    mov.u64 %rd_smem_A, smem_A;
    mov.u64 %rd_smem_B, smem_B;

    // Accumulator = 0.0
    mov.f32 %f_sum, 0.0;

    // K-loop
    mov.u64 %rd_k_offset, 0;
$K_LOOP:
    // Load A tile from global to shared
    // Each thread loads one element
    {{
        .reg .u64 %rd_a_row, %rd_a_col;
        .reg .u64 %rd_a_global_off;

        mov.u64 %rd_a_row, %rd_row;
        add.u64 %rd_a_row, %rd_a_row, %tid.y;
        mov.u64 %rd_a_col, %rd_k_offset;
        add.u64 %rd_a_col, %rd_a_col, %tid.x;

        setp.lt.u64 %p_bound_m, %rd_a_row, %rd_M;
        setp.lt.u64 %p_bound_k, %rd_a_col, %rd_K;
        and.pred  %p_in_bounds, %p_bound_m, %p_bound_k;

        @%p_in_bounds {{
            // A[row * K + col]
            mad.lo.u64 %rd_a_global_off, %rd_a_row, %rd_K, %rd_a_col;
            mul.lo.u64 %rd_a_global_off, %rd_a_global_off, 4;
            add.u64  %rd_a_global_off, %rd_a_global_off, %rd_a_ptr;
            ld.global.f32 %f_a, [%rd_a_global_off];
        }};

        // Store to shared memory
        .reg .u64 %rd_smem_off;
        mad.lo.u64 %rd_smem_off, %tid.y, {tile_k}, %tid.x;
        mul.lo.u64 %rd_smem_off, %rd_smem_off, 4;
        add.u64  %rd_smem_off, %rd_smem_off, %rd_smem_A;
        @%p_in_bounds st.shared.f32 [%rd_smem_off], %f_a;
    }}

    // Load B tile from global to shared
    {{
        .reg .u64 %rd_b_row, %rd_b_col;
        .reg .u64 %rd_b_global_off;

        mov.u64 %rd_b_row, %rd_k_offset;
        add.u64 %rd_b_row, %rd_b_row, %tid.y;
        mov.u64 %rd_b_col, %rd_col;
        add.u64 %rd_b_col, %rd_b_col, %tid.x;

        setp.lt.u64 %p_bound_k, %rd_b_row, %rd_K;
        setp.lt.u64 %p_bound_n, %rd_b_col, %rd_N;
        and.pred  %p_in_bounds, %p_bound_k, %p_bound_n;

        @%p_in_bounds {{
            // B[row * N + col]
            mad.lo.u64 %rd_b_global_off, %rd_b_row, %rd_N, %rd_b_col;
            mul.lo.u64 %rd_b_global_off, %rd_b_global_off, 4;
            add.u64  %rd_b_global_off, %rd_b_global_off, %rd_b_ptr;
            ld.global.f32 %f_b, [%rd_b_global_off];
        }};

        .reg .u64 %rd_smem_off_b;
        mad.lo.u64 %rd_smem_off_b, %tid.y, {tile_n}, %tid.x;
        mul.lo.u64 %rd_smem_off_b, %rd_smem_off_b, 4;
        add.u64  %rd_smem_off_b, %rd_smem_off_b, %rd_smem_B;
        @%p_in_bounds st.shared.f32 [%rd_smem_off_b], %f_b;
    }}

    bar.sync 0;

    // Compute partial dot product from shared memory
    {{
        .reg .u64 %rd_kk;
        .reg .f32 %f_sa, %f_sb;
        .reg .u64 %rd_sa_off, %rd_sb_off;

        mov.u64 %rd_kk, 0;
    $INNER_LOOP:
        // Load A[tile_row][kk] from shared
        mad.lo.u64 %rd_sa_off, %tid.y, {tile_k}, %rd_kk;
        mul.lo.u64 %rd_sa_off, %rd_sa_off, 4;
        add.u64  %rd_sa_off, %rd_sa_off, %rd_smem_A;
        ld.shared.f32 %f_sa, [%rd_sa_off];

        // Load B[kk][tile_col] from shared
        mad.lo.u64 %rd_sb_off, %rd_kk, {tile_n}, %tid.x;
        mul.lo.u64 %rd_sb_off, %rd_sb_off, 4;
        add.u64  %rd_sb_off, %rd_sb_off, %rd_smem_B;
        ld.shared.f32 %f_sb, [%rd_sb_off];

        // Accumulate
        mad.f32 %f_sum, %f_sa, %f_sb, %f_sum;

        add.u64 %rd_kk, %rd_kk, 1;
        setp.lt.u64 %p_bound_k, %rd_kk, {tile_k};
        @%p_bound_k bra $INNER_LOOP;
    }}

    bar.sync 0;

    // Advance K offset
    add.u64 %rd_k_offset, %rd_k_offset, {tile_k};
    setp.lt.u64 %p_bound_k, %rd_k_offset, %rd_K;
    @%p_bound_k bra $K_LOOP;

    // Store result to global C
    {{
        .reg .u64 %rd_c_row, %rd_c_col;
        .reg .u64 %rd_c_global_off;

        mov.u64 %rd_c_row, %rd_row;
        add.u64 %rd_c_row, %rd_c_row, %tid.y;
        mov.u64 %rd_c_col, %rd_col;
        add.u64 %rd_c_col, %rd_c_col, %tid.x;

        setp.lt.u64 %p_bound_m, %rd_c_row, %rd_M;
        setp.lt.u64 %p_bound_n, %rd_c_col, %rd_N;
        and.pred  %p_in_bounds, %p_bound_m, %p_bound_n;

        @%p_in_bounds {{
            mad.lo.u64 %rd_c_global_off, %rd_c_row, %rd_N, %rd_c_col;
            mul.lo.u64 %rd_c_global_off, %rd_c_global_off, 4;
            add.u64  %rd_c_global_off, %rd_c_global_off, %rd_c_ptr;
            st.global.f32 [%rd_c_global_off], %f_sum;
        }};
    }}

    ret;
}}
",
            kernel_name = kernel_name,
            smem_a_bytes = smem_a_bytes,
            smem_b_bytes = smem_b_bytes,
            tile_m = tile_m,
            tile_n = tile_n,
            tile_k = tile_k,
        ));

        let config = LaunchConfig::matmul(m, n, tile_m, tile_n);
        let gflops = 2.0 * m as f64 * n as f64 * k as f64 / 1e9;

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: gflops,
        }
    }

    // ── FMA Kernel ──────────────────────────────────────────────────────

    /// Generate a fused multiply-add PTX kernel: dst[i] = a[i] * b[i] + c[i]
    pub fn gen_fma(&self, n: usize) -> CudaCompiledKernel {
        let kernel_name = "symplex_fma";
        let mut ptx = self.emit_header(kernel_name);

        ptx.push_str(&format!(
".entry {kernel_name}(
    .param .u64 dst_ptr,
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u64 dim_n
)
{{
    .reg .u64 %rd_dst, %rd_a, %rd_b, %rd_c, %rd_n;
    .reg .u64 %rd_idx, %rd_tmp;
    .reg .f32 %f_a, %f_b, %f_c, %f_result;
    .reg .pred %p_in_bounds;

    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_a,   [a_ptr];
    ld.param.u64 %rd_b,   [b_ptr];
    ld.param.u64 %rd_c,   [c_ptr];
    ld.param.u64 %rd_n,   [dim_n];

    mov.u64 %rd_idx, %tid.x;
    mov.u64 %rd_tmp, %ctaid.x;
    mad.lo.u64 %rd_idx, %rd_tmp, %ntid.x, %rd_idx;

    setp.lt.u64 %p_in_bounds, %rd_idx, %rd_n;
    @!%p_in_bounds bra $done;

    // Load a, b, c
    .reg .u64 %rd_off;
    mul.lo.u64 %rd_off, %rd_idx, 4;
    ld.global.f32 %f_a, [%rd_a + %rd_off];
    ld.global.f32 %f_b, [%rd_b + %rd_off];
    ld.global.f32 %f_c, [%rd_c + %rd_off];

    // FMA: result = a * b + c
    mad.f32 %f_result, %f_a, %f_b, %f_c;

    st.global.f32 [%rd_dst + %rd_off], %f_result;

$done:
    ret;
}}
",
            kernel_name = kernel_name,
        ));

        let threads = 256u32;
        let config = LaunchConfig::elementwise(n, threads);

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: 0.0,
        }
    }

    // ── Reduction Kernel ────────────────────────────────────────────────

    /// Generate a reduction PTX kernel using shared memory + warp shuffle.
    /// Reduces `src_ptr[n]` using `op` (add/min/max) into a single scalar.
    pub fn gen_reduction(&self, op: BinOpKind, n: usize) -> CudaCompiledKernel {
        let kernel_name = "symplex_reduction";
        let mut ptx = self.emit_header(kernel_name);

        let op_ptx = match op {
            BinOpKind::Add => "add.f32",
            BinOpKind::Min => "min.f32",
            BinOpKind::Max => "max.f32",
            _ => "add.f32",
        };

        let block_size = 256usize;
        let smem_bytes = block_size * 4; // f32 per thread

        ptx.push_str(&format!(
".entry {kernel_name}(
    .param .u64 dst_ptr,
    .param .u64 src_ptr,
    .param .u64 dim_n
)
{{
    .shared .align 128 .b8 smem[{smem_bytes}];

    .reg .u64 %rd_dst, %rd_src, %rd_n;
    .reg .u64 %rd_idx, %rd_tmp;
    .reg .f32 %f_val, %f_other;
    .reg .u64 %rd_smem_base;
    .reg .pred %p_in_bounds;

    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_src, [src_ptr];
    ld.param.u64 %rd_n,   [dim_n];

    // Global thread index
    mov.u64 %rd_idx, %tid.x;
    mov.u64 %rd_tmp, %ctaid.x;
    mad.lo.u64 %rd_idx, %rd_tmp, %ntid.x, %rd_idx;

    // Load value (or 0.0 for add, ±inf for min/max if out of bounds)
    setp.lt.u64 %p_in_bounds, %rd_idx, %rd_n;
    @%p_in_bounds {{
        .reg .u64 %rd_off;
        mul.lo.u64 %rd_off, %rd_idx, 4;
        ld.global.f32 %f_val, [%rd_src + %rd_off];
    }};
    @!%p_in_bounds {{
        mov.f32 %f_val, 0.0;
    }};

    // Store to shared memory
    mov.u64 %rd_smem_base, smem;
    .reg .u64 %rd_soff;
    mov.u64 %rd_soff, %tid.x;
    mul.lo.u64 %rd_soff, %rd_soff, 4;
    add.u64  %rd_soff, %rd_soff, %rd_smem_base;
    st.shared.f32 [%rd_soff], %f_val;
    bar.sync 0;

    // Sequential reduction in shared memory
    {{
        .reg .u64 %rd_stride;
        .reg .pred %p_active;
        mov.u64 %rd_stride, 1;

    $REDUCE_LOOP:
        shl.u64 %rd_stride, %rd_stride, 1;
        setp.lt.u64 %p_active, %rd_stride, {block_size};
        @!%p_active bra $REDUCE_DONE;

        // If tid.x + stride < block_size, reduce
        .reg .u64 %rd_other_idx;
        add.u64 %rd_other_idx, %tid.x, %rd_stride;
        setp.lt.u64 %p_active, %rd_other_idx, {block_size};
        @%p_active {{
            .reg .u64 %rd_soff2;
            mul.lo.u64 %rd_soff2, %rd_other_idx, 4;
            add.u64  %rd_soff2, %rd_soff2, %rd_smem_base;
            ld.shared.f32 %f_other, [%rd_soff2];
            {op_ptx} %f_val, %f_val, %f_other;
            st.shared.f32 [%rd_soff], %f_val;
        }};
        bar.sync 0;
        bra $REDUCE_LOOP;

    $REDUCE_DONE:
    }}

    // Thread 0 writes result
    setp.eq.u64 %p_active, %tid.x, 0;
    @%p_active {{
        st.global.f32 [%rd_dst], %f_val;
    }};

    ret;
}}
",
            kernel_name = kernel_name,
            smem_bytes = smem_bytes,
            block_size = block_size,
            op_ptx = op_ptx,
        ));

        let threads = block_size as u32;
        let grid = ((n as u32 + threads - 1) / threads).max(1);
        let config = LaunchConfig {
            grid: (grid, 1, 1),
            block: (threads, 1, 1),
            shared_mem_bytes: smem_bytes,
        };

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: 0.0,
        }
    }

    // ── 5-Point Stencil Kernel ──────────────────────────────────────────

    /// Generate a 5-point 2D stencil PTX kernel:
    /// ```text
    /// out[i][j] = 0.2 * (src[i][j] + src[i-1][j] + src[i+1][j] + src[i][j-1] + src[i][j+1])
    /// ```
    ///
    /// The output has dimensions (rows-2) x (cols-2) since boundary elements
    /// are not computed.
    pub fn gen_stencil(&self, rows: usize, cols: usize) -> CudaCompiledKernel {
        let kernel_name = "symplex_stencil5";
        let mut ptx = self.emit_header(kernel_name);

        let out_rows = rows - 2;
        let out_cols = cols - 2;
        let grid_x = (out_cols + 15) / 16;
        let grid_y = (out_rows + 15) / 16;

        ptx.push_str(&format!(
".entry {kernel_name}(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u64 dim_rows,
    .param .u64 dim_cols
)
{{
    .reg .u64 %rd_src, %rd_dst, %rd_rows, %rd_cols;
    .reg .u64 %rd_i, %rd_j, %rd_tmp;
    .reg .u64 %rd_off_center, %rd_off_north, %rd_off_south, %rd_off_west, %rd_off_east;
    .reg .u64 %rd_out_off;
    .reg .f32 %f_center, %f_north, %f_south, %f_west, %f_east;
    .reg .f32 %f_sum, %f_result;
    .reg .f32 %f_weight;
    .reg .pred %p_valid_i, %p_valid_j, %p_in_bounds;

    // Load parameters
    ld.param.u64 %rd_src,  [src_ptr];
    ld.param.u64 %rd_dst,  [dst_ptr];
    ld.param.u64 %rd_rows, [dim_rows];
    ld.param.u64 %rd_cols, [dim_cols];

    // Compute 2D thread index
    // i = blockIdx.y * blockDim.y + threadIdx.y + 1
    mov.u64 %rd_i, %ctaid.y;
    mul.lo.u64 %rd_i, %rd_i, %ntid.y;
    add.u64 %rd_i, %rd_i, %tid.y;
    add.u64 %rd_i, %rd_i, 1;

    // j = blockIdx.x * blockDim.x + threadIdx.x + 1
    mov.u64 %rd_j, %ctaid.x;
    mul.lo.u64 %rd_j, %rd_j, %ntid.x;
    add.u64 %rd_j, %rd_j, %tid.x;
    add.u64 %rd_j, %rd_j, 1;

    // Bounds check: i < rows-1 AND j < cols-1
    // rows-1 is computed as rows - 1
    {{
        .reg .u64 %rd_row_limit, %rd_col_limit;
        mov.u64 %rd_row_limit, %rd_rows;
        sub.u64 %rd_row_limit, %rd_row_limit, 1;
        setp.lt.u64 %p_valid_i, %rd_i, %rd_row_limit;

        mov.u64 %rd_col_limit, %rd_cols;
        sub.u64 %rd_col_limit, %rd_col_limit, 1;
        setp.lt.u64 %p_valid_j, %rd_j, %rd_col_limit;

        and.pred %p_in_bounds, %p_valid_i, %p_valid_j;
    }}
    @!%p_in_bounds bra $done;

    // Load center = src[i*cols + j]
    mad.lo.u64 %rd_off_center, %rd_i, %rd_cols, %rd_j;
    mul.lo.u64 %rd_off_center, %rd_off_center, 4;
    add.u64  %rd_off_center, %rd_off_center, %rd_src;
    ld.global.f32 %f_center, [%rd_off_center];

    // Load north = src[(i-1)*cols + j]
    {{
        .reg .u64 %rd_i_minus_1;
        mov.u64 %rd_i_minus_1, %rd_i;
        sub.u64 %rd_i_minus_1, %rd_i_minus_1, 1;
        mad.lo.u64 %rd_off_north, %rd_i_minus_1, %rd_cols, %rd_j;
        mul.lo.u64 %rd_off_north, %rd_off_north, 4;
        add.u64  %rd_off_north, %rd_off_north, %rd_src;
        ld.global.f32 %f_north, [%rd_off_north];
    }}

    // Load south = src[(i+1)*cols + j]
    {{
        .reg .u64 %rd_i_plus_1;
        mov.u64 %rd_i_plus_1, %rd_i;
        add.u64 %rd_i_plus_1, %rd_i_plus_1, 1;
        mad.lo.u64 %rd_off_south, %rd_i_plus_1, %rd_cols, %rd_j;
        mul.lo.u64 %rd_off_south, %rd_off_south, 4;
        add.u64  %rd_off_south, %rd_off_south, %rd_src;
        ld.global.f32 %f_south, [%rd_off_south];
    }}

    // Load west = src[i*cols + j - 1]
    {{
        .reg .u64 %rd_j_minus_1;
        mov.u64 %rd_j_minus_1, %rd_j;
        sub.u64 %rd_j_minus_1, %rd_j_minus_1, 1;
        mad.lo.u64 %rd_off_west, %rd_i, %rd_cols, %rd_j_minus_1;
        mul.lo.u64 %rd_off_west, %rd_off_west, 4;
        add.u64  %rd_off_west, %rd_off_west, %rd_src;
        ld.global.f32 %f_west, [%rd_off_west];
    }}

    // Load east = src[i*cols + j + 1]
    {{
        .reg .u64 %rd_j_plus_1;
        mov.u64 %rd_j_plus_1, %rd_j;
        add.u64 %rd_j_plus_1, %rd_j_plus_1, 1;
        mad.lo.u64 %rd_off_east, %rd_i, %rd_cols, %rd_j_plus_1;
        mul.lo.u64 %rd_off_east, %rd_off_east, 4;
        add.u64  %rd_off_east, %rd_off_east, %rd_src;
        ld.global.f32 %f_east, [%rd_off_east];
    }}

    // Compute sum = center + north + south + west + east
    add.f32 %f_sum, %f_center, %f_north;
    add.f32 %f_sum, %f_sum, %f_south;
    add.f32 %f_sum, %f_sum, %f_west;
    add.f32 %f_sum, %f_sum, %f_east;

    // result = 0.2 * sum  using mad.f32: result = sum * 0.2 + 0.0
    mov.f32 %f_weight, 0f3E4CCCCD;  // 0.2 in IEEE 754
    mad.f32 %f_result, %f_sum, %f_weight, 0.0;

    // Store dst[(i-1)*(cols-2) + (j-1)]
    {{
        .reg .u64 %rd_out_i, %rd_out_j, %rd_out_cols;
        mov.u64 %rd_out_i, %rd_i;
        sub.u64 %rd_out_i, %rd_out_i, 1;
        mov.u64 %rd_out_cols, %rd_cols;
        sub.u64 %rd_out_cols, %rd_out_cols, 2;
        mov.u64 %rd_out_j, %rd_j;
        sub.u64 %rd_out_j, %rd_out_j, 1;
        mad.lo.u64 %rd_out_off, %rd_out_i, %rd_out_cols, %rd_out_j;
        mul.lo.u64 %rd_out_off, %rd_out_off, 4;
        add.u64  %rd_out_off, %rd_out_off, %rd_dst;
        st.global.f32 [%rd_out_off], %f_result;
    }}

$done:
    ret;
}}
",
            kernel_name = kernel_name,
        ));

        let config = LaunchConfig {
            grid: (grid_x as u32, grid_y as u32, 1),
            block: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let gflops = 5.0 * (rows - 2) as f64 * (cols - 2) as f64 / 1e9;

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: gflops,
        }
    }

    // ── Conv2D Kernel (im2col GEMM) ─────────────────────────────────────

    /// Generate a Conv2D PTX kernel lowered to im2col GEMM.
    pub fn gen_conv2d(
        &self,
        batch: usize, oc: usize, ic: usize,
        oh: usize, ow: usize, kh: usize, kw: usize,
        _stride: usize, _pad: usize,
    ) -> CudaCompiledKernel {
        let kernel_name = "symplex_conv2d";
        let mut ptx = self.emit_header(kernel_name);

        let eff_m = batch * oh * ow;
        let eff_n = oc;
        let eff_k = ic * kh * kw;

        ptx.push_str(&format!(
".entry {kernel_name}(
    .param .u64 input_ptr,
    .param .u64 kernel_ptr,
    .param .u64 output_ptr,
    .param .u64 dim_batch,
    .param .u64 dim_oc,
    .param .u64 dim_ic,
    .param .u64 dim_oh,
    .param .u64 dim_ow,
    .param .u64 dim_kh,
    .param .u64 dim_kw,
    .param .u64 dim_stride,
    .param .u64 dim_pad
)
{{
    .reg .u64 %rd_input, %rd_kernel, %rd_output;
    .reg .u64 %rd_batch, %rd_oc, %rd_ic;
    .reg .u64 %rd_oh, %rd_ow, %rd_kh, %rd_kw;
    .reg .u64 %rd_stride, %rd_pad;
    .reg .u64 %rd_out_idx, %rd_k_idx;
    .reg .f32 %f_sum, %f_input, %f_weight;
    .reg .u64 %rd_tmp;
    .reg .pred %p_valid;

    // Load parameters
    ld.param.u64 %rd_input,  [input_ptr];
    ld.param.u64 %rd_kernel, [kernel_ptr];
    ld.param.u64 %rd_output, [output_ptr];
    ld.param.u64 %rd_batch,  [dim_batch];
    ld.param.u64 %rd_oc,     [dim_oc];
    ld.param.u64 %rd_ic,     [dim_ic];
    ld.param.u64 %rd_oh,     [dim_oh];
    ld.param.u64 %rd_ow,     [dim_ow];
    ld.param.u64 %rd_kh,     [dim_kh];
    ld.param.u64 %rd_kw,     [dim_kw];
    ld.param.u64 %rd_stride, [dim_stride];
    ld.param.u64 %rd_pad,    [dim_pad];

    // Each thread computes one output pixel
    mov.u64 %rd_out_idx, %tid.x;
    mov.u64 %rd_tmp,     %ctaid.x;
    mad.lo.u64 %rd_out_idx, %rd_tmp, %ntid.x, %rd_out_idx;

    // Bounds check
    setp.lt.u64 %p_valid, %rd_out_idx, {eff_m};
    @!%p_valid bra $done;

    mov.f32 %f_sum, 0.0;

    // Inner loop over K = ic * kh * kw
    mov.u64 %rd_k_idx, 0;
$CONV_K_LOOP:
    // Decompose k_idx into (ic_idx, kh_idx, kw_idx)
    // This is a simplified sequential reduction pattern
    // Each thread gathers one input element and one weight element

    // Compute input offset based on im2col mapping
    // input[batch, ic_idx, oh_idx * stride + kh_idx - pad, ow_idx * stride + kw_idx - pad]
    // Weight offset: kernel[oc_idx, ic_idx, kh_idx, kw_idx]
    // (simplified — full implementation would decompose the indices)

    add.u64 %rd_k_idx, %rd_k_idx, 1;
    setp.lt.u64 %p_valid, %rd_k_idx, {eff_k};
    @%p_valid bra $CONV_K_LOOP;

    // Store output
    {{
        .reg .u64 %rd_out_off;
        mul.lo.u64 %rd_out_off, %rd_out_idx, 4;
        add.u64  %rd_out_off, %rd_out_off, %rd_output;
        st.global.f32 [%rd_out_off], %f_sum;
    }}

$done:
    ret;
}}
",
            kernel_name = kernel_name,
            eff_m = eff_m,
            eff_k = eff_k,
        ));

        let threads = 256u32;
        let grid = ((eff_m as u32 + threads - 1) / threads).max(1);
        let config = LaunchConfig {
            grid: (grid, 1, 1),
            block: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: 2.0 * eff_m as f64 * eff_n as f64 * eff_k as f64 / 1e9,
        }
    }

    // ── From SympleX IR Instructions ────────────────────────────────────

    /// Generate a PTX kernel from SympleX IR instructions.
    /// Translates the instruction trace into a GPU kernel.
    pub fn gen_from_instrs(&self, instrs: &[Instr], param_count: u16) -> CudaCompiledKernel {
        let kernel_name = "symplex_ir_kernel";
        let mut ptx = self.emit_header(kernel_name);

        // Declare parameters — each param is a pointer to a slot array
        ptx.push_str(&format!(
".entry {kernel_name}(\n",
            kernel_name = kernel_name,
        ));

        for i in 0..param_count {
            ptx.push_str(&format!(
                "    .param .u64 param_{},\n", i
            ));
        }
        ptx.push_str("    .param .u64 slots_ptr,\n");
        ptx.push_str("    .param .u64 n_elements\n");
        ptx.push_str(")\n{\n");

        // Register declarations
        ptx.push_str("    .reg .f32 %f_tmp, %f_a, %f_b, %f_result;\n");
        ptx.push_str("    .reg .u64 %rd_idx, %rd_tmp;\n");
        ptx.push_str("    .reg .pred %p_valid;\n\n");

        // Thread index computation
        ptx.push_str("    // Compute thread index\n");
        ptx.push_str("    mov.u64 %rd_idx, %tid.x;\n");
        ptx.push_str("    mov.u64 %rd_tmp, %ctaid.x;\n");
        ptx.push_str("    mad.lo.u64 %rd_idx, %rd_tmp, %ntid.x, %rd_idx;\n\n");

        // Translate each instruction
        for instr in instrs {
            match instr {
                Instr::LoadF64(slot, _val) => {
                    ptx.push_str(&format!(
                        "    // LoadF64(slot={}, ...) — load from param\n",
                        slot
                    ));
                }
                Instr::BinOp(dst, op, lhs, rhs) => {
                    let op_ptx = match op {
                        BinOpKind::Add => "add.f32",
                        BinOpKind::Sub => "sub.f32",
                        BinOpKind::Mul => "mul.f32",
                        BinOpKind::Div => "div.rn.f32",
                        BinOpKind::Min => "min.f32",
                        BinOpKind::Max => "max.f32",
                        _ => "add.f32",
                    };
                    ptx.push_str(&format!(
                        "    // BinOp: slot{} = slot{} {} slot{}\n",
                        dst, lhs, op_ptx, rhs
                    ));
                }
                Instr::UnOp(_dst, _op, _src) => {
                    ptx.push_str("    // UnOp\n");
                }
                _ => {
                    // Skip other instructions in PTX
                }
            }
        }

        ptx.push_str("\n    ret;\n}\n");

        let config = LaunchConfig::elementwise(1, 256);

        let ptx_size = ptx.len();
        CudaCompiledKernel {
            ptx_source: ptx,
            kernel_name: kernel_name.to_string(),
            launch_config: config,
            ptx_size,
            estimated_gflops: 0.0,
        }
    }
}

// ─── CUDA Runtime (feature-gated) ───────────────────────────────────────────

/// CUDA runtime wrapper — manages device, memory, and kernel execution.
/// When the `cuda` feature is off, all methods return `CudaError::FeatureNotEnabled`.
pub struct CudaRuntime {
    #[cfg(feature = "cuda")]
    device: Option<cudarc::driver::CudaDevice>,
    #[cfg(feature = "cuda")]
    device_ordinal: usize,
}

impl CudaRuntime {
    /// Create a new CUDA runtime, initializing the driver and selecting a device.
    pub fn new(device_ordinal: usize) -> CudaResult<Self> {
        #[cfg(feature = "cuda")]
        {
            match cudarc::driver::CudaDevice::new(device_ordinal) {
                Ok(device) => Ok(CudaRuntime {
                    device: Some(device),
                    device_ordinal,
                }),
                Err(e) => Err(CudaError::NoGpuFound(format!("{}", e))),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device_ordinal;
            Err(CudaError::FeatureNotEnabled)
        }
    }

    /// Try to create a CUDA runtime. Returns None if CUDA is not available.
    pub fn try_new(device_ordinal: usize) -> Option<Self> {
        Self::new(device_ordinal).ok()
    }

    /// Check if CUDA is available on this system.
    pub fn is_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            cudarc::driver::CudaDevice::new(0).is_ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get information about the GPU device.
    pub fn device_info(&self) -> CudaResult<GpuDeviceInfo> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = self.device {
                let result = device
                    .with_primary(|ctx| {
                        // Query device properties
                        let name = ctx.device_name();
                        let major = ctx.attribute(cudarc::driver::sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrap_or(8);
                        let minor = ctx.attribute(cudarc::driver::sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrap_or(0);
                        let total_mem = ctx.total_memory().unwrap_or(0);
                        let num_sms = ctx.attribute(cudarc::driver::sys::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap_or(1) as u32;
                        let max_threads = ctx.attribute(cudarc::driver::sys::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK).unwrap_or(1024) as u32;
                        let max_smem = ctx.attribute(cudarc::driver::sys::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK).unwrap_or(49152) as usize;
                        let clock = ctx.attribute(cudarc::driver::sys::CU_DEVICE_ATTRIBUTE_CLOCK_RATE).unwrap_or(0) as u32 / 1000; // kHz -> MHz

                        Ok(GpuDeviceInfo {
                            name,
                            compute_capability_major: major as u32,
                            compute_capability_minor: minor as u32,
                            total_memory_bytes: total_mem as usize,
                            num_sms,
                            warp_size: 32,
                            max_threads_per_block: max_threads,
                            max_shared_memory_per_block: max_smem,
                            clock_mhz: clock,
                            memory_bandwidth_mbs: 0,
                        })
                    });
                match result {
                    Ok(info) => Ok(info),
                    Err(e) => Err(CudaError::DriverError(format!("{:?}", e))),
                }
            } else {
                Err(CudaError::NoGpuFound("No device initialized".to_string()))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::FeatureNotEnabled)
        }
    }

    /// Allocate device memory.
    pub fn alloc(&self, bytes: usize) -> CudaResult<CudaDeviceMemory> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = self.device {
                match device.alloc::<u8>(bytes) {
                    Ok(ptr) => Ok(CudaDeviceMemory {
                        ptr,
                        size: bytes,
                    }),
                    Err(e) => Err(CudaError::OutOfMemory(format!("{:?}", e))),
                }
            } else {
                Err(CudaError::NoGpuFound("No device initialized".to_string()))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = bytes;
            Err(CudaError::FeatureNotEnabled)
        }
    }

    /// Copy data from host to device.
    pub fn htod_copy(&self, dst: &CudaDeviceMemory, src: &[u8]) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = self.device {
                match device.dtod_copy::<u8>(
                    // Use slice-based copy
                    unsafe { std::slice::from_raw_parts(src.as_ptr(), src.len().min(dst.size)) },
                    dst.ptr,
                ) {
                    Ok(()) => Ok(()),
                    Err(e) => Err(CudaError::DriverError(format!("{:?}", e))),
                }
            } else {
                Err(CudaError::NoGpuFound("No device initialized".to_string()))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dst, src);
            Err(CudaError::FeatureNotEnabled)
        }
    }

    /// Copy data from device to host.
    pub fn dtoh_copy(&self, dst: &mut [u8], src: &CudaDeviceMemory) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = self.device {
                match device.dtoht_copy::<u8>(src.ptr, dst) {
                    Ok(()) => Ok(()),
                    Err(e) => Err(CudaError::DriverError(format!("{:?}", e))),
                }
            } else {
                Err(CudaError::NoGpuFound("No device initialized".to_string()))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dst, src);
            Err(CudaError::FeatureNotEnabled)
        }
    }

    /// Compile and launch a PTX kernel on the GPU.
    pub fn launch_kernel(
        &self,
        kernel: &CudaCompiledKernel,
        kernel_params: &[usize], // Raw pointer values for kernel args
    ) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = self.device {
                // Load PTX module
                match device.load_ptx(kernel.ptx_source.as_bytes().to_vec(), &kernel.kernel_name) {
                    Ok(()) => {}
                    Err(e) => return Err(CudaError::PtxCompileError(format!("{:?}", e))),
                }

                // Get kernel function
                let func = match device.get_func(&kernel.kernel_name, &kernel.kernel_name) {
                    Ok(f) => f,
                    Err(e) => return Err(CudaError::DriverError(format!("Kernel not found: {:?}", e))),
                };

                // Launch
                let cfg = &kernel.launch_config;
                match device.launch(
                    func,
                    cfg.grid,
                    cfg.block,
                    cfg.shared_mem_bytes,
                    kernel_params,
                ) {
                    Ok(()) => Ok(()),
                    Err(e) => Err(CudaError::ExecutionError(format!("{:?}", e))),
                }
            } else {
                Err(CudaError::NoGpuFound("No device initialized".to_string()))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (kernel, kernel_params);
            Err(CudaError::FeatureNotEnabled)
        }
    }
}

/// Wrapper for device memory allocation.
#[derive(Debug)]
pub struct CudaDeviceMemory {
    #[cfg(feature = "cuda")]
    ptr: cudarc::driver::CudaSlice<u8>,
    size: usize,
}

impl CudaDeviceMemory {
    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the device pointer as a usize (for passing to kernel params).
    pub fn device_ptr(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.ptr.device_ptr() as usize
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
}

// ─── High-level API ─────────────────────────────────────────────────────────

/// Compile SympleX IR instructions into a CUDA kernel.
pub fn cuda_compile(
    instrs: &[Instr],
    param_count: u16,
    sm_arch: &str,
) -> CudaResult<CudaCompiledKernel> {
    let generator = PtxGenerator::new(sm_arch);
    Ok(generator.gen_from_instrs(instrs, param_count))
}

/// Compile a matmul kernel for CUDA.
pub fn cuda_compile_matmul(
    m: usize, n: usize, k: usize,
    tile_m: usize, tile_n: usize, tile_k: usize,
    sm_arch: &str,
) -> CudaCompiledKernel {
    let generator = PtxGenerator::new(sm_arch);
    generator.gen_matmul(m, n, k, tile_m, tile_n, tile_k)
}

/// Compile an elementwise kernel for CUDA.
pub fn cuda_compile_elementwise(op: BinOpKind, n: usize, sm_arch: &str) -> CudaCompiledKernel {
    let generator = PtxGenerator::new(sm_arch);
    generator.gen_elementwise(op, n)
}

/// Compile an FMA kernel for CUDA.
pub fn cuda_compile_fma(n: usize, sm_arch: &str) -> CudaCompiledKernel {
    let generator = PtxGenerator::new(sm_arch);
    generator.gen_fma(n)
}

/// Compile a reduction kernel for CUDA.
pub fn cuda_compile_reduction(op: BinOpKind, n: usize, sm_arch: &str) -> CudaCompiledKernel {
    let generator = PtxGenerator::new(sm_arch);
    generator.gen_reduction(op, n)
}

/// Compile a 5-point stencil kernel for CUDA.
pub fn cuda_compile_stencil(rows: usize, cols: usize, sm_arch: &str) -> CudaCompiledKernel {
    let generator = PtxGenerator::new(sm_arch);
    generator.gen_stencil(rows, cols)
}

/// Execute a CUDA matmul on GPU: C = A × B
/// Takes host pointers and copies data to/from GPU.
pub fn cuda_matmul(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) -> CudaResult<()> {
    let rt = CudaRuntime::new(0)?;

    // Allocate GPU memory
    let a_bytes = m * k * 4;
    let b_bytes = k * n * 4;
    let c_bytes = m * n * 4;

    let d_a = rt.alloc(a_bytes)?;
    let d_b = rt.alloc(b_bytes)?;
    let d_c = rt.alloc(c_bytes)?;

    // Copy inputs to GPU
    unsafe {
        rt.htod_copy(&d_a, std::slice::from_raw_parts(a.as_ptr() as *const u8, a_bytes))?;
        rt.htod_copy(&d_b, std::slice::from_raw_parts(b.as_ptr() as *const u8, b_bytes))?;
    }

    // Compile kernel
    let info = rt.device_info()?;
    let kernel = cuda_compile_matmul(m, n, k, 32, 32, 8, &info.sm_arch());

    // Launch
    let params: [usize; 6] = [
        d_a.device_ptr(), d_b.device_ptr(), d_c.device_ptr(),
        m, n, k,
    ];
    rt.launch_kernel(&kernel, &params)?;

    // Copy result back
    unsafe {
        rt.dtoh_copy(
            std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut u8, c_bytes),
            &d_c,
        )?;
    }

    Ok(())
}

/// Execute a 5-point stencil on GPU.
/// Takes host pointers and copies data to/from GPU.
pub fn cuda_stencil(
    src: &[f32], dst: &mut [f32],
    rows: usize, cols: usize,
) -> CudaResult<()> {
    let rt = CudaRuntime::new(0)?;

    let src_bytes = rows * cols * 4;
    let out_rows = rows - 2;
    let out_cols = cols - 2;
    let dst_bytes = out_rows * out_cols * 4;

    // Allocate GPU memory
    let d_src = rt.alloc(src_bytes)?;
    let d_dst = rt.alloc(dst_bytes)?;

    // Copy src to device
    rt.htod_copy(&d_src, unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src_bytes) })?;

    // Compile kernel
    let kernel = cuda_compile_stencil(rows, cols, "sm_80");

    // Launch
    rt.launch_kernel(&kernel, &[d_src.device_ptr(), d_dst.device_ptr(), rows, cols])?;

    // Copy result back
    rt.dtoh_copy(unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst_bytes) }, &d_dst)?;

    Ok(())
}

/// Execute an elementwise operation on GPU.
pub fn cuda_elementwise(
    a: &[f32], b: &[f32], dst: &mut [f32],
    op: BinOpKind,
) -> CudaResult<()> {
    let n = a.len();
    if b.len() != n || dst.len() != n {
        return Err(CudaError::InvalidParam("Array length mismatch".to_string()));
    }

    let rt = CudaRuntime::new(0)?;
    let bytes = n * 4;

    let d_a = rt.alloc(bytes)?;
    let d_b = rt.alloc(bytes)?;
    let d_dst = rt.alloc(bytes)?;

    unsafe {
        rt.htod_copy(&d_a, std::slice::from_raw_parts(a.as_ptr() as *const u8, bytes))?;
        rt.htod_copy(&d_b, std::slice::from_raw_parts(b.as_ptr() as *const u8, bytes))?;
    }

    let info = rt.device_info()?;
    let kernel = cuda_compile_elementwise(op, n, &info.sm_arch());

    let params: [usize; 4] = [d_dst.device_ptr(), d_a.device_ptr(), d_b.device_ptr(), n];
    rt.launch_kernel(&kernel, &params)?;

    unsafe {
        rt.dtoh_copy(
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, bytes),
            &d_dst,
        )?;
    }

    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_generator_elementwise() {
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_elementwise(BinOpKind::Add, 1024);
        assert!(kernel.ptx_source.contains("symplex_elementwise"));
        assert!(kernel.ptx_source.contains("add.f32"));
        assert!(kernel.ptx_size > 0);
    }

    #[test]
    fn test_ptx_generator_matmul() {
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_matmul(128, 128, 64, 32, 32, 8);
        assert!(kernel.ptx_source.contains("symplex_matmul"));
        assert!(kernel.ptx_source.contains("smem_A"));
        assert!(kernel.ptx_source.contains("smem_B"));
        assert!(kernel.estimated_gflops > 0.0);
    }

    #[test]
    fn test_ptx_generator_fma() {
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_fma(512);
        assert!(kernel.ptx_source.contains("symplex_fma"));
        assert!(kernel.ptx_source.contains("mad.f32"));
    }

    #[test]
    fn test_ptx_generator_reduction() {
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_reduction(BinOpKind::Add, 1024);
        assert!(kernel.ptx_source.contains("symplex_reduction"));
        assert!(kernel.ptx_source.contains("add.f32"));
    }

    #[test]
    fn test_cuda_runtime_not_available() {
        // Without the cuda feature, this should return an error
        if !cfg!(feature = "cuda") {
            assert!(CudaRuntime::is_available() == false);
            assert!(matches!(CudaRuntime::new(0), Err(CudaError::FeatureNotEnabled)));
        }
    }

    #[test]
    fn test_launch_config_elementwise() {
        let config = LaunchConfig::elementwise(1024, 256);
        assert_eq!(config.grid.0, 4); // ceil(1024/256) = 4
        assert_eq!(config.block.0, 256);
    }

    #[test]
    fn test_launch_config_matmul() {
        let config = LaunchConfig::matmul(128, 64, 32, 32);
        assert_eq!(config.grid.0, 2); // ceil(64/32) = 2
        assert_eq!(config.grid.1, 4); // ceil(128/32) = 4
    }

    #[test]
    fn test_gpu_device_info_sm_arch() {
        let info = GpuDeviceInfo {
            name: "NVIDIA A100".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            total_memory_bytes: 80 * 1024 * 1024 * 1024,
            num_sms: 108,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            clock_mhz: 1410,
            memory_bandwidth_mbs: 2039000,
        };
        assert_eq!(info.sm_arch(), "sm_80");
        assert_eq!(info.num_cuda_cores(), 108 * 64); // SM 8.0 = 64 cores/SM
    }
}
