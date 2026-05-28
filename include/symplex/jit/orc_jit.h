// SympleX – Polyhedral Tensor Superoptimizer
// LLVM ORC JIT Integration Layer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
//
// This module provides the JIT compilation infrastructure for SympleX.
// When LLVM is available (SYMPLEX_HAS_LLVM defined), it uses the real
// LLVM ORC JIT v2 API for:
//   - JIT compilation of MLIR → LLVM IR → machine code
//   - AOT + JIT hybrid compilation
//   - CUDA via LLVM PTX backend
//   - Symbol resolution and linking
//   - Lazy compilation via ORC v2
//
// When LLVM is NOT available, it falls back to:
//   - Pre-compiled kernel dispatch via mmap'd executable memory
//   - Interpreted execution for basic tensor ops (add, mul, matmul)

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <chrono>

namespace symplex::ir {
    class SympleXIR;
}

namespace symplex::optimizer {
    struct SuperoptimizerResult;
}

namespace symplex::jit {

// ─────────────────────────────────────────────────────────────────────────
// JIT Target Configuration
// ─────────────────────────────────────────────────────────────────────────

/// JITCompileTarget: what machine code target to generate.
enum class JITTarget {
    X86_64,      // Host CPU
    NVIDIA_GPU,  // PTX for CUDA
};

/// JITConfig: configuration for the JIT compiler.
struct JITConfig {
    JITTarget target = JITTarget::X86_64;
    bool verify_module = true;           // Run LLVM verification
    bool optimize = true;                // Run LLVM optimization passes
    int opt_level = 3;                   // Optimization level (0-3)
    bool enable_debug_info = false;      // Emit debug info
    bool lazy_compilation = true;        // Compile on first call
    size_t code_cache_size = 256 * 1024 * 1024; // 256MB code cache
    std::string cpu_features;            // Target CPU features string
};

// ─────────────────────────────────────────────────────────────────────────
// JITSymbol: Compiled Function Handle
// ─────────────────────────────────────────────────────────────────────────

/// JITSymbol: a compiled function handle.
/// Wraps a raw function pointer obtained from JIT compilation.
class JITSymbol {
public:
    JITSymbol() : address_(nullptr) {}
    explicit JITSymbol(void* addr) : address_(addr) {}

    /// Get the raw function pointer cast to the desired type.
    template<typename Fn>
    Fn as() const { return reinterpret_cast<Fn>(address_); }

    /// Call the compiled function with the given arguments.
    /// Usage: sym.call<int(int)>(42)  or  sym.call<void(float*)>(ptr)
    template<typename Ret, typename... Args>
    Ret call(Args... args) {
        auto fn = as<Ret(*)(Args...)>();
        return fn(args...);
    }

    /// Is this a valid (non-null) symbol?
    bool valid() const { return address_ != nullptr; }

    /// Get the raw address.
    void* raw() const { return address_; }

private:
    void* address_;
};

// ─────────────────────────────────────────────────────────────────────────
// KernelArgs: Argument Bundle for Tensor Kernel Invocation
// ─────────────────────────────────────────────────────────────────────────

/// KernelArgs: argument bundle for tensor kernel invocation.
/// This is the common interface for passing arguments to JIT-compiled
/// tensor kernels. It supports arbitrary-arity tensor operations with
/// shape metadata.
struct KernelArgs {
    void*   data = nullptr;          // Pointer to output tensor data
    int64_t data_size = 0;           // Size of output in bytes
    void**  input_ptrs = nullptr;    // Array of input tensor pointers
    int64_t num_inputs = 0;          // Number of inputs
    int64_t* input_sizes = nullptr;  // Size of each input in bytes
    int64_t M = 0;                   // Matrix dimension M (for matmul)
    int64_t N = 0;                   // Matrix dimension N (for matmul)
    int64_t K = 0;                   // Matrix dimension K (for matmul)
    int64_t batch = 1;               // Batch dimension
    // ... extend as needed
};

// ─────────────────────────────────────────────────────────────────────────
// ORCJIT: LLVM ORC JIT Compilation Engine
// ─────────────────────────────────────────────────────────────────────────

/// ORCJIT: LLVM ORC JIT compilation engine.
///
/// This is the "serious systems software" JIT backend.
/// When LLVM is available, this provides:
///   - JIT compilation of MLIR → LLVM IR → machine code
///   - AOT + JIT hybrid compilation
///   - CUDA via LLVM PTX backend
///   - Symbol resolution and linking
///   - Lazy compilation via ORC v2
///
/// When LLVM is NOT available, this falls back to:
///   - Pre-compiled kernel dispatch
///   - Interpreted execution for debugging
class ORCJIT {
public:
    /// Construct a JIT with the given configuration.
    explicit ORCJIT(JITConfig config = {});

    /// Destructor: releases all JIT resources.
    ~ORCJIT();

    // No copying
    ORCJIT(const ORCJIT&) = delete;
    ORCJIT& operator=(const ORCJIT&) = delete;

    // Move-only
    ORCJIT(ORCJIT&&) noexcept;
    ORCJIT& operator=(ORCJIT&&) noexcept;

    // ── Compilation ─────────────────────────────────────────────────

    /// Compile MLIR assembly text to machine code.
    /// Returns a JITSymbol for the compiled kernel.
    /// The kernel_name must match a func.func in the MLIR module.
    JITSymbol compile_mlir(const std::string& mlir_text,
                           const std::string& kernel_name);

    /// Compile LLVM IR assembly text to machine code.
    JITSymbol compile_llvm_ir(const std::string& llvm_ir_text,
                              const std::string& function_name);

    /// Compile a pre-built kernel from the code generator.
    /// This bypasses MLIR/LLVM and uses the code generator's output directly.
    /// The machine_code bytes are copied into executable memory via mmap.
    JITSymbol compile_native(const std::vector<uint8_t>& machine_code,
                             const std::string& symbol_name);

    // ── Symbol Resolution ───────────────────────────────────────────

    /// Look up a symbol by name.
    JITSymbol lookup(const std::string& name);

    /// Add an external symbol for resolution.
    void add_symbol(const std::string& name, void* address);

    // ── Execution ───────────────────────────────────────────────────

    /// Execute a compiled kernel with the given arguments.
    /// This is the main "run" entry point.
    int execute(const std::string& kernel_name, KernelArgs& args);

    /// Execute with raw function pointer and arguments.
    int execute(JITSymbol sym, KernelArgs& args);

    // ── Diagnostics ─────────────────────────────────────────────────

    /// Get the last error message.
    const std::string& last_error() const;

    /// Is the JIT available? (Returns true if LLVM is linked)
    static bool is_available();

    /// Get the target triple.
    std::string target_triple() const;

    /// Get information about compiled kernels.
    struct KernelInfo {
        std::string name;
        size_t code_size = 0;
        double compile_time_ms = 0.0;
        int call_count = 0;
    };

    std::vector<KernelInfo> compiled_kernels() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;  // Pimpl to avoid LLVM header dependency

    JITConfig config_;
    std::string last_error_;
    std::unordered_map<std::string, JITSymbol> symbol_table_;
    std::vector<KernelInfo> kernel_info_;

    /// Set an error message and return an invalid symbol.
    JITSymbol set_error(const std::string& msg);

    /// Internal: mmap-based native code allocation.
    void* allocate_executable_memory(size_t size);

    /// Internal: fallback interpreter for basic tensor ops.
    int interpret_kernel(const std::string& kernel_name, KernelArgs& args);

    /// Internal: simple CPU matmul implementation for the fallback interpreter.
    static void cpu_matmul_fp32(float* C, const float* A, const float* B,
                                int64_t M, int64_t N, int64_t K);

    /// Internal: simple CPU elementwise add for the fallback interpreter.
    static void cpu_add_fp32(float* C, const float* A, const float* B,
                             int64_t n);

    /// Internal: simple CPU elementwise multiply for the fallback interpreter.
    static void cpu_mul_fp32(float* C, const float* A, const float* B,
                             int64_t n);

    // JITSession needs access to set_error()
    friend class JITSession;
};

// ─────────────────────────────────────────────────────────────────────────
// JITSession: Full Pipeline Session Manager
// ─────────────────────────────────────────────────────────────────────────

/// JITSession: manages a complete JIT compilation session.
/// Ties together the entire pipeline:
///   Python tracer → SympleX IR → e-graph optimization → polyhedral schedule
///   → MLIR lowering → LLVM ORC JIT → execution
class JITSession {
public:
    JITSession();
    ~JITSession();

    // No copying
    JITSession(const JITSession&) = delete;
    JITSession& operator=(const JITSession&) = delete;

    // Move-only
    JITSession(JITSession&&) noexcept;
    JITSession& operator=(JITSession&&) noexcept;

    /// Optimize and compile a tensor expression.
    /// Full pipeline: IR → optimize → lower → JIT → execute
    ///
    /// \param ir        The SympleX IR module to optimize
    /// \param config    JIT configuration
    /// \return          JITSymbol for the compiled kernel
    JITSymbol optimize_and_compile(const ir::SympleXIR& ir,
                                   JITConfig config = {});

    /// Execute the last compiled kernel.
    int execute(KernelArgs& args);

    /// Get the last compiled MLIR text (for debugging).
    const std::string& last_mlir() const;

    /// Get the optimization result.
    const optimizer::SuperoptimizerResult& last_result() const;

    /// Get the ORCJIT instance (for advanced usage).
    ORCJIT& jit();
    const ORCJIT& jit() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace symplex::jit
