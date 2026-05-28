// SympleX — MCMC Compiler Mode Header
//
// MCMC mode is a compiler POLICY LAYER, not a standalone compiler.
// It plugs into the existing IR + JIT pipeline and controls:
//   - Fusion aggressiveness (high for energy functions)
//   - Inlining thresholds (high inside kernel, stop at stochastic boundary)
//   - Vectorization policy (enabled for deterministic math)
//   - Cache policy (shape-based for transition kernels)
//   - Trace boundary (stop at stochastic operations)
//
// CRITICAL RULES:
//   1. NEVER compile the outer MCMC loop
//   2. NEVER include RNG in IR optimization
//   3. NEVER trace across stochastic boundaries
//   4. Compile ONLY the deterministic transition kernel

#pragma once

#include <cstdint>
#include <string>

namespace symplex::core::modes {

enum class FusionAggressiveness : uint8_t {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,  // MCMC default: fuse energy function terms aggressively
};

enum class InliningThreshold : uint8_t {
    None = 0,
    Low = 1,
    Medium = 2,
    High = 3,  // MCMC default: inline everything inside the kernel
};

enum class CachePolicy : uint8_t {
    None = 0,
    ShapeBased = 1,           // MCMC default
    ShapeAndDtypeBased = 2,
};

enum class TraceBoundary : uint8_t {
    None = 0,            // Trace everything
    StochasticStop = 1,  // MCMC default: stop at stochastic ops
};

struct McmcPolicy {
    FusionAggressiveness fusion_aggressiveness = FusionAggressiveness::High;
    InliningThreshold inlining_threshold = InliningThreshold::High;
    bool vectorization = true;
    CachePolicy cache_policy = CachePolicy::ShapeBased;
    TraceBoundary trace_boundary = TraceBoundary::StochasticStop;
};

class McmcMode {
public:
    explicit McmcMode(const McmcPolicy& policy = McmcPolicy{});

    // Apply MCMC policy to the compilation pipeline.
    // This modifies optimization parameters, NOT program structure.
    void apply() const;

    // Check if a trace should stop at this instruction
    // (stochastic boundary detection)
    bool should_stop_trace(const std::string& opcode) const;

    // Get the effective fusion aggressiveness for this mode
    FusionAggressiveness fusion_level() const { return policy_.fusion_aggressiveness; }

    // Get the effective inlining threshold
    InliningThreshold inlining_level() const { return policy_.inlining_threshold; }

    // Should we vectorize in this mode?
    bool should_vectorize() const { return policy_.vectorization; }

    const McmcPolicy& policy() const { return policy_; }

private:
    McmcPolicy policy_;
};

} // namespace symplex::core::modes
