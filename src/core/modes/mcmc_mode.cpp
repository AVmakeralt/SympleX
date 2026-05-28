// SympleX — MCMC Compiler Mode Implementation
//
// Policy-only layer that controls optimization behavior for MCMC workloads.
// Does NOT contain any compiler logic — only configuration that the
// existing IR + JIT pipeline reads.

#include "symplex/core/modes/mcmc_mode.h"

namespace symplex::core::modes {

McmcMode::McmcMode(const McmcPolicy& policy)
    : policy_(policy) {}

void McmcMode::apply() const {
    // The actual effect of this policy is read by:
    //   - The polyhedral optimizer (fusion aggressiveness)
    //   - The inliner (threshold)
    //   - The vectorizer (enabled/disabled)
    //   - The JIT cache (shape-based lookup)
    //   - The tracer (stochastic boundary detection)
    //
    // This method is called by the pipeline before compilation begins.
    // The pipeline reads policy_ fields and adjusts its behavior accordingly.
    //
    // Example: when fusion_aggressiveness == High, the polyhedral optimizer
    // will attempt to fuse all energy function terms into a single kernel,
    // avoiding intermediate array materialization.
}

bool McmcMode::should_stop_trace(const std::string& opcode) const {
    if (policy_.trace_boundary != TraceBoundary::StochasticStop) {
        return false;
    }
    // Stop tracing at stochastic operations:
    //   - random_* (any RNG call)
    //   - sample (sampling from a distribution)
    //   - accept_reject (MCMC accept/reject step)
    //   - any non-deterministic operation
    if (opcode.find("random_") == 0 ||
        opcode == "sample" ||
        opcode == "accept_reject" ||
        opcode == "categorical" ||
        opcode == "poisson" ||
        opcode == "gaussian_sample") {
        return true;
    }
    return false;
}

} // namespace symplex::core::modes
