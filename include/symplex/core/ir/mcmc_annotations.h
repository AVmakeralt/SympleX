// SympleX — MCMC-Aware IR Annotations
//
// These annotations mark IR nodes with metadata that the MCMC compiler
// mode uses to determine optimization behavior:
//   - DeterministicMath: safe to fuse, inline, vectorize
//   - StochasticBoundary: stop tracing, do not optimize across
//   - EnergyFunction: candidate for aggressive fusion
//   - ProposalFunction: deterministic part of proposal distribution

#pragma once

#include <cstdint>
#include <string>

namespace symplex::core::ir {

enum class IrAnnotationKind : uint8_t {
    None = 0,
    DeterministicMath = 1,   // Pure math: safe to optimize aggressively
    StochasticBoundary = 2,  // RNG/sampling: do NOT optimize across
    EnergyFunction = 3,      // Energy function: fuse all terms
    ProposalFunction = 4,    // Proposal distribution: deterministic part only
    AcceptRejectStep = 5,    // Accept/reject: must stay outside JIT
    GradientKernel = 6,      // Gradient of energy: from HMC/Langevin
};

struct IrAnnotation {
    IrAnnotationKind kind = IrAnnotationKind::None;
    uint16_t slot = 0;        // Which slot this annotation applies to
    uint16_t metadata = 0;    // Optional metadata (e.g., energy function ID)

    static IrAnnotation deterministic(uint16_t slot) {
        return { IrAnnotationKind::DeterministicMath, slot, 0 };
    }
    static IrAnnotation stochastic_boundary(uint16_t slot) {
        return { IrAnnotationKind::StochasticBoundary, slot, 0 };
    }
    static IrAnnotation energy_function(uint16_t slot, uint16_t id = 0) {
        return { IrAnnotationKind::EnergyFunction, slot, id };
    }
    static IrAnnotation proposal_function(uint16_t slot) {
        return { IrAnnotationKind::ProposalFunction, slot, 0 };
    }
    static IrAnnotation accept_reject(uint16_t slot) {
        return { IrAnnotationKind::AcceptRejectStep, slot, 0 };
    }
    static IrAnnotation gradient_kernel(uint16_t slot) {
        return { IrAnnotationKind::GradientKernel, slot, 0 };
    }

    bool is_deterministic() const {
        return kind == IrAnnotationKind::DeterministicMath ||
               kind == IrAnnotationKind::EnergyFunction ||
               kind == IrAnnotationKind::ProposalFunction ||
               kind == IrAnnotationKind::GradientKernel;
    }

    bool is_stochastic() const {
        return kind == IrAnnotationKind::StochasticBoundary ||
               kind == IrAnnotationKind::AcceptRejectStep;
    }

    bool should_stop_trace() const {
        return kind == IrAnnotationKind::StochasticBoundary ||
               kind == IrAnnotationKind::AcceptRejectStep;
    }
};

} // namespace symplex::core::ir
