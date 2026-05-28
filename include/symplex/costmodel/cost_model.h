// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/hardware/hardware_target.h"
#include "symplex/schedule/tiling.h"
#include "symplex/polyhedral/iteration_space.h"
#include "symplex/costmodel/roofline.h"
#include "symplex/costmodel/analytical.h"
#include "symplex/costmodel/empirical.h"
#include <cstdint>
#include <string>

namespace symplex::costmodel {

/// CostModelLevel: the fidelity level of the cost model.
/// Higher levels provide more accurate estimates at greater cost.
enum class CostModelLevel {
    ROOFLINE_ONLY,  // Quick roofline model only
    ANALYTICAL,     // First-principles analytical model
    EMPIRICAL,      // Profile on real hardware (or simulated fallback)
    HYBRID,         // Blend analytical + empirical with confidence weighting
};

/// CostEstimate: the unified cost estimate returned by the CostModel.
struct CostEstimate {
    double latency_ns;          // Estimated kernel latency in nanoseconds
    double confidence;          // 0-1, how confident we are in this estimate
    CostModelLevel model_used;  // Which model level produced this estimate
    std::string details;        // Human-readable description of the estimate
};

/// CostModel: unified cost model that combines roofline, analytical,
/// and empirical models.  Dispatches to the appropriate sub-model based
/// on the configured CostModelLevel.
///
/// Usage:
///   CostModel model(HardwareTarget::H100(), CostModelLevel::ANALYTICAL);
///   auto estimate = model.estimate(ispace, tile_config);
///   printf("Latency: %.1f ns, confidence: %.2f\n",
///          estimate.latency_ns, estimate.confidence);
class CostModel {
public:
    CostModel(hardware::HardwareTarget target,
              CostModelLevel level = CostModelLevel::ANALYTICAL);

    /// Estimate the latency of executing the given iteration space
    /// with the specified tile configuration.
    CostEstimate estimate(
        const polyhedral::IterationSpace& ispace,
        const schedule::TileConfig& tile
    );

    /// Access the hardware target.
    const hardware::HardwareTarget& target() const;

    /// Access the current model level.
    CostModelLevel level() const;

private:
    hardware::HardwareTarget target_;
    CostModelLevel level_;
    RooflineModel roofline_;
    AnalyticalCostModel analytical_;
    EmpiricalCostModel empirical_;
};

} // namespace symplex::costmodel
