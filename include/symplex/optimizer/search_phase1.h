// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/optimizer/tile_config.h"
#include "symplex/hardware/hardware_target.h"
#include <vector>
#include <cstddef>

namespace symplex::optimizer {

/// Phase 1: Memory-Bound Analytic Pruning (The "Roofline Filter")
///
/// Enumerates all hardware-aligned tile configurations for the given
/// dimensionality, then computes operational intensity (FLOPS / byte)
/// for each.  Any configuration that remains memory-bound when a
/// compute-bound alternative is feasible is pruned.
///
/// For a matmul C[M,N] += A[M,K] * B[K,N] the tile metrics are:
///   compute_ops  = 2 * M * N * K   (each FMA = multiply + add)
///   bytes_moved  = (M*K + K*N + M*N) * bytes_per_element
///   OI           = compute_ops / bytes_moved
///
/// A tile survives if its OI >= roofline threshold OR if no tile in
/// the enumeration achieves an OI above the threshold (i.e. we keep
/// the best memory-bound tiles when compute-bound is infeasible).
///
/// \param ndim            Number of tiled dimensions (2 or 3).
/// \param target          Hardware target specification.
/// \param max_tensor_dim  Upper bound on any single tile dimension.
/// \return                Pruned vector of ExtendedTileConfig with
///                        cost annotations populated.
std::vector<ExtendedTileConfig> phase1_roofline_pruning(
    size_t ndim,
    const hardware::HardwareTarget& target,
    int64_t max_tensor_dim = 1024
);

} // namespace symplex::optimizer
