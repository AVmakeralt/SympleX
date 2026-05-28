// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
#include "symplex/hardware/hardware_target.h"
namespace symplex::hardware {
// TensorCoreSpec is header-inline. This TU verifies compilation.
int64_t tensor_core_throughput_ops_per_clock(const TensorCoreSpec& tc) {
    return tc.m * tc.n * tc.k * 2; // FMA = 2 ops
}
} // namespace symplex::hardware
