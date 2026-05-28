// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
#include "symplex/hardware/hardware_target.h"
namespace symplex::hardware {
// MemorySpec is header-inline. This TU verifies compilation.
double effective_bandwidth_gbps(const MemorySpec& mem, double utilization) {
    return mem.global_bw_gbps * utilization;
}
} // namespace symplex::hardware
