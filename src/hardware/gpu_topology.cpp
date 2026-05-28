// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.
#include "symplex/hardware/hardware_target.h"
namespace symplex::hardware {
// All GPU topology types are header-inline.
// This TU verifies compilation and provides a runtime smoke test.
bool validate_hardware_target(const HardwareTarget& t) {
    if (t.gpu.num_sms <= 0) return false;
    if (t.gpu.warp_size <= 0) return false;
    if (t.max_sram_bytes <= 0) return false;
    if (t.bytes_per_element <= 0) return false;
    if (t.gpu.tensor_core.m <= 0 || t.gpu.tensor_core.n <= 0 || t.gpu.tensor_core.k <= 0) return false;
    return true;
}
} // namespace symplex::hardware
