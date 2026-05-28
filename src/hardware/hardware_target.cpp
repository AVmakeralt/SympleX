// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/hardware/hardware_target.h"

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

namespace symplex::hardware {

// ── Helpers ──────────────────────────────────────────────────────────────

namespace {

/// Read the entire contents of a file into a string.  Returns empty on failure.
std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return {};
    std::string contents((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());
    return contents;
}

/// Enumerate subdirectories under `dir_path`, returning full paths.
std::vector<std::string> list_subdirs(const std::string& dir_path) {
    std::vector<std::string> result;
    DIR* dir = opendir(dir_path.c_str());
    if (!dir) return result;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;
        std::string full = dir_path;
        if (!full.empty() && full.back() != '/') full += '/';
        full += entry->d_name;
        result.push_back(std::move(full));
    }
    closedir(dir);
    return result;
}

/// Case-insensitive substring search.
bool contains_ci(const std::string& haystack, const std::string& needle) {
    auto it = std::search(
        haystack.begin(), haystack.end(),
        needle.begin(), needle.end(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    return it != haystack.end();
}

/// Extract the GPU model name from a `/proc/driver/nvidia/gpus/*/information` blob.
/// The file looks like:
///   Model:           NVIDIA A100-SXM4-80GB
///   IRQ:             47
///   GPU UUID:        GPU-...
///   Video BIOS:      ...
/// We just need the Model line.
std::string extract_gpu_model(const std::string& info) {
    const std::string prefix = "Model:";
    auto pos = info.find(prefix);
    if (pos == std::string::npos) return {};
    pos += prefix.size();
    // Skip whitespace
    while (pos < info.size() && (info[pos] == ' ' || info[pos] == '\t')) ++pos;
    // Read until newline
    auto end = info.find('\n', pos);
    std::string model = info.substr(pos, (end == std::string::npos) ? std::string::npos : end - pos);
    // Trim trailing whitespace
    while (!model.empty() && (model.back() == ' ' || model.back() == '\t' || model.back() == '\r'))
        model.pop_back();
    return model;
}

/// Parse the NVIDIA driver version from `/proc/driver/nvidia/version`.
/// Returns an empty string on failure.
std::string read_driver_version() {
    std::string ver = read_file("/proc/driver/nvidia/version");
    if (ver.empty()) return {};
    // Typical format: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.129.03  ..."
    auto pos = ver.find("Kernel Module");
    if (pos == std::string::npos) return {};
    pos += 13;  // strlen("Kernel Module")
    while (pos < ver.size() && ver[pos] == ' ') ++pos;
    auto end = ver.find(' ', pos);
    return ver.substr(pos, (end == std::string::npos) ? std::string::npos : end - pos);
}

/// Try to read compute capability via `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
/// Returns e.g. "9.0" or empty string on failure.
std::string query_compute_cap_smi() {
    std::string result;
    // Use popen to run nvidia-smi
    FILE* pipe = popen("nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null", "r");
    if (!pipe) return {};
    char buf[256];
    if (fgets(buf, sizeof(buf), pipe)) {
        result = buf;
        // Trim whitespace / newline
        while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
            result.pop_back();
    }
    pclose(pipe);
    return result;
}

/// Parse "X.Y" compute capability string into a packed integer (major*10 + minor).
/// Returns -1 on parse failure.
int parse_compute_cap(const std::string& cap_str) {
    auto dot = cap_str.find('.');
    if (dot == std::string::npos) return -1;
    try {
        int major = std::stoi(cap_str.substr(0, dot));
        int minor = std::stoi(cap_str.substr(dot + 1));
        return major * 10 + minor;
    } catch (...) {
        return -1;
    }
}

/// Map a GPU model name to the appropriate HardwareTarget factory.
HardwareTarget model_to_target(const std::string& model) {
    // Hopper
    if (contains_ci(model, "H100") || contains_ci(model, "H200"))
        return HardwareTarget::H100();
    // Blackwell
    if (contains_ci(model, "B200") || contains_ci(model, "B100"))
        return HardwareTarget::B200();
    // Ampere
    if (contains_ci(model, "A100") || contains_ci(model, "A30") ||
        contains_ci(model, "A40") || contains_ci(model, "A10"))
        return HardwareTarget::A100();
    // Volta
    if (contains_ci(model, "V100") || contains_ci(model, "Titan V"))
        return HardwareTarget::V100();
    // Turing (no dedicated factory yet – closest to A100 with lower specs)
    if (contains_ci(model, "T4") || contains_ci(model, "RTX 2080") ||
        contains_ci(model, "RTX 2070") || contains_ci(model, "RTX 2060") ||
        contains_ci(model, "GTX 16"))
        return HardwareTarget::Generic();

    // Unknown model – try compute capability as fallback
    std::string cap_str = query_compute_cap_smi();
    if (!cap_str.empty()) {
        int cap = parse_compute_cap(cap_str);
        if (cap >= 90) return HardwareTarget::H100();   // Hopper+
        if (cap >= 80) return HardwareTarget::A100();   // Ampere+
        if (cap >= 70) return HardwareTarget::V100();   // Volta/Turing
    }

    // Truly unknown – use generic
    return HardwareTarget::Generic();
}

} // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────────

HardwareTarget detect_hardware_target() {
    // 1. Scan /proc/driver/nvidia/gpus/*/information for the first GPU.
    std::vector<std::string> gpu_dirs =
        list_subdirs("/proc/driver/nvidia/gpus");

    for (const auto& dir : gpu_dirs) {
        std::string info_path = dir + "/information";
        std::string info = read_file(info_path);
        if (info.empty()) continue;

        std::string model = extract_gpu_model(info);
        if (!model.empty()) {
            // Found an NVIDIA GPU – map to a HardwareTarget preset
            return model_to_target(model);
        }
    }

    // 2. No GPU found via /proc – try nvidia-smi as a fallback.
    //    If nvidia-smi is available, parse the first GPU name.
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null", "r");
    if (pipe) {
        char buf[256];
        if (fgets(buf, sizeof(buf), pipe)) {
            std::string name = buf;
            // Trim trailing whitespace / newline
            while (!name.empty() && (name.back() == '\n' || name.back() == '\r' || name.back() == ' '))
                name.pop_back();
            pclose(pipe);
            if (!name.empty() && name.find("No devices") == std::string::npos) {
                return model_to_target(name);
            }
        } else {
            pclose(pipe);
        }
    }

    // 3. No NVIDIA GPU detected – fall back to generic.
    return HardwareTarget::Generic();
}

} // namespace symplex::hardware
