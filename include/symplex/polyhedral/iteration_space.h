// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include "symplex/polyhedral/integer_polytope.h"
#include "symplex/polyhedral/affine_map.h"
#include "symplex/polyhedral/dependency.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <sstream>

namespace symplex::polyhedral {

/// Statement: a single operation (e.g., a tensor read, write, or compute)
/// inside a loop nest, identified by name and with its own iteration domain.
struct Statement {
    std::string            name;
    IntegerPolytope        domain;          // Iteration domain for this statement
    std::vector<AccessRelation> accesses;   // Memory access patterns

    Statement(std::string n, IntegerPolytope d)
        : name(std::move(n)), domain(std::move(d)) {}

    void add_read_access(AffineMap map, std::string name = "") {
        accesses.emplace_back(std::move(map), AccessRelation::READ, std::move(name));
    }

    void add_write_access(AffineMap map, std::string name = "") {
        accesses.emplace_back(std::move(map), AccessRelation::WRITE, std::move(name));
    }
};

/// IterationSpace: the complete mathematical representation of an AI layer's
/// execution domain.  It contains:
///   - A set of statements, each with their own polyhedral domain
///   - The data dependencies between statements
///   - A name for identification (e.g., "conv2d_layer3", "matmul_proj")
class IterationSpace {
public:
    explicit IterationSpace(std::string name = "")
        : name_(std::move(name)) {}

    // ── Statement management ────────────────────────────────────────

    size_t add_statement(Statement stmt) {
        size_t id = statements_.size();
        statements_.push_back(std::move(stmt));
        return id;
    }

    [[nodiscard]] const Statement& statement(size_t id) const {
        return statements_.at(id);
    }

    [[nodiscard]] Statement& statement(size_t id) {
        return statements_.at(id);
    }

    [[nodiscard]] size_t num_statements() const { return statements_.size(); }

    [[nodiscard]] const std::vector<Statement>& statements() const { return statements_; }

    // ── Dependency computation ──────────────────────────────────────

    /// Compute all RAW (true) dependencies between statements.
    void compute_raw_dependencies() {
        raw_deps_.clear();
        for (size_t s1 = 0; s1 < statements_.size(); ++s1) {
            for (size_t s2 = 0; s2 < statements_.size(); ++s2) {
                if (s1 == s2) {
                    // Self-dependency: only include if there's a write and read
                    // to the same array. This captures loop-carried dependencies
                    // like C[m,n] += A[m,k]*B[k,n].
                    bool has_self_dep = false;
                    for (const auto& a1 : statements_[s1].accesses) {
                        if (a1.mode != AccessRelation::WRITE) continue;
                        for (const auto& a2 : statements_[s2].accesses) {
                            if (a2.mode != AccessRelation::READ) continue;
                            if (a1.array_name == a2.array_name && !a1.array_name.empty()) {
                                has_self_dep = true;
                                break;
                            }
                        }
                        if (has_self_dep) break;
                    }
                    if (!has_self_dep) continue;
                }
                // Check if s1 writes to a location that s2 reads
                for (const auto& acc1 : statements_[s1].accesses) {
                    if (acc1.mode != AccessRelation::WRITE) continue;
                    for (const auto& acc2 : statements_[s2].accesses) {
                        if (acc2.mode != AccessRelation::READ) continue;

                        // For self-dependencies, only consider same-array accesses
                        if (s1 == s2 && acc1.array_name != acc2.array_name) continue;

                        // Check compatibility of access map output dimensions
                        if (acc1.access_map.n_out() != acc2.access_map.n_out()) continue;

                        DependencyPolyhedron dep(
                            statements_[s1].domain,
                            statements_[s2].domain,
                            acc1, acc2,
                            DependencyType::RAW
                        );
                        if (dep.has_dependencies()) {
                            raw_deps_.push_back(std::move(dep));
                        }
                    }
                }
            }
        }
    }

    /// Compute all WAR (anti) dependencies.
    void compute_war_dependencies() {
        war_deps_.clear();
        for (size_t s1 = 0; s1 < statements_.size(); ++s1) {
            for (size_t s2 = 0; s2 < statements_.size(); ++s2) {
                if (s1 == s2) {
                    // Self-dependency: only include if there's a read and write
                    // to the same array (skip if write-only statement).
                    bool has_self_dep = false;
                    for (const auto& a1 : statements_[s1].accesses) {
                        if (a1.mode != AccessRelation::READ) continue;
                        for (const auto& a2 : statements_[s2].accesses) {
                            if (a2.mode != AccessRelation::WRITE) continue;
                            if (a1.array_name == a2.array_name && !a1.array_name.empty()) {
                                has_self_dep = true;
                                break;
                            }
                        }
                        if (has_self_dep) break;
                    }
                    if (!has_self_dep) continue;
                }
                for (const auto& acc1 : statements_[s1].accesses) {
                    if (acc1.mode != AccessRelation::READ) continue;
                    for (const auto& acc2 : statements_[s2].accesses) {
                        if (acc2.mode != AccessRelation::WRITE) continue;

                        // For self-dependencies, only consider same-array accesses
                        if (s1 == s2 && acc1.array_name != acc2.array_name) continue;

                        if (acc1.access_map.n_out() != acc2.access_map.n_out()) continue;

                        DependencyPolyhedron dep(
                            statements_[s1].domain,
                            statements_[s2].domain,
                            acc1, acc2,
                            DependencyType::WAR
                        );
                        if (dep.has_dependencies()) {
                            war_deps_.push_back(std::move(dep));
                        }
                    }
                }
            }
        }
    }

    /// Compute all WAW (output) dependencies.
    void compute_waw_dependencies() {
        waw_deps_.clear();
        for (size_t s1 = 0; s1 < statements_.size(); ++s1) {
            for (size_t s2 = 0; s2 < statements_.size(); ++s2) {
                if (s1 > s2) continue;  // Avoid duplicates, but allow s1 == s2 for self-WAW
                for (const auto& acc1 : statements_[s1].accesses) {
                    if (acc1.mode != AccessRelation::WRITE) continue;
                    for (const auto& acc2 : statements_[s2].accesses) {
                        if (acc2.mode != AccessRelation::WRITE) continue;

                        // For self-WAW, skip if same access (same array)
                        // to avoid trivial zero-distance deps when only one write exists
                        if (s1 == s2 && &acc1 == &acc2) continue;

                        if (acc1.access_map.n_out() != acc2.access_map.n_out()) continue;

                        DependencyPolyhedron dep(
                            statements_[s1].domain,
                            statements_[s2].domain,
                            acc1, acc2,
                            DependencyType::WAW
                        );
                        if (dep.has_dependencies()) {
                            waw_deps_.push_back(std::move(dep));
                        }
                    }
                }
            }
        }
    }

    /// Compute all dependency types.
    void compute_all_dependencies() {
        compute_raw_dependencies();
        compute_war_dependencies();
        compute_waw_dependencies();
    }

    // ── Dependency queries ──────────────────────────────────────────

    [[nodiscard]] const std::vector<DependencyPolyhedron>& raw_deps() const { return raw_deps_; }
    [[nodiscard]] const std::vector<DependencyPolyhedron>& war_deps() const { return war_deps_; }
    [[nodiscard]] const std::vector<DependencyPolyhedron>& waw_deps() const { return waw_deps_; }

    /// All dependencies combined.
    [[nodiscard]] std::vector<DependencyPolyhedron> all_dependencies() const {
        std::vector<DependencyPolyhedron> all;
        all.insert(all.end(), raw_deps_.begin(), raw_deps_.end());
        all.insert(all.end(), war_deps_.begin(), war_deps_.end());
        all.insert(all.end(), waw_deps_.begin(), waw_deps_.end());
        return all;
    }

    /// Check if a given dimension is parallelizable across ALL dependencies.
    [[nodiscard]] bool is_parallelizable(size_t dim) const {
        for (const auto& dep : raw_deps_) {
            if (!dep.is_parallelizable(dim)) return false;
        }
        for (const auto& dep : war_deps_) {
            if (!dep.is_parallelizable(dim)) return false;
        }
        for (const auto& dep : waw_deps_) {
            if (!dep.is_parallelizable(dim)) return false;
        }
        return true;
    }

    /// Get all parallelizable dimensions.
    [[nodiscard]] std::vector<size_t> parallelizable_dims() const {
        if (statements_.empty()) return {};
        size_t ndim = statements_[0].domain.ndim();
        std::vector<size_t> result;
        for (size_t d = 0; d < ndim; ++d) {
            if (is_parallelizable(d)) result.push_back(d);
        }
        return result;
    }

    /// Check if a given affine transformation preserves all dependencies.
    [[nodiscard]] bool is_valid_transformation(const AffineMap& T) const {
        for (const auto& dep : raw_deps_) {
            if (!dep.is_valid_transformation(T)) return false;
        }
        for (const auto& dep : war_deps_) {
            if (!dep.is_valid_transformation(T)) return false;
        }
        for (const auto& dep : waw_deps_) {
            if (!dep.is_valid_transformation(T)) return false;
        }
        return true;
    }

    // ── SRAM footprint estimation ───────────────────────────────────

    /// Estimate the SRAM memory footprint (in bytes) for a given tile configuration.
    /// This considers all tensors accessed within the tile, accounting for
    /// double-buffering.
    ///
    /// For each access, computes the actual number of distinct elements
    /// touched in the tile by examining which tile dimensions contribute
    /// to each array dimension via the access map.
    [[nodiscard]] size_t estimate_sram_footprint(
        const std::vector<int64_t>& tile_sizes,
        size_t bytes_per_element = 2,  // FP16 = 2 bytes
        bool double_buffer = true
    ) const {
        // Collect per-array footprints to avoid double-counting
        // accesses to the same array within one tile.
        std::unordered_map<std::string, size_t> array_footprint;

        for (const auto& stmt : statements_) {
            for (const auto& acc : stmt.accesses) {
                // For each output dimension of the access map, determine
                // which tile dimensions contribute to it.
                // The footprint for this access is the product of tile sizes
                // for dimensions that actually affect the accessed indices.
                size_t elements = 1;
                const auto& mat = acc.access_map.matrix();

                for (size_t out_d = 0; out_d < acc.access_map.n_out(); ++out_d) {
                    // Determine which iteration dimensions contribute to
                    // this array dimension.
                    // The number of distinct values for array dim out_d
                    // is the product of tile_sizes for all iteration dims
                    // that have a non-zero coefficient in row out_d.
                    size_t dim_footprint = 1;
                    for (size_t in_d = 0; in_d < acc.access_map.n_in() && in_d < tile_sizes.size(); ++in_d) {
                        if (mat[out_d][in_d] != 0) {
                            dim_footprint *= static_cast<size_t>(tile_sizes[in_d]);
                        }
                    }
                    elements *= dim_footprint;
                }

                // Use the maximum footprint for each array (read and write
                // access the same data, so don't double-count).
                const std::string& arr = acc.array_name;
                if (!arr.empty()) {
                    auto it = array_footprint.find(arr);
                    if (it == array_footprint.end() || elements > it->second) {
                        array_footprint[arr] = elements;
                    }
                } else {
                    // No array name: can't dedup, add directly
                    array_footprint["__anon_" + std::to_string(array_footprint.size())] = elements;
                }
            }
        }

        size_t total_elements = 0;
        for (const auto& [name, count] : array_footprint) {
            total_elements += count;
        }

        size_t bytes = total_elements * bytes_per_element;
        if (double_buffer) bytes *= 2;
        return bytes;
    }

    // ── Accessors ───────────────────────────────────────────────────

    [[nodiscard]] const std::string& name() const { return name_; }
    void set_name(std::string n) { name_ = std::move(n); }

    // ── String representation ───────────────────────────────────────

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "IterationSpace{name='" << name_ << "', stmts=[";
        for (size_t i = 0; i < statements_.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << statements_[i].name;
        }
        oss << "], raw_deps=" << raw_deps_.size()
            << ", war_deps=" << war_deps_.size()
            << ", waw_deps=" << waw_deps_.size() << "}";
        return oss.str();
    }

private:
    std::string name_;
    std::vector<Statement> statements_;
    std::vector<DependencyPolyhedron> raw_deps_;
    std::vector<DependencyPolyhedron> war_deps_;
    std::vector<DependencyPolyhedron> waw_deps_;
};

/// Factory: create a standard matrix multiplication iteration space.
/// C[M,N] += A[M,K] * B[K,N]
inline IterationSpace make_matmul_iteration_space(
    int64_t M, int64_t N, int64_t K
) {
    IterationSpace ispace("matmul");

    // Statement: compute C[m,n] += A[m,k] * B[k,n]
    Statement compute("compute_C",
        make_rectangular_polytope({{0, M}, {0, N}, {0, K}}));

    // Access patterns (3 iteration vars: m, n, k)
    // C[m,n] <- write
    AffineMap c_access(3, 2);
    c_access.matrix_at(0, 0) = 1;  // m
    c_access.matrix_at(1, 1) = 1;  // n
    compute.add_write_access(c_access, "C");

    // A[m,k] <- read
    AffineMap a_access(3, 2);
    a_access.matrix_at(0, 0) = 1;  // m
    a_access.matrix_at(1, 2) = 1;  // k
    compute.add_read_access(a_access, "A");

    // B[k,n] <- read
    AffineMap b_access(3, 2);
    b_access.matrix_at(0, 2) = 1;  // k
    b_access.matrix_at(1, 1) = 1;  // n
    compute.add_read_access(b_access, "B");

    ispace.add_statement(std::move(compute));
    return ispace;
}

/// Factory: create a 2D convolution iteration space.
/// output[batch, out_ch, oh, ow] += input[batch, in_ch, ih, iw] * kernel[out_ch, in_ch, kh, kw]
inline IterationSpace make_conv2d_iteration_space(
    int64_t batch, int64_t out_ch, int64_t in_ch,
    int64_t oh, int64_t ow, int64_t kh, int64_t kw,
    int64_t stride = 1, int64_t pad = 0
) {
    IterationSpace ispace("conv2d");

    // 7-dim loop nest: batch, out_ch, oh, ow, in_ch, kh, kw
    Statement conv("conv2d_compute",
        make_rectangular_polytope({
            {0, batch}, {0, out_ch}, {0, oh}, {0, ow},
            {0, in_ch}, {0, kh}, {0, kw}
        }));

    // Output write: output[batch, out_ch, oh, ow]
    AffineMap out_access(7, 4);
    out_access.matrix_at(0, 0) = 1;  // batch
    out_access.matrix_at(1, 1) = 1;  // out_ch
    out_access.matrix_at(2, 2) = 1;  // oh
    out_access.matrix_at(3, 3) = 1;  // ow
    conv.add_write_access(out_access, "output");

    // Input read: input[batch, in_ch, oh*stride + kh - pad, ow*stride + kw - pad]
    AffineMap in_access(7, 4);
    in_access.matrix_at(0, 0) = 1;                           // batch
    in_access.matrix_at(1, 4) = 1;                           // in_ch
    in_access.matrix_at(2, 2) = stride;  in_access.offset_at(2) = -pad;  // ih
    in_access.matrix_at(2, 5) = 1;
    in_access.matrix_at(3, 3) = stride;  in_access.offset_at(3) = -pad;  // iw
    in_access.matrix_at(3, 6) = 1;
    conv.add_read_access(in_access, "input");

    // Kernel read: kernel[out_ch, in_ch, kh, kw]
    AffineMap kern_access(7, 4);
    kern_access.matrix_at(0, 1) = 1;  // out_ch
    kern_access.matrix_at(1, 4) = 1;  // in_ch
    kern_access.matrix_at(2, 5) = 1;  // kh
    kern_access.matrix_at(3, 6) = 1;  // kw
    conv.add_read_access(kern_access, "kernel");

    ispace.add_statement(std::move(conv));
    return ispace;
}

} // namespace symplex::polyhedral
