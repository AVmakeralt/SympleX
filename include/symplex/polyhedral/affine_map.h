// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <memory>
#include <cmath>

namespace symplex::polyhedral {

/// AffineMap:  represents a function  f(x) = M*x + c
/// where M is an (n_out x n_in) integer matrix and c is an offset vector.
///
/// Affine maps are the morphisms of the polyhedral category.  Schedule maps,
/// access functions, and layout transformations are all affine maps.
class AffineMap {
public:
    AffineMap()
        : n_in_(0)
        , n_out_(0)
    {}

    AffineMap(size_t n_in, size_t n_out)
        : n_in_(n_in)
        , n_out_(n_out)
        , matrix_(n_out, std::vector<int64_t>(n_in, 0))
        , offset_(n_out, 0)
    {}

    AffineMap(
        std::vector<std::vector<int64_t>> matrix,
        std::vector<int64_t> offset
    )
        : n_in_(matrix.empty() ? 0 : matrix[0].size())
        , n_out_(matrix.size())
        , matrix_(std::move(matrix))
        , offset_(std::move(offset))
    {
        assert(offset_.size() == n_out_);
        for (const auto& row : matrix_) {
            assert(row.size() == n_in_);
        }
    }

    // ── Accessors ───────────────────────────────────────────────────

    [[nodiscard]] size_t n_in()  const { return n_in_;  }
    [[nodiscard]] size_t n_out() const { return n_out_; }

    [[nodiscard]] const std::vector<std::vector<int64_t>>& matrix() const { return matrix_; }
    [[nodiscard]] const std::vector<int64_t>&              offset() const { return offset_; }

    [[nodiscard]] int64_t& matrix_at(size_t out_dim, size_t in_dim) {
        return matrix_[out_dim][in_dim];
    }

    [[nodiscard]] int64_t& offset_at(size_t out_dim) {
        return offset_[out_dim];
    }

    // ── Evaluation ──────────────────────────────────────────────────

    /// Apply this affine map to an input point.
    [[nodiscard]] std::vector<int64_t> apply(const std::vector<int64_t>& x) const {
        std::vector<int64_t> result(n_out_, 0);
        for (size_t o = 0; o < n_out_; ++o) {
            result[o] = offset_[o];
            for (size_t i = 0; i < n_in_ && i < x.size(); ++i) {
                result[o] += matrix_[o][i] * x[i];
            }
        }
        return result;
    }

    // ── Composition ─────────────────────────────────────────────────

    /// Compose:  (this) o (other)  =  this(other(x))
    [[nodiscard]] AffineMap compose(const AffineMap& other) const {
        assert(n_in_ == other.n_out_);
        AffineMap result(other.n_in_, n_out_);

        // result.matrix = this.matrix * other.matrix
        for (size_t o = 0; o < n_out_; ++o) {
            for (size_t i = 0; i < other.n_in_; ++i) {
                int64_t sum = 0;
                for (size_t k = 0; k < n_in_; ++k) {
                    sum += matrix_[o][k] * other.matrix_[k][i];
                }
                result.matrix_[o][i] = sum;
            }
        }

        // result.offset = this.matrix * other.offset + this.offset
        for (size_t o = 0; o < n_out_; ++o) {
            int64_t sum = offset_[o];
            for (size_t k = 0; k < n_in_; ++k) {
                sum += matrix_[o][k] * other.offset_[k];
            }
            result.offset_[o] = sum;
        }

        return result;
    }

    // ── Inverse (where possible) ────────────────────────────────────

    /// Attempt to compute the inverse affine map.
    /// Only possible for bijective square affine maps.
    [[nodiscard]] std::optional<AffineMap> inverse() const {
        if (n_in_ != n_out_) return std::nullopt;

        // Augmented matrix [M | I] using double
        size_t n = n_in_;
        std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                aug[i][j] = static_cast<double>(matrix_[i][j]);
            }
            aug[i][n + i] = 1.0;
        }

        for (size_t col = 0; col < n; ++col) {
            size_t pivot = col;
            for (size_t row = col + 1; row < n; ++row) {
                if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) {
                    pivot = row;
                }
            }
            if (std::abs(aug[pivot][col]) < 1e-12) return std::nullopt;
            std::swap(aug[col], aug[pivot]);

            double pv = aug[col][col];
            for (size_t j = 0; j < 2 * n; ++j) aug[col][j] /= pv;
            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;
                double factor = aug[row][col];
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        AffineMap inv(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inv.matrix_[i][j] = static_cast<int64_t>(std::round(aug[i][n + j]));
            }
            // Inverse offset: -M^{-1} * c
            int64_t off = 0;
            for (size_t j = 0; j < n; ++j) {
                off += inv.matrix_[i][j] * offset_[j];
            }
            inv.offset_[i] = -off;
        }

        return inv;
    }

    // ── Identity ────────────────────────────────────────────────────

    [[nodiscard]] static AffineMap identity(size_t n) {
        AffineMap id(n, n);
        for (size_t i = 0; i < n; ++i) id.matrix_[i][i] = 1;
        return id;
    }

    // ── Tiling map ──────────────────────────────────────────────────

    /// Create a tiling affine map that splits dimension d into (outer, inner)
    /// pairs with the given tile size.
    [[nodiscard]] static AffineMap tile_1d(size_t dim, size_t ndim, int64_t tile_size) {
        // Output has ndim + 1 dimensions: the original dims, but dim is replaced
        // by (dim_outer, dim_inner).
        // Actually: output dims = ndim + 1 (tile coordinate + local coordinate)
        AffineMap tile_map(ndim, ndim + 1);

        for (size_t d = 0; d < ndim; ++d) {
            if (d == dim) {
                // i[dim] = tile_size * tile_coord + local_coord
                tile_map.matrix_at(d, dim) = tile_size;          // tile_coord
                tile_map.matrix_at(d, ndim) = 1;                // local_coord
            } else {
                tile_map.matrix_at(d, d) = 1;  // pass-through
            }
        }

        return tile_map;
    }

    // ── String representation ───────────────────────────────────────

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        for (size_t o = 0; o < n_out_; ++o) {
            if (o > 0) oss << ", ";
            bool first = true;
            for (size_t i = 0; i < n_in_; ++i) {
                if (matrix_[o][i] == 0) continue;
                if (!first) oss << (matrix_[o][i] > 0 ? "+" : "-");
                else if (matrix_[o][i] < 0) oss << "-";
                if (std::abs(matrix_[o][i]) != 1) oss << std::abs(matrix_[o][i]) << "*";
                oss << "i" << i;
                first = false;
            }
            if (offset_[o] != 0 || first) {
                if (!first && offset_[o] > 0) oss << "+";
                oss << offset_[o];
            }
        }
        return oss.str();
    }

private:
    size_t n_in_;
    size_t n_out_;
    std::vector<std::vector<int64_t>> matrix_;
    std::vector<int64_t> offset_;
};

} // namespace symplex::polyhedral
