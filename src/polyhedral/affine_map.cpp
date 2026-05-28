// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/polyhedral/affine_map.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace symplex::polyhedral {

// ── Compile-time verification ────────────────────────────────────────────────

static_assert(std::is_default_constructible_v<AffineMap>);
static_assert(std::is_copy_constructible_v<AffineMap>);
static_assert(std::is_move_constructible_v<AffineMap>);
static_assert(std::is_copy_assignable_v<AffineMap>);
static_assert(std::is_move_assignable_v<AffineMap>);

// ── Non-inline helper functions for AffineMap ────────────────────────────────

/// Compute the determinant of the matrix part of an AffineMap.
/// Only meaningful for square maps (n_in == n_out).
/// Returns 0 for non-square maps.
int64_t determinant(const AffineMap& map) {
    if (map.n_in() != map.n_out() || map.n_in() == 0) return 0;

    size_t n = map.n_in();
    const auto& M = map.matrix();

    // Use cofactor expansion for small matrices,
    // Gaussian elimination for larger ones.
    if (n == 1) return M[0][0];
    if (n == 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];

    // Gaussian elimination with double for stability.
    std::vector<std::vector<double>> aug(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = static_cast<double>(M[i][j]);
        }
    }

    double det = 1.0;
    for (size_t col = 0; col < n; ++col) {
        // Find pivot.
        size_t pivot = col;
        for (size_t row = col + 1; row < n; ++row) {
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) {
                pivot = row;
            }
        }
        if (std::abs(aug[pivot][col]) < 1e-12) return 0;

        if (pivot != col) {
            std::swap(aug[col], aug[pivot]);
            det *= -1.0;
        }

        det *= aug[col][col];
        double pv = aug[col][col];

        for (size_t row = col + 1; row < n; ++row) {
            double factor = aug[row][col] / pv;
            for (size_t j = col; j < n; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    return static_cast<int64_t>(std::round(det));
}

/// Check if an AffineMap represents the identity transformation.
bool is_identity(const AffineMap& map) {
    if (map.n_in() != map.n_out()) return false;

    const auto& M = map.matrix();
    const auto& c = map.offset();

    for (size_t i = 0; i < map.n_out(); ++i) {
        if (c[i] != 0) return false;
        for (size_t j = 0; j < map.n_in(); ++j) {
            int64_t expected = (i == j) ? 1 : 0;
            if (M[i][j] != expected) return false;
        }
    }
    return true;
}

/// Check if an AffineMap is a diagonal transformation (only diagonal entries non-zero).
bool is_diagonal(const AffineMap& map) {
    if (map.n_in() != map.n_out()) return false;

    const auto& M = map.matrix();
    for (size_t i = 0; i < map.n_out(); ++i) {
        for (size_t j = 0; j < map.n_in(); ++j) {
            if (i != j && M[i][j] != 0) return false;
        }
    }
    return true;
}

/// Compute the rank of the matrix part of an AffineMap.
size_t matrix_rank(const AffineMap& map) {
    size_t m = map.n_out();
    size_t n = map.n_in();
    if (m == 0 || n == 0) return 0;

    const auto& M = map.matrix();

    // Gaussian elimination using double.
    std::vector<std::vector<double>> aug(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = static_cast<double>(M[i][j]);
        }
    }

    size_t rank = 0;
    std::vector<bool> row_used(m, false);

    for (size_t col = 0; col < n; ++col) {
        size_t pivot = m;  // invalid
        for (size_t row = 0; row < m; ++row) {
            if (!row_used[row] && std::abs(aug[row][col]) > 1e-12) {
                pivot = row;
                break;
            }
        }
        if (pivot == m) continue;

        row_used[pivot] = true;
        ++rank;

        double pv = aug[pivot][col];
        for (size_t j = col; j < n; ++j) {
            aug[pivot][j] /= pv;
        }

        for (size_t row = 0; row < m; ++row) {
            if (row == pivot || row_used[row]) continue;
            double factor = aug[row][col];
            if (std::abs(factor) < 1e-12) continue;
            for (size_t j = col; j < n; ++j) {
                aug[row][j] -= factor * aug[pivot][j];
            }
        }
    }

    return rank;
}

/// Compute a tiled affine map that splits multiple dimensions.
/// For each dimension d in tile_dims, creates a (outer, inner) pair.
/// The resulting map has ndim + tile_dims.size() output dimensions.
AffineMap make_multi_tile_map(
    size_t ndim,
    const std::vector<size_t>& tile_dims,
    const std::vector<int64_t>& tile_sizes
) {
    size_t n_tiled = tile_dims.size();
    AffineMap tile_map(ndim, ndim + n_tiled);

    // Initialize all to zero.
    std::vector<bool> is_tiled(ndim, false);

    // Pass through non-tiled dimensions.
    for (size_t d = 0; d < ndim; ++d) {
        bool in_tile = false;
        size_t tile_idx = 0;
        for (size_t t = 0; t < n_tiled; ++t) {
            if (tile_dims[t] == d) {
                in_tile = true;
                tile_idx = t;
                break;
            }
        }

        if (!in_tile) {
            // Direct pass-through.
            tile_map.matrix_at(d, d) = 1;
        } else {
            // Tiled dimension: i[d] = tile_size * tile_coord + local_coord
            tile_map.matrix_at(d, d) = tile_sizes[tile_idx];          // tile_coord
            tile_map.matrix_at(d, ndim + tile_idx) = 1;              // local_coord
            is_tiled[d] = true;
        }
    }

    return tile_map;
}

} // namespace symplex::polyhedral
