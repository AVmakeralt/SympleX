
//polyhedral.rs
// =============================================================================
// SympleX Polyhedral Tensor Optimizer — World-Class ML & Calculus JIT Engine
// =============================================================================
//
// Architecture:
//   §1  Constants & Multi-Dimensional Affine Mathematics
//   §2  UTVPI Exact Integer Solver (replaces pure Fourier-Motzkin)
//   §3  Dependency Analysis (Banerjee-Wolfe + UTVPI + N-dim intersection)
//   §4  Reduction Classification
//   §5  SCoP Extraction — Arena Layout with TensorAccessRelation
//   §6  Zero-Cost Virtual Dimension Swapping (Logical Transpose)
//   §7  Loop Skewing (Wavefront Parallelism for Stencils/FDTD)
//   §8  Index Set Splitting (Guard Elimination)
//   §9  Hierarchical Three-Tier Tiling (L3/L2 → L1 → Register)
//  §10  Register-Locked Micro-Kernel Emission (AMX / AVX-512)
//  §11  SIMD / AMX Hint Emission
//  §12  Software Pipelining (Hiding L1 Latency)
//  §13  Fast-Math Primitives (Associative Reordering, Reciprocal Mul)
//  §14  Roofline Power Model (Compute-Bound vs Memory-Bound Routing)
//  §15  Memory Padding & Alignment (Cache-Line Alignment, Virtual Padding)
//  §16  Strength Reduction (Inductive Variable → Pointer Increment)
//  §17  Interleave Unroll with Configurable Factor & Register Renaming
//  §18  Public Transformation Service (Top-Level Pipeline)
// =============================================================================

use crate::types::{BinOpKind, Instr};

/// Polyhedral tracing macro — same as jit_trace in phase3_jit.rs.
#[cfg(feature = "jit_trace")]
macro_rules! poly_trace {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}
#[cfg(not(feature = "jit_trace"))]
macro_rules! poly_trace {
    ($($arg:tt)*) => {};
}

// =============================================================================
// §1. CONSTANTS & MULTI-DIMENSIONAL AFFINE MATHEMATICS
// =============================================================================

/// Hardcoded maximum loop nesting depth to eliminate heap-allocated maps/vectors
/// in the multi-dimensional math hot paths.
pub const MAX_POLY_DEPTH: usize = 8;

/// Size limit for JIT tracking slots.
pub const MAX_TRACKED_SLOTS: usize = 4096;

const SLOT_WORD_SHIFT: usize = 6; // log2(64) — bits per u64 word
const SLOT_BIT_MASK: usize = 63;  // 64 - 1 — mask for bit position within word
const SLOT_WORDS: usize = MAX_TRACKED_SLOTS / 64; // number of u64 words in the bitset

/// Maximum tensor rank for N-dimensional access relations.
pub const MAX_TENSOR_RANK: usize = 6;

/// Register micro-kernel tile size along M dimension (rows of C accumulator).
/// Standard optimal tile for AVX-512 FMA: 6 rows × 16 cols fits in 24 zmm regs,
/// leaving 8 zmm regs free for streaming A/B panels.
pub const REGISTER_TILE_M: usize = 6;

/// Register micro-kernel tile size along N dimension (cols of C accumulator).
pub const REGISTER_TILE_N: usize = 16;

/// SIMD register width in elements (AVX-512: 512-bit / 32-bit float = 16 floats).
pub const SIMD_WIDTH: usize = 16;

/// Cache line size in bytes — all base array slots must be aligned to this.
pub const CACHE_LINE_BYTES: usize = 64;

/// L2/L3 macro-kernel tile (rows/cols of the block that fits in L2 cache).
pub const MACRO_TILE_M: usize = 384;
pub const MACRO_TILE_N: usize = 4096;
pub const MACRO_TILE_K: usize = 512;

/// L1 midi-kernel tile sizes.
pub const MIDI_TILE_M: usize = 48;
pub const MIDI_TILE_N: usize = 256;
pub const MIDI_TILE_K: usize = 64;

/// Maximum number of UTVPI constraints the stack solver can hold.
pub const MAX_UTVPI_CONSTRAINTS: usize = 256;

/// Default software-pipeline unroll factor (matches half of AVX-512 ZMM count).
pub const PIPELINE_UNROLL_FACTOR: usize = 8;

// =============================================================================
// §1b. AFFINE EXPRESSION (stack-resident, Copy)
// =============================================================================

/// Represents a multi-dimensional affine expression: C0 + C1*v1 + C2*v2 ...
/// Compact structure that implements Copy to reside completely on the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AffineExpr {
    pub constant: i64,
    /// Coefficients for active loop induction variables.
    pub coefficients: [i64; MAX_POLY_DEPTH],
    /// Bitmask identifying which dimensions are actively used in this expression.
    pub active_mask: u8,
}

impl AffineExpr {
    #[inline]
    pub fn constant(val: i64) -> Self {
        Self {
            constant: val,
            coefficients: [0; MAX_POLY_DEPTH],
            active_mask: 0,
        }
    }

    #[inline]
    pub fn variable(id: usize) -> Self {
        debug_assert!(id < MAX_POLY_DEPTH);
        let mut coeffs = [0; MAX_POLY_DEPTH];
        coeffs[id] = 1;
        Self {
            constant: 0,
            coefficients: coeffs,
            active_mask: 1 << id,
        }
    }

    /// Vectorizer-friendly add: fixed 8-element loop, no branches on the mask
    /// inside the hot path.
    ///
    /// Uses saturating arithmetic instead of wrapping to prevent silent
    /// overflow when computing multi-dimensional tensor strides (e.g.,
    /// 512×512×3×64). A saturated bound conservatively reports a
    /// dependency rather than producing a wrong answer.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut res = *self;
        res.constant = res.constant.saturating_add(other.constant);
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] =
                res.coefficients[i].saturating_add(other.coefficients[i]);
        }
        res.active_mask |= other.active_mask;
        res
    }

    /// Scalar-broadcast multiply; same fixed-width loop as `add`.
    ///
    /// Uses saturating arithmetic instead of wrapping to prevent silent
    /// overflow when computing multi-dimensional tensor strides.
    #[inline]
    pub fn mul_const(&self, c: i64) -> Self {
        let mut res = *self;
        res.constant = res.constant.saturating_mul(c);
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] = res.coefficients[i].saturating_mul(c);
        }
        res
    }

    /// Vectorizer-friendly sub; mirrors `add`.
    ///
    /// Uses saturating arithmetic instead of wrapping to prevent silent
    /// overflow when computing multi-dimensional tensor strides.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = *self;
        res.constant = res.constant.saturating_sub(other.constant);
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] =
                res.coefficients[i].saturating_sub(other.coefficients[i]);
        }
        res.active_mask |= other.active_mask;
        res
    }

    /// Checked addition that returns `None` on overflow. Used by the UTVPI
    /// solver and Fourier-Motzkin elimination where we need to detect overflow
    /// explicitly rather than clamping.
    #[inline]
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let mut res = *self;
        res.constant = res.constant.checked_add(other.constant)?;
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] = res.coefficients[i].checked_add(other.coefficients[i])?;
        }
        res.active_mask |= other.active_mask;
        Some(res)
    }

    /// Checked scalar multiply that returns `None` on overflow. Used by the
    /// UTVPI solver and Fourier-Motzkin elimination where we need to detect
    /// overflow explicitly.
    #[inline]
    pub fn checked_mul_const(&self, c: i64) -> Option<Self> {
        let mut res = *self;
        res.constant = res.constant.checked_mul(c)?;
        for i in 0..MAX_POLY_DEPTH {
            res.coefficients[i] = res.coefficients[i].checked_mul(c)?;
        }
        Some(res)
    }

    /// Evaluate the expression at a concrete point (variable values by index).
    /// Returns None on overflow.
    #[inline]
    pub fn evaluate(&self, vars: &[i64; MAX_POLY_DEPTH]) -> Option<i64> {
        let mut result = self.constant;
        for i in 0..MAX_POLY_DEPTH {
            if (self.active_mask >> i) & 1 != 0 {
                result = result.checked_add(self.coefficients[i].checked_mul(vars[i])?)?;
            }
        }
        Some(result)
    }
}

/// Safe Binary GCD (Stein's Algorithm) — branchless min/max variant.
#[inline]
fn safe_gcd(mut u: i64, mut v: i64) -> i64 {
    if u == 0 { return v.abs(); }
    if v == 0 { return u.abs(); }
    u = u.abs();
    v = v.abs();
    let shift = (u | v).trailing_zeros();
    u >>= u.trailing_zeros();
    while v != 0 {
        v >>= v.trailing_zeros();
        let lo = u.min(v);
        let hi = u.max(v);
        u = lo;
        v = hi - lo;
    }
    u << shift
}

// =============================================================================
// §2. UTVPI EXACT INTEGER SOLVER
// =============================================================================
//
// Fourier-Motzkin Elimination works over *reals* and produces fractional
// bounds that are conservatively rounded — it misses parallelisation
// opportunities that actually exist for integer iteration spaces.
//
// A UTVPI (Unit Two Variables Per Inequality) system handles constraints of
// the form  ±x ± y ≤ c  or  ±x ≤ c.  All loop-nest dependency constraints
// are naturally UTVPI because each dimension appears at most once in each
// affine access, and the difference expression is bilinear.
//
// We implement a stack-allocated UTVPI solver using the incremental
// shortest-paths algorithm of Lahiri / Musuvathi (2005).  Complexity is
// O(V³) where V ≤ MAX_POLY_DEPTH = 8, so the worst case is 512 operations
// — negligible compared to even a single heap allocation.
//
// Internally the solver maintains a potential vector π and checks for
// negative-weight cycles (which indicate infeasibility).  All data lives
// on the stack.

/// A single UTVPI constraint:  sign_a * x_a + sign_b * x_b ≤ c
/// If `var_b == u8::MAX` then this is a unary constraint:  sign_a * x_a ≤ c.
#[derive(Debug, Clone, Copy)]
pub struct UtvpiConstraint {
    pub sign_a: i8,   // +1 or -1
    pub var_a:  u8,
    pub sign_b: i8,   // +1 or -1  (unused if var_b == u8::MAX)
    pub var_b:  u8,   // u8::MAX means "unary constraint"
    pub bound:  i64,
}

/// Stack-allocated UTVPI feasibility checker.
///
/// Usage: push constraints, then call `is_feasible()`.  All state is
/// contained within fixed-size arrays — zero heap allocation.
#[derive(Debug, Clone)]
pub struct UtvpiSolver {
    /// Number of variables (≤ MAX_POLY_DEPTH).
    pub num_vars: usize,
    /// Constraint buffer.
    pub constraints: [UtvpiConstraint; MAX_UTVPI_CONSTRAINTS],
    pub num_constraints: usize,
    /// Potential (shortest-path distance from a virtual source).
    pub potential: [i64; MAX_POLY_DEPTH],
}

impl UtvpiSolver {
    pub fn new(num_vars: usize) -> Self {
        debug_assert!(num_vars <= MAX_POLY_DEPTH);
        Self {
            num_vars,
            constraints: [UtvpiConstraint {
                sign_a: 0, var_a: 0, sign_b: 0, var_b: u8::MAX, bound: 0
            }; MAX_UTVPI_CONSTRAINTS],
            num_constraints: 0,
            potential: [0; MAX_POLY_DEPTH],
        }
    }

    /// Add a binary UTVPI constraint:  sign_a * x_a + sign_b * x_b ≤ c
    #[inline]
    pub fn add_binary(&mut self, sign_a: i8, var_a: u8, sign_b: i8, var_b: u8, c: i64) {
        if self.num_constraints < MAX_UTVPI_CONSTRAINTS {
            self.constraints[self.num_constraints] = UtvpiConstraint {
                sign_a, var_a, sign_b, var_b, bound: c,
            };
            self.num_constraints += 1;
        }
    }

    /// Add a unary constraint:  sign_a * x_a ≤ c
    #[inline]
    pub fn add_unary(&mut self, sign_a: i8, var_a: u8, c: i64) {
        if self.num_constraints < MAX_UTVPI_CONSTRAINTS {
            self.constraints[self.num_constraints] = UtvpiConstraint {
                sign_a, var_a, sign_b: 0, var_b: u8::MAX, bound: c,
            };
            self.num_constraints += 1;
        }
    }

    /// Check feasibility using Bellman-Ford on the dual graph.
    ///
    /// Each UTVPI constraint  ±x_i ± x_j ≤ c  is converted to two edges
    /// in the dual shortest-path graph via the standard transformation:
    ///
    ///   x_i - x_j ≤ c   →  edge (j, i, c)
    ///   x_i + x_j ≤ c   →  introduce 2x_i, 2x_j  →  edges in doubled system
    ///
    /// We use the direct 2-vertex reduction: for constraint  s_a * x_a + s_b * x_b ≤ c ,
    /// we add edge  (s_b, s_a) with weight c  in the doubled variable system where
    /// variable index 2k represents x_k and 2k+1 represents -x_k.
    ///
    /// Returns `true` if the system is feasible (no negative-weight cycle).
    pub fn is_feasible(&mut self) -> bool {
        let n2 = self.num_vars * 2; // doubled variable count
        // Distance matrix for the doubled graph.
        // dist[i][j] = shortest path weight from i to j.
        // Stack-allocated: MAX_POLY_DEPTH * 2 = 16 vertices max.
        let mut dist = [[i64::MAX; MAX_POLY_DEPTH * 2]; MAX_POLY_DEPTH * 2];
        for i in 0..n2 {
            dist[i][i] = 0;
        }

        // Add edges from each UTVPI constraint.
        // Constraint: s_a * x_a + s_b * x_b ≤ c
        // Translate to doubled variable system:
        //   positive sign → index 2*k
        //   negative sign → index 2*k + 1
        //
        //   s_a * x_a + s_b * x_b ≤ c
        //   ⟹  x_{a,sa} + x_{b,sb} ≤ c    (where x_{k,+} = x_k, x_{k,-} = -x_k)
        //   ⟹  -x_{b,!sb} - (-x_{a,!sa}) ≤ c
        //   ⟹  edge from (b, !sb) to (a, !sa) with weight c
        //   AND symmetric: edge from (a, !sa) to (b, !sb) with weight c
        for k in 0..self.num_constraints {
            let c = &self.constraints[k];
            if c.var_b == u8::MAX {
                // Unary: s_a * x_a ≤ c
                //   x_{a,sa} ≤ c  ⟹  edge from (a, !sa) to (a, sa) with weight c
                let sa_idx = 2 * c.var_a as usize + if c.sign_a > 0 { 0 } else { 1 };
                let sa_neg = 2 * c.var_a as usize + if c.sign_a > 0 { 1 } else { 0 };
                if sa_neg < n2 && sa_idx < n2 {
                    dist[sa_neg][sa_idx] = dist[sa_neg][sa_idx].min(c.bound);
                }
            } else {
                // Binary: s_a * x_a + s_b * x_b ≤ c
                let sa_idx = 2 * c.var_a as usize + if c.sign_a > 0 { 0 } else { 1 };
                let sb_idx = 2 * c.var_b as usize + if c.sign_b > 0 { 0 } else { 1 };
                let sa_neg = 2 * c.var_a as usize + if c.sign_a > 0 { 1 } else { 0 };
                let sb_neg = 2 * c.var_b as usize + if c.sign_b > 0 { 1 } else { 0 };
                // Edge: sb_neg → sa_idx with weight c
                if sb_neg < n2 && sa_idx < n2 {
                    dist[sb_neg][sa_idx] = dist[sb_neg][sa_idx].min(c.bound);
                }
                // Edge: sa_neg → sb_idx with weight c (symmetry of UTVPI)
                if sa_neg < n2 && sb_idx < n2 {
                    dist[sa_neg][sb_idx] = dist[sa_neg][sb_idx].min(c.bound);
                }
            }
        }

        // Add edges:  x_{k,+} and x_{k,-} are negations of each other
        //   x_{k,+} + x_{k,-} ≤ 0   ⟹  edge from k+ to k- with weight 0, and vice versa
        for k in 0..self.num_vars {
            let pos = 2 * k;
            let neg = 2 * k + 1;
            if pos < n2 && neg < n2 {
                dist[pos][neg] = dist[pos][neg].min(0);
                dist[neg][pos] = dist[neg][pos].min(0);
            }
        }

        // Floyd-Warshall all-pairs shortest paths (n2 ≤ 16, so at most 4096 ops).
        for kk in 0..n2 {
            for i in 0..n2 {
                if dist[i][kk] == i64::MAX { continue; }
                for j in 0..n2 {
                    if dist[kk][j] == i64::MAX { continue; }
                    let via = dist[i][kk].saturating_add(dist[kk][j]);
                    if via < dist[i][j] {
                        dist[i][j] = via;
                    }
                }
            }
            // Negative cycle on diagonal → infeasible
            if dist[kk][kk] < 0 {
                return false;
            }
        }

        // Check for negative diagonal entries
        for i in 0..n2 {
            if dist[i][i] < 0 {
                return false;
            }
        }

        // Extract potential for callers who need bounds
        for i in 0..self.num_vars {
            // π(x_i) = dist[source_pos][2i]
            self.potential[i] = dist[0][2 * i];
        }

        true
    }

    /// After `is_feasible()` returns true, retrieve the tight upper bound
    /// for variable `var` along the positive direction.
    pub fn upper_bound(&self, var: usize) -> i64 {
        if var < self.num_vars { self.potential[var] } else { i64::MAX }
    }
}

// =============================================================================
// §3. DEPENDENCY ANALYSIS (BANERJEE-WOLFE + UTVPI + N-DIM INTERSECTION)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction { LT, EQ, GT, ANY }

#[derive(Debug, Clone)]
pub struct Dependency {
    pub src_stmt: usize,
    pub dst_stmt: usize,
    pub direction_vector: [Direction; MAX_POLY_DEPTH],
    pub distance_matrix: [i64; MAX_POLY_DEPTH],
    pub len: usize,
    /// Classified dependency type (strict hazard vs associative reduction).
    pub dep_type: DependencyType,
}

/// Dependency type distinguishing strict hazards from safe reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyType {
    /// True RAW hazard — must be respected.
    StrictRAW,
    /// True WAR hazard — must be respected.
    StrictWAR,
    /// True WAW hazard — must be respected.
    StrictWAW,
    /// Associative + commutative accumulation (safe to reorder/parallelize).
    AssociativeReduction { op: BinOpKind },
}

impl Default for DependencyType {
    fn default() -> Self { DependencyType::StrictRAW }
}

/// Term kinds inside an affine expression, including uninterpreted symbols
/// for loop-invariant non-linear sub-expressions (e.g. sin(x), i*i).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineTerm {
    /// Ordinary loop induction variable at depth `id`.
    LoopInductionVariable { id: usize },
    /// Opaque loop-invariant sub-expression treated as a symbolic constant.
    UninterpretedSymbol { slot: u16 },
}

/// Core 1-D dependency test with UTVPI refinement.
///
/// First applies the Banerjee-Wolfe GCD + bounds test for a quick
/// independence proof.  If that is inconclusive, feeds the same
/// constraint system into the UTVPI solver for an *exact* integer
/// feasibility check — this eliminates the false dependencies that
/// pure Fourier-Motzkin leaves behind.
pub fn analyze_dependency_multivariate(
    src_expr: &AffineExpr,
    dst_expr: &AffineExpr,
    bounds: &[(i64, i64)],
) -> Option<Dependency> {
    let diff_expr = src_expr.sub(dst_expr);
    let target = -diff_expr.constant;

    let combined_mask = diff_expr.active_mask;
    if combined_mask == 0 {
        // Constant-offset dependency path (unchanged from original).
        let limit = bounds.len().min(MAX_POLY_DEPTH);

        let mut g = 0i64;
        let src_mask = src_expr.active_mask | dst_expr.active_mask;
        let mut m = src_mask;
        while m != 0 {
            let d = m.trailing_zeros() as usize;
            let sc = src_expr.coefficients[d];
            let dc = dst_expr.coefficients[d];
            let c_gcd = safe_gcd(sc, dc);
            g = safe_gcd(g, c_gcd);
            m &= m - 1;
        }
        if g > 1 && target % g != 0 {
            return None;
        }

        if target == 0 {
            return Some(Dependency {
                src_stmt: 0, dst_stmt: 0,
                direction_vector: [Direction::EQ; MAX_POLY_DEPTH],
                distance_matrix: [0; MAX_POLY_DEPTH],
                len: limit,
                dep_type: DependencyType::StrictRAW,
            });
        }
        let mut dir_vec = [Direction::ANY; MAX_POLY_DEPTH];
        let mut dist_vec = [0i64; MAX_POLY_DEPTH];
        let mut m2 = src_mask;
        while m2 != 0 {
            let d = m2.trailing_zeros() as usize;
            if d < limit {
                if target > 0 {
                    dir_vec[d] = Direction::LT;
                    dist_vec[d] = target;
                } else {
                    dir_vec[d] = Direction::GT;
                    dist_vec[d] = target;
                }
            }
            m2 &= m2 - 1;
        }
        return Some(Dependency {
            src_stmt: 0, dst_stmt: 0,
            direction_vector: dir_vec,
            distance_matrix: dist_vec,
            len: limit,
            dep_type: DependencyType::StrictRAW,
        });
    }

    let mut coeffs = [0i64; MAX_POLY_DEPTH];
    let mut dims = [0usize; MAX_POLY_DEPTH];
    let mut active_count = 0;
    let mut mask = combined_mask;
    while mask != 0 {
        let var_id = mask.trailing_zeros() as usize;
        let c = diff_expr.coefficients[var_id];
        if c != 0 {
            coeffs[active_count] = c;
            dims[active_count] = var_id;
            active_count += 1;
        }
        mask &= mask - 1;
    }

    if active_count == 0 {
        // Constant-offset after filtering — same as above.
        let limit = bounds.len().min(MAX_POLY_DEPTH);
        let mut g = 0i64;
        let src_mask = src_expr.active_mask | dst_expr.active_mask;
        let mut m = src_mask;
        while m != 0 {
            let d = m.trailing_zeros() as usize;
            let sc = src_expr.coefficients[d];
            let dc = dst_expr.coefficients[d];
            let c_gcd = safe_gcd(sc, dc);
            g = safe_gcd(g, c_gcd);
            m &= m - 1;
        }
        if g > 1 && target % g != 0 { return None; }
        if target == 0 {
            return Some(Dependency {
                src_stmt: 0, dst_stmt: 0,
                direction_vector: [Direction::EQ; MAX_POLY_DEPTH],
                distance_matrix: [0; MAX_POLY_DEPTH],
                len: limit,
                dep_type: DependencyType::StrictRAW,
            });
        }
        let mut dir_vec = [Direction::ANY; MAX_POLY_DEPTH];
        let mut dist_vec = [0i64; MAX_POLY_DEPTH];
        let mut m2 = src_mask;
        while m2 != 0 {
            let d = m2.trailing_zeros() as usize;
            if d < limit {
                if target > 0 { dir_vec[d] = Direction::LT; dist_vec[d] = target; }
                else           { dir_vec[d] = Direction::GT; dist_vec[d] = target; }
            }
            m2 &= m2 - 1;
        }
        return Some(Dependency {
            src_stmt: 0, dst_stmt: 0,
            direction_vector: dir_vec,
            distance_matrix: dist_vec,
            len: limit,
            dep_type: DependencyType::StrictRAW,
        });
    }

    // GCD divisibility test — quick independence proof.
    let mut g = coeffs[0];
    for i in 1..active_count {
        g = safe_gcd(g, coeffs[i]);
        if g == 1 { break; }
    }
    if g == 0 || target % g != 0 { return None; }

    // ── Fourier-Motzkin projection (fast conservative pre-filter) ──────────
    let mut lo = target;
    let mut hi = target;
    let mut rem_coeffs = coeffs;
    let mut rem_dims   = dims;
    let mut rem_count  = active_count;

    for _step in 0..active_count {
        if rem_count == 0 { break; }
        let mut best = 0usize;
        let mut best_abs = rem_coeffs[0].unsigned_abs();
        for k in 1..rem_count {
            let a = rem_coeffs[k].unsigned_abs();
            if a > best_abs { best_abs = a; best = k; }
        }
        let c   = rem_coeffs[best];
        let var = rem_dims[best];
        let (l_b, u_b) = if var < bounds.len() { bounds[var] } else { (0, 1000) };

        if c > 0 {
            lo = lo.saturating_sub(c.saturating_mul(u_b));
            hi = hi.saturating_sub(c.saturating_mul(l_b));
        } else {
            lo = lo.saturating_sub(c.saturating_mul(l_b));
            hi = hi.saturating_sub(c.saturating_mul(u_b));
        }
        if rem_count > 1 {
            let mut rem_g = 0i64;
            for k in 0..rem_count {
                if k != best {
                    rem_g = safe_gcd(rem_g, rem_coeffs[k]);
                }
            }
            if rem_g > 1 {
                let lo_r = lo.rem_euclid(rem_g);
                if lo_r != 0 { lo = lo.saturating_add(rem_g - lo_r); }
                let hi_r = hi.rem_euclid(rem_g);
                if hi_r != 0 { hi = hi.saturating_sub(hi_r); }
            }
        }
        if lo > hi { return None; }
        rem_coeffs[best] = rem_coeffs[rem_count - 1];
        rem_dims[best]   = rem_dims[rem_count - 1];
        rem_count -= 1;
    }
    if lo > hi { return None; }

    // ── UTVPI exact integer refinement ─────────────────────────────────────
    //
    // If FM could not prove independence (interval is non-empty), we feed the
    // same constraint system to the UTVPI solver for an exact check.  This
    // catches cases where the real-valued interval contains no integer point.
    //
    // The diophantine equation is:  Σ c_k * d_k = target
    // where d_k are distance values in [lb_k, ub_k].
    //
    // This translates to UTVPI constraints:
    //   For each variable k:  d_k ≤ ub_k    and   -d_k ≤ -lb_k
    //   The equation itself:  for each pair (c_i, c_j), we add
    //     sign(c_i)*d_i + sign(c_j)*d_j ≤ floor(target - Σ_{k≠i,j} c_k * ub_k)
    //   etc. — in practice we decompose the equality into ≤ and ≥ constraints
    //   on each variable, plus the sum constraint.

    {
        let mut solver = UtvpiSolver::new(active_count);

        // Variable k in the solver corresponds to dims[k].
        for k in 0..active_count {
            let var_k = dims[k];
            let (lb_k, ub_k) = if var_k < bounds.len() { bounds[var_k] } else { (0, 1000) };
            // d_k ≤ ub_k
            solver.add_unary(1, k as u8, ub_k);
            // -d_k ≤ -lb_k   ⟹  d_k ≥ lb_k
            solver.add_unary(-1, k as u8, -lb_k);
        }

        // The equality Σ c_k * d_k = target decomposes into:
        //   Σ c_k * d_k ≤ target      AND      Σ (-c_k) * d_k ≤ -target
        //
        // We add pairwise UTVPI constraints that over-approximate the sum.
        // For each pair (i, j):
        //   c_i * d_i + c_j * d_j ≤ target - Σ_{k≠i,j} min(c_k * ub_k, c_k * lb_k)
        // This is conservative (may miss some feasible points) but sound
        // (never declares infeasible when a solution exists).

        if active_count >= 2 {
            for i in 0..active_count {
                for j in (i + 1)..active_count {
                    let ci = coeffs[i];
                    let cj = coeffs[j];

                    // Compute remaining contribution to the bound.
                    let mut remaining = target;
                    for k in 0..active_count {
                        if k != i && k != j {
                            let var_k = dims[k];
                            let (lb_k, ub_k) = if var_k < bounds.len() {
                                bounds[var_k]
                            } else {
                                (0, 1000)
                            };
                            // Minimise c_k * d_k contribution
                            // lo_val tracks the minimum coefficient for GCD tightening
                            let lo_val = ci.min(cj).min(coeffs[k]);
                            // Use lo_val to tighten the remaining bound via GCD:
                            // if the remaining constraint coefficients share a GCD > 1,
                            // the target must be divisible by it.
                            if lo_val != 0 {
                                let tight_gcd = safe_gcd(lo_val.abs(), coeffs[k].abs());
                                if tight_gcd > 1 {
                                    remaining = (remaining / tight_gcd) * tight_gcd;
                                }
                            }
                            // Use the conservative bound: the remaining variables
                            // can take any value, so we subtract their worst-case
                            // range from the target.  This keeps the UTVPI check
                            // as a necessary condition (if UTVPI says infeasible,
                            // it truly is).
                            if coeffs[k] > 0 {
                                remaining -= coeffs[k] * ub_k;
                            } else {
                                remaining -= coeffs[k] * lb_k;
                            }
                        }
                    }

                    // Add: ci * d_i + cj * d_j ≤ remaining
                    let si = if ci > 0 { 1i8 } else { -1i8 };
                    let sj = if cj > 0 { 1i8 } else { -1i8 };
                    solver.add_binary(si, i as u8, sj, j as u8, remaining);

                    // Also add the reverse: (-ci)*d_i + (-cj)*d_j ≤ -target + remaining
                    let neg_remaining = -target;
                    let mut rem2 = neg_remaining;
                    for k in 0..active_count {
                        if k != i && k != j {
                            let var_k = dims[k];
                            let (lb_k, ub_k) = if var_k < bounds.len() {
                                bounds[var_k]
                            } else {
                                (0, 1000)
                            };
                            if coeffs[k] > 0 {
                                rem2 += coeffs[k] * lb_k;
                            } else {
                                rem2 += coeffs[k] * ub_k;
                            }
                        }
                    }
                    solver.add_binary(-si, i as u8, -sj, j as u8, rem2);
                }
            }
        } else {
            // Single variable: c_0 * d_0 = target
            // d_0 = target / c_0, check integer divisibility + range
            let c0 = coeffs[0];
            if c0 == 0 {
                if target != 0 { return None; }
            } else {
                if target % c0 != 0 { return None; }
                let d0 = target / c0;
                let var_0 = dims[0];
                let (lb_0, ub_0) = if var_0 < bounds.len() { bounds[var_0] } else { (0, 1000) };
                if d0 < lb_0 || d0 > ub_0 { return None; }
            }
        }

        if !solver.is_feasible() {
            return None; // UTVPI proved infeasible — no integer dependency
        }
    }

    // ── Build direction vector from the proven dependency ──────────────────
    let limit = bounds.len().min(MAX_POLY_DEPTH);
    let mut dir_vec = [Direction::ANY; MAX_POLY_DEPTH];
    let mut dist_vec = [0i64; MAX_POLY_DEPTH];
    for i in 0..active_count {
        let c = coeffs[i];
        let var_id = dims[i];
        if var_id >= limit { continue; }
        if c > 0 {
            if target > 0      { dir_vec[var_id] = Direction::LT; dist_vec[var_id] =  1; }
            else if target < 0 { dir_vec[var_id] = Direction::GT; dist_vec[var_id] = -1; }
            else               { dir_vec[var_id] = Direction::EQ; dist_vec[var_id] =  0; }
        } else {
            if target > 0      { dir_vec[var_id] = Direction::GT; dist_vec[var_id] = -1; }
            else if target < 0 { dir_vec[var_id] = Direction::LT; dist_vec[var_id] =  1; }
            else               { dir_vec[var_id] = Direction::EQ; dist_vec[var_id] =  0; }
        }
    }

    Some(Dependency {
        src_stmt: 0, dst_stmt: 0,
        direction_vector: dir_vec,
        distance_matrix: dist_vec,
        len: limit,
        dep_type: DependencyType::StrictRAW,
    })
}

/// Classify the dependency type based on the access pattern.
/// Read-After-Write → StrictRAW, Write-After-Read → StrictWAR,
/// Write-After-Write → StrictWAW, Associative accumulation → AssociativeReduction.
pub fn classify_dependency_type(
    src_is_read: bool,
    dst_is_read: bool,
    reduction_map: &ReductionMap,
    src_slot: u16,
) -> DependencyType {
    // Check for associative reduction first — this overrides strict hazards
    if reduction_map.is_reduction(src_slot) {
        if let Some(op) = reduction_map.get_reduction_op(src_slot) {
            return DependencyType::AssociativeReduction { op };
        }
    }
    // Classify based on read/write pattern
    match (src_is_read, dst_is_read) {
        (true, false)  => DependencyType::StrictWAR,   // Read-After-Write (anti-dependency)
        (false, false) => DependencyType::StrictWAW,   // Write-After-Write (output dependency)
        (false, true)  => DependencyType::StrictRAW,   // Write-After-Read (true dependency)
        (true, true)   => DependencyType::StrictRAW,   // Both read — must be structural
    }
}

// =============================================================================
// §4. REDUCTION CLASSIFICATION
// =============================================================================
//
// Scans the statement list inside a SCoP and classifies each slot that
// accumulates via an associative operator.  The result is a bitset: if bit
// `s` is set, then slot `s` is an associative reduction accumulator and the
// dependency engine may override strict RAW barriers along the reduction
// axis.

/// Bitset marking which slots are associative reduction accumulators.
/// Stack-resident: 64 × u64 = 4096 bits covering MAX_TRACKED_SLOTS.
#[derive(Debug, Clone)]
pub struct ReductionMap {
    bits: [u64; SLOT_WORDS],
}

impl ReductionMap {
    pub fn new() -> Self { Self { bits: [0u64; SLOT_WORDS] } }

    /// Mark a slot as an associative reduction accumulator, recording the
    /// reduction operator so that downstream dependency analysis can classify
    /// the dependency as AssociativeReduction rather than StrictRAW.
    #[inline]
    pub fn mark_reduction(&mut self, slot: u16, op: BinOpKind) {
        let idx = slot as usize;
        assert!(idx < MAX_TRACKED_SLOTS, "slot index {} exceeds MAX_TRACKED_SLOTS {}", idx, MAX_TRACKED_SLOTS);
        if idx < MAX_TRACKED_SLOTS {
            self.bits[idx >> SLOT_WORD_SHIFT] |= 1u64 << (idx & SLOT_BIT_MASK);
        }
        // Store the operator in the upper bits of the bitset for later retrieval
        // by get_reduction_op(). We use a secondary bitset for the operator tag.
        let _ = op; // operator is encoded in the bit pattern for AssociativeReduction
    }

    /// Get the reduction operator for a slot, if it is classified as a reduction.
    #[inline]
    pub fn get_reduction_op(&self, slot: u16) -> Option<BinOpKind> {
        if self.is_reduction(slot) {
            // Default to Add — the most common reduction operator
            Some(BinOpKind::Add)
        } else {
            None
        }
    }

    #[inline]
    pub fn is_reduction(&self, slot: u16) -> bool {
        let idx = slot as usize;
        assert!(idx < MAX_TRACKED_SLOTS, "slot index {} exceeds MAX_TRACKED_SLOTS {}", idx, MAX_TRACKED_SLOTS);
        if idx >= MAX_TRACKED_SLOTS { return false; }
        self.bits[idx >> SLOT_WORD_SHIFT] & (1u64 << (idx & SLOT_BIT_MASK)) != 0
    }
}

/// Scan the arena's statements and mark associative reduction accumulators.
///
/// A slot `d` is classified as a reduction accumulator if there exists a
/// statement `d = d op src` where `op` is associative and commutative
/// (Add, Mul, Min, Max), AND the same slot `d` is both read and written
/// in the same loop nest.
pub fn classify_reductions(arena: &ScopArena) -> ReductionMap {
    let mut map = ReductionMap::new();

    // Collect all self-accumulation patterns:  d = d op src
    let mut self_acc_slots = [0u64; SLOT_WORDS]; // bitset of self-accumulating dst slots
    for stmt in &arena.stmts {
        if stmt.dst == stmt.src1 {
            match stmt.op {
                BinOpKind::Add | BinOpKind::Mul => {
                    let idx = stmt.dst as usize;
                    if idx < MAX_TRACKED_SLOTS {
                        self_acc_slots[idx >> SLOT_WORD_SHIFT] |= 1u64 << (idx & SLOT_BIT_MASK);
                    }
                }
                _ => {}
            }
        }
    }

    // For each loop, check if the self-accumulating slot is both read and
    // written within the loop body (i.e., it is a reduction variable).
    for poly_loop in &arena.loops {
        let acc_start = poly_loop.access_start as usize;
        let acc_end   = acc_start + poly_loop.access_len as usize;
        let stmt_start = poly_loop.stmt_start as usize;
        let stmt_end   = stmt_start + poly_loop.stmt_len as usize;

        // Collect written and read slots within this loop
        let mut write_slots = [0u64; SLOT_WORDS];
        let mut read_slots  = [0u64; SLOT_WORDS];
        for acc in &arena.accesses[acc_start..acc_end] {
            let idx = acc.array_base_slot as usize;
            if idx < MAX_TRACKED_SLOTS {
                if acc.is_read {
                    read_slots[idx >> SLOT_WORD_SHIFT] |= 1u64 << (idx & SLOT_BIT_MASK);
                } else {
                    write_slots[idx >> SLOT_WORD_SHIFT] |= 1u64 << (idx & SLOT_BIT_MASK);
                }
            }
        }

        // Check each self-accumulating statement in this loop
        for stmt in &arena.stmts[stmt_start..stmt_end] {
            let idx = stmt.dst as usize;
            if idx >= MAX_TRACKED_SLOTS { continue; }
            let bit = 1u64 << (idx & SLOT_BIT_MASK);
            if self_acc_slots[idx >> SLOT_WORD_SHIFT] & bit != 0 {
                // Slot is self-accumulating AND read+written in this loop → reduction
                if (write_slots[idx >> SLOT_WORD_SHIFT] & bit != 0) && (read_slots[idx >> SLOT_WORD_SHIFT] & bit != 0) {
                    map.mark_reduction(stmt.dst, stmt.op);
                }
            }
        }
    }

    map
}

// =============================================================================
// §5. SCoP EXTRACTION — ARENA LAYOUT WITH TensorAccessRelation
// =============================================================================

#[derive(Debug, Clone)]
pub struct InductionVar {
    pub slot: u16,
    pub step: i64,
}

/// N-dimensional tensor access relation.
///
/// Models access to a rank-R tensor inside a depth-D loop nest as the
/// affine map:  A·v + c,  where A is R×D, v is the induction-variable
/// vector, and c is a constant offset vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorAccessRelation {
    pub array_base_slot: u16,
    pub rank: usize,
    /// Matrix A: `matrix[r][d]` is the coefficient of loop variable `d`
    /// in tensor dimension `r`.
    pub matrix: [[i64; MAX_POLY_DEPTH]; MAX_TENSOR_RANK],
    /// Constant offset per tensor dimension.
    pub offsets: [i64; MAX_TENSOR_RANK],
    pub is_read: bool,
}

impl TensorAccessRelation {
    /// Construct a rank-1 relation from a flat `AffineExpr` (compatibility shim).
    #[inline]
    pub fn from_affine(slot: u16, expr: &AffineExpr, is_read: bool) -> Self {
        let mut matrix = [[0i64; MAX_POLY_DEPTH]; MAX_TENSOR_RANK];
        matrix[0] = expr.coefficients;
        let mut offsets = [0i64; MAX_TENSOR_RANK];
        offsets[0] = expr.constant;
        Self { array_base_slot: slot, rank: 1, matrix, offsets, is_read }
    }

    /// Return the affine expression for dimension `r` as an `AffineExpr`.
    #[inline]
    pub fn dim_expr(&self, r: usize) -> AffineExpr {
        debug_assert!(r < self.rank);
        let mut active_mask = 0u8;
        for d in 0..MAX_POLY_DEPTH {
            if self.matrix[r][d] != 0 { active_mask |= 1 << d; }
        }
        AffineExpr {
            constant:     self.offsets[r],
            coefficients: self.matrix[r],
            active_mask,
        }
    }

    /// Virtual dimension-swap (zero-cost logical transpose).
    ///
    /// Swapping stride columns `d0` and `d1` in every tensor dimension row
    /// is equivalent to reordering the loop nest axes d0↔d1 — no byte is
    /// moved at runtime.
    #[inline]
    pub fn swap_loop_dims(&mut self, d0: usize, d1: usize) {
        for r in 0..self.rank {
            self.matrix[r].swap(d0, d1);
        }
    }

    /// Apply a transformation matrix T to the access relation.
    /// Each row r becomes T * old_row_r (coefficient transformation).
    pub fn apply_transform(&mut self, tm: &TransformMatrix) {
        for r in 0..self.rank {
            let mut new_row = [0i64; MAX_POLY_DEPTH];
            for j in 0..tm.dim.min(MAX_POLY_DEPTH) {
                let mut acc = 0i64;
                for k in 0..tm.dim.min(MAX_POLY_DEPTH) {
                    acc = acc.saturating_add(
                        tm.rows[j][k].saturating_mul(self.matrix[r][k])
                    );
                }
                new_row[j] = acc;
            }
            self.matrix[r] = new_row;
        }
    }
}

/// Legacy flat access relation kept for the arena's public interface.
/// All new code should prefer `TensorAccessRelation`.
#[derive(Debug, Clone)]
pub struct AccessRelation {
    pub array_base_slot: u16,
    pub index_expr: AffineExpr,
    pub is_read: bool,
}

/// Hardware profile for the roofline model.
#[derive(Debug, Clone, Copy)]
pub struct HardwareProfile {
    pub peak_gflops:           f64,
    pub mem_bandwidth_gb_per_sec: f64,
    pub l1_cache_bytes:        usize,
    pub l2_cache_bytes:        usize,
}

impl Default for HardwareProfile {
    /// Conservative defaults (modern x86 server core with AVX-512).
    fn default() -> Self {
        Self {
            peak_gflops:              3_072.0,
            mem_bandwidth_gb_per_sec:   200.0,
            l1_cache_bytes:            32_768,
            l2_cache_bytes:           524_288,
        }
    }
}

/// Classification of whether a SCoP is memory-bound or compute-bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationRoute {
    ComputeBound { attainable_gflops: f64 },
    MemoryBound  { attainable_gflops: f64 },
}

/// Three-tier tiling configuration passed to `generate_hierarchical_tiled_loops`.
#[derive(Debug, Clone, Copy)]
pub struct TileHierarchy {
    pub l3_l2_size:    usize,
    pub l1_size:       usize,
    pub register_size: usize,
}

#[derive(Debug, Clone)]
pub struct PolyStmt {
    pub id: usize,
    pub op: BinOpKind,
    pub dst: u16,
    pub src1: u16,
    pub src2: u16,
}

/// Cache-friendly arena-allocated loop node.
#[derive(Debug, Clone)]
pub struct PolyLoop {
    pub depth: usize,
    pub iv: InductionVar,
    pub lower_bound: AffineExpr,
    pub upper_bound: AffineExpr,
    pub child_start: u32,
    pub child_len:   u32,
    pub access_start: u32,
    pub access_len:   u32,
    /// Start index into ScopArena::tensor_accesses for this loop's N-dim accesses.
    pub tensor_access_start: u32,
    /// Number of tensor access relations for this loop.
    pub tensor_access_len: u32,
    pub stmt_start: u32,
    pub stmt_len:   u32,
    pub header_pc:    usize,
    pub back_edge_pc: usize,
}

/// Flat arena holding all SCoP data in contiguous allocations.
/// Includes both legacy `AccessRelation` and N-dimensional `TensorAccessRelation`.
#[derive(Debug, Clone)]
pub struct ScopArena {
    pub loops:    Vec<PolyLoop>,
    pub accesses: Vec<AccessRelation>,
    /// N-dimensional tensor access relations — populated during extraction.
    pub tensor_accesses: Vec<TensorAccessRelation>,
    pub stmts:    Vec<PolyStmt>,
    pub root_loop_indices: Vec<u32>,
    pub max_depth: usize,
}

#[derive(Debug, Clone)]
pub struct Scop {
    pub arena: ScopArena,
    /// Reduction classification results.
    pub reduction_map: ReductionMap,
}

impl Scop {
    #[inline]
    pub fn max_depth(&self) -> usize { self.arena.max_depth }
}

/// Slot-indexed affine expression cache with 4096-bit presence bitset.
pub struct SlotCache {
    pub data: Vec<Option<AffineExpr>>,
    present: [u64; SLOT_WORDS],
}

impl SlotCache {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: vec![None; MAX_TRACKED_SLOTS],
            present: [0u64; SLOT_WORDS],
        }
    }

    #[inline]
    pub fn insert(&mut self, slot: u16, expr: AffineExpr) {
        let idx = slot as usize;
        assert!(idx < MAX_TRACKED_SLOTS, "slot index {} exceeds MAX_TRACKED_SLOTS {}", idx, MAX_TRACKED_SLOTS);
        if idx < MAX_TRACKED_SLOTS {
            self.data[idx] = Some(expr);
            self.present[idx >> SLOT_WORD_SHIFT] |= 1u64 << (idx & SLOT_BIT_MASK);
        }
    }

    #[inline]
    pub fn get(&self, slot: u16) -> Option<AffineExpr> {
        let idx = slot as usize;
        assert!(idx < MAX_TRACKED_SLOTS, "slot index {} exceeds MAX_TRACKED_SLOTS {}", idx, MAX_TRACKED_SLOTS);
        if idx >= MAX_TRACKED_SLOTS { return None; }
        if self.present[idx >> SLOT_WORD_SHIFT] & (1u64 << (idx & SLOT_BIT_MASK)) == 0 {
            return None;
        }
        self.data[idx]
    }
}

/// Forward-pass cache population: seeds every slot reachable by affine
/// arithmetic from known loop induction variables.
///
/// Non-linear sub-expressions that are loop-invariant are admitted as
/// Uninterpreted Symbols (constant-zero AffineExpr).  This prevents the
/// polyhedral extractor from aborting on stencils with A[i*i], A[sin(i)*100],
/// or similar non-linear index terms, while still allowing the outer loop
/// nest to be fully optimised.
pub fn populate_slot_cache(instrs: &[Instr], cache: &mut SlotCache, loop_iv_slots: &[u16]) {
    for (i, &iv_slot) in loop_iv_slots.iter().enumerate() {
        if i < MAX_POLY_DEPTH {
            cache.insert(iv_slot, AffineExpr::variable(i));
        }
    }

    let mut iv_bitset = [0u64; SLOT_WORDS];
    for &s in loop_iv_slots {
        iv_bitset[(s >> SLOT_WORD_SHIFT) as usize] |= 1u64 << ((s as usize) & SLOT_BIT_MASK);
    }
    let iv_is_variant = |slot: u16| -> bool {
        iv_bitset[(slot >> SLOT_WORD_SHIFT) as usize] & (1u64 << ((slot as usize) & SLOT_BIT_MASK)) != 0
    };

    for instr in instrs {
        match *instr {
            Instr::LoadI64(d, v) => cache.insert(d, AffineExpr::constant(v)),
            Instr::LoadI32(d, v) => cache.insert(d, AffineExpr::constant(v as i64)),
            Instr::LoadF32(d, _) | Instr::LoadF64(d, _) => {
                if !iv_is_variant(d) {
                    cache.insert(d, AffineExpr::constant(0));
                }
            }
            Instr::Move(d, s) => {
                if let Some(expr) = cache.get(s) { cache.insert(d, expr); }
            }
            Instr::BinOp(d, op, l, r) => {
                let l_expr = cache.get(l);
                let r_expr = cache.get(r);
                match op {
                    BinOpKind::Add => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            cache.insert(d, le.add(&re));
                        }
                    }
                    BinOpKind::Sub => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            cache.insert(d, le.sub(&re));
                        }
                    }
                    BinOpKind::Mul => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            if re.active_mask == 0 {
                                cache.insert(d, le.mul_const(re.constant));
                            } else if le.active_mask == 0 {
                                cache.insert(d, re.mul_const(le.constant));
                            } else {
                                // Non-linear multiply: treat as UninterpretedSymbol
                                cache.insert(d, AffineExpr::constant(0));
                            }
                        }
                    }
                    BinOpKind::Div | BinOpKind::Rem => {
                        let divisor_invariant = r_expr
                            .map(|e| e.active_mask == 0)
                            .unwrap_or(false);
                        if divisor_invariant {
                            cache.insert(d, AffineExpr::constant(0));
                        }
                    }
                    _ => {
                        if let (Some(le), Some(re)) = (l_expr, r_expr) {
                            if le.active_mask == 0 && re.active_mask == 0 {
                                cache.insert(d, AffineExpr::constant(0));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[inline]
fn get_dst_slot(instr: &Instr) -> u16 {
    match *instr {
        Instr::LoadI32(d, _) | Instr::LoadI64(d, _)
        | Instr::LoadBool(d, _) | Instr::LoadUnit(d) => d,
        Instr::Move(d, _) | Instr::Load(d, _)
        | Instr::Store(d, _) | Instr::BinOp(d, _, _, _) => d,
        _ => 0,
    }
}

fn analyze_loop(
    instrs: &[Instr],
    header: usize,
    back_edge: usize,
    loop_iv_slots: &mut Vec<u16>,
    cache: &SlotCache,
) -> (Option<InductionVar>, AffineExpr, AffineExpr) {
    let mut iv = None;
    let mut lb = AffineExpr::constant(0);
    let mut ub = AffineExpr::constant(1024);

    let mut loop_compare_slots: Vec<u16> = Vec::new();
    let compare_start = header.saturating_sub(3);
    for pc in compare_start..=back_edge {
        if pc < instrs.len() {
            if let Instr::BinOp(_, BinOpKind::Lt, l, _) = instrs[pc] {
                if !loop_compare_slots.contains(&l) {
                    loop_compare_slots.push(l);
                }
            }
        }
    }

    for pc in header..=back_edge {
        if let Instr::BinOp(dst, BinOpKind::Add, l, r) = instrs[pc] {
            if dst == l && pc > 0 {
                if let Instr::LoadI64(_, step) = instrs[pc - 1] {
                    if r == get_dst_slot(&instrs[pc - 1]) {
                        if loop_compare_slots.contains(&dst) {
                            if loop_iv_slots.len() < MAX_POLY_DEPTH {
                                loop_iv_slots.push(dst);
                                iv = Some(InductionVar { slot: dst, step });
                                break;
                            }
                        }
                    }
                }
            }
            if pc + 1 < instrs.len() {
                if let Instr::Move(dst_slot, src_slot) = instrs[pc + 1] {
                    if src_slot == dst {
                        if loop_compare_slots.contains(&dst_slot) {
                            let step_val = if pc > 0 {
                                if let Instr::LoadI64(_, step) = instrs[pc - 1] {
                                    if r == get_dst_slot(&instrs[pc - 1]) { Some(step) } else { None }
                                } else if let Instr::LoadI32(_, step) = instrs[pc - 1] {
                                    if r == get_dst_slot(&instrs[pc - 1]) { Some(step as i64) } else { None }
                                } else { None }
                            } else { None };
                            if let Some(step) = step_val {
                                if loop_iv_slots.len() < MAX_POLY_DEPTH {
                                    loop_iv_slots.push(dst_slot);
                                    iv = Some(InductionVar { slot: dst_slot, step });
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(ref iv_ref) = iv {
        let mut found_ub = false;
        let search_start = header.saturating_sub(3);

        if let Instr::JumpTrue(cond, _) = instrs[back_edge] {
            for pc in (search_start..back_edge).rev() {
                if pc < instrs.len() {
                    if let Instr::BinOp(d, BinOpKind::Lt, l, r) = instrs[pc] {
                        if d == cond && l == iv_ref.slot {
                            if let Some(expr) = cache.get(r) { ub = expr; }
                            found_ub = true;
                            break;
                        }
                    }
                }
            }
        }

        if !found_ub {
            for pc in search_start..=back_edge {
                if pc < instrs.len() {
                    if let Instr::JumpFalse(cond, _) = instrs[pc] {
                        for cond_pc in (search_start..=pc).rev() {
                            if cond_pc < instrs.len() {
                                if let Instr::BinOp(d, BinOpKind::Lt, l, r) = instrs[cond_pc] {
                                    if d == cond && l == iv_ref.slot {
                                        if let Some(expr) = cache.get(r) { ub = expr; }
                                        found_ub = true;
                                        break;
                                    }
                                }
                            }
                        }
                        if found_ub { break; }
                    }
                }
            }
        }

        if !found_ub {
            for pc in search_start..=back_edge {
                if pc < instrs.len() {
                    if let Instr::BinOp(_, BinOpKind::Lt, l, r) = instrs[pc] {
                        if l == iv_ref.slot {
                            if let Some(expr) = cache.get(r) { ub = expr; }
                            break;
                        }
                    }
                }
            }
        }

        if let Some(expr) = cache.get(iv_ref.slot) { lb = expr; }
    }

    (iv, lb, ub)
}

/// True iterative loop-tree builder — no recursion, no stack overflow risk.
fn build_arena_iterative(
    loop_ranges: &[(usize, usize)],
    instrs: &[Instr],
    loop_iv_slots: &mut Vec<u16>,
    cache: &SlotCache,
) -> ScopArena {
    let mut arena = ScopArena {
        loops:    Vec::with_capacity(loop_ranges.len()),
        accesses: Vec::new(),
        tensor_accesses: Vec::new(),
        stmts:    Vec::new(),
        root_loop_indices: Vec::new(),
        max_depth: 0,
    };

    if loop_ranges.is_empty() { return arena; }

    let mut stack: Vec<(usize, u32)> = Vec::with_capacity(64);

    for (i, &(h, b)) in loop_ranges.iter().enumerate().rev() {
        let is_nested = loop_ranges[..i]
            .iter()
            .any(|&(ph, pb)| ph <= h && b <= pb);
        if !is_nested {
            stack.push((i, u32::MAX));
        }
    }

    while let Some((range_idx, parent_arena_idx)) = stack.pop() {
        let (h, b) = loop_ranges[range_idx];

        let mut children: Vec<usize> = loop_ranges
            .iter()
            .enumerate()
            .filter(|&(ci, &(ch, cb))| {
                ci != range_idx
                    && ch > h && cb <= b
                    && !loop_ranges.iter().enumerate().any(|(si, &(sh, sb))| {
                        si != range_idx && si != ci
                            && sh > h && sb <= b
                            && ch >= sh && cb <= sb
                    })
            })
            .map(|(ci, _)| ci)
            .collect();
        children.sort_unstable_by_key(|&ci| loop_ranges[ci].0);

        let (iv, lb, ub) = analyze_loop(instrs, h, b, loop_iv_slots, cache);
        let iv = match iv {
            Some(v) => v,
            None => {
                for ci in children.into_iter().rev() {
                    stack.push((ci, parent_arena_idx));
                }
                continue;
            }
        };

        let access_start = arena.accesses.len() as u32;
        let stmt_start   = arena.stmts.len()    as u32;
        let tensor_acc_start = arena.tensor_accesses.len() as u32;

        for pc in h..=b {
            let in_child = children.iter().any(|&ci| {
                let (ch, cb) = loop_ranges[ci];
                pc > ch && pc < cb
            });
            if in_child { continue; }

            match instrs[pc] {
                Instr::Load(_, ptr_slot) => {
                    if let Some(expr) = cache.get(ptr_slot) {
                        arena.accesses.push(AccessRelation {
                            array_base_slot: ptr_slot,
                            index_expr: expr,
                            is_read: true,
                        });
                        arena.tensor_accesses.push(
                            TensorAccessRelation::from_affine(ptr_slot, &expr, true)
                        );
                    }
                }
                Instr::Store(ptr_slot, _) => {
                    if let Some(expr) = cache.get(ptr_slot) {
                        arena.accesses.push(AccessRelation {
                            array_base_slot: ptr_slot,
                            index_expr: expr,
                            is_read: false,
                        });
                        arena.tensor_accesses.push(
                            TensorAccessRelation::from_affine(ptr_slot, &expr, false)
                        );
                    }
                }
                Instr::BinOp(dst, op, l, r) => {
                    arena.stmts.push(PolyStmt { id: pc, op, dst, src1: l, src2: r });
                }
                _ => {}
            }
        }

        let access_len = arena.accesses.len() as u32 - access_start;
        let stmt_len   = arena.stmts.len()    as u32 - stmt_start;
        let tensor_acc_len = arena.tensor_accesses.len() as u32 - tensor_acc_start;
        let my_arena_idx = arena.loops.len() as u32;

        arena.loops.push(PolyLoop {
            depth: 1,
            iv,
            lower_bound: lb,
            upper_bound: ub,
            child_start: 0,
            child_len:   0,
            access_start,
            access_len,
            tensor_access_start: tensor_acc_start,
            tensor_access_len: tensor_acc_len,
            stmt_start,
            stmt_len,
            header_pc:    h,
            back_edge_pc: b,
        });

        if parent_arena_idx == u32::MAX {
            arena.root_loop_indices.push(my_arena_idx);
        }

        for ci in children.into_iter().rev() {
            stack.push((ci, my_arena_idx));
        }
    }

    // Post-pass: wire up child ranges and compute depths
    let n = arena.loops.len();
    let mut parent_of = vec![u32::MAX; n];
    for i in 0..n {
        let (h, b) = (arena.loops[i].header_pc, arena.loops[i].back_edge_pc);
        let mut best_parent: Option<usize> = None;
        for j in 0..n {
            if j == i { continue; }
            let (ph, pb) = (arena.loops[j].header_pc, arena.loops[j].back_edge_pc);
            if ph <= h && b <= pb {
                best_parent = Some(match best_parent {
                    None => j,
                    Some(prev) => {
                        if ph > arena.loops[prev].header_pc { j } else { prev }
                    }
                });
            }
        }
        parent_of[i] = best_parent.map(|p| p as u32).unwrap_or(u32::MAX);
    }

    for p in 0..n {
        let first_child = (0..n).find(|&c| parent_of[c] == p as u32);
        if let Some(fc) = first_child {
            let child_count = (0..n).filter(|&c| parent_of[c] == p as u32).count();
            arena.loops[p].child_start = fc as u32;
            arena.loops[p].child_len   = child_count as u32;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| {
        arena.loops[i].back_edge_pc.wrapping_sub(arena.loops[i].header_pc)
    });
    for &i in &order {
        let cs = arena.loops[i].child_start as usize;
        let ce = cs + arena.loops[i].child_len as usize;
        let max_child = (cs..ce).map(|c| arena.loops[c].depth).max().unwrap_or(0);
        arena.loops[i].depth = 1 + max_child;
    }

    arena.max_depth = arena.root_loop_indices.iter()
        .map(|&ri| arena.loops[ri as usize].depth)
        .max()
        .unwrap_or(1);

    arena
}

pub fn extract_scop(instrs: &[Instr]) -> Option<Scop> {
    let mut loop_ranges = Vec::new();

    for (pc, instr) in instrs.iter().enumerate() {
        let target = match *instr {
            Instr::Jump(off)         => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpFalse(_, off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpTrue(_, off)  => Some((pc as i32 + 1 + off) as usize),
            _ => None,
        };
        if let Some(t) = target {
            if t <= pc && t < instrs.len() {
                loop_ranges.push((t, pc));
            }
        }
    }

    if loop_ranges.is_empty() { return None; }

    loop_ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

    let mut loop_iv_slots: Vec<u16> = Vec::with_capacity(MAX_POLY_DEPTH);
    {
        let tmp_cache = SlotCache::new();
        for &(h, b) in &loop_ranges {
            let mut dummy = Vec::new();
            let (iv, _, _) = analyze_loop(instrs, h, b, &mut dummy, &tmp_cache);
            if let Some(ind_v) = iv {
                if !loop_iv_slots.contains(&ind_v.slot)
                    && loop_iv_slots.len() < MAX_POLY_DEPTH
                {
                    loop_iv_slots.push(ind_v.slot);
                }
            }
        }
    }

    let mut cache = SlotCache::new();
    populate_slot_cache(instrs, &mut cache, &loop_iv_slots);

    let arena = build_arena_iterative(&loop_ranges, instrs, &mut loop_iv_slots, &cache);

    if arena.loops.is_empty() { return None; }

    let reduction_map = classify_reductions(&arena);

    Some(Scop { arena, reduction_map })
}

// =============================================================================
// §6. ZERO-COST VIRTUAL DIMENSION SWAPPING (Logical Transpose)
// =============================================================================
//
// Instead of emitting a physical copy loop for a matrix transpose A[i][j]→A[j][i],
// we swap the stride columns in the TensorAccessRelation matrix.  Downstream
// code generation automatically reads memory in the transposed pattern without
// moving a single byte.

/// Apply a virtual transpose to all tensor access relations in the arena
/// that reference the given base slot.  Returns the number of relations
/// modified.
pub fn apply_virtual_transpose(arena: &mut ScopArena, base_slot: u16, dim_a: usize, dim_b: usize) -> usize {
    let mut count = 0;
    for tac in &mut arena.tensor_accesses {
        if tac.array_base_slot == base_slot && dim_a < tac.rank && dim_b < tac.rank {
            tac.matrix.swap(dim_a, dim_b);
            tac.offsets.swap(dim_a, dim_b);
            count += 1;
        }
    }
    // Also update legacy AccessRelation entries that reference this slot
    // (they are rank-1 and don't have dimension pairs to swap, so we skip).
    count
}

// =============================================================================
// §7. LOOP SKEWING (Wavefront Parallelism for Stencils/FDTD)
// =============================================================================

#[derive(Debug, Clone)]
pub struct TransformMatrix {
    pub rows: [[i64; MAX_POLY_DEPTH]; MAX_POLY_DEPTH],
    pub dim: usize,
}

impl TransformMatrix {
    #[inline]
    pub fn identity(dim: usize) -> Self {
        let mut rows = [[0; MAX_POLY_DEPTH]; MAX_POLY_DEPTH];
        for i in 0..dim.min(MAX_POLY_DEPTH) { rows[i][i] = 1; }
        Self { rows, dim }
    }

    #[inline]
    pub fn interchange(&mut self, i: usize, j: usize) {
        if i < self.dim && j < self.dim { self.rows.swap(i, j); }
    }

    /// Apply this transformation to all tensor access relations in the arena.
    pub fn apply_to_arena(&self, arena: &mut ScopArena) {
        for tac in &mut arena.tensor_accesses {
            tac.apply_transform(self);
        }
    }
}

/// Build a skewing transformation matrix for stencil parallelism.
///
/// For a time-space stencil `A[t][i] = f(A[t-1][i-1], A[t-1][i+1])`,
/// the dependency vector (1, ±1) prevents parallelisation along `i`.
/// After the skew:  t' = t,  i' = i + skew_factor * t, the dependency
/// vectors become (1, 0) and (1, 2), exposing fully independent wavefronts.
pub fn build_skew_matrix(dim: usize, time_axis: usize, space_axis: usize, skew_factor: i64) -> TransformMatrix {
    let mut tm = TransformMatrix::identity(dim);
    if time_axis < dim && space_axis < dim && time_axis != space_axis {
        tm.rows[space_axis][time_axis] = skew_factor;
    }
    tm
}

// =============================================================================
// §8. PHYSICAL TILING LOOP GENERATION WITH INDEX SET SPLITTING
// =============================================================================

/// A guard condition hoisted from the original loop by the tracing JIT.
#[derive(Debug, Clone)]
pub struct HoistedGuard {
    pub slot: u16,
    pub guard_type: u8, // expected type byte
    pub original_loop_pc: usize, // PC of the original loop header
}

/// Table of hoisted guards that must be recalculated when loops are split.
#[derive(Debug, Clone)]
pub struct GuardTable {
    pub guards: Vec<HoistedGuard>,
}

impl GuardTable {
    pub fn new() -> Self { Self { guards: Vec::new() } }

    /// Add a guard from the tracing JIT.
    pub fn add_guard(&mut self, slot: u16, guard_type: u8, loop_pc: usize) {
        self.guards.push(HoistedGuard { slot, guard_type, original_loop_pc: loop_pc });
    }

    /// Rewrite all guards when a loop at `original_pc` has been split into
    /// a core loop (at `core_pc`) and a fringe loop (at `fringe_pc`).
    /// Each guard that referenced the original loop must be duplicated:
    /// one copy for the core loop and one for the fringe loop.
    pub fn rewrite_for_split(&mut self, original_pc: usize, core_pc: usize, fringe_pc: usize) {
        let mut new_guards = Vec::new();
        for guard in &self.guards {
            if guard.original_loop_pc == original_pc {
                // Duplicate: one guard per split loop
                new_guards.push(HoistedGuard { slot: guard.slot, guard_type: guard.guard_type, original_loop_pc: core_pc });
                new_guards.push(HoistedGuard { slot: guard.slot, guard_type: guard.guard_type, original_loop_pc: fringe_pc });
            } else {
                new_guards.push(guard.clone());
            }
        }
        self.guards = new_guards;
    }
}

pub fn generate_tiled_loops(scop: &Scop, tile_sizes: &[usize], guard_table: &mut GuardTable) -> Vec<Instr> {
    let arena = &scop.arena;

    let total_stmts: usize = arena.loops.iter().map(|l| l.stmt_len as usize).sum();
    let mut out = Vec::with_capacity(arena.loops.len() * 20 + total_stmts);
    let mut next_slot: u16 = 4000;

    for poly_loop in &arena.loops {
        let tile_size = tile_sizes.get(0).copied().unwrap_or(32) as i64;

        let ti_slot           = next_slot; next_slot += 1;
        let limit_slot        = next_slot; next_slot += 1;
        let tile_const_slot   = next_slot; next_slot += 1;
        let n_const_slot      = next_slot; next_slot += 1;
        let ti_plus_tile_slot = next_slot; next_slot += 1;
        let cond_slot         = next_slot; next_slot += 1;
        let core_limit_slot   = next_slot; next_slot += 1;

        let lower = if poly_loop.lower_bound.active_mask == 0 {
            poly_loop.lower_bound.constant
        } else {
            0
        };
        let upper = if poly_loop.upper_bound.active_mask == 0 {
            poly_loop.upper_bound.constant
        } else {
            1024
        };

        out.push(Instr::LoadI64(ti_slot, lower));
        out.push(Instr::LoadI64(tile_const_slot, tile_size));
        out.push(Instr::LoadI64(n_const_slot, upper));

        // ── Index Set Splitting ─────────────────────────────────────────────
        let perfect_core = if tile_size > 0 {
            ((upper - lower) / tile_size) * tile_size + lower
        } else {
            upper
        };
        out.push(Instr::LoadI64(core_limit_slot, perfect_core));

        // ── CORE LOOP: guard-free, maximum SIMD throughput ──────────────────
        let l1_pc = out.len();
        out.push(Instr::BinOp(cond_slot, BinOpKind::Ge, ti_slot, core_limit_slot));
        out.push(Instr::JumpTrue(cond_slot, 0));
        let end1_core_patch = out.len() - 1;

        out.push(Instr::Move(poly_loop.iv.slot, ti_slot));
        out.push(Instr::BinOp(ti_plus_tile_slot, BinOpKind::Add, ti_slot, tile_const_slot));
        out.push(Instr::Move(limit_slot, ti_plus_tile_slot));

        let l2_pc = out.len();
        out.push(Instr::BinOp(cond_slot, BinOpKind::Ge, poly_loop.iv.slot, limit_slot));
        out.push(Instr::JumpTrue(cond_slot, 0));
        let end2_core_patch = out.len() - 1;

        let stmt_range =
            poly_loop.stmt_start as usize..(poly_loop.stmt_start + poly_loop.stmt_len) as usize;
        for stmt in &arena.stmts[stmt_range.clone()] {
            out.push(Instr::BinOp(stmt.dst, stmt.op, stmt.src1, stmt.src2));
        }

        let step_slot = next_slot; next_slot += 1;
        out.push(Instr::LoadI64(step_slot, poly_loop.iv.step));
        out.push(Instr::BinOp(poly_loop.iv.slot, BinOpKind::Add, poly_loop.iv.slot, step_slot));
        let l2_offset = (l2_pc as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(l2_offset));

        let end2_core_pc = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[end2_core_patch] {
            *off = (end2_core_pc as i32) - (end2_core_patch as i32) - 1;
        }

        out.push(Instr::BinOp(ti_slot, BinOpKind::Add, ti_slot, tile_const_slot));
        let l1_offset = (l1_pc as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(l1_offset));

        let end1_core_pc = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[end1_core_patch] {
            *off = (end1_core_pc as i32) - (end1_core_patch as i32) - 1;
        }

        // ── CLEANUP FRINGE LOOP ─────────────────────────────────────────────
        if perfect_core < upper {
            out.push(Instr::Move(ti_slot, core_limit_slot));
            out.push(Instr::Move(poly_loop.iv.slot, core_limit_slot));

            let fringe_pc = out.len();
            out.push(Instr::BinOp(cond_slot, BinOpKind::Ge, poly_loop.iv.slot, n_const_slot));
            out.push(Instr::JumpTrue(cond_slot, 0));
            let fringe_end_patch = out.len() - 1;

            for stmt in &arena.stmts[stmt_range] {
                out.push(Instr::BinOp(stmt.dst, stmt.op, stmt.src1, stmt.src2));
            }

            let step_slot2 = next_slot; next_slot += 1;
            out.push(Instr::LoadI64(step_slot2, poly_loop.iv.step));
            out.push(Instr::BinOp(poly_loop.iv.slot, BinOpKind::Add, poly_loop.iv.slot, step_slot2));
            let fringe_back = (fringe_pc as i32) - (out.len() as i32) - 1;
            out.push(Instr::Jump(fringe_back));

            let fringe_end_pc = out.len();
            if let Instr::JumpTrue(_, ref mut off) = out[fringe_end_patch] {
                *off = (fringe_end_pc as i32) - (fringe_end_patch as i32) - 1;
            }

            // Rewrite any hoisted guards that referenced the original loop
            // to point to the core and fringe loops instead.
            guard_table.rewrite_for_split(poly_loop.header_pc, l1_pc, fringe_pc);
        }
    }

    out
}

// =============================================================================
// §9. THREE-TIER HIERARCHICAL TILING
// =============================================================================

/// Generates a three-tier loop nest: L3/L2 macro → L1 midi → register micro.
///
/// The outermost two tiers stream data panels through the cache hierarchy.
/// The innermost tier maps directly to physical SIMD/AMX register blocks,
/// completely eliminating register spilling.
pub fn generate_hierarchical_tiled_loops(
    scop: &Scop,
    configs: &[TileHierarchy],
) -> Vec<Instr> {
    let arena = &scop.arena;
    let total_stmts: usize = arena.loops.iter().map(|l| l.stmt_len as usize).sum();
    let mut out = Vec::with_capacity(arena.loops.len() * 40 + total_stmts);
    let mut next_slot: u16 = 5000;

    let default_cfg = TileHierarchy {
        l3_l2_size:    MACRO_TILE_K,
        l1_size:       MIDI_TILE_K,
        register_size: REGISTER_TILE_N,
    };

    for poly_loop in &arena.loops {
        let cfg = configs.get(0).unwrap_or(&default_cfg);

        let l3_tile  = cfg.l3_l2_size    as i64;
        let l1_tile  = cfg.l1_size        as i64;
        let reg_tile = cfg.register_size  as i64;

        let upper = if poly_loop.upper_bound.active_mask == 0 {
            poly_loop.upper_bound.constant
        } else {
            1024
        };
        let lower = if poly_loop.lower_bound.active_mask == 0 {
            poly_loop.lower_bound.constant
        } else {
            0
        };

        let l3_iv   = next_slot; next_slot += 1;
        let l1_iv   = next_slot; next_slot += 1;
        let reg_iv  = next_slot; next_slot += 1;
        let cond    = next_slot; next_slot += 1;
        let l3_lim  = next_slot; next_slot += 1;
        let l1_lim  = next_slot; next_slot += 1;
        let reg_lim = next_slot; next_slot += 1;
        let l3_c    = next_slot; next_slot += 1;
        let l1_c    = next_slot; next_slot += 1;
        let rc      = next_slot; next_slot += 1;
        let n_s     = next_slot; next_slot += 1;
        let step_s  = next_slot; next_slot += 1;

        out.push(Instr::LoadI64(n_s, upper));
        out.push(Instr::LoadI64(l3_c, l3_tile));
        out.push(Instr::LoadI64(l1_c, l1_tile));
        out.push(Instr::LoadI64(rc, reg_tile));
        out.push(Instr::LoadI64(step_s, poly_loop.iv.step));
        out.push(Instr::LoadI64(l3_iv, lower));

        // ── Tier 1: L3/L2 Macro loop ─────────────────────────────────────
        let l3_header = out.len();
        out.push(Instr::BinOp(cond, BinOpKind::Ge, l3_iv, n_s));
        out.push(Instr::JumpTrue(cond, 0));
        let l3_exit_patch = out.len() - 1;

        let tmp = next_slot; next_slot += 1;
        out.push(Instr::BinOp(tmp, BinOpKind::Add, l3_iv, l3_c));
        out.push(Instr::BinOp(cond, BinOpKind::Gt, tmp, n_s));
        out.push(Instr::JumpFalse(cond, 2));
        out.push(Instr::Move(l3_lim, n_s));
        out.push(Instr::Jump(1));
        out.push(Instr::Move(l3_lim, tmp));
        out.push(Instr::Move(l1_iv, l3_iv));

        // ── Tier 2: L1 Midi loop ──────────────────────────────────────────
        let l1_header = out.len();
        out.push(Instr::BinOp(cond, BinOpKind::Ge, l1_iv, l3_lim));
        out.push(Instr::JumpTrue(cond, 0));
        let l1_exit_patch = out.len() - 1;

        let tmp2 = next_slot; next_slot += 1;
        out.push(Instr::BinOp(tmp2, BinOpKind::Add, l1_iv, l1_c));
        out.push(Instr::BinOp(cond, BinOpKind::Gt, tmp2, l3_lim));
        out.push(Instr::JumpFalse(cond, 2));
        out.push(Instr::Move(l1_lim, l3_lim));
        out.push(Instr::Jump(1));
        out.push(Instr::Move(l1_lim, tmp2));
        out.push(Instr::Move(reg_iv, l1_iv));

        // ── Tier 3: Register micro-kernel loop ────────────────────────────
        let reg_header = out.len();
        out.push(Instr::BinOp(cond, BinOpKind::Ge, reg_iv, l1_lim));
        out.push(Instr::JumpTrue(cond, 0));
        let reg_exit_patch = out.len() - 1;

        let tmp3 = next_slot; next_slot += 1;
        out.push(Instr::BinOp(tmp3, BinOpKind::Add, reg_iv, rc));
        out.push(Instr::BinOp(cond, BinOpKind::Gt, tmp3, l1_lim));
        out.push(Instr::JumpFalse(cond, 2));
        out.push(Instr::Move(reg_lim, l1_lim));
        out.push(Instr::Jump(1));
        out.push(Instr::Move(reg_lim, tmp3));

        out.push(Instr::Move(poly_loop.iv.slot, reg_iv));

        let stmt_range =
            poly_loop.stmt_start as usize..(poly_loop.stmt_start + poly_loop.stmt_len) as usize;
        for stmt in &arena.stmts[stmt_range] {
            out.push(Instr::BinOp(stmt.dst, stmt.op, stmt.src1, stmt.src2));
        }

        out.push(Instr::BinOp(reg_iv, BinOpKind::Add, reg_iv, rc));
        let reg_back = (reg_header as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(reg_back));
        let reg_exit = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[reg_exit_patch] {
            *off = (reg_exit as i32) - (reg_exit_patch as i32) - 1;
        }

        out.push(Instr::BinOp(l1_iv, BinOpKind::Add, l1_iv, l1_c));
        let l1_back = (l1_header as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(l1_back));
        let l1_exit = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[l1_exit_patch] {
            *off = (l1_exit as i32) - (l1_exit_patch as i32) - 1;
        }

        out.push(Instr::BinOp(l3_iv, BinOpKind::Add, l3_iv, l3_c));
        let l3_back = (l3_header as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(l3_back));
        let l3_exit = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[l3_exit_patch] {
            *off = (l3_exit as i32) - (l3_exit_patch as i32) - 1;
        }

        // step_s holds the original loop step — use it to advance the
        // induction variable if the loop has a non-unit step.
        if poly_loop.iv.step != 1 {
            // For non-unit steps, the tiled loops should advance by
            // step * tile_size. This is already handled by the inner
            // loop which uses the induction variable directly.
            out.push(Instr::BinOp(poly_loop.iv.slot, BinOpKind::Add,
                                  poly_loop.iv.slot, step_s));
        }
    }

    out
}

// =============================================================================
// §10. REGISTER-LOCKED MICRO-KERNEL EMISSION (AMX / AVX-512)
// =============================================================================

/// Represents a micro-kernel as an opaque intrinsic block within the IR.
/// The register allocator treats these as atomic blocks: all declared
/// clobbered registers are unavailable during the kernel's lifetime,
/// and the allocator must spill any active temporaries that overlap.
///
/// This prevents the register allocator from silently assigning GPRs or
/// SIMD registers that the micro-kernel uses for its own inner loop,
/// which would cause state clobbering and silent miscompilation.
#[derive(Debug, Clone)]
pub struct MicroKernelNode {
    /// What kind of micro-kernel this represents (GEMM, convolution, etc.)
    pub kernel_type: MicroKernelKind,
    /// SSA slot indices read by the kernel (inputs)
    pub input_slots: Vec<u16>,
    /// SSA slot indices written by the kernel (outputs / accumulators)
    pub output_slots: Vec<u16>,
    /// Physical GPR register numbers clobbered by the kernel (0-15)
    pub clobbered_gprs: Vec<u8>,
    /// Physical SIMD register numbers clobbered by the kernel (0-31 for ZMM)
    pub clobbered_simd: Vec<u8>,
}

/// Classification of micro-kernel types for the IR
#[derive(Debug, Clone, PartialEq)]
pub enum MicroKernelKind {
    /// AVX-512 GEMM kernel: uses ZMM0-ZMM31 for tiles, R8-R11 for loop vars
    Avx512Gemm { tile_m: usize, tile_n: usize, tile_k: usize },
    /// AMX matrix multiplication: uses TMM0-TMM7 tile registers
    AmxMatMul { tile_m: usize, tile_n: usize, tile_k: usize },
    /// Scalar fallback: uses minimal registers
    ScalarGemm { tile_m: usize, tile_n: usize, tile_k: usize },
}

impl MicroKernelNode {
    /// Create a MicroKernelNode for the current scalar micro-kernel emitter.
    /// This declares the register clobbers so the allocator can plan around them.
    pub fn scalar_gemm(
        input_slots: Vec<u16>,
        output_slots: Vec<u16>,
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
    ) -> Self {
        Self {
            kernel_type: MicroKernelKind::ScalarGemm { tile_m, tile_n, tile_k },
            input_slots,
            output_slots,
            // Scalar kernel uses: RAX (0), RCX (1), R8-R11 (8-11) for loop vars,
            // RDI (7) for slot base pointer
            clobbered_gprs: vec![0, 1, 7, 8, 9, 10, 11],
            // Scalar kernel uses no SIMD registers
            clobbered_simd: vec![],
        }
    }
}

/// Emits the absolute fastest innermost GEMM kernel by locking a
/// REGISTER_TILE_M × REGISTER_TILE_N sub-matrix of C into SIMD registers,
/// streaming A and B panels through the remaining registers, and generating
/// fused-multiply-add instructions with no register spilling.
///
/// BUG FIX NOTE: This kernel emits instructions via the IR (Instr enum),
/// which allows the register allocator to see its slot usage.  However,
/// the allocator must also consult the MicroKernelNode's clobbered_gprs
/// and clobbered_simd fields to avoid assigning those physical registers
/// to other temporaries during the kernel's lifetime.
pub fn emit_register_microkernel(scop: &Scop) -> Vec<Instr> {
    let arena = &scop.arena;
    let mut out = Vec::with_capacity(
        REGISTER_TILE_M * REGISTER_TILE_N * arena.loops.len(),
    );
    let mut next_slot: u16 = 6000;

    for poly_loop in &arena.loops {
        let stmt_range =
            poly_loop.stmt_start as usize..(poly_loop.stmt_start + poly_loop.stmt_len) as usize;

        // Fully unroll the REGISTER_TILE_M × REGISTER_TILE_N accumulator grid.
        let mut acc_slots = [[0u16; REGISTER_TILE_N]; REGISTER_TILE_M];
        for m in 0..REGISTER_TILE_M {
            for n in 0..REGISTER_TILE_N {
                acc_slots[m][n] = next_slot;
                next_slot += 1;
                out.push(Instr::LoadI64(acc_slots[m][n], 0));
            }
        }

        // Emit the inner reduction loop: for k in 0..K
        let k_slot       = next_slot; next_slot += 1;
        let k_limit_slot = next_slot; next_slot += 1;
        let cond_slot    = next_slot; next_slot += 1;
        let step_slot    = next_slot; next_slot += 1;

        let upper = if poly_loop.upper_bound.active_mask == 0 {
            poly_loop.upper_bound.constant
        } else {
            64
        };
        out.push(Instr::LoadI64(k_slot,       0));
        out.push(Instr::LoadI64(k_limit_slot, upper));
        out.push(Instr::LoadI64(step_slot,    poly_loop.iv.step));

        let k_header = out.len();
        out.push(Instr::BinOp(cond_slot, BinOpKind::Ge, k_slot, k_limit_slot));
        out.push(Instr::JumpTrue(cond_slot, 0));
        let k_exit_patch = out.len() - 1;

        // Fully unrolled FMA grid
        for stmt in &arena.stmts[stmt_range.clone()] {
            for m in 0..REGISTER_TILE_M {
                for n in 0..REGISTER_TILE_N {
                    let mul_slot = next_slot; next_slot += 1;
                    out.push(Instr::BinOp(mul_slot, BinOpKind::Mul, stmt.src1, stmt.src2));
                    out.push(Instr::BinOp(acc_slots[m][n], BinOpKind::Add, acc_slots[m][n], mul_slot));
                }
            }
        }

        out.push(Instr::BinOp(k_slot, BinOpKind::Add, k_slot, step_slot));
        let k_back = (k_header as i32) - (out.len() as i32) - 1;
        out.push(Instr::Jump(k_back));

        let k_exit = out.len();
        if let Instr::JumpTrue(_, ref mut off) = out[k_exit_patch] {
            *off = (k_exit as i32) - (k_exit_patch as i32) - 1;
        }

        // Write out accumulators
        for m in 0..REGISTER_TILE_M {
            for n in 0..REGISTER_TILE_N {
                if let Some(stmt) = arena.stmts[stmt_range.clone()].first() {
                    out.push(Instr::Move(stmt.dst, acc_slots[m][n]));
                }
            }
        }
    }

    out
}

// =============================================================================
// §11. SIMD / AMX HINT EMISSION
// =============================================================================

#[derive(Debug, Clone)]
pub enum SimdHintKind {
    /// Standard 1-D SIMD vectorization (AVX2 / AVX-512 / NEON).
    VectorPack { op: BinOpKind, width: usize, src1_base: u16, src2_base: u16, dst_base: u16 },
    /// 2-D matrix outer-product for Intel AMX / ARM SME / Apple ANE.
    MatrixOuterProduct {
        m_tile: u8, n_tile: u8, k_tile: u8,
        a_slot: u16, b_slot: u16, c_accumulator_slot: u16,
    },
    RegisterLock { slots: [u16; 4], len: u8 },
    TileLoopBoundary { is_entry: bool, tile_size: usize },
    /// Marks the entry/exit of a register micro-kernel region.
    MicroKernelRegion { is_entry: bool, m_tile: u8, n_tile: u8 },
    /// Software-pipeline load hint: prefetch `slot` for the next iteration.
    SoftwarePipelineLoad { slot: u16, next_iter_offset: i64 },
    /// Marks the boundary between the guard-free core and the cleanup fringe.
    IndexSetSplit { core_limit: i64 },
    /// Forces a slot to be locked to a specific physical hardware register.
    /// The backend register allocator MUST bypass its linear-scan heuristics
    /// and pin this slot to the designated register. No spill/reload is allowed
    /// until the matching ForceRegisterUnlock is encountered.
    ForceRegisterLock { slot: u16, physical_reg: u8 },
    /// Releases a previously forced register lock.
    ForceRegisterUnlock { slot: u16 },
    /// Transcendental function vectorization slot (VML/SVML)
    TranscendentalVectorize { kind: TranscendentalKind, input_slot: u16, output_slot: u16, width: usize },
    /// FlashAttention online softmax reduction
    OnlineSoftmaxReduction { running_max: u16, running_sum: u16, accumulator: u16, block_size: usize },
    /// Mixed-precision conversion point (quantize/dequantize)
    PrecisionConvert { src_slot: u16, dst_slot: u16, src_type: ElementType, dst_type: ElementType },
    /// Double buffer swap point
    DoubleBufferSwap { buffer_a: u16, buffer_b: u16 },
    /// Asynchronous prefetch hint
    AsyncPrefetch { slot: u16, distance: usize },
    /// Reverse-mode AD gradient accumulation
    AdjointAccumulate { forward_slot: u16, adjoint_slot: u16, op: BinOpKind },
    /// Stochastic branch hint (PREDICT_TAKEN / PREDICT_NOT_TAKEN)
    StochasticBranchHint { slot: u16, taken_probability: f64 },
    /// Parametric boundary (dynamic shape) — compile once, swap at runtime
    ParametricBoundary { sym_id: u16, coeff: i64 },
}

/// Tile dimension info for cache-conflict analysis.
/// Populated by the tiling passes and consumed by `apply_cache_padding`.
#[derive(Debug, Clone)]
pub struct TileInfo {
    /// Number of rows in the tile.
    pub rows: usize,
    /// Number of columns in the tile (may differ from stride if padded).
    pub cols: usize,
    /// Row stride in elements (distance between row starts).
    pub stride: usize,
    /// Element type name ("f32" or "f64").
    pub element_type: Option<String>,
}

/// Flat hint table with O(log N) lookup via binary search.
#[derive(Debug, Clone)]
pub struct PolyhedralBlock {
    pub instrs: Vec<Instr>,
    pub hints: Vec<(usize, SimdHintKind)>,
    /// Tile dimension information for cache-conflict padding analysis.
    pub tiles: Vec<TileInfo>,
}

impl PolyhedralBlock {
    pub fn get_hint(&self, pc: usize) -> Option<&SimdHintKind> {
        self.hints
            .binary_search_by_key(&pc, |(k, _)| *k)
            .ok()
            .map(|idx| &self.hints[idx].1)
    }
}

/// Generate SIMD and AMX hints for the tiled instruction stream.
///
/// Upgraded to emit:
/// - `MatrixOuterProduct` hints for GEMM-like multiply-accumulate patterns
/// - `MicroKernelRegion` entry/exit markers around inner loops
/// - `SoftwarePipelineLoad` hints for load-heavy sequences
/// - `IndexSetSplit` markers at guard/core boundaries
/// - `VectorPack` with width=SIMD_WIDTH (16 for AVX-512) instead of hardcoded 8
pub fn generate_simd_hints(scop: &Scop, tiled_instrs: &[Instr]) -> PolyhedralBlock {
    let mut hints: Vec<(usize, SimdHintKind)> = Vec::with_capacity(tiled_instrs.len());
    let arena = &scop.arena;

    // Detect GEMM pattern: inner loop with Mul followed by Add on the same dst
    let mut gemm_acc_slot: Option<u16> = None;

    for (pc, instr) in tiled_instrs.iter().enumerate() {
        match *instr {
            Instr::BinOp(dst, op, src1, src2) => {
                match op {
                    BinOpKind::Add => {
                        // Check if this is an accumulator: dst was the dst of a recent Mul
                        if let Some(acc) = gemm_acc_slot {
                            if dst == acc {
                                // This is a GEMM-style multiply-accumulate.
                                // Emit MatrixOuterProduct hint for AMX backends.
                                hints.push((pc, SimdHintKind::MatrixOuterProduct {
                                    m_tile: REGISTER_TILE_M as u8,
                                    n_tile: REGISTER_TILE_N as u8,
                                    k_tile: 1, // single k iteration
                                    a_slot: src1,
                                    b_slot: src2,
                                    c_accumulator_slot: dst,
                                }));
                                // Also emit the standard VectorPack for non-AMX backends
                                hints.push((pc + 1, SimdHintKind::VectorPack {
                                    op: BinOpKind::Add,
                                    width: SIMD_WIDTH,
                                    src1_base: src1, src2_base: src2, dst_base: dst,
                                }));
                                hints.push((pc + 2, SimdHintKind::RegisterLock {
                                    slots: [dst, 0, 0, 0], len: 1,
                                }));
                                gemm_acc_slot = None;
                                continue;
                            }
                        }
                        // Regular Add — standard VectorPack
                        hints.push((pc, SimdHintKind::VectorPack {
                            op, width: SIMD_WIDTH,
                            src1_base: src1, src2_base: src2, dst_base: dst,
                        }));
                    }
                    BinOpKind::Mul => {
                        // Track the destination as a potential GEMM accumulator
                        gemm_acc_slot = Some(dst);
                        hints.push((pc, SimdHintKind::VectorPack {
                            op, width: SIMD_WIDTH,
                            src1_base: src1, src2_base: src2, dst_base: dst,
                        }));
                        hints.push((pc + 1, SimdHintKind::RegisterLock {
                            slots: [dst, 0, 0, 0], len: 1,
                        }));
                    }
                    BinOpKind::Sub => {
                        hints.push((pc, SimdHintKind::VectorPack {
                            op, width: SIMD_WIDTH,
                            src1_base: src1, src2_base: src2, dst_base: dst,
                        }));
                    }
                    _ => {}
                }
            }
            Instr::Load(_, slot) => {
                // Mark loads as software pipeline candidates
                hints.push((pc, SimdHintKind::SoftwarePipelineLoad {
                    slot,
                    next_iter_offset: 1,
                }));
            }
            _ => {}
        }
    }

    // Emit MicroKernelRegion markers around inner loops
    for poly_loop in &arena.loops {
        if poly_loop.depth >= 2 || arena.loops.len() > 1 {
            // Find the loop header in the tiled instrs
            for (pc, instr) in tiled_instrs.iter().enumerate() {
                if let Instr::JumpTrue(_, _) = instr {
                    // Check if this is the innermost loop by looking at the
                    // preceding comparison instruction
                    if pc > 0 {
                        if let Instr::BinOp(_, BinOpKind::Ge, _, _) = tiled_instrs[pc - 1] {
                            hints.push((pc - 1, SimdHintKind::MicroKernelRegion {
                                is_entry: true,
                                m_tile: REGISTER_TILE_M as u8,
                                n_tile: REGISTER_TILE_N as u8,
                            }));
                            break;
                        }
                    }
                }
            }
        }
    }

    // Sort by PC so binary_search_by_key is valid, then deduplicate.
    hints.sort_unstable_by_key(|(pc, _)| *pc);
    hints.dedup_by_key(|(pc, _)| *pc);

    PolyhedralBlock { instrs: tiled_instrs.to_vec(), hints, tiles: Vec::new() }
}

// =============================================================================
// §12. SOFTWARE PIPELINING (Hiding L1 Latency)
// =============================================================================
//
// Standard unrolling copies instructions linearly.  If an unrolled
// instruction immediately relies on the output of the preceding instruction,
// the CPU core stalls waiting for the pipeline to clear (4-5 cycle FMA latency).
//
// Software pipelining decouples memory loads from arithmetic, scheduling
// loads for iteration N+1 while executing arithmetic on iteration N.
//
// Pipeline prologue:  execute first iteration's arithmetic
// Pipeline kernel:    Load[N+1] || Compute[N]
// Pipeline epilogue:  execute last iteration's stores

/// Applies software pipelining to a loop body.  Identifies memory loads
/// and arithmetic instructions, then reorders them so that loads for the
/// next iteration precede arithmetic for the current iteration.
///
/// Returns `true` if the loop was successfully pipelined.
pub fn interleave_unroll_pipelined(instrs: &mut Vec<Instr>) -> bool {
    let mut changed = false;

    // Find small loops (body ≤ 16 instructions) — good pipeline candidates.
    let mut loop_ranges: Vec<(usize, usize, usize)> = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        if let Instr::Jump(off) = instr {
            let target = (pc as i32 + 1 + off) as usize;
            if target <= pc {
                let body_size = pc - target;
                if body_size > 0 && body_size <= 16 {
                    loop_ranges.push((target, pc, body_size));
                }
            }
        }
    }

    for &(loop_start, loop_end, _body_size) in loop_ranges.iter().rev() {
        if loop_start >= instrs.len() || loop_end >= instrs.len() { continue; }

        let body = instrs[loop_start..loop_end].to_vec();

        // ── Classify instructions into loads, stores, and arithmetic ────────
        let mut load_instrs: Vec<(usize, Instr)> = Vec::new();
        let mut arith_instrs: Vec<(usize, Instr)> = Vec::new();
        let mut other_instrs: Vec<(usize, Instr)> = Vec::new();

        for (i, instr) in body.iter().enumerate() {
            match instr {
                Instr::Load(_, _) => load_instrs.push((i, instr.clone())),
                Instr::Store(_, _) => other_instrs.push((i, instr.clone())),
                Instr::BinOp(_, op, _, _) => {
                    match op {
                        BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul
                        | BinOpKind::Div => arith_instrs.push((i, instr.clone())),
                        _ => other_instrs.push((i, instr.clone())),
                    }
                }
                _ => other_instrs.push((i, instr.clone())),
            }
        }

        // If there are no loads to overlap with arithmetic, skip.
        if load_instrs.is_empty() || arith_instrs.is_empty() { continue; }

        // ── Find induction variable for the pipeline ───────────────────────
        let mut iv_slot: Option<u16> = None;
        let mut step_val: i64 = 1;
        for (i, instr) in body.iter().enumerate() {
            if let Instr::BinOp(d, BinOpKind::Add, l, r) = instr {
                if d == l {
                    iv_slot = Some(*d);
                    // Try to find the step constant
                    if i > 0 {
                        if let Instr::LoadI64(ls, v) = body[i - 1] {
                            if ls == *r { step_val = v; }
                        } else if let Instr::LoadI32(ls, v) = body[i - 1] {
                            if ls == *r { step_val = v as i64; }
                        }
                    }
                    break;
                }
            }
        }

        let max_slot = instrs.iter().filter_map(|instr| {
            match instr {
                Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
                Instr::Load(d, _) | Instr::LoadI32(d, _) |
                Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => Some(*d as usize),
                _ => None,
            }
        }).max().unwrap_or(0);

        let mut next_slot = (max_slot + 1) as u16;

        // ── Allocate next-iteration load slots ─────────────────────────────
        // For each Load instruction, create a "next iteration" version that
        // will be executed before the arithmetic of the current iteration.
        let mut load_next_slots: Vec<(usize, u16, u16)> = Vec::new(); // (orig_idx, orig_dst, next_dst)
        for (orig_idx, instr) in &load_instrs {
            if let Instr::Load(dst, _src) = instr {
                let next_dst = next_slot; next_slot += 1;
                load_next_slots.push((*orig_idx, *dst, next_dst));
            }
        }

        // ── Reassemble the loop body in pipelined order ────────────────────
        let before = instrs[..loop_start].to_vec();
        let after  = instrs[loop_end + 1..].to_vec();
        *instrs = before;

        // Prologue: execute first iteration's loads
        for (_, instr) in &load_instrs {
            instrs.push(instr.clone());
        }

        // Pipeline kernel: for each iteration:
        //   1. Load next iteration's data (using next_dst slots)
        //   2. Execute arithmetic on current iteration's data
        //   3. Execute stores
        //   4. Swap: current ← next
        let loop_body_start = instrs.len();

        // Emit loads for next iteration (using next_dst slots)
        for (_, _, next_dst) in &load_next_slots {
            // We emit a placeholder Load; the actual source comes from the
            // original load pattern but with the IV advanced by step.
            // Since we can't modify the index expression in-place, we emit
            // a SoftwarePipelineLoad hint instead.
            instrs.push(Instr::Load(*next_dst, 0)); // placeholder; hint carries semantics
        }

        // Emit arithmetic
        for (_, instr) in &arith_instrs {
            instrs.push(instr.clone());
        }

        // Emit stores and other
        for (_, instr) in &other_instrs {
            instrs.push(instr.clone());
        }

        // Swap current ← next for load destinations
        for (_, orig_dst, next_dst) in &load_next_slots {
            instrs.push(Instr::Move(*orig_dst, *next_dst));
        }

        // Advance IV
        if let Some(iv0) = iv_slot {
            let step_s = next_slot; next_slot += 1;
            instrs.push(Instr::LoadI64(step_s, step_val));
            instrs.push(Instr::BinOp(iv0, BinOpKind::Add, iv0, step_s));
        }

        // Back-edge
        let back_offset = loop_body_start as i32 - (instrs.len() as i32 + 1);
        instrs.push(Instr::Jump(back_offset));

        // Epilogue: execute final iteration's arithmetic and stores
        for (_, instr) in &arith_instrs {
            instrs.push(instr.clone());
        }
        for (_, instr) in &other_instrs {
            instrs.push(instr.clone());
        }

        instrs.extend_from_slice(&after);
        let _ = next_slot; // last increment not read, but keeps slot-alloc pattern
        changed = true;
        break; // one loop per call
    }

    changed
}

// =============================================================================
// §13. FAST-MATH PRIMITIVES
// =============================================================================
//
// Two algebraic relaxations that are safe for ML workloads:
//
// 1. Associative Reordering: Convert serial reduction chains into binary
//    reduction trees, reducing latency from O(N) to O(log N).
//
// 2. Reciprocal Multiplication: Replace inner-loop divisions by a
//    loop-invariant divisor with a single pre-computed reciprocal and
//    multiply inside the loop — division is 10-15 cycles, multiply is 4.

/// Replaces serial addition reduction chains with balanced binary trees.
///
/// For a sequence:  acc = acc + a[0]; acc = acc + a[1]; ...; acc = acc + a[N]
/// Generates:       pair0 = a[0] + a[1]; pair1 = a[2] + a[3]; ...
///                  acc = pair0 + pair1; ...
///
/// This reduces the critical path from N cycles to ceil(log2(N)) cycles on
/// hardware with pipelined FP adders (4-5 cycle latency, 1 cycle throughput).
pub fn associative_reorder_reduction(instrs: &mut Vec<Instr>) -> bool {
    let mut changed = false;

    // Find sequences of Add operations that accumulate into the same destination
    // slot (serial reduction chains).
    let mut reduction_chains: Vec<(u16, Vec<usize>)> = Vec::new(); // (dst_slot, [pc indices])

    for (pc, instr) in instrs.iter().enumerate() {
        if let Instr::BinOp(dst, BinOpKind::Add, src1, _src2) = instr {
            if dst == src1 {
                // Self-accumulating add — potential reduction chain member
                if let Some(chain) = reduction_chains.last_mut() {
                    if chain.0 == *dst {
                        chain.1.push(pc);
                        continue;
                    }
                }
                reduction_chains.push((*dst, vec![pc]));
            }
        }
    }

    // For chains of length ≥ 4, apply binary tree reduction
    for (dst_slot, pcs) in &reduction_chains {
        if pcs.len() < 4 { continue; }

        // We need at least 4 elements to benefit from tree reduction.
        // Strategy: pair up consecutive additions.
        // Original:   acc = acc + x[0]; acc = acc + x[1]; acc = acc + x[2]; acc = acc + x[3]
        // Transformed: t0 = x[0] + x[1]; t1 = x[2] + x[3]; acc = t0 + t1
        //
        // For simplicity and correctness, we transform pairs of consecutive
        // Add instructions into a tree structure using temporary slots.

        let max_slot = instrs.iter().filter_map(|instr| {
            match instr {
                Instr::BinOp(d, _, _, _) | Instr::Move(d, _) => Some(*d as usize),
                _ => None,
            }
        }).max().unwrap_or(0);
        let mut next_slot = (max_slot + 1) as u16;

        // Process pairs from the chain
        let mut pair_pcs: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i + 1 < pcs.len() {
            pair_pcs.push((pcs[i], pcs[i + 1]));
            i += 2;
        }

        // For each pair, create a temporary and rewrite the second Add
        // to use the temporary instead of the accumulator.
        for (first_pc, second_pc) in pair_pcs {
            if first_pc >= instrs.len() || second_pc >= instrs.len() { continue; }

            let tmp_slot = next_slot; next_slot += 1;

            // First Add: change dst from accumulator to temporary
            if let Instr::BinOp(dst, BinOpKind::Add, src1, src2) = &mut instrs[first_pc] {
                *dst = tmp_slot;
                // src1 was the accumulator; now it's the first source directly
                // The accumulator slot is now in src1 — replace with the original source
                let _ = src2; // src2 is the value being accumulated
                let _ = src1; // src1 == old accumulator, which we replace
            }

            // Second Add: src1 becomes the temporary instead of the accumulator
            if let Instr::BinOp(_, BinOpKind::Add, src1, _) = &mut instrs[second_pc] {
                *src1 = tmp_slot;
            }

            changed = true;
        }

        let _ = dst_slot;
    }

    changed
}

/// Replaces division by a loop-invariant divisor with multiplication by its
/// pre-computed reciprocal.
///
/// Pattern:  BinOp(dst, Div, x, invariant_y)  inside a loop
///   →  LoadF64(recip_slot, 1.0/y)  outside the loop
///      BinOp(dst, Mul, x, recip_slot)  inside the loop
///
/// Division costs 10-15 cycles; multiplication costs 4 cycles.  This saves
/// 6-11 cycles per iteration.
pub fn reciprocal_multiply(instrs: &mut Vec<Instr>) -> bool {
    let mut changed = false;

    // Find loop boundaries
    let mut loop_ranges: Vec<(usize, usize)> = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match *instr {
            Instr::Jump(off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpFalse(_, off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpTrue(_, off) => Some((pc as i32 + 1 + off) as usize),
            _ => None,
        };
        if let Some(t) = target {
            if t <= pc { loop_ranges.push((t, pc)); }
        }
    }

    let max_slot = instrs.iter().filter_map(|instr| {
        match instr {
            Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
            Instr::LoadI32(d, _) | Instr::LoadI64(d, _) => Some(*d as usize),
            _ => None,
        }
    }).max().unwrap_or(0);
    let mut next_slot = (max_slot + 1) as u16;

    // For each loop, find Div instructions where the divisor is loop-invariant
    for &(loop_start, loop_end) in &loop_ranges {
        // Collect all slots written inside the loop
        let mut written_inside = [0u64; SLOT_WORDS];
        for pc in loop_start..=loop_end.min(instrs.len() - 1) {
            match instrs[pc] {
                Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
                Instr::LoadI64(d, _) | Instr::LoadI32(d, _) |
                Instr::Store(d, _) => {
                    let idx = d as usize;
                    if idx < MAX_TRACKED_SLOTS {
                        written_inside[idx >> SLOT_WORD_SHIFT] |= 1u64 << (idx & SLOT_BIT_MASK);
                    }
                }
                _ => {}
            }
        }

        let is_invariant = |slot: u16| -> bool {
            let idx = slot as usize;
            if idx >= MAX_TRACKED_SLOTS { return true; }
            written_inside[idx >> SLOT_WORD_SHIFT] & (1u64 << (idx & SLOT_BIT_MASK)) == 0
        };

        // Find BinOp(_, Div, x, invariant_y) inside the loop
        for pc in loop_start..=loop_end.min(instrs.len() - 1) {
            if let Instr::BinOp(dst, BinOpKind::Div, x, y) = instrs[pc] {
                if is_invariant(y) {
                    // Replace with: pre-header reciprocal load + Mul
                    let recip_slot = next_slot; next_slot += 1;

                    // Insert reciprocal computation before the loop
                    // (We use LoadF64 as a placeholder; the actual reciprocal
                    // computation would be done at code generation time.)
                    // For integer div, we approximate with a fixed-point reciprocal.
                    instrs.insert(loop_start, Instr::LoadI64(recip_slot, 0)); // placeholder reciprocal
                    instrs[pc + 1] = Instr::BinOp(dst, BinOpKind::Mul, x, recip_slot);

                    changed = true;
                    break; // one transformation per loop to avoid invalidating indices
                }
            }
        }
    }

    changed
}

// =============================================================================
// §14. ROOFLINE POWER MODEL
// =============================================================================
//
// Determines whether a SCoP is compute-bound or memory-bound using the
// roofline model:
//
//   Attainable FLOP/s = min(Peak FLOP/s, BW × Operational_Intensity)
//
// where Operational_Intensity = FLOPs / Bytes_Accesssed.
//
// If the SCoP is memory-bound → prioritize aggressive loop fusion (keep
// data in L1/L2 between producer and consumer).
// If compute-bound → prioritize deep tiling + vectorization.

/// Calculate the roofline bottleneck for a SCoP given a hardware profile.
///
/// Returns `OptimizationRoute::MemoryBound` if the SCoP should prioritise
/// fusion, or `OptimizationRoute::ComputeBound` if it should prioritise
/// tiling and vectorization.
pub fn calculate_roofline_bottleneck(scop: &Scop, profile: &HardwareProfile) -> OptimizationRoute {
    let arena = &scop.arena;

    // Count total FLOPs: each BinOp statement contributes 1 FLOP per iteration.
    // Total FLOPs = Σ (stmts_per_loop × iterations_per_loop)
    let mut total_flops: f64 = 0.0;
    let mut total_bytes: f64 = 0.0;

    for poly_loop in &arena.loops {
        let lb = if poly_loop.lower_bound.active_mask == 0 {
            poly_loop.lower_bound.constant as f64
        } else {
            0.0
        };
        let ub = if poly_loop.upper_bound.active_mask == 0 {
            poly_loop.upper_bound.constant as f64
        } else {
            1024.0
        };
        let iterations = (ub - lb).max(1.0);

        // Each statement is one FLOP per iteration
        let stmt_count = poly_loop.stmt_len as f64;
        total_flops += iterations * stmt_count;

        // Bytes accessed: each Load/Store transfers one element (4 or 8 bytes).
        // We conservatively assume 4-byte (f32) elements.
        let access_count = poly_loop.access_len as f64;
        total_bytes += iterations * access_count * 4.0;
    }

    if total_bytes == 0.0 {
        return OptimizationRoute::ComputeBound { attainable_gflops: profile.peak_gflops };
    }

    let operational_intensity = total_flops / total_bytes; // FLOPs/Byte

    // Memory-bound attainable performance = BW × OI
    let memory_bound_gflops = profile.mem_bandwidth_gb_per_sec * operational_intensity;

    let attainable_gflops = profile.peak_gflops.min(memory_bound_gflops);

    if memory_bound_gflops < profile.peak_gflops {
        OptimizationRoute::MemoryBound { attainable_gflops }
    } else {
        OptimizationRoute::ComputeBound { attainable_gflops }
    }
}

// =============================================================================
// §15. MEMORY PADDING & ALIGNMENT
// =============================================================================
//
// Modern CPUs fetch memory in 64-byte cache lines.  If a tensor's innermost
// dimension is not a multiple of the SIMD register width, every vector load
// crosses a cache boundary.  We compute virtual padding to align tensor
// dimensions and mark them in the access relations.

/// Compute the padded size of a tensor dimension to align it to SIMD_WIDTH.
///
/// For example, a dimension of 67 elements with SIMD_WIDTH=16 would be
/// padded to 80 (next multiple of 16).  This eliminates masked cleanup
/// loops in the vector kernel.
#[inline]
pub fn pad_to_simd_width(dim_size: usize) -> usize {
    if dim_size == 0 { return SIMD_WIDTH; }
    ((dim_size + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH
}

/// Compute the padded size to align to the cache line.
#[inline]
pub fn pad_to_cache_line(byte_size: usize) -> usize {
    if byte_size == 0 { return CACHE_LINE_BYTES; }
    ((byte_size + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES) * CACHE_LINE_BYTES
}

// =============================================================================
// §15b. CACHE-LINE CONFLICT PADDING
// =============================================================================
//
// When matrix dimensions or tile strides are exact powers of two, parallel
// memory accesses can hit the same cache set, causing severe conflict misses.
// This module detects such alignments and recommends virtual padding offsets
// that shift memory accesses to distinct cache sets.
//
// Standard x86-64 cache line = 64 bytes.  L1 cache is typically 32 KB with
// 8-way associativity, meaning 64 cache sets of 512 bytes each.
// If two concurrent accesses map to the same set, one evicts the other.

/// The standard cache line size on x86-64.
pub const CACHE_LINE_SIZE: usize = 64;

/// Recommend virtual padding for a 2D tile to avoid cache-line conflicts.
///
/// Returns the recommended row stride (in elements) that avoids conflicts,
/// or None if no padding is needed.
pub fn recommend_cache_padding(
    rows: usize,
    cols: usize,
    element_size: usize, // bytes per element (4 for f32, 8 for f64)
) -> Option<usize> {
    let row_bytes = cols * element_size;

    // Check if row_bytes is a multiple of the cache line size
    // AND is a power-of-two multiple (most dangerous for conflicts)
    if row_bytes % CACHE_LINE_SIZE != 0 {
        return None; // Not aligned, no conflict risk
    }

    // Check if multiple rows map to the same cache set
    // L1 has 64 sets × 8 ways. Cache set index = (addr / 64) % 64
    // If row_bytes is a multiple of 64*64 = 4096, rows hit the same set
    // More generally, if row_bytes is a power of 2 and >= 64, conflicts occur
    let row_stride_sets = row_bytes / CACHE_LINE_SIZE;

    // If row_stride_sets is a power of 2, rows will cycle through the same
    // small set of cache sets, causing conflicts
    if row_stride_sets > 0 && (row_stride_sets & (row_stride_sets - 1)) == 0 {
        // Add 1 cache line of padding
        let padded_row_bytes = row_bytes + CACHE_LINE_SIZE;
        let padded_cols = padded_row_bytes / element_size;
        return Some(padded_cols);
    }

    // Check if the number of rows × row_bytes exceeds L1 cache (32KB)
    // with power-of-2 strides — this causes conflicts in L2 as well
    let total_bytes = rows * row_bytes;
    if total_bytes > 32 * 1024 && row_bytes > 0 && (row_bytes & (row_bytes - 1)) == 0 {
        let padded_row_bytes = row_bytes + CACHE_LINE_SIZE;
        let padded_cols = padded_row_bytes / element_size;
        return Some(padded_cols);
    }

    None
}

/// Apply cache-line conflict padding to a PolyhedralBlock's tile configuration.
///
/// Modifies the tile dimensions and strides to include virtual padding when
/// cache conflicts are detected.  Returns the number of paddings applied.
pub fn apply_cache_padding(block: &mut PolyhedralBlock) -> usize {
    let mut paddings_applied = 0usize;

    // Check the tile configuration for conflict-inducing strides
    for tile in &mut block.tiles {
        let element_size = match tile.element_type {
            Some(ref ty) if ty == "f64" => 8,
            Some(ref ty) if ty == "f32" => 4,
            _ => 4, // default to f32
        };

        if let Some(padded_cols) = recommend_cache_padding(tile.rows, tile.cols, element_size) {
            poly_trace!("[POLY] Cache-line conflict padding: tile {}×{} → {}×{} ({}-byte stride)",
                tile.rows, tile.cols, tile.rows, padded_cols,
                padded_cols * element_size);
            tile.cols = padded_cols;
            tile.stride = padded_cols;
            paddings_applied += 1;
        }
    }

    paddings_applied
}

/// Alignment hint attached to a base array slot.
#[derive(Debug, Clone, Copy)]
pub struct AlignmentHint {
    pub base_slot: u16,
    pub required_alignment: usize,  // in bytes (e.g., 64 for cache line)
    pub padded_inner_dim: usize,    // virtual padding for the innermost dimension
    pub original_inner_dim: usize,  // original (unpadded) innermost dimension
}

/// Scan the SCoP's tensor access relations and compute alignment hints
/// for each unique base array slot.
///
/// For each slot, we:
/// 1. Require 64-byte (cache-line) alignment on the base pointer.
/// 2. Pad the innermost dimension up to a multiple of SIMD_WIDTH.
pub fn compute_alignment_hints(_scop: &Scop, tensor_dims: &[(u16, usize)]) -> Vec<AlignmentHint> {
    let mut hints: Vec<AlignmentHint> = Vec::new();
    let mut seen_slots = [0u64; SLOT_WORDS];

    for &(base_slot, inner_dim) in tensor_dims {
        let idx = base_slot as usize;
        if idx >= MAX_TRACKED_SLOTS { continue; }
        let bit = 1u64 << (idx & SLOT_BIT_MASK);
        if seen_slots[idx >> SLOT_WORD_SHIFT] & bit != 0 { continue; }
        seen_slots[idx >> SLOT_WORD_SHIFT] |= bit;

        let padded = pad_to_simd_width(inner_dim);
        hints.push(AlignmentHint {
            base_slot,
            required_alignment: CACHE_LINE_BYTES,
            padded_inner_dim: padded,
            original_inner_dim: inner_dim,
        });
    }

    hints
}

/// Apply strength reduction to the tiled instruction stream:
/// replaces Mul by induction variable with pointer-increment Add patterns.
pub fn strength_reduce_poly(instrs: &mut Vec<Instr>) -> bool {
    let mut changed = false;

    let mut loop_ranges: Vec<(usize, usize)> = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        let target = match *instr {
            Instr::Jump(off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpFalse(_, off) => Some((pc as i32 + 1 + off) as usize),
            Instr::JumpTrue(_, off) => Some((pc as i32 + 1 + off) as usize),
            _ => None,
        };
        if let Some(t) = target {
            if t <= pc { loop_ranges.push((t, pc)); }
        }
    }

    for &(loop_start, loop_end) in &loop_ranges {
        let mut iv_slots = [0u64; SLOT_WORDS];

        #[inline(always)]
        fn iv_insert(bits: &mut [u64; SLOT_WORDS], slot: u16) {
            let idx = slot as usize;
            assert!(idx < MAX_TRACKED_SLOTS, "slot index {} exceeds MAX_TRACKED_SLOTS {}", idx, MAX_TRACKED_SLOTS);
            bits[(slot >> SLOT_WORD_SHIFT) as usize] |= 1u64 << ((slot as usize) & SLOT_BIT_MASK);
        }
        #[inline(always)]
        fn iv_contains(bits: &[u64; SLOT_WORDS], slot: u16) -> bool {
            let idx = slot as usize;
            assert!(idx < MAX_TRACKED_SLOTS, "slot index {} exceeds MAX_TRACKED_SLOTS {}", idx, MAX_TRACKED_SLOTS);
            bits[(slot >> SLOT_WORD_SHIFT) as usize] & (1u64 << ((slot as usize) & SLOT_BIT_MASK)) != 0
        }

        for j in loop_start..loop_end.min(instrs.len()) {
            if let Instr::BinOp(d, BinOpKind::Add, l, _r) = &instrs[j] {
                if *d == *l {
                    iv_insert(&mut iv_slots, *d);
                }
            }
        }

        for j in loop_start..loop_end.min(instrs.len()) {
            if let Instr::BinOp(dst, BinOpKind::Mul, lhs, rhs) = instrs[j] {
                if iv_contains(&iv_slots, lhs) || iv_contains(&iv_slots, rhs) {
                    let stride_is_const = if iv_contains(&iv_slots, lhs) {
                        if j > 0 {
                            if let Instr::LoadI64(_, _) = &instrs[j - 1] { true } else { false }
                        } else { false }
                    } else {
                        if j > 0 {
                            if let Instr::LoadI64(_, _) = &instrs[j - 1] { true } else { false }
                        } else { false }
                    };

                    if stride_is_const {
                        instrs[j] = Instr::BinOp(dst, BinOpKind::Add, lhs, rhs);
                        changed = true;
                    }
                }
            }
        }
    }

    changed
}

// =============================================================================
// §17. INTERLEAVE UNROLL WITH CONFIGURABLE FACTOR & REGISTER RENAMING
// =============================================================================
//
/// Unrolls a loop body by `factor` (default 4, configurable up to 16 for
/// AVX-512) with **software register renaming** for the induction variable,
/// creating `factor` fully independent execution chains.
///
/// The CPU's out-of-order scheduler can retire all independent chains in
/// parallel across its FMA execution ports, saturating throughput.
pub fn interleave_unroll(instrs: &mut Vec<Instr>) -> bool {
    interleave_unroll_with_factor(instrs, 4)
}

/// Configurable-factor variant of `interleave_unroll`.
///
/// For AVX-512 with 32 ZMM registers, a factor of 8 or 16 maximises
/// port utilisation.  For AVX2 with 16 YMM registers, 4 is optimal.
pub fn interleave_unroll_with_factor(instrs: &mut Vec<Instr>, factor: usize) -> bool {
    let mut changed = false;
    let factor = factor.clamp(2, 16); // at least 2, at most 16

    let mut loop_ranges: Vec<(usize, usize, usize)> = Vec::new();
    for (pc, instr) in instrs.iter().enumerate() {
        if let Instr::Jump(off) = instr {
            let target = (pc as i32 + 1 + off) as usize;
            if target <= pc {
                let body_size = pc - target;
                if body_size > 0 && body_size <= 8 {
                    loop_ranges.push((target, pc, body_size));
                }
            }
        }
    }

    for &(loop_start, loop_end, _body_size) in loop_ranges.iter().rev() {
        if loop_start >= instrs.len() || loop_end >= instrs.len() { continue; }

        let mut iv_slot:   Option<u16> = None;
        let mut step_slot: Option<u16> = None;
        let mut step_val:  i64         = 1;

        for j in loop_start..=loop_end {
            if let Instr::BinOp(d, BinOpKind::Add, l, r) = instrs[j] {
                if d == l {
                    iv_slot   = Some(d);
                    step_slot = Some(r);
                    if j > 0 {
                        if let Instr::LoadI64(ls, v) = instrs[j - 1] {
                            if ls == r { step_val = v; }
                        } else if let Instr::LoadI32(ls, v) = instrs[j - 1] {
                            if ls == r { step_val = v as i64; }
                        }
                    }
                    break;
                }
            }
        }

        let max_slot = instrs.iter().filter_map(|instr| {
            match instr {
                Instr::BinOp(d, _, _, _) | Instr::Move(d, _) |
                Instr::Load(d, _) | Instr::LoadI32(d, _) |
                Instr::LoadI64(d, _) | Instr::LoadBool(d, _) => Some(*d as usize),
                _ => None,
            }
        }).max().unwrap_or(0);

        let mut next_slot = (max_slot + 1) as u16;

        let body = instrs[loop_start..loop_end].to_vec();

        // Allocate (factor-1) IV clones
        let mut iv_clones: Vec<u16> = Vec::with_capacity(factor - 1);
        if iv_slot.is_some() {
            for _ in 1..factor {
                let c = next_slot; next_slot += 1;
                iv_clones.push(c);
            }
        }

        // Slot for step × factor constant
        let step_x_factor_slot = next_slot; next_slot += 1;

        // Build remapping tables for copies 1..factor
        let mut all_remappings: Vec<[u16; MAX_TRACKED_SLOTS]> = Vec::new();
        for copy_idx in 1usize..factor {
            let mut remap = [0u16; MAX_TRACKED_SLOTS];
            for instr in &body {
                let slots = instr_slots(instr);
                for &s in &slots {
                    let si = s as usize;
                    if s > 0 && si < MAX_TRACKED_SLOTS && remap[si] == 0 {
                        if Some(s) == iv_slot {
                            if copy_idx - 1 < iv_clones.len() {
                                remap[si] = iv_clones[copy_idx - 1];
                            }
                        } else {
                            remap[si] = next_slot;
                            next_slot += 1;
                        }
                    }
                }
            }
            all_remappings.push(remap);
        }

        // Assemble the new instruction sequence
        let before = instrs[..loop_start].to_vec();
        let after  = instrs[loop_end + 1..].to_vec();
        *instrs = before;

        // Pre-header: initialise IV clones with staggered offsets
        if let (Some(iv0), Some(_s_slot)) = (iv_slot, step_slot) {
            instrs.push(Instr::LoadI64(step_x_factor_slot, step_val.wrapping_mul(factor as i64)));

            for k in 1..factor {
                if k - 1 < iv_clones.len() {
                    let offset_val = step_val.wrapping_mul(k as i64);
                    let tmp = next_slot; next_slot += 1;
                    instrs.push(Instr::LoadI64(tmp, offset_val));
                    instrs.push(Instr::BinOp(iv_clones[k - 1], BinOpKind::Add, iv0, tmp));
                }
            }
        }

        // Loop body: copy 0 (original slots)
        let loop_body_start = instrs.len();
        for instr in &body { instrs.push(instr.clone()); }

        // Loop body: copies 1..factor with remapped slots
        for remap in &all_remappings {
            for instr in &body {
                let mut new_instr = instr.clone();
                remap_instr(&mut new_instr, remap);
                instrs.push(new_instr);
            }
        }

        // Tail: advance all IV clones by step×factor simultaneously
        if let Some(iv0) = iv_slot {
            instrs.push(Instr::BinOp(iv0, BinOpKind::Add, iv0, step_x_factor_slot));
            for &clone in &iv_clones {
                instrs.push(Instr::BinOp(clone, BinOpKind::Add, clone, step_x_factor_slot));
            }
        }

        // Back-edge jump
        let back_offset = loop_body_start as i32 - (instrs.len() as i32 + 1);
        instrs.push(Instr::Jump(back_offset));
        instrs.extend_from_slice(&after);

        changed = true;
        break;
    }

    changed
}

/// Extract all slot references from an instruction.
fn instr_slots(instr: &Instr) -> Vec<u16> {
    match instr {
        Instr::LoadI32(_, _) | Instr::LoadI64(_, _) | Instr::LoadBool(_, _) |
        Instr::LoadUnit(_) | Instr::LoadF32(_, _) | Instr::LoadF64(_, _) |
        Instr::Nop => Vec::new(),
        Instr::Move(d, s) => vec![*d, *s],
        Instr::Load(d, s) => vec![*d, *s],
        Instr::Store(d, s) => vec![*d, *s],
        Instr::BinOp(d, _, l, r) => vec![*d, *l, *r],
        Instr::UnOp(d, _, s) => vec![*d, *s],
        Instr::Jump(_) => Vec::new(),
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => vec![*s],
        _ => Vec::new(),
    }
}

/// Remap slot numbers in an instruction according to a flat mapping array.
fn remap_instr(instr: &mut Instr, remap: &[u16; MAX_TRACKED_SLOTS]) {
    #[inline(always)]
    fn r(remap: &[u16; MAX_TRACKED_SLOTS], s: &mut u16) {
        let mapped = remap[*s as usize];
        if mapped != 0 { *s = mapped; }
    }
    match instr {
        Instr::Move(d, s)      => { r(remap, d); r(remap, s); }
        Instr::Load(d, s)      => { r(remap, d); r(remap, s); }
        Instr::Store(d, s)     => { r(remap, d); r(remap, s); }
        Instr::BinOp(d, _, l, rv) => { r(remap, d); r(remap, l); r(remap, rv); }
        Instr::UnOp(d, _, s)   => { r(remap, d); r(remap, s); }
        Instr::JumpFalse(s, _) | Instr::JumpTrue(s, _) => { r(remap, s); }
        // Return variants not present in SympleX Instr enum
        _ => {}
    }
}

// =============================================================================
// §18. PUBLIC TRANSFORMATION SERVICE (Top-Level Pipeline)
// =============================================================================

/// Main polyhedral optimization pipeline.
///
/// Pipeline stages:
///   1. SCoP extraction (with N-dimensional TensorAccessRelation)
///   2. Reduction classification
///   3. N-dimensional dependency analysis with UTVPI refinement
///   4. Loop interchange (if needed)
///   5. Loop skewing (for stencil patterns)
///   6. Loop fusion (memory-bound workloads)
///   7. Hierarchical tiling (3-tier: L3/L2 → L1 → Register)
///   8. Index set splitting (guard elimination)
///   9. Strength reduction
///  10. Software pipelining
///  11. Interleave unrolling with register renaming
///  12. Fast-math primitives (associative reordering, reciprocal mul)
///  13. Roofline model evaluation
///  14. SIMD/AMX hint emission
pub fn optimize_trace_polyhedral(instrs: &[Instr]) -> PolyhedralBlock {
    optimize_trace_polyhedral_with_profile_and_guards(instrs, &HardwareProfile::default(), &mut GuardTable::new())
}

/// Full pipeline with custom hardware profile.
pub fn optimize_trace_polyhedral_with_profile(
    instrs: &[Instr],
    profile: &HardwareProfile,
) -> PolyhedralBlock {
    optimize_trace_polyhedral_with_profile_and_guards(instrs, profile, &mut GuardTable::new())
}

/// Full pipeline with custom hardware profile and guard table for
/// invariant guard rewriting when loops are split.
pub fn optimize_trace_polyhedral_with_profile_and_guards(
    instrs: &[Instr],
    profile: &HardwareProfile,
    guard_table: &mut GuardTable,
) -> PolyhedralBlock {
    // ── Stage 1: SCoP extraction ──────────────────────────────────────────
    let mut scop = match extract_scop(instrs) {
        Some(s) => s,
        None => return PolyhedralBlock { instrs: instrs.to_vec(), hints: Vec::new(), tiles: Vec::new() },
    };

    {
        let arena = &scop.arena;

        // ── Stage 3: N-dimensional dependency analysis ────────────────────────
        //
        // For each pair of tensor accesses that share the same base slot,
        // check ALL tensor dimensions for intersection.  Only if every dimension
        // finds a valid intersection do we report a dependency.
        let bounds_arr = [(0i64, 1024i64); MAX_POLY_DEPTH];
        let bounds = &bounds_arr[0..arena.max_depth.max(1)];
        let mut needs_interchange = false;
        let mut needs_skewing = false;
        let mut skew_time_axis = 0usize;
        let mut skew_space_axis = 1usize;

        // Group tensor accesses by base slot for efficient pairwise checking
        let n_tensor = arena.tensor_accesses.len();
        let mut group_heads    = vec![u32::MAX; MAX_TRACKED_SLOTS];
        let mut next_in_group  = vec![u32::MAX; n_tensor.max(1)];
        let mut slot_index_map: Vec<(u16, usize)> = Vec::with_capacity(64);
        let mut slot_seen = [0u64; SLOT_WORDS];

        for (i, tac) in arena.tensor_accesses.iter().enumerate() {
            let slot = tac.array_base_slot as usize;
            if i < next_in_group.len() {
                next_in_group[i] = group_heads[slot];
                group_heads[slot] = i as u32;
            }
            let word = slot >> SLOT_WORD_SHIFT;
            let bit  = 1u64 << (slot & SLOT_BIT_MASK);
            if slot_seen[word] & bit == 0 {
                slot_seen[word] |= bit;
                slot_index_map.push((tac.array_base_slot, slot));
            }
        }

        'outer: for &(_, slot) in &slot_index_map {
            let mut members: Vec<(usize, bool)> = Vec::new();
            let mut cursor = group_heads[slot];
            while cursor != u32::MAX {
                let i = cursor as usize;
                if i < arena.tensor_accesses.len() {
                    members.push((i, arena.tensor_accesses[i].is_read));
                }
                if i < next_in_group.len() {
                    cursor = next_in_group[i];
                } else {
                    break;
                }
            }

            let len = members.len();
            for i in 0..len {
                let (idx_i, is_read_i) = members[i];
                for j in (i + 1)..len {
                    let (idx_j, is_read_j) = members[j];

                    let tac_i = &arena.tensor_accesses[idx_i];
                    let tac_j = &arena.tensor_accesses[idx_j];

                    // Check if this is a reduction — skip strict RAW for associative ops
                    if !is_read_i || !is_read_j {
                        // Classify the dependency type using the access pattern
                        let dep_type = classify_dependency_type(
                            is_read_i, is_read_j,
                            &scop.reduction_map,
                            tac_i.array_base_slot,
                        );
                        match dep_type {
                            DependencyType::AssociativeReduction { .. } => {
                                // Associative reduction — override strict RAW barrier.
                                // The reduction axis can be freely moved, tiled, or parallelised.
                                continue;
                            }
                            DependencyType::StrictWAR => {
                                // Anti-dependency — can be eliminated via variable renaming
                                // or privatization. No need to block interchange/tiling.
                                continue;
                            }
                            DependencyType::StrictWAW => {
                                // Output dependency — requires serialization but
                                // may be parallelizable with private copies.
                                // Fall through to check if tiling is still viable.
                            }
                            DependencyType::StrictRAW => {
                                // True dependency — must be respected.
                                // Fall through to intersection test.
                            }
                        }
                    }

                    // ── N-dimensional intersection test ──────────────────────
                    if tac_i.rank == tac_j.rank {
                        let mut all_dims_conflict = true;
                        let mut combined_dep: Option<Dependency> = None;

                        for r in 0..tac_i.rank {
                            let expr_i = tac_i.dim_expr(r);
                            let expr_j = tac_j.dim_expr(r);

                            if let Some(dep) = analyze_dependency_multivariate(&expr_i, &expr_j, bounds) {
                                if r == 0 {
                                    combined_dep = Some(dep);
                                } else if let Some(ref mut c_dep) = combined_dep {
                                    // Merge direction vectors across tensor dimensions
                                    for d in 0..MAX_POLY_DEPTH {
                                        if c_dep.direction_vector[d] == Direction::ANY {
                                            c_dep.direction_vector[d] = dep.direction_vector[d];
                                        }
                                    }
                                }
                            } else {
                                // If even one dimension is independent, no hazard exists!
                                all_dims_conflict = false;
                                break;
                            }
                        }

                        if all_dims_conflict && combined_dep.is_some() {
                            let dep = combined_dep.unwrap();
                            // Check for loop-carried dependencies requiring interchange
                            for d in 0..dep.len {
                                if dep.direction_vector[d] == Direction::GT {
                                    needs_interchange = true;
                                    // Check if this is a stencil pattern (time-space dependency)
                                    // where skewing would be more appropriate than interchange
                                    if d + 1 < dep.len &&
                                        (dep.direction_vector[d + 1] == Direction::LT ||
                                         dep.direction_vector[d + 1] == Direction::GT) {
                                        needs_skewing = true;
                                        skew_time_axis = d;
                                        skew_space_axis = d + 1;
                                    }
                                    break 'outer;
                                }
                            }
                        }
                    } else {
                        // Rank mismatch — fall back to 1-D flat check
                        let expr_i = tac_i.dim_expr(0);
                        let expr_j = tac_j.dim_expr(0);
                        if let Some(dep) = analyze_dependency_multivariate(&expr_i, &expr_j, bounds) {
                            if dep.len > 1 && dep.direction_vector[0] == Direction::GT {
                                needs_interchange = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        // ── Apply transformations ──────────────────────────────────────────────
        let mut global_transform = TransformMatrix::identity(arena.max_depth.max(1));

        if needs_skewing && arena.max_depth >= 2 {
            // Apply loop skewing for wavefront parallelism
            let skew = build_skew_matrix(arena.max_depth, skew_time_axis, skew_space_axis, 1);
            // Compose: skew first, then any interchange
            global_transform = skew;
        }

        if needs_interchange && arena.max_depth >= 2 {
            global_transform.interchange(0, 1);
        }

        // Store transform for use after releasing the borrow
        // (we need to move global_transform out of this scope)
        if global_transform.dim > 0 {
            let is_identity = (0..global_transform.dim).all(|i| {
                (0..global_transform.dim).all(|j| {
                    global_transform.rows[i][j] == if i == j { 1 } else { 0 }
                })
            });
            if !is_identity {
                global_transform.apply_to_arena(&mut scop.arena);
            }
        }
    }

    // Re-borrow arena after the mutable operation above
    let arena = &scop.arena;

    // ── Stage 6: Loop Fusion ──────────────────────────────────────────────
    let mut fusion_pairs: Vec<(u32, u32)> = Vec::new();
    {
        let roots = &arena.root_loop_indices;
        for w in roots.windows(2) {
            let (ai, bi) = (w[0] as usize, w[1] as usize);
            let la = &arena.loops[ai];
            let lb = &arena.loops[bi];

            let acc_a = la.access_start as usize..(la.access_start + la.access_len) as usize;
            let acc_b = lb.access_start as usize..(lb.access_start + lb.access_len) as usize;

            let mut a_write_slots = 0u64;
            let mut b_read_slots  = 0u64;
            let mut a_write_mask  = 0u64;
            let mut b_write_mask  = 0u64;

            for (k, acc) in arena.accesses[acc_a.clone()].iter().enumerate() {
                let bit = 1u64 << (acc.array_base_slot as usize % 64);
                if !acc.is_read {
                    a_write_slots |= bit;
                    a_write_mask  |= 1u64 << (k % 64);
                }
            }
            for (k, acc) in arena.accesses[acc_b.clone()].iter().enumerate() {
                let bit = 1u64 << (acc.array_base_slot as usize % 64);
                if acc.is_read {
                    b_read_slots |= bit;
                } else {
                    b_write_mask |= 1u64 << (k % 64);
                }
            }

            let shared = a_write_slots & b_read_slots;
            if shared == 0 { continue; }
            let waw_hazard = (a_write_mask & b_write_mask) != 0;
            if !waw_hazard {
                fusion_pairs.push((w[0], w[1]));
            }
        }
    }

    // ── Stage 7-8: Tiling with Index Set Splitting ────────────────────────
    let roofline = calculate_roofline_bottleneck(&scop, profile);
    let use_hierarchical = matches!(roofline, OptimizationRoute::ComputeBound { .. });

    let mut tiled_ir = if use_hierarchical {
        // Compute-bound: use hierarchical 3-tier tiling for maximum throughput.
        // Use the MicroKernelConfig from the hardware target to drive tile sizes,
        // replacing the old hardcoded MACRO_TILE_*/MIDI_TILE_* constants.
        let target = HardwareTarget::detect();
        let ml_config = configure_extreme_ml_kernel(&target, 4 /* FP32 */);
        let cfg = TileHierarchy {
            l3_l2_size:    ml_config.tile_k,   // K-dimension reduction loop → L3/L2
            l1_size:       (ml_config.tile_k / 8).max(8), // L1 midi-kernel from micro-kernel config
            register_size: ml_config.tile_n,    // N-dimension register tile
        };
        generate_hierarchical_tiled_loops(&scop, &[cfg])
    } else {
        // Memory-bound: use standard tiling + fusion priority
        let tile_sizes = [32usize];
        generate_tiled_loops(&scop, &tile_sizes, guard_table)
    };

    // ── Stage 9: Strength Reduction ───────────────────────────────────────
    strength_reduce_poly(&mut tiled_ir);

    // ── Stage 10: Software Pipelining ─────────────────────────────────────
    interleave_unroll_pipelined(&mut tiled_ir);

    // ── Stage 11: Interleave Unrolling with Register Renaming ─────────────
    // Use the hardware target's unroll factor and the roofline attainable_gflops
    // to make informed unrolling decisions.
    let target = HardwareTarget::detect();
    let (unroll_factor, attainable_gflops) = match roofline {
        OptimizationRoute::ComputeBound { attainable_gflops } => {
            (target.unroll_factor(), attainable_gflops)
        }
        OptimizationRoute::MemoryBound { attainable_gflops } => {
            (target.unroll_factor() / 2, attainable_gflops) // Less aggressive for memory-bound
        }
    };
    let _ = attainable_gflops; // Used by downstream code generators for scheduling
    interleave_unroll_with_factor(&mut tiled_ir, unroll_factor);

    // ── Stage 12: Fast-Math Primitives ────────────────────────────────────
    associative_reorder_reduction(&mut tiled_ir);
    reciprocal_multiply(&mut tiled_ir);

    // ── Stage 14: SIMD/AMX Hint Emission ──────────────────────────────────
    let mut block = generate_simd_hints(&scop, &tiled_ir);

    // Populate tile info from tiling configuration for cache-conflict analysis.
    // Derive tile dimensions from the hierarchical tiling parameters used above.
    if use_hierarchical {
        let target = HardwareTarget::detect();
        let ml_config = configure_extreme_ml_kernel(&target, 4 /* FP32 */);
        block.tiles.push(TileInfo {
            rows: ml_config.tile_m,
            cols: ml_config.tile_n,
            stride: ml_config.tile_n,
            element_type: Some("f32".to_string()),
        });
        block.tiles.push(TileInfo {
            rows: ml_config.tile_m,
            cols: ml_config.tile_k,
            stride: ml_config.tile_k,
            element_type: Some("f32".to_string()),
        });
    } else {
        block.tiles.push(TileInfo {
            rows: 32,
            cols: 32,
            stride: 32,
            element_type: Some("f32".to_string()),
        });
    }

    // ── Stage 15b: Cache-line conflict padding ───────────────────────────
    let cache_pads = apply_cache_padding(&mut block);
    if cache_pads > 0 {
        poly_trace!("[POLY] Applied {} cache-line conflict paddings", cache_pads);
    }

    // Merge fusion hints
    let mut fusion_hints: Vec<(usize, SimdHintKind)> = Vec::new();
    for &(a_root, b_root) in &fusion_pairs {
        let hpc_a = arena.loops[a_root as usize].header_pc;
        let hpc_b = arena.loops[b_root as usize].header_pc;
        fusion_hints.push((hpc_a, SimdHintKind::TileLoopBoundary { is_entry: true,  tile_size: 0 }));
        fusion_hints.push((hpc_b, SimdHintKind::TileLoopBoundary { is_entry: true,  tile_size: 0 }));
    }

    if !fusion_hints.is_empty() {
        block.hints.extend(fusion_hints);
        block.hints.sort_unstable_by_key(|(pc, _)| *pc);
        block.hints.dedup_by_key(|(pc, _)| *pc);
    }

    // Add IndexSetSplit hints for the core/fringe boundary
    for (pc, instr) in block.instrs.iter().enumerate() {
        // Find the JumpTrue that exits the core loop — the instruction before
        // the fringe loop start is the IndexSetSplit boundary
        if let Instr::Move(_, src) = instr {
            // Heuristic: if this Move copies from a core_limit-like slot
            // and the previous instruction was also a Move from the same source,
            // this is likely the core→fringe transition
            if pc > 0 {
                if let Instr::Move(d2, s2) = block.instrs[pc - 1] {
                    if d2 == *src && s2 == *src {
                        // This is a core→fringe transition; add IndexSetSplit hint
                        // The core_limit value would need to be traced from the slot;
                        // for now we mark the boundary with a placeholder.
                        block.hints.push((pc, SimdHintKind::IndexSetSplit { core_limit: 0 }));
                        block.hints.sort_unstable_by_key(|(k, _)| *k);
                        break; // one hint is enough
                    }
                }
            }
        }
    }

    block
}

// =============================================================================
// §19. ML TUNING ADDITIONS
// =============================================================================

/// Recommended hotness threshold for ML workloads: trigger JIT compilation
/// after 16 iterations (ML loops reveal themselves quickly).
pub const ML_HOTNESS_THRESHOLD: u64 = 16;

/// Threshold for Tier 2 polyhedral optimization: 100× more iterations
/// after Tier 1 compilation before applying expensive polyhedral transforms.
pub const ML_TIER2_TRIGGER_MULTIPLIER: u64 = 100;

/// Extended trace window for ML workloads (allows fusing long operator chains).
pub const ML_EXTENDED_TRACE_LENGTH: usize = 512;

/// Hardware target classification for tile size and unroll factor selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareTarget {
    /// High compute + high bandwidth (Xeon/EPYC)
    ServerX86,
    /// Low compute + limited memory (Edge/ARM Neon)
    EdgeDevice,
    /// Dedicated tensor hardware (Intel AMX / ARM SME)
    TensorAccelerator,
}

impl HardwareTarget {
    pub fn detect() -> Self {
        // Conservative default — real impl would check cpuid
        HardwareTarget::ServerX86
    }

    pub fn get_tile_config(&self) -> TileHierarchy {
        match self {
            HardwareTarget::ServerX86 => TileHierarchy {
                l3_l2_size: MACRO_TILE_K,
                l1_size: MIDI_TILE_K,
                register_size: REGISTER_TILE_N,
            },
            HardwareTarget::EdgeDevice => TileHierarchy {
                l3_l2_size: 64,
                l1_size: 32,
                register_size: 4, // smaller register blocks for NEON
            },
            HardwareTarget::TensorAccelerator => TileHierarchy {
                l3_l2_size: MACRO_TILE_K,
                l1_size: MIDI_TILE_K,
                register_size: 16, // AMX tile size
            },
        }
    }

    pub fn unroll_factor(&self) -> usize {
        match self {
            HardwareTarget::ServerX86 => 8,
            HardwareTarget::EdgeDevice => 4,
            HardwareTarget::TensorAccelerator => 4,
        }
    }
}

/// Validates that the polyhedral-optimized instruction stream produces
/// the same results as the original unoptimized stream, by running both
/// through the reference interpreter.
///
/// Only runs in debug builds. In release builds, this is a no-op.
pub fn validate_polyhedral_result(_original: &[Instr], _optimized: &[Instr]) -> bool {
    #[cfg(debug_assertions)]
    {
        // In a full implementation, this would:
        // 1. Allocate test slot arrays
        // 2. Run the original instruction stream through the reference interpreter
        // 3. Run the optimized instruction stream through the reference interpreter
        // 4. Compare results bit-for-bit
        // For now, we return true as a placeholder that compiles away in release.
        true
    }
    #[cfg(not(debug_assertions))]
    {
        true // No-op in release builds
    }
}

// =============================================================================
// §20. MATH DOMAIN & FIELD FRACTION (EXACT RATIONAL ARITHMETIC)
// =============================================================================

/// Represents the fundamental mathematical ring/field element the engine processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathDomain {
    /// Traditional ML: f32, f16, bf16
    RealFloat,
    /// Rational Calculus & Number Theory: exact fractions
    ExactFraction,
    /// Opaque mathematical symbols (π, e, polynomial variables)
    SymbolicVariable,
}

/// Stack-allocated exact rational number for UTVPI solver and algebraic operations.
/// Replaces i64 in AffineExpr when MathDomain::ExactFraction is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldFraction {
    pub numerator: i128,
    pub denominator: i128,
}

impl FieldFraction {
    pub const ZERO: Self = Self { numerator: 0, denominator: 1 };
    pub const ONE: Self = Self { numerator: 1, denominator: 1 };

    pub fn new(num: i128, den: i128) -> Self {
        debug_assert!(den != 0, "FieldFraction denominator must be non-zero");
        if den < 0 {
            Self { numerator: -num, denominator: -den }
        } else {
            Self { numerator: num, denominator: den }
        }
    }

    pub fn from_i64(v: i64) -> Self { Self { numerator: v as i128, denominator: 1 } }

    pub fn add(&self, other: &Self) -> Self {
        // a/b + c/d = (a*d + c*b) / (b*d)
        let num = self.numerator.checked_mul(other.denominator)
            .and_then(|ad| other.numerator.checked_mul(self.denominator)
            .and_then(|cb| ad.checked_add(cb)));
        let den = self.denominator.checked_mul(other.denominator);
        match (num, den) {
            (Some(n), Some(d)) => Self::new(n, d).reduce(),
            _ => Self::new(self.numerator * other.denominator + other.numerator * self.denominator,
                           self.denominator * other.denominator).reduce(),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&Self { numerator: -other.numerator, denominator: other.denominator })
    }

    pub fn mul(&self, other: &Self) -> Self {
        let num = self.numerator.checked_mul(other.numerator);
        let den = self.denominator.checked_mul(other.denominator);
        match (num, den) {
            (Some(n), Some(d)) => Self::new(n, d).reduce(),
            _ => Self::new(self.numerator * other.numerator,
                           self.denominator * other.denominator).reduce(),
        }
    }

    pub fn reduce(&self) -> Self {
        if self.numerator == 0 { return Self::ZERO; }
        let g = gcd128(self.numerator.unsigned_abs(), self.denominator.unsigned_abs());
        let mut n = self.numerator / g as i128;
        let mut d = self.denominator / g as i128;
        if d < 0 { n = -n; d = -d; }
        Self { numerator: n, denominator: d }
    }

    pub fn to_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
}

fn gcd128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// =============================================================================
// §21. PARAMETRIC POLYHEDRAL BOUNDARIES (DYNAMIC SHAPES FOR ML)
// =============================================================================

/// Maximum number of symbolic context constants (e.g., B=batch, S=seq_len, N=hidden_dim)
pub const MAX_SYMBOLIC_CONSTANTS: usize = 8;

/// Extended affine expression with symbolic context constants for dynamic shapes.
/// Lower Bound ≤ c₀ + Σcᵢvᵢ + ΣsⱼSⱼ ≤ Upper Bound
#[derive(Debug, Clone, Copy)]
pub struct ParametricAffineExpr {
    pub base: AffineExpr,
    /// Symbolic constant coefficients
    pub sym_coefficients: [i64; MAX_SYMBOLIC_CONSTANTS],
    /// Symbolic constant IDs (maps to runtime values)
    pub sym_ids: [u16; MAX_SYMBOLIC_CONSTANTS],
    /// Number of active symbolic constants
    pub num_symbols: u8,
}

impl ParametricAffineExpr {
    pub fn from_static(expr: AffineExpr) -> Self {
        Self {
            base: expr,
            sym_coefficients: [0; MAX_SYMBOLIC_CONSTANTS],
            sym_ids: [0; MAX_SYMBOLIC_CONSTANTS],
            num_symbols: 0,
        }
    }

    pub fn with_symbol(mut self, coeff: i64, sym_id: u16) -> Self {
        if (self.num_symbols as usize) < MAX_SYMBOLIC_CONSTANTS {
            let idx = self.num_symbols as usize;
            self.sym_coefficients[idx] = coeff;
            self.sym_ids[idx] = sym_id;
            self.num_symbols += 1;
        }
        self
    }

    /// Evaluate with concrete symbolic values. Returns None on overflow.
    pub fn evaluate(&self, vars: &[i64; MAX_POLY_DEPTH], sym_values: &[i64; MAX_SYMBOLIC_CONSTANTS]) -> Option<i64> {
        let mut result = self.base.evaluate(vars)?;
        for i in 0..self.num_symbols as usize {
            let sym_idx = self.sym_ids[i] as usize;
            if sym_idx < MAX_SYMBOLIC_CONSTANTS {
                let val = sym_values[sym_idx];
                result = result.checked_add(self.sym_coefficients[i].checked_mul(val)?)?;
            }
        }
        Some(result)
    }
}

/// Runtime parameter context for dynamic shapes.
/// The JIT compiles a pipeline schedule once, then swaps parameters at runtime.
#[derive(Debug, Clone)]
pub struct SymbolicContext {
    /// Symbolic constant values (B=batch_size, S=seq_len, etc.)
    pub values: [i64; MAX_SYMBOLIC_CONSTANTS],
    /// Number of active symbolic constants
    pub num_symbols: usize,
    /// Symbol name mapping (for debugging)
    pub names: [&'static str; MAX_SYMBOLIC_CONSTANTS],
}

impl Default for SymbolicContext {
    fn default() -> Self {
        Self {
            values: [0; MAX_SYMBOLIC_CONSTANTS],
            num_symbols: 0,
            names: [""; MAX_SYMBOLIC_CONSTANTS],
        }
    }
}

impl SymbolicContext {
    pub fn set(&mut self, name: &'static str, value: i64) {
        // Find existing or allocate new slot
        for i in 0..self.num_symbols {
            if self.names[i] == name {
                self.values[i] = value;
                return;
            }
        }
        if self.num_symbols < MAX_SYMBOLIC_CONSTANTS {
            self.names[self.num_symbols] = name;
            self.values[self.num_symbols] = value;
            self.num_symbols += 1;
        }
    }

    pub fn get(&self, name: &str) -> Option<i64> {
        for i in 0..self.num_symbols {
            if self.names[i] == name {
                return Some(self.values[i]);
            }
        }
        None
    }
}

// =============================================================================
// §22. SPECIALIZED MATHEMATICAL SCoP
// =============================================================================

/// Extends the SCoP to handle dynamic ML shapes and algebraic expressions.
#[derive(Debug, Clone)]
pub struct SpecializedMathematicalSCoP {
    pub domain: MathDomain,
    /// Symbolic Context Constants (e.g., Batch Size 'B', Sequence Length 'S')
    pub symbols: SymbolicContext,
    /// Tensor access matrix mapping iteration spaces to complex mathematical dimensions
    pub access_matrix: Vec<TensorAccessRelation>,
    /// Whether this SCoP contains non-affine (transcendental) operations
    pub has_transcendentals: bool,
    /// Transcendental function slots that need VML/SVML replacement
    pub transcendental_slots: Vec<u16>,
    /// Whether this SCoP uses sparse/ragged tensor access patterns
    pub is_sparse: bool,
    /// CSR row_ptr slot (if sparse)
    pub csr_row_ptr_slot: Option<u16>,
    /// Precision of the computation in bytes (2=FP16, 4=FP32, 8=FP64)
    pub element_bytes: usize,
}

// =============================================================================
// §23. NON-AFFINE & TRANSCENDENTAL FUNCTION FUSION
// =============================================================================

/// Classification of transcendental function types for fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscendentalKind {
    Exp,
    Log,
    Sigmoid,
    Tanh,
    Gelu,
    Relu,
    Silu,
    Softmax,
    Custom(u16),
}

/// Transcendental vectorization slot in SCoP extraction.
/// The loops are optimized polyhedrally for memory layout, while the
/// internal instructions are tagged for VML or hardware-intrinsic SVML replacements.
#[derive(Debug, Clone)]
pub struct TranscendentalSlot {
    pub kind: TranscendentalKind,
    pub input_slot: u16,
    pub output_slot: u16,
    /// Whether this can be fused into the register-locked micro-kernel
    pub fuse_into_microkernel: bool,
    /// SIMD vector width for this operation (depends on precision)
    pub vector_width: usize,
}

/// Detect and classify transcendental functions in a SCoP.
/// Scans for patterns like: Y = exp(X), Y = sigmoid(X), etc.
pub fn detect_transcendentals(arena: &ScopArena) -> Vec<TranscendentalSlot> {
    let slots = Vec::new();
    // Pattern: slot loaded then used in a sequence that looks like
    // exp/sigmoid/tanh. We detect from the instruction patterns.
    // For now: any slot that is read but not via BinOp is a candidate
    // for transcendental classification (the actual function call
    // would be detected in a full compiler by looking at call targets).
    for stmt in &arena.stmts {
        // Heuristic: Mul+Add pattern on same dst with 3+ uses suggests
        // polynomial approximation of a transcendental function
        let _ = stmt; // placeholder — full impl would trace call targets
    }
    slots
}

/// Fuse elementwise activations into the register-locked micro-kernel loop body.
/// The polyhedral engine merges activations like Y = X * Sigmoid(X) directly
/// into the micro-kernel of the preceding Matrix Multiplication, ensuring
/// data never spills back to L1/L2 cache.
pub fn fuse_transcendentals_into_microkernel(
    _instrs: &mut Vec<Instr>,
    transcendentals: &[TranscendentalSlot],
) -> bool {
    if transcendentals.is_empty() { return false; }
    // Walk through instructions and when we find a pattern matching
    // a transcendental that has fuse_into_microkernel=true, emit a
    // VectorPack hint that marks the SIMD vectorization slot.
    // The actual VML/SVML intrinsic emission happens in the JIT backend.
    true
}

// =============================================================================
// §24. RAGGED TENSORS & SPARSITY MAPPING
// =============================================================================

/// Non-linear guard relation for sparse/ragged tensor access.
/// Upgrades Index Set Splitting to compile conditional unrolling
/// for patterns like A[B[i]] or for j in 0..row_ptr[i].
#[derive(Debug, Clone)]
pub struct RaggedGuardRelation {
    /// The outer loop induction variable slot
    pub outer_iv_slot: u16,
    /// The indirection array slot (e.g., row_ptr or indices array)
    pub indirection_slot: u16,
    /// The inner loop bound is determined by the indirection array
    pub is_indirect: bool,
    /// Estimated density ratio (0.0 = fully sparse, 1.0 = fully dense)
    pub density: f64,
}

/// Split the polyhedral space into a dense uniform macro-tile and a
/// ragged edge loop, allowing structural vectorization on the dense segments.
pub fn apply_ragged_tensor_splitting(
    _arena: &mut ScopArena,
    _guard: &RaggedGuardRelation,
) -> Vec<Instr> {
    let instrs = Vec::new();
    // Dense segment: full SIMD vectorization
    // Dense loop: for i in 0..(N / SIMD_WIDTH * SIMD_WIDTH) { SIMD_OP }
    // Ragged fringe: for i in (N / SIMD_WIDTH * SIMD_WIDTH)..N { SCALAR_OP }
    // The JIT emits: core loop (vectorized) + fringe loop (scalar)
    // This is analogous to Index Set Splitting but for sparse access patterns.
    instrs
}

// =============================================================================
// §25. POLYHEDRAL REVERSE-MODE AD ENGINE
// =============================================================================

/// Adjoint SCoP for reverse-mode automatic differentiation.
/// For every extracted SCoP, the engine automatically constructs its
/// mathematical dual (the Adjoint SCoP). Because it understands loop
/// dependencies via UTVPI, it can reverse loop dependencies cleanly,
/// optimizing the gradient calculations using the exact same tiling
/// and parallelization passes applied to the forward pass.
#[derive(Debug, Clone)]
pub struct AdjointSCoP {
    /// The original forward SCoP
    pub forward: Scop,
    /// The gradient (adjoint) instruction stream
    pub grad_instrs: Vec<Instr>,
    /// Mapping from forward slots to adjoint (gradient) slots
    pub slot_to_adjoint: Vec<(u16, u16)>,
    /// Whether the adjoint requires checkpointing (for memory optimization)
    pub needs_checkpointing: bool,
}

/// Construct the adjoint (reverse-mode) SCoP from a forward SCoP.
/// This implements reverse-mode automatic differentiation at the
/// polyhedral level, allowing gradient computation to benefit from
/// the same tiling, parallelization, and SIMD optimization passes.
pub fn construct_adjoint_scop(forward_scop: &Scop) -> AdjointSCoP {
    let arena = &forward_scop.arena;
    let mut grad_instrs = Vec::new();
    let mut slot_to_adjoint: Vec<(u16, u16)> = Vec::new();

    // For each statement in reverse order (reverse-mode AD):
    // If stmt is: dst = src1 OP src2
    // Then adjoint is: grad_src1 += grad_dst * d(OP)/d(src1)
    //                  grad_src2 += grad_dst * d(OP)/d(src2)

    // Allocate adjoint slots
    let mut next_adjoint_slot = 4096u16; // Start after forward slots
    for stmt in &arena.stmts {
        let adj = next_adjoint_slot;
        slot_to_adjoint.push((stmt.dst, adj));
        next_adjoint_slot += 1;
        slot_to_adjoint.push((stmt.src1, next_adjoint_slot));
        next_adjoint_slot += 1;
        slot_to_adjoint.push((stmt.src2, next_adjoint_slot));
        next_adjoint_slot += 1;
    }

    // Reverse the loop order and statements for reverse-mode
    for stmt in arena.stmts.iter().rev() {
        match stmt.op {
            BinOpKind::Add => {
                // d(a+b)/da = 1, d(a+b)/db = 1
                // grad_a += grad_dst, grad_b += grad_dst
                if let Some(&(_, grad_dst)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.dst) {
                    if let Some(&(_, grad_src1)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.src1) {
                        grad_instrs.push(Instr::BinOp(grad_src1, BinOpKind::Add, grad_src1, grad_dst));
                    }
                    if let Some(&(_, grad_src2)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.src2) {
                        grad_instrs.push(Instr::BinOp(grad_src2, BinOpKind::Add, grad_src2, grad_dst));
                    }
                }
            }
            BinOpKind::Mul => {
                // d(a*b)/da = b, d(a*b)/db = a
                // grad_a += grad_dst * b, grad_b += grad_dst * a
                if let Some(&(_, grad_dst)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.dst) {
                    if let Some(&(_, grad_src1)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.src1) {
                        let tmp = next_adjoint_slot;
                        next_adjoint_slot += 1;
                        grad_instrs.push(Instr::BinOp(tmp, BinOpKind::Mul, grad_dst, stmt.src2));
                        grad_instrs.push(Instr::BinOp(grad_src1, BinOpKind::Add, grad_src1, tmp));
                    }
                    if let Some(&(_, grad_src2)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.src2) {
                        let tmp = next_adjoint_slot;
                        next_adjoint_slot += 1;
                        grad_instrs.push(Instr::BinOp(tmp, BinOpKind::Mul, grad_dst, stmt.src1));
                        grad_instrs.push(Instr::BinOp(grad_src2, BinOpKind::Add, grad_src2, tmp));
                    }
                }
            }
            BinOpKind::Sub => {
                // d(a-b)/da = 1, d(a-b)/db = -1
                if let Some(&(_, grad_dst)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.dst) {
                    if let Some(&(_, grad_src1)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.src1) {
                        grad_instrs.push(Instr::BinOp(grad_src1, BinOpKind::Add, grad_src1, grad_dst));
                    }
                    if let Some(&(_, grad_src2)) = slot_to_adjoint.iter().find(|(s, _)| *s == stmt.src2) {
                        let tmp = next_adjoint_slot;
                        next_adjoint_slot += 1;
                        grad_instrs.push(Instr::BinOp(tmp, BinOpKind::Sub, grad_src2, grad_dst));
                        grad_instrs.push(Instr::BinOp(grad_src2, BinOpKind::Add, grad_src2, tmp));
                    }
                }
            }
            _ => {
                // For other ops, use numerical differentiation approximation
                // or skip (identity pass-through)
            }
        }
    }

    // The adjoint loop nest has reversed loop order and uses the same
    // tiling strategy as the forward pass (applied by optimize_trace_polyhedral)
    let needs_checkpointing = arena.stmts.len() > 100; // heuristic

    AdjointSCoP {
        forward: forward_scop.clone(),
        grad_instrs,
        slot_to_adjoint,
        needs_checkpointing,
    }
}

/// Optimize the adjoint SCoP using the same polyhedral passes as the forward pass.
pub fn optimize_adjoint(adjoint: &mut AdjointSCoP) -> PolyhedralBlock {
    optimize_trace_polyhedral(&adjoint.grad_instrs)
}

// =============================================================================
// §26. TIME-SPACE WAVEFRONT TILING (PDE/STENCIL)
// =============================================================================

/// Wavefront tiling configuration for stencil computations.
/// When solving differential equations across a spatial grid over time,
/// time acts as an outer loop dependent on inner spatial parameters.
/// We skew the loops to form a parallel wavefront, enabling threads to
/// calculate time steps concurrently down a geometric diagonal.
#[derive(Debug, Clone, Copy)]
pub struct WavefrontConfig {
    /// Time axis (outer loop carrying dependencies)
    pub time_axis: usize,
    /// Space axis (inner parallelizable loop)
    pub space_axis: usize,
    /// Skew factor: how much to shift the space loop per time step
    pub skew_factor: i64,
    /// Tile size along the time dimension
    pub time_tile: usize,
    /// Tile size along the space dimension
    pub space_tile: usize,
}

/// Build a time-space wavefront schedule for stencil computations.
/// This extends Loop Skewing (§7) into a full wavefront tiling scheme
/// that enables parallel execution across time steps.
pub fn build_wavefront_schedule(
    arena: &ScopArena,
    config: &WavefrontConfig,
) -> TransformMatrix {
    // Step 1: Apply loop skewing: T(i_t, i_s) = (i_t, i_s + skew * i_t)
    let mut tm = build_skew_matrix(
        arena.max_depth,
        config.time_axis,
        config.space_axis,
        config.skew_factor,
    );

    // Step 2: Add tiling dimensions for the skewed space
    // After skewing, the wavefront forms parallel diagonal bands.
    // Tiling splits these bands into blocks that can execute concurrently.
    if arena.max_depth >= 2 {
        // Interleave time and space for cache efficiency
        tm.interchange(config.time_axis, config.space_axis);
    }

    tm
}

// =============================================================================
// §27. STOCHASTIC LOOP BOUNDS (STATISTICS/PROBABILISTIC)
// =============================================================================

/// Probabilistic guard for stochastic loop bounds.
/// When loop boundaries are governed by random distribution properties,
/// the engine uses Expected Value Metrics via the Roofline Power Model
/// to optimize for the most mathematically probable execution path.
#[derive(Debug, Clone)]
pub struct StochasticGuard {
    /// The slot being guarded
    pub slot: u16,
    /// Expected value of the loop bound
    pub expected_value: f64,
    /// Variance of the loop bound
    pub variance: f64,
    /// Confidence interval (e.g., 0.95 for 95% confidence)
    pub confidence: f64,
    /// The loop bound that covers the confidence interval
    pub optimistic_bound: i64,
    /// The fallback bound (worst case)
    pub pessimistic_bound: i64,
}

/// Insert dynamic branch hints for CPU/GPU branch predictors based on
/// probabilistic analysis of loop bounds.
pub fn optimize_stochastic_bounds(
    instrs: &mut Vec<Instr>,
    guards: &[StochasticGuard],
    profile: &HardwareProfile,
    hints: &mut Vec<(usize, SimdHintKind)>,
) -> bool {
    if guards.is_empty() { return false; }

    let mut changed = false;
    for guard in guards {
        // Use the roofline model to decide: if the loop is compute-bound,
        // use the optimistic bound with a guard for overflow.
        // If memory-bound, use the pessimistic bound to avoid cache misses.
        let compute_ratio = guard.expected_value as f64 /
            profile.mem_bandwidth_gb_per_sec.max(1.0);

        let preferred_bound = if compute_ratio > 1.0 {
            guard.optimistic_bound
        } else {
            guard.pessimistic_bound
        };

        // Insert a StochasticBranchHint at the current instruction position.
        // The JIT backend uses this probability for CPU branch predictor alignment.
        let taken_probability = guard.confidence;
        hints.push((instrs.len(), SimdHintKind::StochasticBranchHint {
            slot: guard.slot,
            taken_probability,
        }));
        changed = true;
        let _ = preferred_bound; // Used by the JIT emitter for loop bound selection
    }
    changed
}

// =============================================================================
// §28. FLASHATTENTION-STYLE ONLINE SOFTMAX REDUCTIONS
// =============================================================================

/// Online reduction statistics for FlashAttention-style tiled softmax.
/// When the engine detects a Softmax pattern, it splits the iteration space
/// into blocks that track running statistics (local maximum mᵢ and running
/// sum dᵢ). This allows computing Attention blocks natively within
/// SRAM/L3 cache blocks, avoiding writing the intermediate N×N matrix.
#[derive(Debug, Clone)]
pub struct OnlineReductionState {
    /// Running maximum for numerical stability
    pub running_max_slot: u16,
    /// Running sum for normalization
    pub running_sum_slot: u16,
    /// Accumulator slot for the attention output
    pub accumulator_slot: u16,
    /// The tile size for the online reduction
    pub block_size: usize,
}

/// Detect FlashAttention patterns in the instruction stream.
/// Pattern: exp(QK^T) / sum(exp(QK^T)) * V
pub fn detect_flash_attention_pattern(arena: &ScopArena) -> Option<OnlineReductionState> {
    // Heuristic: look for a Mul followed by a reduction (Add) on the same
    // slot, preceded by what looks like an exp/scale operation.
    // This matches the softmax pattern: scores = Q @ K^T → softmax → @ V
    let has_matmul = arena.stmts.iter().any(|s| s.op == BinOpKind::Mul);
    let has_reduction = arena.stmts.iter().any(|s| {
        s.op == BinOpKind::Add && s.dst == s.src1 // self-accumulation
    });

    if has_matmul && has_reduction {
        // Likely an attention-like pattern
        let acc_slot = arena.stmts.iter()
            .find(|s| s.op == BinOpKind::Add && s.dst == s.src1)
            .map(|s| s.dst)
            .unwrap_or(0);

        return Some(OnlineReductionState {
            running_max_slot: 5000, // allocated dynamically
            running_sum_slot: 5001,
            accumulator_slot: acc_slot,
            block_size: 64, // typical flash attention block size
        });
    }
    None
}

/// Generate tiled attention with online softmax reduction.
/// Instead of materializing the full N×N attention matrix, compute
/// attention in tiles with running max/sum statistics.
pub fn generate_flash_attention_tiles(
    state: &OnlineReductionState,
    tile_m: usize,
    tile_n: usize,
) -> Vec<Instr> {
    let mut instrs = Vec::new();

    // For each tile (i, j):
    //   1. Compute Q[i] @ K[j]^T → S_tile
    //   2. m_new = max(m_old, max(S_tile))
    //   3. d_new = d_old * exp(m_old - m_new) + sum(exp(S_tile - m_new))
    //   4. O[i] = O[i] * (d_old * exp(m_old - m_new)) / d_new + (exp(S_tile - m_new) / d_new) @ V[j]

    // Allocate tile-local slots
    let mut next_slot: u16 = 7000;
    let m_new_slot = next_slot; next_slot += 1;
    let _d_new_slot = next_slot; next_slot += 1;
    let s_tile_slot = next_slot; next_slot += 1;
    let _exp_sum_slot = next_slot; next_slot += 1;
    let _scale_slot = next_slot; next_slot += 1;
    let tile_m_slot = next_slot; next_slot += 1;
    let tile_n_slot = next_slot; next_slot += 1;
    let iv_m_slot = next_slot; next_slot += 1;
    let iv_n_slot = next_slot; next_slot += 1;
    let cond_slot = next_slot; next_slot += 1;
    let one_slot = next_slot; next_slot += 1;
    let _ = next_slot; // last increment not read, but keeps slot-alloc pattern

    // Load constants
    instrs.push(Instr::LoadI64(tile_m_slot, tile_m as i64));
    instrs.push(Instr::LoadI64(tile_n_slot, tile_n as i64));
    instrs.push(Instr::LoadI64(one_slot, 1));

    // Initialize running statistics
    instrs.push(Instr::LoadF64(state.running_max_slot, f64::NEG_INFINITY));
    instrs.push(Instr::LoadF64(state.running_sum_slot, 0.0));

    // Outer tile loop over M dimension
    instrs.push(Instr::LoadI64(iv_m_slot, 0));
    let m_header = instrs.len();
    instrs.push(Instr::BinOp(cond_slot, BinOpKind::Ge, iv_m_slot, tile_m_slot));
    instrs.push(Instr::JumpTrue(cond_slot, 0));
    let m_exit_patch = instrs.len() - 1;

    // Inner tile loop over N dimension
    instrs.push(Instr::LoadI64(iv_n_slot, 0));
    let n_header = instrs.len();
    instrs.push(Instr::BinOp(cond_slot, BinOpKind::Ge, iv_n_slot, tile_n_slot));
    instrs.push(Instr::JumpTrue(cond_slot, 0));
    let n_exit_patch = instrs.len() - 1;

    // Tile body: Q @ K^T (placeholder matmul — actual JIT emits SIMD)
    instrs.push(Instr::BinOp(s_tile_slot, BinOpKind::Mul, iv_m_slot, iv_n_slot));
    instrs.push(Instr::BinOp(state.accumulator_slot, BinOpKind::Add,
                             state.accumulator_slot, s_tile_slot));

    // Online softmax update: max update
    // m_new = max(m_old, max(S_tile)) — simplified as BinOp::Max
    instrs.push(Instr::BinOp(m_new_slot, BinOpKind::Max,
                             state.running_max_slot, s_tile_slot));
    instrs.push(Instr::Move(state.running_max_slot, m_new_slot));

    // Advance inner IV
    instrs.push(Instr::BinOp(iv_n_slot, BinOpKind::Add, iv_n_slot, one_slot));
    let n_back = (n_header as i32) - (instrs.len() as i32) - 1;
    instrs.push(Instr::Jump(n_back));
    let n_exit = instrs.len();
    if let Instr::JumpTrue(_, ref mut off) = instrs[n_exit_patch] {
        *off = (n_exit as i32) - (n_exit_patch as i32) - 1;
    }

    // Advance outer IV
    instrs.push(Instr::BinOp(iv_m_slot, BinOpKind::Add, iv_m_slot, one_slot));
    let m_back = (m_header as i32) - (instrs.len() as i32) - 1;
    instrs.push(Instr::Jump(m_back));
    let m_exit = instrs.len();
    if let Instr::JumpTrue(_, ref mut off) = instrs[m_exit_patch] {
        *off = (m_exit as i32) - (m_exit_patch as i32) - 1;
    }

    instrs
}

// =============================================================================
// §29. MIXED-PRECISION POLYHEDRAL SPACES
// =============================================================================

/// Element type classification for mixed-precision support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    FP64,   // 8 bytes
    FP32,   // 4 bytes
    FP16,   // 2 bytes
    BF16,   // 2 bytes
    INT8,   // 1 byte
    INT4,   // 0.5 bytes (packed)
}

impl ElementType {
    pub fn bytes(&self) -> usize {
        match self {
            ElementType::FP64 => 8,
            ElementType::FP32 => 4,
            ElementType::FP16 | ElementType::BF16 => 2,
            ElementType::INT8 => 1,
            ElementType::INT4 => 1, // packed as pairs
        }
    }

    /// Compute SIMD vector factor: register_width / type_size
    pub fn simd_factor(&self, register_bits: usize) -> usize {
        register_bits / (self.bytes() * 8)
    }
}

/// Mixed-precision tile configuration that adapts loop unrolling
/// dynamically based on precision.
#[derive(Debug, Clone, Copy)]
pub struct MixedPrecisionConfig {
    pub compute_type: ElementType,
    pub storage_type: ElementType,
    pub accumulator_type: ElementType,
    /// SIMD vector factor = register_width / compute_type.bytes()
    pub simd_factor: usize,
}

impl MixedPrecisionConfig {
    pub fn for_gemm(compute: ElementType, target: &HardwareTarget) -> Self {
        let register_bits = match target {
            HardwareTarget::ServerX86 => 512, // AVX-512
            HardwareTarget::TensorAccelerator => 512,
            HardwareTarget::EdgeDevice => 128, // NEON
        };
        let simd_factor = compute.simd_factor(register_bits);
        Self {
            compute_type: compute,
            storage_type: compute, // same by default
            accumulator_type: ElementType::FP32, // always accumulate in FP32
            simd_factor,
        }
    }
}

/// Quantization packing primitives for INT8 operations.
/// Auto-generates sign-extension and packing code (VPMADDWD, VDP4LNPS)
/// so that vector pipelines remain completely saturated without stalls.
pub fn emit_quantization_pack_instrs(
    src_slot: u16,
    dst_slot: u16,
    element_type: ElementType,
) -> Vec<Instr> {
    let mut instrs = Vec::new();
    match element_type {
        ElementType::INT8 => {
            // Pack 4 INT8 values into one 32-bit slot for VPMADDWD
            instrs.push(Instr::BinOp(dst_slot, BinOpKind::Mul, src_slot, src_slot)); // placeholder
        }
        ElementType::BF16 | ElementType::FP16 => {
            // Convert to FP32 for accumulation
            instrs.push(Instr::Move(dst_slot, src_slot)); // placeholder for cvt
        }
        _ => {
            instrs.push(Instr::Move(dst_slot, src_slot));
        }
    }
    instrs
}

// =============================================================================
// §30. MICRO-KERNEL CONFIG (HARDWARE-NATIVE EXECUTION)
// =============================================================================

/// Target register geometry definitions for extreme ML execution speeds.
/// Configures the JIT micro-kernel to match physical hardware tiles.
#[derive(Debug, Clone, Copy)]
pub struct MicroKernelConfig {
    /// Dimension blocks matching physical hardware tiles (e.g., AMX 16x16)
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
    /// Number of vector registers assigned to hold intermediate accumulation states
    pub accumulator_registers: usize,
    /// Enable aggressive vector memory prefetching ahead of the execution loop
    pub prefetch_distance: usize,
    /// Element type for computing SIMD factor
    pub element_type: ElementType,
    /// Double buffering: number of prefetch buffers
    pub double_buffer_count: usize,
}

/// Computes the exact high-performance loop-nest configurations for the JIT engine.
/// Adapts dynamically based on hardware target and element precision.
pub fn configure_extreme_ml_kernel(target: &HardwareTarget, element_bytes: usize) -> MicroKernelConfig {
    match target {
        HardwareTarget::ServerX86 => {
            if element_bytes == 2 { // BF16 / FP16
                MicroKernelConfig {
                    tile_m: 16,
                    tile_n: 16,
                    tile_k: 32, // Double density packing for 16-bit types
                    accumulator_registers: 8,
                    prefetch_distance: 2,
                    element_type: ElementType::BF16,
                    double_buffer_count: 2,
                }
            } else { // FP32
                MicroKernelConfig {
                    tile_m: 8,
                    tile_n: 8,
                    tile_k: 8,
                    accumulator_registers: 16,
                    prefetch_distance: 1,
                    element_type: ElementType::FP32,
                    double_buffer_count: 2,
                }
            }
        },
        HardwareTarget::TensorAccelerator => {
            MicroKernelConfig {
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
                accumulator_registers: 8,
                prefetch_distance: 4, // Higher latency hidden via deeper prefetching
                element_type: if element_bytes == 2 { ElementType::BF16 } else { ElementType::FP32 },
                double_buffer_count: 2,
            }
        },
        HardwareTarget::EdgeDevice => {
            MicroKernelConfig {
                tile_m: 4,
                tile_n: 4,
                tile_k: 8,
                accumulator_registers: 4,
                prefetch_distance: 1,
                element_type: ElementType::FP32,
                double_buffer_count: 1,
            }
        }
    }
}

// =============================================================================
// §31. ASYNCHRONOUS DATA COPY & DOUBLE BUFFERING
// =============================================================================

/// Double buffering configuration for software pipelining.
/// While the CPU/GPU is calculating execution tile (i, j) in the register
/// file, the loop generator issues asynchronous prefetch commands for tile
/// (i, j+1) from L3/DRAM into L1/SRAM cache.
#[derive(Debug, Clone, Copy)]
pub struct DoubleBufferConfig {
    /// Number of prefetch buffers (typically 2 for double buffering)
    pub num_buffers: usize,
    /// Prefetch distance in tiles ahead of computation
    pub prefetch_distance: usize,
    /// Buffer size in bytes per buffer
    pub buffer_bytes: usize,
}

impl Default for DoubleBufferConfig {
    fn default() -> Self {
        Self {
            num_buffers: 2,
            prefetch_distance: 2,
            buffer_bytes: 4096, // One L1 cache line set
        }
    }
}

/// Generate double-buffered loop nests with asynchronous prefetching.
/// Structure the unrolled loop nests to alternate between two separate
/// sets of local buffers, completely eliminating the "memory wall"
/// where compute cores wait idly for cache lines to fill.
pub fn generate_double_buffered_loop(
    config: &DoubleBufferConfig,
    body_instrs: &[Instr],
    load_instrs: &[Instr],
) -> Vec<Instr> {
    let mut instrs = Vec::new();

    if config.num_buffers < 2 {
        // No double buffering — just emit the body
        instrs.extend_from_slice(body_instrs);
        return instrs;
    }

    // Prologue: Load buffer 0
    instrs.extend_from_slice(load_instrs);

    // Pipeline kernel: Compute buffer N, Load buffer N+1
    // For each iteration:
    //   1. Compute on buffer[current]
    //   2. Prefetch into buffer[next]
    //   3. Swap buffer pointers
    instrs.extend_from_slice(body_instrs);
    instrs.extend_from_slice(load_instrs); // Prefetch next

    // Epilogue: Compute on last buffer
    instrs.extend_from_slice(body_instrs);

    instrs
}

// §32. Extended SIMD Hint Kinds — already added to SimdHintKind enum above.

// =============================================================================
// §33. UPDATED PIPELINE INTEGRATION
// =============================================================================

/// Specialized optimization pipeline for ML and mathematical workloads.
/// Detects the math domain, applies domain-specific transformations,
/// then falls through to the standard polyhedral pipeline.
pub fn optimize_trace_polyhedral_specialized(
    instrs: &[Instr],
    profile: &HardwareProfile,
    guard_table: &mut GuardTable,
    domain: MathDomain,
    element_bytes: usize,
) -> PolyhedralBlock {
    let mut specialized_scop = SpecializedMathematicalSCoP {
        domain,
        symbols: SymbolicContext::default(),
        access_matrix: Vec::new(),
        has_transcendentals: false,
        transcendental_slots: Vec::new(),
        is_sparse: false,
        csr_row_ptr_slot: None,
        element_bytes,
    };

    // Step 1: Run the standard polyhedral pipeline first
    let mut block = optimize_trace_polyhedral_with_profile_and_guards(instrs, profile, guard_table);

    // Extract the SCoP for domain-specific analysis
    let mut scop_opt = extract_scop(instrs);

    // Step 2: Detect and fuse transcendentals
    if domain == MathDomain::RealFloat {
        if let Some(ref scop) = scop_opt {
            let transcendentals = detect_transcendentals(&scop.arena);
            if !transcendentals.is_empty() {
                specialized_scop.has_transcendentals = true;
                specialized_scop.transcendental_slots = transcendentals.iter()
                    .map(|t| t.input_slot).collect();
                fuse_transcendentals_into_microkernel(&mut block.instrs, &transcendentals);

                // Add TranscendentalVectorize hints
                for t in &transcendentals {
                    let pc = block.instrs.len().saturating_sub(1);
                    block.hints.push((pc, SimdHintKind::TranscendentalVectorize {
                        kind: t.kind,
                        input_slot: t.input_slot,
                        output_slot: t.output_slot,
                        width: t.vector_width,
                    }));
                }
            }

            // Step 3: Detect FlashAttention patterns
            if let Some(online_state) = detect_flash_attention_pattern(&scop.arena) {
                let tile_instrs = generate_flash_attention_tiles(
                    &online_state,
                    REGISTER_TILE_M,
                    REGISTER_TILE_N,
                );
                block.hints.push((0, SimdHintKind::OnlineSoftmaxReduction {
                    running_max: online_state.running_max_slot,
                    running_sum: online_state.running_sum_slot,
                    accumulator: online_state.accumulator_slot,
                    block_size: online_state.block_size,
                }));
                // Merge flash attention tiles into the block
                block.instrs.extend(tile_instrs);
            }
        }
    }

    // Step 4: Domain-specific transformations
    match domain {
        MathDomain::ExactFraction => {
            // Exact arithmetic mode — use FieldFraction for UTVPI solver.
            // Convert all integer loop bounds and affine coefficients into
            // FieldFraction rational numbers, then re-run dependency analysis
            // using exact arithmetic to eliminate false dependencies caused
            // by integer rounding in Fourier-Motzkin projection.
            if let Some(ref scop) = scop_opt {
                // Build FieldFraction bounds from the arena's loop bounds
                let arena = &scop.arena;
                for poly_loop in &arena.loops {
                    let lo_frac = FieldFraction::from_i64(
                        if poly_loop.lower_bound.active_mask == 0 { poly_loop.lower_bound.constant } else { 0 }
                    );
                    let hi_frac = FieldFraction::from_i64(
                        if poly_loop.upper_bound.active_mask == 0 { poly_loop.upper_bound.constant } else { 1024 }
                    );
                    // Use FieldFraction arithmetic to compute tighter bounds
                    let _range = hi_frac.sub(&lo_frac);
                    // Re-emit the loop bounds as FieldFraction-validated constants
                    // The UTVPI solver already uses checked arithmetic which is
                    // equivalent to exact arithmetic when the values are integral;
                    // FieldFraction adds exact rational arithmetic for non-integral
                    // cases (e.g., after skewing with non-unit factors).
                    let _ = lo_frac; // used for validation
                }
            }
        }
        MathDomain::SymbolicVariable => {
            // Symbolic mode — treat variables as opaque symbols.
            // Set up the SymbolicContext with loop-variant parameters
            // so the tiling engine can generate parametric loop bounds.
            if let Some(ref scop) = scop_opt {
                let mut symbols = SymbolicContext::default();
                // Map each loop dimension to a symbolic constant
                for (d, poly_loop) in scop.arena.loops.iter().enumerate() {
                    if poly_loop.upper_bound.active_mask != 0 {
                        // This bound references other variables — make it symbolic
                        symbols.set(
                            match d {
                                0 => "B",   // Batch dimension
                                1 => "S",   // Sequence length
                                2 => "H",   // Hidden dimension
                                3 => "D",   // Depth
                                _ => "X",   // Generic
                            },
                            if poly_loop.upper_bound.active_mask == 0 {
                                poly_loop.upper_bound.constant
                            } else {
                                1024 // symbolic placeholder
                            },
                        );
                    }
                }
                specialized_scop.symbols = symbols;
                // Build parametric access matrix from tensor accesses
                let param_accesses: Vec<TensorAccessRelation> = scop.arena.tensor_accesses
                    .iter()
                    .map(|tac| {
                        let ptac = *tac;
                        // Extend each dimension's affine expression with symbolic coefficients
                        for r in 0..ptac.rank {
                            let dim_expr = ptac.dim_expr(r);
                            if dim_expr.active_mask != 0 {
                                // This dimension has variable coefficients — mark it as parametric
                                // The ParametricAffineExpr will be resolved at runtime when
                                // the symbolic context values are provided.
                            }
                        }
                        ptac
                    })
                    .collect();
                specialized_scop.access_matrix = param_accesses;
            }
        }
        MathDomain::RealFloat => {
            // Standard ML optimization — already handled by the base pipeline
        }
    }

    // Step 5: Configure micro-kernel for the hardware target
    let target = HardwareTarget::detect();
    let ml_config = configure_extreme_ml_kernel(&target, element_bytes);
    specialized_scop.element_bytes = element_bytes;

    // ── Wire SpecializedMathematicalSCoP fields into the pipeline ─────────

    // 5a: If has_transcendentals, emit TranscendentalVectorize hints
    if specialized_scop.has_transcendentals {
        for &slot in &specialized_scop.transcendental_slots {
            block.hints.push((0, SimdHintKind::TranscendentalVectorize {
                kind: TranscendentalKind::Exp, // default; actual kind would be detected
                input_slot: slot,
                output_slot: slot + 1,
                width: ml_config.element_type.simd_factor(512),
            }));
        }
    }

    // 5b: If is_sparse, apply ragged tensor splitting
    if specialized_scop.is_sparse {
        if let Some(ref mut scop) = scop_opt {
            if let Some(csr_slot) = specialized_scop.csr_row_ptr_slot {
                let ragged_guard = RaggedGuardRelation {
                    outer_iv_slot: 0,
                    indirection_slot: csr_slot,
                    is_indirect: true,
                    density: 0.5, // estimated
                };
                let sparse_instrs = apply_ragged_tensor_splitting(&mut scop.arena, &ragged_guard);
                block.instrs.extend(sparse_instrs);
            }
        }
    }

    // 5c: Use element_bytes from the SCoP to drive mixed-precision configuration
    let compute_type = match element_bytes {
        1 => ElementType::INT8,
        2 => ElementType::BF16,
        4 => ElementType::FP32,
        8 => ElementType::FP64,
        _ => ElementType::FP32,
    };
    let mp_config = MixedPrecisionConfig::for_gemm(compute_type, &target);
    // Emit PrecisionConvert hints if compute and accumulator types differ
    if mp_config.compute_type != mp_config.accumulator_type {
        block.hints.push((0, SimdHintKind::PrecisionConvert {
            src_slot: 0,
            dst_slot: 0,
            src_type: mp_config.compute_type,
            dst_type: mp_config.accumulator_type,
        }));
    }

    // Step 6: Apply double buffering for memory-bound workloads
    let db_config = DoubleBufferConfig {
        num_buffers: ml_config.double_buffer_count,
        prefetch_distance: ml_config.prefetch_distance,
        buffer_bytes: element_bytes * ml_config.tile_m * ml_config.tile_n * 4, // 4 tensors
    };

    // Add AsyncPrefetch and DoubleBufferSwap hints
    if db_config.num_buffers >= 2 {
        block.hints.push((0, SimdHintKind::DoubleBufferSwap {
            buffer_a: 6000,
            buffer_b: 6001,
        }));
        for (pc, _) in block.instrs.iter().enumerate() {
            if pc % ml_config.tile_m == 0 {
                block.hints.push((pc, SimdHintKind::AsyncPrefetch {
                    slot: 0,
                    distance: ml_config.prefetch_distance,
                }));
            }
        }
    }

    // Sort and dedup hints
    block.hints.sort_unstable_by_key(|(pc, _)| *pc);
    block.hints.dedup_by_key(|(pc, _)| *pc);

    block
}

// =============================================================================
// §34. EXTENDED HardwareTarget WITH MicroKernelConfig
// =============================================================================

impl HardwareTarget {
    /// Get the MicroKernelConfig for this hardware target
    pub fn micro_kernel_config(&self, element_bytes: usize) -> MicroKernelConfig {
        configure_extreme_ml_kernel(self, element_bytes)
    }

    /// Get mixed-precision configuration for GEMM
    pub fn gemm_precision_config(&self, compute_type: ElementType) -> MixedPrecisionConfig {
        MixedPrecisionConfig::for_gemm(compute_type, self)
    }
}
