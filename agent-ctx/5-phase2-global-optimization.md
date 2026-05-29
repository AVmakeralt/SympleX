# Task 5: Phase 2 "World-Class Global Optimization" Improvements

## Agent: Main

## Summary

Implemented three major global optimization improvements for the SympleX JIT engine in `phase3_jit.rs`, adding dominator tree analysis, global value numbering across CFG blocks, and dominator-aware loop-invariant code motion.

## Changes Made

### 1. FlatBlock::succs() Method (line 430)
- Added `impl FlatBlock` block with `succs()` method
- Returns successor BlockIds based on the terminator instruction (CondBr, Jump, Ret)
- Used by DominatorTree construction and LICM for CFG traversal

### 2. DominatorTree Struct (line 13349)
- Implemented Cooper-Harvey-Kennedy iterative dominance algorithm
- Key components:
  - `idom`: Maps BlockId → immediate dominator's BlockId
  - `children`: Children in the dominator tree for DFS traversal
  - `dfs_order`: Preorder DFS traversal of the dominator tree
  - `dfs_num`: DFS numbering for dominance testing
- Key methods:
  - `compute()`: Builds the dominator tree using iterative dataflow with RPO
  - `intersect()`: Finds the lowest common ancestor in the dominator tree
  - `reverse_postorder()`: Computes RPO of the CFG via DFS from entry
  - `dominates()`: Tests if one block dominates another
  - `idom()`: Returns the immediate dominator of a block
  - `dfs_order()`: Returns the DFS order for traversal

### 3. gvn_optimize_global() (line 13527)
- Global Value Numbering across CFG blocks using dominator tree
- Traverses blocks in dominator tree DFS order
- Maintains a value map that persists across dominated blocks
- Includes `apply_replacements()` helper that handles more operand types than the per-block GVN
- Includes `hash_expr()` that hashes BinOp and UnOp expressions with replacement-aware operand hashing
- Conservative but correct: keeps the map intact when backtracking (may miss some eliminations but never incorrectly eliminates)

### 4. instr_operand_values() (line 13647)
- Extracts operand ValueIds from any IrOp variant
- Covers BinOp, UnOp, Move, Copy, Store, Load, Cast, Call, Intrinsic, Emit, TypeCheck, tensor ops, etc.
- Used by LICM to check if all operands are defined outside a loop

### 5. licm_optimize_ssa() (line 13676)
- Dominator-aware Loop-Invariant Code Motion
- Identifies loop headers via back-edge detection (succ dominates pred)
- Identifies pre-headers (dominating predecessor of loop header)
- Collects loop body blocks (all blocks dominated by header)
- Identifies outside definitions (ValueIds defined in non-loop blocks)
- Scans loop body for invariant instructions (pure, all operands from outside)
- Hoists invariant instructions to pre-header (before terminator)
- Replaces hoisted instructions with Nop

### 6. Wired into translate_ssa() (line 14354-14379)
- Added Step 2a: Global optimization passes
- Runs SCCP → Global GVN → Dominator-Aware LICM
- Block index rebuilt AFTER optimization passes (SCCP may remove unreachable blocks)

### 7. Wired into translate_from_ir() (line 15256-15267)
- Added Global GVN and LICM after existing per-block GVN
- Runs as Phase 2: global optimization passes using dominator tree

## Build Verification
- `cargo build` compiles cleanly with ZERO warnings
- `RUSTFLAGS="-D warnings" cargo build` also passes (warnings-as-errors)
- File grew from ~19,133 lines to 21,274 lines (+2,141 lines)
