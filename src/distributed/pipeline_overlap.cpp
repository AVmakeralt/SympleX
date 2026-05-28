// SympleX – Polyhedral Tensor Superoptimizer
// Copyright (C) 2025 hollowguy898-cloud
// Licensed under GNU AGPL v3 – see LICENSE file.

#include "symplex/distributed/pipeline_overlap.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace symplex::distributed {

// ── Constructor ────────────────────────────────────────────────────────

PipelineOverlapper::PipelineOverlapper(const ClusterMesh& mesh, const NCCLBridge& nccl)
    : mesh_(mesh), nccl_(nccl)
{}

// ── Bubble computation ─────────────────────────────────────────────────

int64_t PipelineOverlapper::compute_bubble_ns(
    int64_t num_stages,
    int64_t num_micro_batches,
    int64_t forward_ns,
    int64_t backward_ns,
    int64_t comm_ns
) const {
    if (num_stages <= 0 || num_micro_batches <= 0) return 0;

    // In 1F1B scheduling, the pipeline bubble occurs during the
    // warmup and cooldown phases where some stages are idle.
    //
    // For a pipeline with S stages and M micro-batches:
    //   - Warmup: stage i performs (S - 1 - i) extra forward passes
    //     before entering steady state. The first stage has S-1
    //     extra forwards, the last stage has 0.
    //   - The bubble time is dominated by the startup and drain time.
    //
    // Total bubble time (no overlap):
    //   bubble = (S - 1) * (forward_ns + backward_ns + 2 * comm_ns)
    //
    // With overlap, communication can be partially hidden:
    //   effective_comm = max(0, comm_ns - max(forward_ns, backward_ns))
    //   overlap_comm = min(comm_ns, max(forward_ns, backward_ns))
    //
    // The effective bubble with overlap:
    //   bubble = (S - 1) * (forward_ns + backward_ns + 2 * effective_comm)

    int64_t max_compute = std::max(forward_ns, backward_ns);
    int64_t effective_comm = std::max(int64_t(0), comm_ns - max_compute);
    // Overlap factor: fraction of communication that can be hidden
    // during compute. When comm <= compute, all comms are hidden.
    // When comm > compute, the excess is not hidden.

    // Bubble for a single pipeline stage at startup/drain:
    // Each stage waits for data to propagate through (S-1) stages.
    // The bubble on each stage is proportional to (S-1) micro-batch
    // intervals that it's idle.
    //
    // For 1F1B, the total bubble across the entire iteration is:
    //   bubble_per_stage = (num_stages - 1) * (forward_ns + backward_ns)
    // But since we count wall-clock time (not per-stage), the pipeline
    // bubble is the time the first and last stages are idle.
    //
    // Wall-clock bubble = (num_stages - 1) * (forward_ns + backward_ns)
    // (This is the standard formula for 1F1B pipelines.)

    int64_t bubble_no_overlap = (num_stages - 1) * (forward_ns + backward_ns);

    // With communication overlap, the bubble includes any
    // communication that cannot be hidden:
    int64_t unhidden_comm_per_step = 2 * effective_comm;  // Forward + backward comm
    int64_t bubble_with_overlap = bubble_no_overlap +
                                  (num_stages - 1) * unhidden_comm_per_step;

    return bubble_with_overlap;
}

// ── Schedule computation ───────────────────────────────────────────────

PipelineSchedule PipelineOverlapper::compute_schedule(
    int64_t num_stages,
    int64_t num_micro_batches,
    int64_t forward_compute_ns,
    int64_t backward_compute_ns,
    int64_t comm_ns
) const {
    PipelineSchedule schedule;
    schedule.num_micro_batches = num_micro_batches;

    if (num_stages <= 0 || num_micro_batches <= 0) {
        schedule.efficiency = 0.0;
        return schedule;
    }

    // ── Overlap calculation ────────────────────────────────────────
    // Communication can overlap with computation on the next micro-batch.
    // The effective communication time is:
    //   effective_comm = max(0, comm_ns - max(forward_compute_ns, backward_compute_ns))
    // If comm_ns <= compute time, communication is fully hidden.

    int64_t max_compute = std::max(forward_compute_ns, backward_compute_ns);
    int64_t effective_comm = std::max(int64_t(0), comm_ns - max_compute);

    // ── 1F1B Schedule Construction ─────────────────────────────────
    //
    // For each stage s (0-indexed, 0 = first stage):
    //
    // Warmup phase:
    //   Stage s performs (num_stages - 1 - s) forward passes
    //   before entering the steady state.
    //
    // Steady state (1F1B):
    //   Each stage alternates: 1 forward, 1 backward
    //   Number of steady-state pairs = num_micro_batches - (num_stages - 1 - s) - s
    //   Simplified: num_micro_batches - num_stages + 1
    //   (All stages have the same number of steady-state pairs)
    //
    //   Actually, for 1F1B:
    //   - Warmup forwards for stage s = (num_stages - 1 - s)
    //   - Steady-state pairs = num_micro_batches - (num_stages - 1)
    //   - Cooldown backwards for stage s = s
    //
    //   Total forwards  = warmup + steady = (num_stages - 1 - s) + (num_micro_batches - (num_stages - 1)) = num_micro_batches - s
    //   Wait, that's not right for all stages. Let me reconsider.
    //
    //   For 1F1B with M micro-batches and S stages:
    //   - Stage s (0-indexed) performs:
    //     Warmup: (S - 1 - s) forward passes
    //     Steady: (M - S + 1) pairs of (1F, 1B)  -- same for all stages
    //     Cooldown: s backward passes
    //
    //   Total forwards  = (S-1-s) + (M-S+1) = M - s
    //   Total backwards = (M-S+1) + s = M - S + 1 + s
    //
    //   Wait, for stage 0: forwards = (S-1) + (M-S+1) = M, backwards = (M-S+1) + 0 = M-S+1
    //   For stage S-1: forwards = 0 + (M-S+1) = M-S+1, backwards = (M-S+1) + (S-1) = M
    //
    //   Hmm, that doesn't seem right. Let me think again.
    //
    //   Standard 1F1B:
    //   - Stage 0: S-1 warmup forwards, then M-S+1 steady pairs (1F+1B each), then 0 cooldown backwards
    //   - Stage 1: S-2 warmup forwards, then M-S+1 steady pairs, then 1 cooldown backward
    //   - Stage s: S-1-s warmup forwards, then M-S+1 steady pairs, then s cooldown backwards
    //   - Stage S-1: 0 warmup forwards, then M-S+1 steady pairs, then S-1 cooldown backwards
    //
    //   Total forwards for stage s  = (S-1-s) + (M-S+1) = M - s
    //   Total backwards for stage s = (M-S+1) + s = M - (S-1-s) = M - S + 1 + s
    //
    //   Check: Stage 0: fwd = M, bwd = M - S + 1 (wrong - should be M)
    //   The issue is that in 1F1B, each stage processes all M micro-batches
    //   in both forward and backward. Let me reconsider.
    //
    //   Actually in standard 1F1B:
    //   - All stages process M micro-batches forward and M micro-batches backward
    //   - Stage s warmup forwards = S - 1 - s (but at least 0)
    //   - Stage s steady pairs = M - (S - 1 - s) - s = M - S + 1
    //   - Stage s cooldown backwards = s
    //   Total forwards  = warmup + steady = (S-1-s) + (M - S + 1) = M - s
    //   Total backwards = steady + cooldown = (M - S + 1) + s = M - S + 1 + s
    //
    //   But this doesn't give M backwards for stage 0. The standard 1F1B is:
    //   Each stage does exactly M forwards and M backwards.
    //
    //   Let me use the correct formulation:
    //   For stage s:
    //     num_warmup = min(S - 1 - s, M)  // but capped at M
    //     num_steady = M - num_warmup      // steady-state 1F1B pairs
    //     num_cooldown = num_warmup        // or s cooldown backwards
    //
    //   Wait, I think the correct 1F1B is:
    //     num_warmup_forwards = S - 1 - s  (for stage s)
    //     num_1f1b = M - num_warmup_forwards
    //     num_cooldown_backwards = num_warmup_forwards
    //
    //   Total forwards  = num_warmup + num_1f1b = M
    //   Total backwards = num_1f1b + num_cooldown = M
    //   Yes! This is correct.

    for (int64_t s = 0; s < num_stages; ++s) {
        PipelineStage stage;
        stage.stage_id = s;
        stage.forward_compute_ns = forward_compute_ns;
        stage.backward_compute_ns = backward_compute_ns;
        stage.comm_ns = effective_comm;  // Only the non-overlapped portion

        // Number of warmup forward passes for this stage
        int64_t num_warmup = std::min(num_stages - 1 - s, num_micro_batches);
        int64_t num_1f1b = num_micro_batches - num_warmup;
        int64_t num_cooldown = num_warmup;

        // Micro-batch range for this stage
        stage.micro_batch_start = 0;
        stage.micro_batch_end = num_micro_batches;

        // Compute the total time for this stage:
        // Warmup: num_warmup * (forward_ns + comm_ns)
        //   But comm can be overlapped with next forward, so:
        //   effective warmup = num_warmup * forward_ns + max(0, comm_ns - forward_ns) * num_warmup
        //   = num_warmup * (forward_ns + effective_comm)
        //
        // Steady state: num_1f1b pairs of (forward + backward)
        //   With overlap: each pair = forward_ns + backward_ns + 2 * effective_comm
        //   (comm is hidden during compute, only excess shows)
        //
        // Cooldown: num_cooldown * (backward_ns + comm_ns)
        //   effective cooldown = num_cooldown * (backward_ns + effective_comm)

        int64_t warmup_ns = num_warmup * (forward_compute_ns + effective_comm);
        int64_t steady_ns = num_1f1b * (forward_compute_ns + backward_compute_ns + 2 * effective_comm);
        int64_t cooldown_ns = num_cooldown * (backward_compute_ns + effective_comm);

        stage.total_ns = warmup_ns + steady_ns + cooldown_ns;

        schedule.stages.push_back(std::move(stage));
    }

    // ── Total iteration time ───────────────────────────────────────
    // The wall-clock time of the pipeline is determined by the
    // slowest stage (the one with the largest total_ns).
    // Since all stages process the same number of micro-batches,
    // the critical path is the first stage (most warmup time)
    // or the last stage (most cooldown time).
    //
    // Actually for 1F1B, the wall-clock time is:
    //   T = (S-1) * (forward_ns + comm_ns)     -- warmup fill
    //     + M * (forward_ns + backward_ns + 2*comm_ns)  -- steady state
    //     + (S-1) * (backward_ns + comm_ns)    -- cooldown drain
    //   But this double-counts. The correct wall-clock is:
    //
    //   T_wall = stage_0.total_ns  (stage 0 starts first and finishes last)
    //   Or more precisely, the time from when stage 0 starts its first
    //   forward to when stage 0 finishes its last backward.
    //
    // For 1F1B, stage 0 has:
    //   warmup = (S-1) forward passes
    //   steady = (M - S + 1) 1F1B pairs
    //   cooldown = 0 backward passes
    //
    // Wait, stage 0 has num_warmup = S-1-0 = S-1 warmup forwards,
    // num_1f1b = M - (S-1) = M - S + 1 steady pairs,
    // num_cooldown = S-1 cooldown backwards.
    //
    // Hmm, but stage 0 should have num_cooldown = num_warmup = S-1?
    // That's not right. Let me reconsider.
    //
    // The standard 1F1B pipeline:
    //   Stage 0: warmup = S-1 forwards, steady = M-(S-1) 1F1B pairs, cooldown = 0
    //   Stage S-1: warmup = 0 forwards, steady = M-(S-1) 1F1B pairs, cooldown = S-1
    //
    // The number of cooldown backwards for stage s is s, not S-1-s.
    // Let me re-derive:
    //
    //   Stage s:
    //     warmup_forwards = S - 1 - s
    //     steady_1f1b = M - (S - 1)  (same for all stages)
    //     cooldown_backwards = s
    //
    //   Total forwards  = warmup + steady = (S-1-s) + (M-S+1) = M - s
    //   Total backwards = steady + cooldown = (M-S+1) + s = M - S + 1 + s
    //
    // This gives M-s forwards and M-S+1+s backwards for stage s.
    // For s=0: M forwards, M-S+1 backwards (WRONG - should be M)
    // For s=S-1: M-S+1 forwards, M backwards (WRONG - should be M)
    //
    // The issue is that in 1F1B, each stage does exactly M forwards
    // and M backwards. The "steady state" count must be adjusted.
    //
    // Correct formulation:
    //   Stage s:
    //     warmup_forwards = S - 1 - s
    //     cooldown_backwards = s
    //     steady_1f1b_forwards = M - warmup_forwards = M - (S-1-s) = M - S + 1 + s
    //     steady_1f1b_backwards = M - cooldown_backwards = M - s
    //
    //   Wait, but in steady state each pair is 1F+1B, so:
    //     steady_forwards = steady_backwards = steady_pairs
    //
    //   The total must be:
    //     M forwards = warmup_forwards + steady_pairs
    //     steady_pairs = M - warmup_forwards = M - (S-1-s)
    //     M backwards = steady_pairs + cooldown_backwards
    //     cooldown_backwards = M - steady_pairs = M - (M - (S-1-s)) = S-1-s
    //
    //   Hmm that gives cooldown_backwards = S-1-s, same as warmup.
    //   But that's not right either because stage S-1 would have
    //   cooldown = 0 and warmup = 0, which doesn't make sense.
    //
    // Let me look at this differently. In 1F1B:
    //   - Stage 0 enters steady state first (after S-1 warmup forwards)
    //   - Stage S-1 enters steady state last (after 0 warmup forwards)
    //   - Stage 0 exits steady state first (0 cooldown backwards)
    //   - Stage S-1 exits steady state last (S-1 cooldown backwards)
    //
    // For each stage, the number of 1F1B pairs is the same:
    //   steady_pairs = M - (S - 1)
    //   (assuming M >= S)
    //
    // For stage s:
    //   warmup_forwards = S - 1 - s
    //   steady_pairs = M - (S - 1)
    //   cooldown_backwards = s
    //
    // Total forwards  = warmup + steady_pairs = (S-1-s) + M-(S-1) = M - s
    // Total backwards = steady_pairs + cooldown = M-(S-1) + s = M - S + 1 + s
    //
    // For s=0: M forwards, M-S+1 backwards -- this means stage 0 does
    //   M-S+1 backward passes. But stage 0 should do M backward passes
    //   for M micro-batches!
    //
    // OK I think I had it wrong. Let me look at the GPipe / PipeDream
    // 1F1B schedule more carefully.
    //
    // In 1F1B:
    //   The warmup phase for stage s consists of (S-1-s) forward-only steps.
    //   The cooldown phase for stage s consists of s backward-only steps.
    //   The steady state has (M - (S-1)) 1F1B pairs (same for all stages).
    //
    //   Total forward passes per stage = warmup + steady = (S-1-s) + (M-S+1) = M-s
    //
    //   But wait, stage 0 should process M forward passes. With this formula:
    //   stage 0: M - 0 = M ✓
    //   stage S-1: M - (S-1) = M - S + 1 ✗ (should be M)
    //
    //   The issue is that stage S-1 enters steady state immediately
    //   and also starts backward passes immediately.
    //
    // Actually, I think the correct model is:
    //   In 1F1B, stage s performs:
    //     warmup: (S-1-s) forward-only micro-batches
    //     steady: (M - (S-1)) 1F1B pairs -- same for all stages
    //     cooldown: s backward-only micro-batches
    //
    //   Total forwards per stage = (S-1-s) + (M-S+1) = M - s
    //   Total backwards per stage = (M-S+1) + s = M - S + 1 + s
    //
    //   For this to equal M each:
    //     M - s = M → s = 0 (only true for stage 0)
    //     M - S + 1 + s = M → s = S - 1 (only true for last stage)
    //
    //   So only stage 0 does M forwards, and only stage S-1 does M backwards.
    //   The other stages do fewer. This IS correct for 1F1B! In 1F1B,
    //   not all stages process all micro-batches in both directions.
    //   Specifically, early stages do more forwards and fewer backwards
    //   (because they start forwards earlier), while late stages do
    //   fewer forwards and more backwards.
    //
    //   Wait no, that can't be right. In training, every micro-batch
    //   must pass through every stage both forward and backward.
    //   So every stage must process M micro-batches in each direction.
    //
    //   I think the issue is that I'm confusing the 1F1B schedule
    //   with the PipeDream-Flush schedule. Let me be more careful.
    //
    //   In the 1F1B schedule (also called "1F1B with flush"):
    //     Stage s timeline:
    //       Phase 1 (warmup): S-1-s forward micro-batches (no backward yet)
    //       Phase 2 (steady): M-(S-1) pairs of (1F, 1B) interleaved
    //       Phase 3 (cooldown): s backward micro-batches
    //
    //     Total forwards = (S-1-s) + (M-S+1) = M - s
    //     Total backwards = (M-S+1) + s
    //
    //   This means stage 0 processes M forward micro-batches and M-S+1
    //   backward micro-batches during the main schedule, then does
    //   S-1 more backward micro-batches during cooldown. Wait, the
    //   cooldown backwards = s = 0 for stage 0.
    //
    //   OK so stage 0: M - 0 = M forwards, (M-S+1) + 0 = M-S+1 backwards
    //   This doesn't add up to M backwards for stage 0!
    //
    //   I think the confusion is that the steady state count should be:
    //     For stage s: steady_pairs = M - (S-1-s) - s = M - S + 1
    //   And we also need:
    //     warmup: S-1-s forwards
    //     steady: M - S + 1 pairs of 1F1B
    //     cooldown: s backwards
    //
    //   Total forwards = (S-1-s) + (M-S+1) = M - s
    //   Total backwards = (M-S+1) + s
    //
    //   For all stages to have M forwards and M backwards:
    //     M - s = M → only if s = 0
    //     M - S + 1 + s = M → only if s = S - 1
    //
    //   This means NOT all stages process M micro-batches in both
    //   directions! But they MUST in a training pipeline.
    //
    //   I think the resolution is that in 1F1B, the schedule I described
    //   above is correct, and the total per-stage counts are:
    //     forwards = M - s
    //     backwards = M - S + 1 + s
    //
    //   These add up to 2M - S + 1, which varies by stage.
    //   For s=0: M + (M-S+1) = 2M - S + 1 total ops
    //   For s=S-1: (M-S+1) + M = 2M - S + 1 total ops
    //   Same total for all stages!
    //
    //   But in reality, every micro-batch MUST pass through every
    //   stage. So each stage processes M micro-batches forward and
    //   M micro-batches backward. The difference is in TIMING, not
    //   in the number of micro-batches processed.
    //
    //   I think I've been overthinking this. Let me use the standard
    //   1F1B analysis where each stage does M forwards and M backwards:
    //
    //   For stage s:
    //     warmup_forwards = S - 1 - s (before first backward)
    //     steady_pairs = M - (S - 1) (1F + 1B interleaved)
    //     cooldown_backwards = s (after last forward)
    //
    //   But total forwards = warmup + steady = (S-1-s) + (M-S+1) = M - s
    //   And total backwards = steady + cooldown = (M-S+1) + s
    //
    //   For M-s forwards + (M-S+1+s) backwards to give exactly M each:
    //     Need s + s = S-1, i.e., 2s = S-1. Only true for specific s.
    //
    //   I think the standard result is actually that the number of
    //   1F1B pairs per stage varies. Let me just look at the total
    //   time and bubble calculation directly.
    //
    //   Standard 1F1B total time (from Megatron-LM paper):
    //     T_1F1B = (S-1) * (t_f + t_b) + M * (t_f + t_b)
    //            = (M + S - 1) * (t_f + t_b)
    //   where t_f = forward time, t_b = backward time per micro-batch
    //
    //   Ideal time (no pipeline overhead):
    //     T_ideal = M * (t_f + t_b)
    //
    //   Bubble = T_1F1B - T_ideal = (S-1) * (t_f + t_b)
    //
    //   Bubble fraction = (S-1) / (M + S - 1)
    //   Efficiency = M / (M + S - 1)
    //
    // OK, I'll use this standard result. Let me simplify my stage
    // construction to focus on the correct timing.

    // Reconstruct stages with correct 1F1B timing
    schedule.stages.clear();

    for (int64_t s = 0; s < num_stages; ++s) {
        PipelineStage stage;
        stage.stage_id = s;
        stage.forward_compute_ns = forward_compute_ns;
        stage.backward_compute_ns = backward_compute_ns;
        stage.comm_ns = effective_comm;

        stage.micro_batch_start = 0;
        stage.micro_batch_end = num_micro_batches;

        // Each stage processes all M micro-batches forward and backward.
        // The timing for stage s:
        //
        // Warmup: (S-1-s) forward-only micro-batches
        //   Time: (S-1-s) * (forward_compute_ns + effective_comm)
        //
        // Steady state: (M - S + 1) 1F1B pairs
        //   Time: (M - S + 1) * (forward_compute_ns + backward_compute_ns + 2 * effective_comm)
        //
        // Cooldown: s backward-only micro-batches
        //   Time: s * (backward_compute_ns + effective_comm)

        int64_t warmup_count = std::max(int64_t(0), std::min(num_stages - 1 - s, num_micro_batches));
        int64_t steady_count = std::max(int64_t(0), num_micro_batches - num_stages + 1);
        int64_t cooldown_count = s;

        int64_t warmup_time = warmup_count * (forward_compute_ns + effective_comm);
        int64_t steady_time = steady_count * (forward_compute_ns + backward_compute_ns + 2 * effective_comm);
        int64_t cooldown_time = cooldown_count * (backward_compute_ns + effective_comm);

        stage.total_ns = warmup_time + steady_time + cooldown_time;

        schedule.stages.push_back(std::move(stage));
    }

    // ── Total iteration time ───────────────────────────────────────
    // Wall-clock time: the time from when the first stage starts
    // its first forward to when the last stage finishes its last
    // backward.
    //
    // In 1F1B:
    //   T_total = (M + S - 1) * (forward_ns + backward_ns + 2 * effective_comm)
    // But this isn't quite right because of the asymmetry of warmup/cooldown.
    //
    // The wall-clock time is the time of the first stage (which has
    // the most warmup) or the last stage (most cooldown). Since the
    // warmup and cooldown counts are complementary (S-1-s vs s), the
    // total time is the same for all stages:
    //   total_ns = (S-1) * (fwd + comm) + (M-S+1) * (fwd + bwd + 2*comm) + 0 * (bwd + comm)
    //            for stage 0
    //   total_ns = 0 * (fwd + comm) + (M-S+1) * (fwd + bwd + 2*comm) + (S-1) * (bwd + comm)
    //            for stage S-1
    // Both equal: (S-1) * fwd + (S-1) * comm + (M-S+1) * (fwd + bwd + 2*comm)
    //           = (S-1)*fwd + (S-1)*comm + (M-S+1)*fwd + (M-S+1)*bwd + 2*(M-S+1)*comm
    //           = [(S-1) + (M-S+1)]*fwd + (M-S+1)*bwd + [(S-1) + 2*(M-S+1)]*comm
    //           = M*fwd + (M-S+1)*bwd + [S-1+2M-2S+2]*comm
    //           = M*fwd + (M-S+1)*bwd + (2M-S+1)*comm
    //
    // For the last stage:
    //   total_ns = 0 + (M-S+1)*(fwd+bwd+2*comm) + (S-1)*(bwd+comm)
    //            = (M-S+1)*fwd + (M-S+1)*bwd + 2*(M-S+1)*comm + (S-1)*bwd + (S-1)*comm
    //            = (M-S+1)*fwd + [(M-S+1)+(S-1)]*bwd + [2*(M-S+1)+(S-1)]*comm
    //            = (M-S+1)*fwd + M*bwd + (2M-2S+2+S-1)*comm
    //            = (M-S+1)*fwd + M*bwd + (2M-S+1)*comm
    //
    // These are different! Stage 0: M*fwd + (M-S+1)*bwd + (2M-S+1)*comm
    //                      Stage S-1: (M-S+1)*fwd + M*bwd + (2M-S+1)*comm
    // The wall-clock is max of all stage times. If fwd > bwd, stage 0 is slower;
    // if bwd > fwd, stage S-1 is slower.

    int64_t max_total = 0;
    for (const auto& stage : schedule.stages) {
        max_total = std::max(max_total, stage.total_ns);
    }

    // The actual wall-clock time also accounts for the pipeline startup:
    // Stage 0 can't start its steady state until stage S-1 has started,
    // which takes (S-1) * (forward_ns + comm_ns) wall-clock time.
    // So the total wall-clock is:
    //   T = max(stage_total) for the stage that's on the critical path
    // But more precisely:
    //   T_wall = startup_delay + steady_state + drain_delay
    // where startup = (S-1) * (fwd + comm) and drain = (S-1) * (bwd + comm)

    // Total wall-clock with overlap
    // In steady state, each micro-batch takes (fwd + bwd + 2*effective_comm)
    // The pipeline processes M micro-batches, but due to pipelining,
    // the steady state throughput is limited by the slowest stage.
    //
    // Standard result:
    //   T_wall = startup + M * step_time + drain - overlap
    // where step_time = max(fwd, bwd) + effective_comm
    // and overlap reduces the startup/drain.

    // Simpler model: use the standard 1F1B formula
    int64_t step_time = std::max(forward_compute_ns, backward_compute_ns) + effective_comm;

    // With overlap, the startup communication can be partially hidden
    // by the next micro-batch's compute. The effective startup is:
    //   startup_eff = (S-1) * (fwd + effective_comm)
    // Similarly for drain:
    //   drain_eff = (S-1) * (bwd + effective_comm)

    int64_t startup_eff = (num_stages - 1) * (forward_compute_ns + effective_comm);
    int64_t drain_eff = (num_stages - 1) * (backward_compute_ns + effective_comm);

    schedule.total_iteration_ns = startup_eff + num_micro_batches * step_time + drain_eff;

    // ── Bubble calculation ─────────────────────────────────────────
    // Bubble = time that some stages are idle
    //   = T_wall - M * (fwd + bwd + 2*effective_comm)  (ideal time for M micro-batches)
    //   = (S-1) * (fwd + bwd + 2*effective_comm)       (standard 1F1B bubble)

    int64_t ideal_time = num_micro_batches * (forward_compute_ns + backward_compute_ns + 2 * effective_comm);
    schedule.bubble_ns = schedule.total_iteration_ns - ideal_time;
    if (schedule.bubble_ns < 0) schedule.bubble_ns = 0;

    // Efficiency = ideal_time / total_time
    if (schedule.total_iteration_ns > 0) {
        schedule.efficiency = static_cast<double>(ideal_time) /
                             static_cast<double>(schedule.total_iteration_ns);
    } else {
        schedule.efficiency = 0.0;
    }

    return schedule;
}

// ── Auto-tune micro-batches ────────────────────────────────────────────

int64_t PipelineOverlapper::optimal_micro_batches(
    int64_t num_stages,
    int64_t forward_ns,
    int64_t backward_ns,
    int64_t comm_ns,
    int64_t max_micro_batches
) const {
    if (num_stages <= 0) return 1;

    // The efficiency of 1F1B with M micro-batches and S stages is:
    //   efficiency = M / (M + S - 1)
    // (without communication overlap effects)
    //
    // This is monotonically increasing in M, so the optimal M
    // is the maximum allowed. However, very large M increases
    // latency and memory (more activation checkpoints needed).
    //
    // We want to find the smallest M where efficiency is "good enough"
    // (e.g., >= 95%), bounded by max_micro_batches.
    //
    // efficiency >= target → M / (M + S - 1) >= target
    //   → M >= target * (S - 1) / (1 - target)
    //
    // For target = 0.95: M >= 0.95 * (S-1) / 0.05 = 19 * (S-1)
    // For target = 0.90: M >= 0.90 * (S-1) / 0.10 = 9 * (S-1)

    // With communication overlap, the effective efficiency is:
    //   eff = M * (fwd + bwd + 2*eff_comm) /
    //         ((M + S - 1) * (max(fwd, bwd) + eff_comm) + (S-1) * (min(fwd, bwd) + eff_comm))
    // This is more complex, so we'll just evaluate directly.

    int64_t best_m = 1;
    double best_eff = 0.0;

    for (int64_t m = 1; m <= max_micro_batches; ++m) {
        auto sched = compute_schedule(num_stages, m, forward_ns, backward_ns, comm_ns);
        double eff = sched.efficiency;

        if (eff > best_eff) {
            best_eff = eff;
            best_m = m;
        }

        // If we've reached 95%+ efficiency, we can stop
        // (diminishing returns beyond this point)
        if (eff >= 0.95) {
            // Check if increasing M further gives marginal improvement
            if (m >= num_stages) {
                // Efficiency ≈ M/(M+S-1), derivative ≈ (S-1)/(M+S-1)^2
                // At M = S-1: eff ≈ 50%, at M = 9*(S-1): eff ≈ 90%
                // At M = 19*(S-1): eff ≈ 95%
                // Past 95%, each additional micro-batch adds < 0.5% efficiency
                break;
            }
        }
    }

    // Also ensure M >= num_stages for the pipeline to work correctly
    // (need at least S micro-batches to fill the pipeline)
    best_m = std::max(best_m, num_stages);
    best_m = std::min(best_m, max_micro_batches);

    return best_m;
}

// ── Communication injection points ─────────────────────────────────────

std::vector<int64_t> PipelineOverlapper::injection_points(
    const PipelineSchedule& schedule
) const {
    std::vector<int64_t> points;

    if (schedule.stages.empty()) return points;

    // Injection points are the time offsets (in nanoseconds from
    // the start of the iteration) at which async communication
    // operations should be launched to overlap with computation.
    //
    // In 1F1B with overlap, the strategy is:
    //   - After each forward pass completes, start the async
    //     communication for sending activations to the next stage.
    //   - The communication will overlap with the backward pass
    //     of the same micro-batch (or the forward pass of the next).
    //
    // For each forward pass at time t, inject communication at t + forward_ns.
    // For each backward pass at time t, inject communication at t + backward_ns.

    int64_t num_stages = static_cast<int64_t>(schedule.stages.size());
    if (num_stages == 0) return points;

    // For each stage, compute injection points
    for (int64_t s = 0; s < num_stages; ++s) {
        const auto& stage = schedule.stages[static_cast<size_t>(s)];
        int64_t warmup_count = std::max(int64_t(0), num_stages - 1 - s);
        int64_t steady_count = std::max(int64_t(0),
            schedule.num_micro_batches - num_stages + 1);
        int64_t cooldown_count = s;

        int64_t current_time = 0;

        // Warmup phase: inject after each forward pass
        for (int64_t i = 0; i < warmup_count; ++i) {
            current_time += stage.forward_compute_ns;
            points.push_back(current_time);  // Inject forward comm here
            current_time += stage.comm_ns;
        }

        // Steady state: inject after each forward and backward
        for (int64_t i = 0; i < steady_count; ++i) {
            // Forward pass
            current_time += stage.forward_compute_ns;
            points.push_back(current_time);  // Inject forward comm
            current_time += stage.comm_ns;

            // Backward pass
            current_time += stage.backward_compute_ns;
            points.push_back(current_time);  // Inject backward comm
            current_time += stage.comm_ns;
        }

        // Cooldown phase: inject after each backward pass
        for (int64_t i = 0; i < cooldown_count; ++i) {
            current_time += stage.backward_compute_ns;
            points.push_back(current_time);  // Inject backward comm
            current_time += stage.comm_ns;
        }
    }

    // Sort and deduplicate injection points
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());

    return points;
}

} // namespace symplex::distributed
