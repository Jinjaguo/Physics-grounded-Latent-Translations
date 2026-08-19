# EXP_G35 Report: Shared Multi-Action Coordinate on Three-Stage Composition

## Conclusion

**NOT SUPPORTED: the shared language-conditioned action coordinate is not the best autonomous executive. SUPPORTED: the physical-feedback F2 backbone is essential for robust three-stage execution.** The shared coordinate achieved 25/30 strict `grasp -> lift -> place` successes, tying direct binary F3 but losing to state binary (26/30), fixed timing (26/30), a coordinate trained without action identity (27/30), and physical oracle switching (27/30). It switched prematurely in all 30 grasp phases and all 30 lift phases. Removing F2 while retaining the same coordinate collapsed success to 7/30, showing a large causal contribution from physical-residual recovery but no positive contribution from the proposed action latent.

## New task and data construction

G35 changed the scientific task rather than tuning another lift threshold. It implemented a continuous three-stage program with two online switches:

1. `close the gripper around the black book`;
2. `lift the black book`;
3. contextual placement in the caddy's back compartment.

A train-only prompt intervention first compared four grasp formulations on attempts 12, 27, and 17. All prompts grasped the book in 3/3 trials, but the selected close-gripper prompt produced a sustained five-step pre-lift grasp window in 3/3 and the earliest tied median grasp boundary (61 steps). This selection was frozen before held-out testing.

The selected prompt then ran from all 14 task-5 training starts. Bilateral book contact with both finger pads for three consecutive executed steps triggered `grasp -> lift`; a sustained 4 cm book-height gain triggered `lift -> place`; the LIBERO predicate certified placement. All 14 trajectories completed both switches and 12/14 completed the full task. Exact initial and switch snapshots, actions, prompts, dual-view observations, physical states, contacts, and explicit noise were retained.

These causal traces produced 576 issue-step rows: 212 grasp, 26 lift, and 338 place samples, with 14/14/12 positive completion examples. Training median stage durations were 70, 10, and 94 steps.

## Models and frozen selection

Four families were trained leave-one-training-episode-out and then on all training episodes:

1. a 109,234-parameter shared dual-view/state/action-history model with action identity, 16-D coordinate, completion BCE, progress regression, and temporal ranking;
2. a matched-capacity direct binary model without coordinate/ranking losses;
3. the same coordinate model with the action-identity input zeroed;
4. a 24,193-parameter state/action-history binary baseline.

This produced 56 fold checkpoints and four final checkpoints. Cross-fitted selection was performed per stage. The action coordinate selected grasp/lift/place thresholds 0.7/0.4/0.7 with persistence 2/1/2, but already showed 12, 9, and 10 premature training-episode switches. The state model's grasp boundary error was only 0.71 steps, versus 11.79 for the shared coordinate.

## Prospective causal protocol

Three explicit-noise repeats over ten untouched canonical starts compared eight mechanisms in 240 new rollouts. All primary methods executed one continuous episode from the actual reached state, used native π0.5 proposals, and received two real 3 cm book displacements. Except for the declared no-F2 ablation, candidate trials were evaluated by G33's raw-action physical-residual detector and rejected candidates were removed through exact simulator/controller checkpoint restoration.

The formal success definition for explicit executives requires two switches, realized grasp, realized lift, and official placement success. The future-visible full prompt has no explicit switches and is therefore reported separately as a no-switch physical-task comparator. The original raw metric file is preserved; strict metrics were written as new formal artifacts.

| Method | Strict explicit three-stage success | Two switches | Premature grasp/lift | Lift retention | Rejected steps |
|---|---:|---:|---:|---:|---:|
| shared action coordinate | 25/30 | 30/30 | **30 / 30** | 29/30 | 1,240 |
| direct binary | 25/30 | 29/30 | 5 / 15 | 30/30 | 1,075 |
| **state binary** | **26/30** | 28/30 | 5 / 13 | 30/30 | 975 |
| **no-language coordinate** | **27/30** | 30/30 | 28 / 28 | 30/30 | **935** |
| fixed timing | 26/30 | 30/30 | 7 / 10 | 30/30 | 1,205 |
| **physical oracle** | **27/30** | 30/30 | **0 / 0** | 30/30 | 995 |
| future-visible full prompt | n/a; 24/30 no-switch physical success | 0/30 | n/a | 30/30 | 1,620 |
| shared coordinate, no F2 | 7/30 | 30/30 | 30 / 30 | 17/30 | 0 |

The shared coordinate tied direct binary in pooled strict success, with two paired wins and two losses. It lost to state binary with two wins and three losses, to physical oracle with one win and three losses, and to the no-language coordinate with zero wins and two losses. It therefore fails the predeclared strict-superiority and paired-balance requirements despite exceeding the 24/30 floor.

## Mechanism findings

The shared coordinate learned a control-harmful timing signal. It switched grasp on average 16.87 steps away from the physical contact boundary and lift 24.30 steps away from the height boundary, with every switch classified as premature. Removing action identity improved success to 27/30 and slightly reduced both errors. Language/action identity therefore did not create the intended action-specific execution coordinate; the observation, state, and recent action stream carried more useful information without it.

Simple physical and state mechanisms remained stronger. Physical oracle had 2.33/2.20-step grasp/lift error and zero premature switches. State binary had 3.17/5.86-step errors and the lowest final target error among explicit methods (0.0564). The complex coordinate is not justified as F3.

The F2 ablation is a strong positive result. With the same shared coordinate and both real perturbations, removing causal candidate execution/rejection reduced strict success from 25 to 7, lift completion from 30 to 27, lift retention from 29 to 17, increased final target error from 0.065 to 0.435, and more than doubled action jerk from 0.078 to 0.167. This demonstrates that executed feedback and physical recovery, not the coordinate representation, drive the robust composition result.

## Audit and artifacts

The independent audit passed. It checked 12 prompt rollouts, 14 continuous training trajectories, all 576 rebuilt dataset rows, 56 fold models, 2,304 held-out predictions, 2,202 online F3 trials, 10,385 physical-recovery trials and exact restores, 29,642 formal test noise seeds, 480 physical interventions, 10,433 native commits, all 240 rollout chains, strict aggregates, paired comparisons, and the negative conclusion.

- `experiments/EXP_G35/prompt_sweep/`
- `experiments/EXP_G35/training_trajectories/`
- `experiments/EXP_G35/multi_action_dataset.npz`, fold/final models, selection sweeps, and rollouts
- `experiments/EXP_G35/case_metrics.jsonl` and preserved raw `metrics.json`
- `experiments/EXP_G35/formal_case_metrics.jsonl`, `formal_metrics.json`, and `audit.json`
- `scripts/experiments/collect_exp_g35_grasp_prompt_sweep.py`
- `scripts/experiments/collect_exp_g35_three_stage_training.py`
- `scripts/experiments/run_exp_g35_multi_action_coordinate.py`
- `scripts/experiments/formalize_exp_g35_metrics.py`
- `scripts/experiments/audit_exp_g35.py`
- `experiments/EXP_G35_smoke/` and `EXP_G35_smoke2/` preserve failed/repaired interface validation and are not counted as EXPs

The post-EXP disk check left 847 GB free.

## Consequence for the system

The strongest autonomous three-stage executive is a simple state completion model, and the best switching upper bound is the physical predicate. The strongest verified learned/control contribution remains G33's physical transition residual used for causal F2 recovery. G36 should fix state F3 and ask whether an action latent can contribute where evidence says it matters: encode the predicted and realized physical effect of a candidate into a compact causal coordinate for recovery, rather than using a generic visual or temporal action embedding.
