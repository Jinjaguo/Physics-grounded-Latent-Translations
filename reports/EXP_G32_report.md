# EXP_G32 Report: Direct-Utility Sequential Recovery

## Conclusion

**SUPPORTED for the predeclared direct sequential-recovery criterion; NOT SUPPORTED as an overall best-controller or latent-control result.** The stage-specific direct utility rule completed 27/30 ordered tasks, compared with 24/30 for exhaustive physical shooting and 26/30 for single pi0.5, while using 9,720 rejected candidate steps rather than shooting's 18,933. It therefore satisfies G32's frozen success rule. However, G28 aligned latent-point ranking also completed 27/30 with no simulator trial cost, and single pi0.5 tied the lower-cost improvement rule at 26/30. G32 supports transparent measured-effect rollback as one useful controller, but does not establish that trial recovery is globally preferable or that a learned latent causes the gain.

## Why this experiment was necessary

G31 established exact adaptive checkpoint recovery, but its action/effect latent verifier achieved only 22/30 while raw-visual, physical, and even shuffled-latent verifiers achieved 25/30. G32 tested whether the useful part was simply causal measurement followed by a transparent stopping decision. It removed learned representations entirely from its three primary F2 variants and selected their stopping rules using the audited G24 matched-state branch dataset before opening the new prospective test.

## Frozen policies and data

The development data were all 414 three-candidate matched-state groups (1,242 executed branches) in `experiments/EXP_G24_retry1/visual_branch_dataset.npz`. The experiment swept thresholds `[-0.02, -0.01, 0, 0.005, 0.01, 0.02, 0.05]` within this EXP and froze:

- absolute stopping at realized utility 0.05;
- improvement stopping at margin 0.0, accepting candidate zero when its utility was positive;
- stage-specific stopping at 0.01 during lift and 0.05 during place.

Selection maximized mean realized branch utility, then ranking accuracy, then fewer trials. The selected policies respectively obtained development mean utilities 0.032978, 0.032809, and 0.032998. The stage policy's branch ranking accuracy was 0.5435, so it was never treated as a reliable oracle; its value had to be established prospectively.

## Causal controller protocol

For every decision, the three direct controllers requested three π0.5 native action chunks under an explicit common-noise schedule. They captured the complete simulator, environment, observable, controller, and integration state; executed candidate prefixes sequentially; measured the realized end-effector and book displacement; and computed stage-aware task utility from the reached physical state. An accepted trial remained the current simulator trajectory and was not executed again. A rejected trial caused exact restoration before the next candidate, and the final candidate was the fallback. The direct decision used no image/action latent, although before/after pixels were preserved as intervention evidence.

The prospective test used three independent noise repeats across all ten canonical held-out starts, producing 240 new rollouts. It compared three direct policies, G31's learned physical verifier, exhaustive physical shooting, G28 aligned latent-point ranking, single pi0.5, and initial-observation open loop. Autonomous F3 and current-state replanning were held fixed.

| Method | Ordered success | Per-repeat | Rejected candidate steps | Endpoint error | Jerk |
|---|---:|---:|---:|---:|---:|
| absolute direct utility | 25/30 | 9, 8, 8 | 10,875 | 0.07005 | 0.07953 |
| improvement direct utility | 26/30 | 8, 9, 9 | **2,420** | 0.06782 | 0.07980 |
| stage-specific direct utility | **27/30** | 10, 8, 9 | 9,720 | **0.05865** | **0.07934** |
| learned physical verifier | 24/30 | 8, 8, 8 | 6,430 | 0.07079 | 0.08016 |
| exhaustive physical shooting | 24/30 | 8, 8, 8 | 18,933 | 0.07062 | 0.08320 |
| G28 aligned latent point | **27/30** | 10, 9, 8 | 0 | 0.06151 | 0.07941 |
| single raw pi0.5 | 26/30 | 8, 9, 9 | 0 | 0.06790 | 0.08438 |
| initial-observation open loop | 0/30 | 0, 0, 0 | 0 | 1.07211 | 0.11365 |

The stage rule gained three paired successes and no losses against both learned physical recovery and shooting. Against aligned point it had one win, one loss, and 28 ties; against single it had two wins, one loss, and 27 ties. The lower-cost improvement rule tied single's pooled 26 successes and therefore did not satisfy the strict predeclared gain over single.

## Failure analysis

The recovery mechanisms can actively replace a viable first proposal. In repeat 1, attempt 44, single succeeded while every recovery/shooting/point method failed. In repeat 2 on the same start, single again succeeded while absolute recovery, learned physical recovery, shooting, and aligned point failed; improvement and stage recovered successfully. Thus neither “more trials” nor one fixed stopping rule dominates across inference noise. The strongest zero-trial latent point baseline tied stage utility, which means G32 does not justify paying branch-execution cost on clean starts without a perturbation or uncertainty trigger.

The learned physical verifier fell to 24/30 and used 6,430 rejected steps. It was worse than all three direct rules on pooled success and no better than shooting. It should be removed from the current best F2. The direct result also supplies no new positive evidence for an action latent: its primary policies intentionally omit latent features.

## Audit and artifacts

The independent audit passed. It reselected all three policy families from the source branch data, recomputed 8,300 direct trial utilities/features/acceptance decisions and 2,512 learned physical verifier trials, checked 10,812 exact restore events with maximum integration-state error 0, regenerated 28,335 explicit noise seeds, verified 8,672 native committed prefixes and all 240 rollout chains, and rebuilt every aggregate, paired comparison, winner, and support decision.

- `experiments/EXP_G32/policy_selection.json`, `common_noise_manifest.json`, and `frozen_system_manifest.json`
- `experiments/EXP_G32/rollouts/`, `case_metrics.jsonl`, `metrics.json`, and `audit.json`
- `scripts/experiments/run_exp_g32_direct_utility_recovery.py`
- `scripts/experiments/audit_exp_g32.py`

The exact executed command is recorded in `experiments/EXP_G32/run_metadata.json`. The post-EXP disk check left 848 GB free.

## Consequence for the system

The strongest defensible clean-start backbone is now the zero-trial G28 aligned point controller: it ties the best pooled success without speculative simulator execution. Stage utility recovery is retained as an intervention mechanism that can be triggered when an external disturbance is detected, because G32 proves it can causally restore and continue with exact checkpoint state. G33 must therefore move to controlled perturbations rather than retuning clean-start thresholds. It will test whether action-latent memory contributes to detecting or localizing execution disruption beyond raw visual and physical memory, while fixing the clean controller and autonomous F3.
