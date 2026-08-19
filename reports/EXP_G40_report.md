# EXP_G40 Report: Final Integrated Acceptance and Mechanism Falsification

## Conclusion

**SUPPORTED under G40's preregistered operational rule: the simplified direct/state closed-loop system is robust. NOT SUPPORTED: a learned action latent is control-relevant, so the final Actions-as-Coordinates paper goal remains unmet.** The full system—native π0.5 proposals, direct action-conditioned physical-residual F2 with exact branch restore, and autonomous state F3—achieved 28/30 strict `grasp -> lift -> place` successes. It exactly matched physical-oracle F3 at 28/30 and strictly beat no-F2/no-memory (4/30), restart-from-initial (21/30), future-visible unprotected control (21/30), and no explicit F3 (0/30 under the declared ordered-switch criterion). The existing latent corrective intervention achieved 26/30, with one paired win and three losses against direct control, so it fails the required latent contribution test.

There is an important qualification. The no-F3 controller achieved 29/30 official LIBERO task successes while receiving only the grasp prompt, but by construction made zero explicit program switches and therefore scored 0/30 on strict ordered-program success. Consequently G40 proves that explicit F3 can enforce and log the requested program semantics; it does **not** prove that F3 improves the benchmark's ordinary task reward. In fact, π0.5's behavior collapses the intended atomic decomposition by often completing the whole manipulation from a nominally atomic prompt. This is a central feasibility problem for the original paper story.

## Hypothesis and frozen design

G40 tested two separate hypotheses. First, the simplified non-latent system should reach at least 24/30 strict successes, strictly beat open-loop/no-F2, restart, future-unprotected, no-F3, and no-memory mechanisms, and remain within three successes of oracle F3. Second, a learned action latent should strictly exceed this direct system with positive paired balance and no greater rejected work. The final action-coordinate goal required both conditions.

Ten held-out canonical task-5 initial states, three method-neutral explicit-noise repeats, and nine mechanisms produced 270 new rollouts. Each rollout used a 300-step horizon, five native proposals per decision, five-step replanning, autonomous `grasp -> lift -> place` execution where applicable, and real 3 cm book displacements at decisions 8 and 24. Candidate trials were actual LIBERO interventions; accepted trials became the current physical state, while rejected trials were removed by exact simulator/controller restore before the next proposal.

The nine mechanisms were:

1. full direct/state closed loop;
2. G39 latent corrective action coordinate;
3. one-proposal open-loop/no-F2 recovery;
4. observation-independent teacher-forced progress coordinate based on train-only median stage durations;
5. exact restart to the canonical initial snapshot at the autonomous `lift -> place` switch;
6. the full future-visible prompt at every low-level decision;
7. physical-predicate oracle F3 with the same direct F2;
8. no F3, which never changed the active prompt;
9. direct physical failure detection without checkpoint recovery authority.

## Results

| Method | Strict ordered success | Official task success | Two switches | Lift retention | Rejected steps | Final error |
|---|---:|---:|---:|---:|---:|---:|
| **full direct/state** | **28/30** | 28/30 | 30/30 | 30/30 | 1,455 | 0.05678 |
| latent action coordinate | 26/30 | 28/30 | 28/30 | 30/30 | **1,075** | **0.05386** |
| open-loop / no F2 | 4/30 | 6/30 | 28/30 | 12/30 | 0 | 0.45012 |
| teacher-forced coordinate | 26/30 | 26/30 | 30/30 | 30/30 | 1,540 | 0.05782 |
| restart from initial | 21/30 | 21/30 | 30/30 | 14/30 | 1,070 | 0.09172 |
| future-visible unprotected | 21/30 | 26/30 | 25/30 | 30/30 | 1,675 | 0.05566 |
| physical-oracle F3 | **28/30** | 28/30 | 30/30 | 30/30 | 1,440 | 0.05356 |
| no F3 / never switch | 0/30 strict | **29/30 official** | 0/30 | 30/30 | 2,185 | 0.05626 |
| no checkpoint memory | 4/30 | 6/30 | 28/30 | 12/30 | 0 | 0.45012 |

The full system had 24 paired wins and no losses against no-F2 and no-memory, seven wins and no losses against restart, eight wins and one loss against future-visible unprotected control, two wins and no losses against teacher-forced progression, and exact 30-case ties with oracle F3. Against the latent intervention it had three wins, one loss, and 26 ties. The latent used fewer rejected steps and had slightly lower endpoint error, but the preregistered primary outcome was strict ordered success; it therefore cannot be called the winner.

## Mechanism conclusions

**F2 and checkpoint branch memory are the strongest supported mechanisms.** Removing causal proposal evaluation and recovery reduced strict success from 28 to 4, official success from 28 to 6, lift retention from 30 to 12, and increased final error from 0.0568 to 0.4501. The no-memory method still computed the direct physical residual after the disturbed proposal, but without authority to restore and retry it produced exactly the same physical trajectories and outcomes as no-F2. Detection alone is not recovery; the action-changing checkpoint branch is essential in this simulator protocol.

**Current-state continuation is supported over restart.** Restoring the canonical initial state at every `lift -> place` switch reduced strict success from 28 to 21, lift retention from 30 to 14, increased mean switch error and final error, and never beat the full system in a paired case. This directly supports retargeting from the physically reached state rather than regenerating the next action from the episode start.

**Current-action prompt protection helps the explicit program, but the benchmark does not cleanly identify atomic authority.** Giving π0.5 the full future instruction throughout reduced strict success from 28 to 21 and missed the required two switches in five cases. Yet it still had 26 official successes, and the no-F3 single-prompt controller had 29. Thus protection improves compliance with the declared execution program, not necessarily ordinary benchmark reward. The low-level policy already contains prerequisite and long-horizon behavior that crosses the proposed F1/F3 boundary.

**Autonomous state F3 is sufficient but not established as task-necessary.** It exactly matched oracle F3 on all 30 strict outcomes and exceeded the teacher-forced progress coordinate by two successes. This is strong evidence that state F3 can replace oracle switching with bounded degradation—here, zero degradation. However, no-F3's 29/30 official success means the task does not require explicit switching to obtain reward. F3 should be claimed as an interpretable program-execution mechanism, not as the source of task success on this benchmark.

**The action latent is unnecessary.** It generated genuinely non-native controls in 495 audited decisions, but lost strict success 26 to 28 and had a negative 1–3 paired balance. G39 also showed that direct corrective regression beat latent generation. G40 therefore closes the remaining escape route: action latents did not become a necessary control interface even when allowed to create new executable actions inside the strongest closed loop.

## Audit and artifacts

The independent audit passed. It checked all 270 rollouts; reconstructed 12,412 proposal transforms, active-stage decisions, F3 sequences, and committed action chains; recomputed 3,381 learned F3 predictions and 12,821 direct physical scores; checked 14,500 exact candidate restores, 540 real interventions, 30 initial-state restarts, 62,060 explicit noise seeds, 495 nontrivial latent-generated decisions, every strict outcome, aggregate, paired comparison, and both acceptance decisions. The single post-EXP disk check left 847 GB free.

- `experiments/EXP_G40/frozen_protocol.json` and `common_noise_manifest.json`
- `experiments/EXP_G40/rollouts/`, `case_metrics.jsonl`, `metrics.json`, `run_metadata.json`, and `audit.json`
- `scripts/experiments/run_exp_g40_final_integrated_acceptance.py`
- `scripts/experiments/audit_exp_g40.py`
- `experiments/EXP_G40_smoke/` preserves the missing-server gate; `EXP_G40_smoke2/` preserves successful branch validation; neither is a separate EXP

## Final decision

The G series stops at G40 as requested. The simplified direct/state system satisfies the operational closed-loop composition rule, but the final goal in `ACTIONS_AS_COORDINATES_FINAL_METHOD_AND_GOAL.md` explicitly also requires a learned action latent to contribute beyond a non-latent reproduction. That requirement is contradicted by G40 and by the accumulated G-series evidence. The correct final status is therefore: **robust causal closed-loop control achieved; Actions-as-Coordinates latent contribution not achieved.**
