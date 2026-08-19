# EXP_G22 Report: Counterfactual-Regret Autonomous F3

## Conclusion

**NOT SUPPORTED.** The selected autoencoding counterfactual-regret latent did not beat the existing state-window F3. In the formal auditable run it completed 8/10 ordered tasks with 21.2 mean absolute switch-error steps, while state F3 completed 10/10 with 6.0 steps. The latent beat the matched direct regret model (7/10), but that is insufficient under the preregistered rule. The first complete run showed the same decisive ordering—latent 9/10 versus state 10/10—so the conclusion is not an artifact of the retry's stochastic pi0.5 samples.

## Scientific intervention

G22 froze the strongest G21 F2, single-proposal receding pi0.5 feedback, and changed F3 supervision from retrospective height labels to downstream causal regret. For each of the ten G18 perturbed initial states, it followed the successful G21 single-proposal trajectory to two checkpoints around its autonomous boundary. From each exact simulator/controller checkpoint it restored the same state and executed three real continuations: switch to place immediately, continue lift for 5 steps then switch, or continue for 10 steps then switch. All continuations began from complete checkpoints and sent actions through LIBERO; none were teacher-forced.

This produced 20 counterfactual examples and 60 physical continuation rollouts. The best delay was 0 steps for 5 examples, 5 steps for 8, and 10 steps for 7. Every group had a nonzero best-versus-second-best cost margin, with mean margin 0.00726, but only one group changed binary success across delays. Delay-wise success was 19/20, 19/20, and 18/20. Thus timing had measurable continuous downstream effects, but the supervised cohort contained little binary failure contrast.

Three models were trained in ten leave-one-attempt-out folds: a 16-D contrastive regret latent, a 16-D autoencoding regret latent, and a direct 64-D MLP. Each predicted the realized costs of the three delay interventions. The latent-family rule selected the autoencoder. Held-out delay-choice accuracy was 0.45 for that model, 0.40 for the contrastive latent, and 0.60 for the direct MLP; cost MAE was 0.70768, 0.56913, and 0.63363 respectively. The selected latent therefore already lost to direct prediction on the principal offline choice metric.

## Prospective closed-loop result

All methods used autonomous, continuous current-state execution from the ten matched perturbed snapshots. The two regret models made F3 decisions from leave-one-attempt-out checkpoints; state F3, oracle-height behavior, fixed-time F3, and an unprotected future-visible prompt were frozen controls. F2 was disabled to a single proposal so candidate selection could not mask the switching mechanism.

| Method | Ordered success | Composition success | Switches | Mean switch error | Endpoint error |
|---|---:|---:|---:|---:|---:|
| counterfactual latent F3 | 8/10 | 8/10 | 10 | 21.20 | 0.09573 |
| direct regret F3 | 7/10 | 6/10 | 9 | 20.56 | 0.07541 |
| state-window F3 | **10/10** | **10/10** | 10 | **6.00** | **0.05249** |
| oracle-height control | 8/10 | 0/10 | 0 | n/a | 0.07256 |
| fixed-time F3 | 8/10 | 8/10 | 10 | 13.60 | 0.08633 |
| future-visible full prompt | 7/10 | 7/10 | 0 | n/a | 0.12208 |

The `oracle_f3` implementation is an isolation control that retains lift prompting rather than an autonomous program switch, hence its zero composition count despite task reward in 8 cases. The decisive control result is state F3: it preserves lift, switches autonomously, retargets from the reached physical state, and succeeds in all ten cases. Counterfactual-regret compression does not improve that controller.

## Preserved retries and audit

The original complete directory, `experiments/EXP_G22`, contains the 20 checkpoints, 60 matched branches, dataset, 30 fold checkpoints, and its first 60 prospective rollouts. It saved external F3 probabilities but not their exact 64-D inputs, so it could not alone support independent model-forward auditing. `EXP_G22_retry1` records a sandbox-denied pre-rollout start. `EXP_G22_retry2` records an intentionally interrupted start after a provenance-name defect was found. Neither consumes an EXP ID.

`EXP_G22_retry3` is the formal result. It reused the already audited 60 branches, 30 models, and 40 control rollouts, and generated 20 new latent/direct rollouts that additionally save every F3 input, predicted three-delay cost, latent, and probability. The first audit attempt is preserved as `audit_failed_observable_reconstruction.json`: it revealed that checkpoint restoration intentionally refreshes proprioceptive observables by about 1e-3. The corrected audit reconstructed the exact original input from the observable buffers stored inside each checkpoint and passed.

The passing audit independently checked 20 checkpoints and input vectors, 60 branch rollouts, 20 counterfactual samples, 30 fold models, 407 external-F3 decisions, 2,584 proposal-to-commit decisions, all 60 end-to-end chains, and all six aggregate tables.

- `scripts/experiments/run_exp_g22_counterfactual_f3.py`
- `scripts/experiments/run_exp_g22_auditable_retry.py`
- `scripts/experiments/audit_exp_g22.py`
- `experiments/EXP_G22/boundary_checkpoints/`, `branch_rollouts/`, `counterfactual_dataset.json`, `fold_models/`
- `experiments/EXP_G22_retry3/eval_rollouts/`, `case_metrics.jsonl`, `metrics.json`, `audit.json`

The post-EXP disk check left 850 GB free.

## Consequence for the system

G22 strengthens the integrated non-latent system: state F3 plus single-proposal closed-loop pi0.5 achieved repeatable 10/10 autonomous `lift -> place`, while hard future-goal exposure was worse. It does not satisfy final acceptance because the tested latent is unnecessary and loses to state control. G23 therefore stops adding post-hoc latent heads to F2/F3. It will train action compression and realized transition prediction jointly, intervene through the decoded action representation before execution, and compare that mechanism against a capacity-matched direct world model and physical branch shooting.
