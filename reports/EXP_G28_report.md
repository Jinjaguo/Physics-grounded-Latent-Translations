# EXP_G28 Report: Common-Noise Matched Representation Trial

## Conclusion

**NOT SUPPORTED.** Explicit common random numbers removed proposal-noise luck as an explanation for the previous representation comparisons, but the aligned action coordinate still did not improve control. Across three paired repeats, aligned latent point ranking completed 23/30 ordered lift-to-place tasks, versus 27/30 for the equal-capacity unaligned latent point ranker and 27/30 for one unranked native pi0.5 proposal. The raw-action point ranker also reached 23/30. Alignment therefore neither beats the raw representation nor the simpler policy baseline.

## New causal protocol

G28 extended the local pi0.5 server with explicit 10x32 flow-matching noise for every request. A determinism gate verified that the same observation and noise produced bitwise-identical raw and postprocessed action chunks, while changing the noise changed the proposal. Every prospective decision used a seed determined only by `(repeat, attempt, decision, candidate)`, never by method identity. Thus corresponding methods received the same stochastic innovation, although their later observations could diverge because they executed different actions.

Thirty leave-one-attempt-out point rankers were trained: aligned latent, unaligned latent, and raw-action families for ten held-out attempts. Architecture, padded 128-D input width, parameter count, optimizer, epochs, losses, and split were matched. Their held-out ranking accuracy was 0.7475, 0.7075, and 0.6481 respectively. The higher offline accuracy of aligned features did not translate into better closed-loop success.

## Prospective results

The frozen systems executed 240 fresh autonomous simulator rollouts: three repeats, ten canonical initial states, and eight methods. Every closed-loop decision selected and committed an exact native pi0.5 action prefix, then replanned from the physically realized observation.

| Method | Ordered success | Per-repeat successes | Endpoint error | Jerk | Candidate steps |
|---|---:|---:|---:|---:|---:|
| aligned latent point | 23/30 | 6, 9, 8 | 0.06078 | 0.08148 | 0 |
| unaligned latent point | **27/30** | 8, 9, 10 | **0.05209** | 0.08116 | 0 |
| raw-action point | 23/30 | 8, 7, 8 | 0.06492 | 0.08589 | 0 |
| aligned set ranker | 26/30 | 8, 9, 9 | 0.05910 | **0.08004** | 0 |
| raw set ranker | 26/30 | 8, 9, 9 | 0.05319 | 0.08480 | 0 |
| physical shooting | 24/30 | 8, 9, 7 | 0.06817 | 0.08250 | 18,696 |
| single raw pi0.5 | **27/30** | 9, 10, 8 | 0.05865 | 0.08458 | 0 |
| initial-observation open loop | 0/30 | 0, 0, 0 | 1.07497 | 0.11285 | 0 |

Against unaligned point ranking, aligned point ranking had 2 paired wins, 6 paired losses, and 22 ties. Against the single-proposal policy it had 0 wins, 4 losses, and 26 ties. It also lost three paired cases to aligned set ranking without winning any. These are direct causal comparisons under the declared common-noise protocol, not post-hoc score analysis.

The experiment also exposes an important data property. Among the 414 matched-state groups used for training, candidate zero was not the realized best candidate in 225 groups, but only 12 groups offered more than 0.01 realized utility improvement over candidate zero. Frequent learned reordering therefore faces a low-margin decision problem and can easily destroy a strong base proposal.

## Audit and artifacts

The independent audit passed with zero failures. It recomputed all 30 fold models' held-out results (1,242 held-out decisions), regenerated and checked 28,498 explicit noise seeds, rebuilt 6,219 online learned scores from saved observations/actions and frozen checkpoints, verified 8,713 native proposal commits, checked all 240 rollout chains, and reconstructed pooled, per-repeat, paired, and acceptance metrics.

- `experiments/EXP_G28/fold_models/`, `model_selection.json`, and `common_noise_manifest.json`
- `experiments/EXP_G28/rollouts/`, `case_metrics.jsonl`, and `metrics.json`
- `experiments/EXP_G28/audit.json`
- `scripts/experiments/run_exp_g28_common_noise_trial.py`
- `scripts/experiments/audit_exp_g28.py`
- `scripts/dynamics/start_wave19_pi05_server.py`

The post-EXP disk check left 849 GB free.

## Consequence for the system

The aligned visual/action coordinate should not remain the primary F2 mechanism. G29 will instead learn a compact coordinate directly from realized action effects and use conservative baseline bootstrapping: candidate zero remains authoritative unless the learned effect representation supplies cross-fitted evidence of a meaningful advantage. This changes both the representation target and the controller intervention rule. It is motivated by the observed sparse high-margin improvements and by the principle of falling back to a behavior policy when offline evidence is weak, as in [Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html). The paper is used only as methodological guidance; G29's claim will depend on new simulator interventions.
