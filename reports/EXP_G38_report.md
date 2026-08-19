# EXP_G38 Report: Causal Action-Support Proposal Ranking

## Conclusion

**NOT SUPPORTED under the predeclared rule.** The main support latent achieved 24/30 strict three-stage successes, exceeding direct classifier and native-order ranking at 23/30 and shuffled-label latent at 22/30. Its paired balances were positive against direct (1–0), native order (2–1), and shuffled (3–1). However, it used 1,240 rejected candidate steps versus 1,205 for native order and lost to the pairwise latent's 25/30. The required success-plus-efficiency claim is therefore false.

## New mechanism

G38 stopped using a latent as the physical verifier. It rebuilt 13,330 actually executed G36 candidate trials into a support dataset with 11,665 direct-residual accepts and 1,665 rejects. Ten held-attempt-out folds trained 40 models: an eight-dimensional support latent with action reconstruction, a pairwise-ranking latent, a matched direct classifier, and a shuffled-label latent. The frozen G33 direct physical residual remained the verifier for every learned and heuristic ranker.

At test time the G35 autonomous state F3 and continuous `grasp -> lift -> place` executor generated five common-noise native proposals per decision. Candidate 0 was always tried first. Each method only changed the order of candidates 1–4 after a rejection: support latent probability, pairwise latent score, direct classifier, shuffled latent, raw-action diversity, or unchanged native order. A physical oracle recovery comparator used the same proposal order. Thus the intervention point was proposal selection, not thresholding.

## Prospective results

Three repeats over ten canonical starts produced 210 new causal rollouts with two real 3 cm book interventions per rollout.

| Ranking method | Strict success | Paired main-latent wins/losses | Rejected steps |
|---|---:|---:|---:|
| support latent | 24/30 | — | 1,240 |
| **pairwise latent** | **25/30** | 0 / 1 | 1,255 |
| direct classifier | 23/30 | 1 / 0 | 1,225 |
| shuffled-label latent | 22/30 | 3 / 1 | 1,440 |
| raw-action diversity | 24/30 | 2 / 2 | 1,220 |
| native order | 23/30 | 2 / 1 | **1,205** |
| physical oracle | 28/30 | 0 / 4 | 300 |

The mechanism has a real but insufficient signal. Destroying support labels reduced success from 24 to 22 and increased rejected work. The main latent also produced small paired gains over both direct and native rankers. But a no-learning raw-diversity rule tied its success with lower cost, and pairwise training produced the best learned success. The main hypothesis demanded strict superiority and lower recovery cost, so it is not supported.

## Audit and artifacts

The independent audit passed without failures. It rebuilt all 13,330 labels, checked 40 checkpoints, recomputed 53,320 held-out predictions, 8,580 five-candidate rankings, 42,900 explicit noise seeds, 10,157 direct physical scores and exact restores, 420 interventions, 8,580 native commits, all 210 rollout chains, strict aggregates, paired comparisons, and the negative conclusion.

- `experiments/EXP_G38/support_dataset.npz`, `model_selection.json`, and 40 checkpoints
- `experiments/EXP_G38/rollouts/`, `case_metrics.jsonl`, `metrics.json`, and `audit.json`
- `scripts/experiments/run_exp_g38_action_support_ranking.py`
- `scripts/experiments/audit_exp_g38.py`
- `experiments/EXP_G38_smoke/` and `EXP_G38_smoke_retry1/` preserve interface evidence and are not separate EXPs

The single post-EXP disk check reported 846 GB free.

## Consequence

Learned support ranking is the first latent intervention to show positive paired balances against matched direct and native baselines, but its magnitude and efficiency do not justify the final claim. The strongest learned variant is pairwise ranking; the strongest simple ranker is raw-action diversity. G39 should test whether a latent decoder can generate a new corrective proposal outside the fixed π0.5 sample set, which is the remaining control role not reducible to reordering or residual detection.
