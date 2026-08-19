# EXP_G17 Report: Identity-Preserving Causal Residual Latent

## Conclusion

**PARTIALLY SUPPORTED, but the final latent claim remains NOT SUPPORTED.** The identity-safe causal residual system reached 10/10 on ten validated 7 cm shifts. It strictly beat shuffled residual codes (8/10), single raw π0.5 (9/10), outcome-head-only residual control (7/10), open loop (0/10), and replay (1/10). Thus residual code ranking and real branch feedback each had measurable value. However, equal-budget independent raw π0.5 shooting also reached 10/10 and had lower endpoint error and jerk. Because the latent method did not strictly beat this conventional causal F2 baseline, the preregistered latent contribution and integrated acceptance are false.

## New causal representation

G15/G16 matched branch artifacts yielded 2,340 residual samples. Each linked the exact pre-decision state and base proposal to an alternate-minus-base action residual, both candidates' physically realized endpoints, value difference, and success difference. Splitting by source attempt selected a deterministic conditional AE over a beta-VAE. An eight-code residual codebook was ranked separately for lift/place using realized progress. The decoder was identity-safe: code zero always produced zero residual and candidate zero remained byte-identical to raw π0.5.

The first run failed before rollout on NumPy integer JSON serialization. Retry1 produced 70 valid rollouts but lacked the exact observation context needed to independently re-decode saved residuals. Both are preserved. Retry2 added `residual_context` to every decision and is the sole formal result.

## Results

| Method | Success | Switch error ↓ | Endpoint error ↓ | Candidate steps |
|---|---:|---:|---:|---:|
| ranked causal residual latent | 10/10 | 6.6 | 0.067858 | 4,824 |
| outcome-head-only residual | 7/10 | 12.22 | 0.185746 | 0 |
| raw π0.5 shooting | **10/10** | 7.1 | **0.046888** | 5,448 |
| shuffled residual codes | 8/10 | 7.6 | 0.062277 | 5,902 |
| single raw π0.5 | 9/10 | 7.6 | 0.047694 | 0 |
| initial-observation open loop | 0/10 | n/a | 1.124779 | 0 |
| unperturbed replay | 1/10 | n/a | 0.447016 | 0 |

The ranked/shuffled gap shows learned residual semantics can matter. The causal/predicted gap shows model-predicted outcomes cannot replace physical candidate execution. Yet raw shooting matches success with smoother and more accurate endpoints. The appropriate conclusion is that causal multi-proposal F2 is supported, while this residual action latent remains unnecessary.

## Audit and artifacts

The independent audit passed: it rebuilt all 2,340 samples from source branches, recomputed selection, validated ten snapshots, 70 rollouts and 50 chains, checked 1,121 identity candidates and exact decodes, 400 predicted selections, replay provenance, and every metric.

- `scripts/experiments/run_exp_g17_causal_residual_latent.py`
- `scripts/experiments/audit_exp_g17.py`
- `experiments/EXP_G17/`, `EXP_G17_retry1/` — preserved gate/incomplete-audit evidence
- `experiments/EXP_G17_retry2/causal_residual_dataset.npz`
- `experiments/EXP_G17_retry2/residual_ae.pt`, `residual_vae.pt`, `residual_codebook.npz`
- `experiments/EXP_G17_retry2/test_snapshots/`, `test_rollouts/`, `metrics.json`, `audit.json`

The post-EXP disk check left 849 GB free; formal retry2 occupies about 20 MB.

## Next decision

G18 will stop decoding or perturbing actions. It will learn a transition latent from actual candidate outcomes and use that latent only to choose among untouched native π0.5 proposals after real branch execution. This tests whether action-conditioned transition coordinates add value without damaging the strong proposal manifold.
