# EXP_G16 Report: Decodable Latent-Chunk Shooting

## Conclusion

**NOT SUPPORTED: phase-conditioned action-chunk coordinates substantially harmed closed-loop control.** The frozen PCA latent shooter achieved 2/10 on physically valid 6 cm perturbations, versus 10/10 for equal-budget raw π0.5 shooting and 10/10 for a single raw π0.5 proposal. Decoded-base-only achieved 1/10, showing that most damage came from projecting an observation/language-conditioned π0.5 chunk through an unconditional demonstration manifold. Latent perturbation rescued two cases relative to projection, but it also destroyed a projected success on another case and never beat raw control.

Feedback remains strongly supported: open loop and unperturbed replay were both 0/10. Equal-budget raw Gaussian shooting reached only 1/10, so the strong result is not generic action-space search; it is preservation of observation/language-conditioned π0.5 proposals.

## Training and development selection

Fourteen task-5 training demonstrations yielded 2,658 overlapping 10×7 chunks: 1,096 lift and 1,562 place. Three distinct 4-D phase-conditioned coordinate families were fit: PCA, deterministic AE with reconstruction/temporal losses, and beta-VAE with reconstruction/temporal/KL losses. No development or test transition entered representation fitting.

Each family was causally evaluated at scales 0.5 and 1.0 on frozen G15 attempts 40, 44, and 47, with three actually branch-executed candidates per decision. Five variants obtained 2/3 and AE scale 1.0 obtained 1/3. The preregistered success-then-error rule selected PCA scale 1.0, whose mean endpoint error was 0.234689. The system and ten new test snapshots were frozen before test rollout.

## Frozen 6 cm test

All ten predeclared cardinal 6 cm book shifts reproduced the requested displacement within 5 mm and kept Z drift below 5 mm. Every method used state-window autonomous F3 and the same exact snapshots.

| Method | Success | Endpoint error ↓ | Candidate steps | Committed steps |
|---|---:|---:|---:|---:|
| PCA latent-coordinate shooting | 2/10 | 0.401359 | 8,503 | 2,833 |
| independent raw π0.5 shooting | **10/10** | **0.037443** | 5,290 | 1,762 |
| raw-action noise shooting | 1/10 | 0.430079 | 8,639 | 2,879 |
| decoded base only | 1/10 | 0.395083 | 2,877 | 2,877 |
| single raw π0.5 | **10/10** | 0.052260 | 0 | 1,852 |
| initial-observation open loop | 0/10 | 1.102846 | 0 | 3,000 |
| unperturbed replay | 0/10 | 0.466901 | 0 | 1,852 |

Raw shooting reduced endpoint error relative to single but did not improve success, so on this cohort its extra branch computation is not required for task completion. PCA latent shooting did beat raw noise and decoded-only by one success, but it lost eight successes to both appropriate raw baselines. The latent-control hypothesis and integrated acceptance are false.

## Mechanism diagnosis

The decoder was trained only on demonstration chunks and phase, while the π0.5 proposal already encodes the current images, proprioception, language, and instance geometry. Passing every proposal through PCA discards components that are rare in demonstrations but necessary for the current state. The 1/10 decoded-only result isolates this projection cost. Latent intervention sometimes repaired that projection (attempts42 and46), but was not stable: on attempt47 decoded-only succeeded while latent shooting failed. A useful next representation must preserve the raw proposal exactly and express only an optional state-conditioned residual.

## Audit and artifacts

The independent audit passed. It checked 14 training memberships, 18 development rollouts and the metric-selected winner, freeze ordering, ten perturbations, 70 test rollouts, 50 causal chains, ten replay prefixes, 576 raw-noise decisions, and independently decoded 1,143 saved latent decisions back to their proposal actions.

- `scripts/experiments/run_exp_g16_latent_chunk_shooting.py`
- `scripts/experiments/audit_exp_g16.py`
- `experiments/EXP_G16/` — preserved pre-training normalization gate
- `experiments/EXP_G16_retry1/chunk_dataset.npz`, `chunk_dataset_manifest.json`
- `experiments/EXP_G16_retry1/phase_pca.npz`, `phase_chunk_ae.pt`, `phase_chunk_vae.pt`
- `experiments/EXP_G16_retry1/development_rollouts/`, `development_selection.json`
- `experiments/EXP_G16_retry1/test_snapshots/`, `test_rollouts/`
- `experiments/EXP_G16_retry1/metrics.json`, `audit.json`, `run_metadata.json`

The post-EXP disk check left 849 GB free. The valid directory occupies about 23 MB.

## Next decision

The phase-only replacement latent is rejected. EXP_G17 will learn state-conditioned residual coordinates from the actual matched candidate interventions saved by G15/G16. Candidate zero will always be the untouched π0.5 chunk; decoded latent residuals can add alternatives but can no longer erase the base policy. A realized-outcome head and residual codebook will ground the representation in candidate-specific physical effects.
