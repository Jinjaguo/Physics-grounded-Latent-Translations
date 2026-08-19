# Next Experiment from EXP_G23: EXP_G24 Native-Action Visual Transition Coordinates

## Hypothesis

G23 shows that action decoding is the dominant failure: even decoded-first control loses four tasks relative to native single-proposal control, and continuous/VQ interventions introduce large action shifts and transition residuals. A latent may still contribute without replacing low-level controls if it represents the controllable visual consequence of a native action chunk. The hypothesis is that aligning an action bottleneck to actual dual-view observation change can improve ranking of untouched native pi0.5 proposals over both a direct visual action model and the existing low-dimensional direct state/action model.

This is a mechanism change from G18 and G23. G18 encoded state/action/outcome only after simulator candidate execution; G23 decoded latents into new controls. G24 will predict visual outcomes before selection, but the chosen proposal bytes will remain native pi0.5 output. The construction follows the visual-transition grounding motivation of [LAWM](https://arxiv.org/abs/2509.18428) while retaining the frozen-policy predictive-control comparison emphasized by [GPC](https://arxiv.org/abs/2502.00622).

## Real visual branch dataset

Use the ten audited G18 snapshots and G23 physical-shooting artifacts. Replay each committed physical-shooting trajectory from its complete initial checkpoint. At every decision:

1. capture the exact pre-decision simulator/controller checkpoint and both agent-view and wrist RGB observations;
2. restore that checkpoint separately for every saved native candidate prefix;
3. execute the exact saved 5-step candidate bytes through LIBERO;
4. capture both realized post-intervention views, physical displacement, success, and progress;
5. restore and commit the originally selected action, verifying that the replayed physical trajectory stays within a declared tolerance of the saved G23 chain.

Save images at a compact fixed resolution plus exact checkpoint/action/provenance. A replay mismatch is a G24 data-construction failure to fix inside G24, not a reason to create G25. This collection must produce new actual simulator interventions; merely extracting frames from demonstrations is insufficient.

## Models and held-out protocol

Split all samples leave-one-attempt-out. Train and compare:

1. an action-conditioned dual-view transition latent whose action bottleneck is contrastively aligned with the encoded realized before-to-after visual change, with future-latent, progress, and success heads;
2. a dual-view joint latent without action/visual-delta alignment, isolating the alignment mechanism;
3. a direct dual-view state-plus-action predictor without an action bottleneck;
4. the G23-style direct low-dimensional state-plus-action world model.

The image encoder, action encoder, and transition head must be trained only on the nine non-held attempts for each fold. Save exact images, normalization, fold membership, model checkpoints, and held-out future-latent, progress-ranking, success, and calibration metrics. Select the aligned latent architecture before prospective execution.

## Prospective native-action evaluation

Run fresh closed-loop `lift -> place` rollouts from all ten perturbed snapshots with pi0.5 F1 and the audited state-window F3. Compare:

1. aligned visual-transition action latent ranking;
2. unaligned visual latent ablation;
3. direct visual action world model;
4. direct low-dimensional state/action world model;
5. shuffled action-latent alignment as a mechanism ablation;
6. physical matched-branch shooting;
7. single raw pi0.5;
8. initial-observation open loop.

All model-based rankers query matched native proposals and set only their scores. They must never decode, perturb, interpolate, or otherwise alter proposal bytes. Save input images, raw actions, action/visual latents, predicted after-latent and progress, selected index, exact commit chain, realized images/state, prediction residual, F3 state, outcome, and computation time.

## Decision rule

The visual action latent contributes only if it strictly beats both direct visual and direct state/action world models, the unaligned/shuffled ablations, and single raw pi0.5 in ordered success. It must additionally match or beat physical shooting success with fewer candidate-executed steps, or strictly beat physical shooting. Ties do not support the latent claim; prediction error is only a tie-break after equal success. If the aligned latent loses, G25 must attempt policy-side latent conditioning or redefine the paper around the experimentally superior non-latent controller rather than return to decoded action perturbations.
