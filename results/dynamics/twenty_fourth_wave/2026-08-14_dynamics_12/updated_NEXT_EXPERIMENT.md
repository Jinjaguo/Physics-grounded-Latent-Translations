# Twenty-fourth wave next experiment

## Decision from Wave 24

Wave 25 should remain on the language-conditioned latent-dynamics research line, but replace deterministic neighborhood averaging with a compact conditional distribution over displacement modes. Wave24 found reliable state-conditioned direction (full cosine 0.627, execution cosine 0.648) and large gains over a goal+horizon mean, while D2 produced only 56–66% of true displacement magnitude and failed endpoint/continuity gates. This is the signature expected when heterogeneous local displacement modes are averaged.

## Recommended experiment: conditional mode diagnosis before a new model

First, use train only to cluster normalized displacement direction and log-magnitude separately within each `(goal,horizon)` cell. Condition mode probabilities on `(z_previous,z_current,language,horizon)`. On development, compare the deterministic mean, nearest mode, a compact mixture-density head, and an oracle best mode. Authorize a learned model only if a non-oracle mode selector improves H2 full MSE, H4 decoded MSE, endpoint identity, and continuity while preserving the Wave21 language effect. This directly tests whether multimodality—not missing language or source state—is causing mean cancellation.

If authorized, train a small `LCT-MD` mixture-displacement head rather than a large end-to-end policy: predict categorical mode probabilities plus per-mode direction and log-norm residuals, then decode the selected/sampled latent trajectory. Keep the representation, decoder, text projection, Wave21 split, six seeds, source-session bootstrap, and all historical rejection decisions frozen. Do not add endpoint classification, cycle loss, F2, DEL, or closed-loop execution.

## Relation to recent methods

The distributional direction is consistent with [Diffusion Policy](https://arxiv.org/abs/2303.04137), which models multimodal high-dimensional robot actions rather than regressing their mean. More recent flow approaches make the same issue explicit: [VFP](https://arxiv.org/abs/2508.01622) adds a variational prior, optimal-transport alignment, and mixture-of-experts specialization for task/path multimodality; [LG-Flow Policy](https://arxiv.org/abs/2601.23087) performs flow matching in a temporally regularized latent action space to improve smoothness; and [Latent Action Guided Flow Matching](https://arxiv.org/abs/2606.23420) replaces one global Gaussian with state-selected learned priors for fragmented, heteroscedastic action spaces. These papers motivate distributional modeling, but the present dataset has only 257 train transitions, so Wave25 should begin with a compact mixture head and an oracle/non-oracle mode diagnostic rather than immediately adopting a high-capacity diffusion or flow model.

If the compact mixture diagnostic fails, the next conclusion should be that the frozen latent representation lacks sufficient phase/contact information. Only then should the project add a phase variable or learn a temporally regularized latent action representation; it should not keep stacking geometric attraction losses onto the current coordinates.
