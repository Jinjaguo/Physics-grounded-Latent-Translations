# Next Experiment from EXP_G16: EXP_G17 Identity-Preserving Causal Residual Latent

## Hypothesis

G16 failed because its decoder replaced a fully observation/language-conditioned π0.5 action chunk with an unconditional phase-manifold projection. A useful action coordinate should instead represent optional residual changes around the raw base, keep zero residual exactly equal to the base, and be trained on physically realized differences between matched candidate interventions. Such a residual latent may improve difficult states without sacrificing base-policy successes.

## Causal residual dataset

Construct a new dataset from saved real candidate branches in formal G15 and G16 raw-π0.5 rollouts. For every decision with at least two proposals, reconstruct the pre-decision robot/book state from the committed-action cursor; record phase, base proposal, alternate-minus-base action residual, base and alternate realized endpoint states, success indicators, value/progress difference, and the exact source artifact/decision index. All entries must come from candidates actually executed from the same restored snapshot.

Fit and compare in this EXP:

1. a state/phase-conditioned residual autoencoder with an identity-safe decoder `base + D(z,s,g) - D(0,s,g)`;
2. a state/phase-conditioned residual beta-VAE with the same identity path;
3. a discrete residual codebook obtained by clustering encoded causal residuals and ranking codes by held-out realized progress within phase/state neighborhoods.

Jointly predict the alternate-minus-base realized end-effector/book displacement and success/value advantage. Split by source attempt, not individual decisions, so validation cannot share trajectories with fitting. Select family, latent dimension/codebook size, and candidate codes on existing G15/G16 evidence only.

## Frozen prospective intervention test

Freeze the model and selection before creating controller outcomes. Construct ten physically validated 7 cm cardinal perturbations from G14 snapshots, using directions not used for the same attempt in G16. If any snapshot is physically invalid before rollout, apply a single predeclared 6.5 cm fallback to the entire cohort; do not adapt to controller results.

At every decision, query one raw π0.5 base chunk. Compare equal three-candidate budgets:

1. causal residual-latent shooting: untouched base plus two decoded state-conditioned residual codes;
2. outcome-ranked residual codes without branch execution, to isolate the learned outcome head;
3. equal-budget independent raw π0.5 shooting;
4. base plus shuffled residual codes, a mechanism ablation;
5. single untouched raw π0.5;
6. initial-observation open loop;
7. unperturbed replay.

For shooting methods, actually restore and execute every candidate before selecting and committing. Log base/residual/code/decoded actions, pre-state, predicted outcome, realized candidate state, selected index, committed action, next state/observation latent, F3 state, and end-to-end metrics.

## Decision rule

Primary metric is ordered lift→place success. Residual latent contributes only if causal residual shooting strictly beats equal-budget raw π0.5 shooting, shuffled residuals, and single raw control. It must never do worse than single on a case solely because the identity candidate was omitted; candidate zero must be byte-identical to the base proposal. Endpoint error and efficiency are secondary tie diagnostics, not substitutes for success.

If residual latent still ties raw/single, the action-latent control claim remains unsupported and G18 should move the representation to state-transition goals or recovery memory rather than another decoder sweep. Save dataset provenance, checkpoints/logs, frozen manifest, snapshots, every rollout, metrics, and an independent residual/decode/commit audit under `experiments/EXP_G17/`.
