# Next Experiment from EXP_G17: EXP_G18 Realized-Transition Latent Selection

## Hypothesis

G17 shows latent residuals have signal but modifying π0.5 actions costs accuracy. A latent may be useful as a coordinate over physically realized candidate transitions rather than as an action decoder. Encoding `pre-state + raw proposal + realized next-state` and learning phase-conditioned progress geometry may select better untouched π0.5 proposals than the existing state-value scorer.

## Training

From all audited G15–G17 causal branch traces, build candidate-level transitions containing pre-state, untouched raw π0.5 chunk, realized endpoint/delta, phase, success, retrospective lift/place progress, and source attempt. Fit and compare:

1. supervised contrastive transition encoder with phase-progress and success heads;
2. deterministic transition autoencoder plus progress head;
3. direct state-transition MLP without a latent bottleneck.

Split by source attempt. Select by held-out candidate-ranking accuracy and realized success-regret, not embedding reconstruction alone. Save exact membership and checkpoints.

## Causal test

Create a frozen ten-case 8 cm cardinal perturbation cohort, falling back once to 7.5 cm for the entire cohort if pre-rollout physics is invalid. At each decision query three untouched π0.5 proposals, actually execute each from the same complete snapshot, then compare selectors over identical realized candidates:

1. selected transition-latent progress score;
2. direct transition MLP score;
3. existing state-value score;
4. random/first candidate;
5. single raw π0.5;
6. open loop;
7. unperturbed replay.

All action chunks must remain byte-identical across selectors before branch execution. Save candidate set identifiers/actions, realized transitions, latent coordinates/scores, selected/committed actions, observations, F3 state, and final metrics.

## Decision

Transition latent contributes only if it strictly beats both direct MLP and existing state-value selection in ordered success, and beats first/single. Ties do not support the claim. If direct/raw baselines remain best, G19 must pivot to checkpoint recovery/state memory, where compression could provide a distinct control function, rather than another proposal scorer.
