# Next Experiment from EXP_G22: EXP_G23 Joint Action-Bottleneck Predictive Control

## Hypothesis

G15--G22 repeatedly reject latents trained after a controller decision has already been defined. A latent may become control-relevant only when its encoder and executable action decoder are trained jointly with an action-conditioned world model, so that changing a coordinate necessarily changes a predicted and then realized physical transition. The hypothesis is that a jointly trained action bottleneck can rank or refine native pi0.5 chunks with better end-to-end success than a capacity-matched direct action world model, while avoiding the candidate-execution cost of matched simulator shooting.

This mechanism follows the action-token causal bias of [QueST](https://arxiv.org/abs/2407.15840), the policy-plus-predictive-model structure of [Generative Predictive Control](https://arxiv.org/abs/2502.00622), and the joint latent-action/world-model objective of [LAWM](https://arxiv.org/abs/2509.18428), but will be implemented on the project's existing audited LIBERO transitions rather than importing unverified benchmark claims.

## Dataset and leakage control

Rebuild candidate-level transitions from audited G15--G18 branch artifacts. Each item must contain the exact pre-decision EEF/object state and active-action one-hot, the first five actually executed controls of a native 10x7 pi0.5 chunk, the physically realized EEF/object displacement after that prefix, success, retrospective value, source artifact, decision, candidate, and attempt. Never use an unexecuted padded suffix as a transition target.

Create ten leave-one-attempt-out folds. Every checkpoint used for a prospective attempt must exclude all prior transitions from that attempt. Save exact fold membership and normalization. Compare at least these model families under comparable widths and training budgets:

1. a vector-quantized joint action world model with an action encoder, discrete codebook, executable action decoder, and state-conditioned transition/value heads;
2. a continuous joint action bottleneck with the same decoder and heads;
3. an action-only autoencoder followed by a separately fitted transition head, isolating joint training;
4. a direct state-plus-action world model without a latent bottleneck.

Training losses must include executed-prefix reconstruction, realized displacement, success/value prediction, and the appropriate VQ or bottleneck regularizer. Select between VQ and continuous joint latents using only fold-level realized-transition and within-group ranking metrics, before reading prospective outcomes.

## New causal controller evaluation

Freeze the G22 winner, state-window F3, and keep pi0.5 as F1. On the ten matched G18 perturbed snapshots, run fresh stochastic closed-loop rollouts for:

1. selected joint-latent predictive control: encode native proposals, make a bounded latent/code intervention, decode executable chunks, predict their outcomes, execute the best decoded prefix, observe the realized state, and replan;
2. the other joint latent family as a model-family ablation;
3. action-only latent plus post-hoc transition head;
4. direct world-model ranking of untouched native proposals;
5. decoded-first-proposal without latent optimization, measuring reconstruction damage;
6. physical matched-branch state-value shooting;
7. single raw pi0.5;
8. initial-observation open loop.

The model-based methods must not silently use simulator candidate outcomes for selection. They must save pre-state, raw proposals, exact latent/code, intervened latent/code, decoded action bytes, predicted displacement/value, selected index, committed actions, realized next state, prediction residual, F3 state, and final task outcome. Physical shooting must continue to save all matched branch outcomes. Every selected action must be executed through LIBERO and followed by a new observation; offline ranking alone is invalid.

## Decision rule

The latent contributes only if the preregistered selected joint-latent method strictly beats the direct world model and decoded-only/action-only controls in ordered success. It must additionally either match or beat physical shooting in success while using fewer candidate-executed steps, or strictly beat physical shooting in success. Endpoint error, transition residual, jerk, and compute are tie-breakers only after equal success. If both joint latent families lose, the hypothesis is not supported; G24 must move to observation-transition latent actions or policy-side finetuning rather than another post-hoc state/action head.
