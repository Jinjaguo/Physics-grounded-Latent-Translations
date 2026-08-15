# Twenty-fifth wave next experiment

## Decision from Wave 25

The oracle is strong and Phase-flow passed five of six development gates, but endpoint identity conflicts with continuity. Wave26 should enrich the causal state with longer recent history and current contact/gripper phase, then test whether the same latent flow can resolve that trade-off before any closed-loop claim.

Keep the Actions-as-Coordinates main line: current latent + next atomic language should produce a local editable transition. Do not reopen DEL, static endpoint attraction, or global cycle projection. Any retarget/return experiment remains incremental and must distinguish stored-waypoint recovery from physical time reversal.

## Recommended Wave 26 implementation

Freeze `phase-aware latent flow with richer causal history/contact state and explicit identity-continuity mechanism diagnostics` as the empirical starting point. Add only causal information available online: at least three recent latent/action chunks, current gripper/contact proxies measured at or before the query, and an explicit transition-phase state. First diagnose whether this state separates the cases where endpoint identity and continuity disagree; then retrain the same compact flow and matched F2-C/RAT-C controls. Include an additional-data condition because 257 train transitions may limit causal phase inference. Use source-session separation and a single frozen held-out evaluation.

## Relation to recent work

[LG-Flow Policy](https://arxiv.org/abs/2601.23087) motivates temporally regularized latent action flow when raw-space flow is insufficiently smooth. [Latent Action Guided Flow Matching](https://arxiv.org/abs/2606.23420) motivates state-selected learned priors for fragmented, heteroscedastic action spaces. [3D FlowMatch Actor](https://arxiv.org/abs/2508.11002) shows that targeted flow architectures can retain fast inference, while [BAKU](https://arxiv.org/abs/2406.07539) reports gains from a multimodal VQ-BeT action head. These methods motivate the next implementation only where Wave25's oracle/causal gap supports it; they do not override the small-data evidence here.
