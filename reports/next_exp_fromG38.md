# Next Experiment from EXP_G38: EXP_G39 Latent Corrective Proposal Generation

## Hypothesis

G38 found weak support-ranking signal but showed that reordering fixed π0.5 samples is bounded by the proposal set. A conditional action latent trained on matched-state executed branches can decode a new corrective five-action prefix after candidate-0 failure that improves recovery over five-sample native search, raw-action interpolation, and a direct action-regression generator.

## New model family

Train grouped held-attempt-out generators from causal candidate sets:

1. a conditional variational action autoencoder whose latent is shaped by realized physical effect and direct-verifier acceptance;
2. a conditional diffusion/flow-style low-dimensional latent generator if feasible with local dependencies;
3. a deterministic direct corrective-action regressor;
4. raw control-space interpolation between accepted proposals;
5. shuffled effect/support conditioning and latent-prior ablations.

The generator must produce controls not already present in the five native proposals. Select latent dimension, decoder loss, and sampling rule using training/development checkpoints within G39, then freeze before test.

## Causal evaluation

Retain G35 state F3 and G33 direct physical verification. Generate and execute candidate 0 from common π0.5 noise. If rejected, compare a decoded latent correction, direct generated correction, interpolated correction, best G38 pairwise-ranked native proposal, five-sample native search, shuffled generator, and physical oracle. Restore the exact checkpoint before every probe, execute the generated actions through LIBERO, observe realized feedback, and continue from the accepted state. Apply two real object interventions per rollout and save generated controls plus nearest-native distances to prove novelty.

## Decision rule

The latent generator contributes only if its decoded proposals are measurably distinct from all native samples, it strictly exceeds direct generation and five-sample native search in pooled strict success with positive paired balance, reduces rejected work or recovery latency, and loses the advantage when effect/support conditioning is shuffled. A representation-only reconstruction gain is insufficient.

If it fails, G40 should stop adding action-latent F2 variants and run a final integrated simplification/acceptance study that explicitly contrasts the best verified direct/state system against all required open-loop, teacher-forced, restart, no-F2, no-F3, no-protection, and no-memory baselines; the paper conclusion would then be that action latents did not survive causal closed-loop ablation.

## Required evidence

Save grouped causal membership, generator checkpoints/logs, generated actions, nearest-native distances, matched noise, every executed probe and restore, physical residuals, autonomous switches, strict outcomes, paired comparisons, runtime metadata, and an independent audit of generation novelty and the complete action-to-feedback chain.
