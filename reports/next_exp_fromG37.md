# Next Experiment from EXP_G37: EXP_G38 Causal Action-Support Proposal Selection

## Hypothesis

G33/G36/G37 show that direct physical residuals are the strongest learned verifier, while latent residuals duplicate them poorly. A learned action representation may still contribute at a different intervention point: after direct F2 rejects a candidate, a causal action-support latent trained from successful and failed executed branches can rank the remaining native proposals by recoverability and downstream support better than their arbitrary server order or raw-action distance.

## New model and labels

Build a training set from train-only matched-state branches plus executed candidate evidence from G33, G36, and G37. A row contains current state, active action semantics, native prefix, physically realized effect, direct-residual accept/reject, intervention status, and continuation outcome when available. Train within G38:

1. a supervised causal action-support latent with a compact embedding and recoverability head;
2. a pairwise/ranking latent that orders candidates from the same checkpoint;
3. a direct non-latent support classifier with matched inputs;
4. a shuffled outcome/support-label latent;
5. raw-action and random/native-order rankers.

Use grouped train/development splits by source checkpoint. Sweep representation dimension and loss weights only within G38. Freeze the selected ranker and direct residual verifier before prospective testing.

## Prospective intervention

Use the robust G35 three-stage `grasp -> lift -> place` system with autonomous state F3, exact current-state retargeting, and G33 direct physical residual verification. At each decision generate at least five common-noise native proposals. Execute candidate 0 first. When direct F2 rejects it, compare mechanisms that choose the next candidate by learned support latent, pairwise latent, direct classifier, raw-action diversity, native order, shuffled latent, and oracle realized recovery. Every selected proposal must be executed in LIBERO; rejected probes must be exactly restored; two real object interventions must be applied per rollout.

## Decision rule

The action-support latent contributes only if it strictly exceeds native-order and matched direct-classifier ranking in pooled strict three-stage success, has positive paired balance against both, reduces rejected candidate steps or improves recovery latency, and loses its advantage under shuffled support labels. It must operate on the same direct physical verifier, so any gain is attributable to proposal selection rather than a changed acceptance threshold.

If it ties or loses, the action-latent contribution remains unsupported. G39 must then test an action-latent proposal generator/decoder that creates new controls rather than selecting among fixed π0.5 samples, or simplify the paper claim if that also fails.

## Required evidence

Save source membership, grouped splits, causal labels, checkpoints/logs, cross-fitted rankings, frozen selection, common-noise proposal sets, every probe/restore/intervention, direct verifier outputs, selected proposal identities, autonomous F3 switches, strict three-stage outcomes, paired comparisons, runtime metadata, and an independent audit of the complete ranking-to-action-to-feedback chain.
