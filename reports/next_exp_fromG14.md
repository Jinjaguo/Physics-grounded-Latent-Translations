# Next Experiment from EXP_G14: EXP_G15 Frozen Perturbation-Cohort Ablations

## Hypothesis

Outcome-latent F3 only separated from state F3 through switch timing on an all-success cohort. Under prospective object-position shifts, the complete causal system should preserve successful lift-to-place composition more often than state-only F3 and should strictly outperform F2-disabled, F3-disabled, restart, open-loop, and action-replay ablations.

## Frozen perturbation cohort

Use the ten audited EXP_G14 snapshots for official init indices 40–49. Before any policy action, modify the verified `black_book_1_joint0` free-joint XY coordinates by a predeclared 4 cm offset, zero its free-joint velocity, advance MuJoCo with five dummy controls, and capture a new exact snapshot. Construct and validate every snapshot before executing any evaluated controller. A perturbation is invalid if the realized XY displacement differs from the request by more than 5 mm, its Z changes by more than 5 mm, or any coordinate is non-finite; such a construction failure does not consume EXP_G15 and must be fixed inside it.

The first mixed cardinal/diagonal construction is retained in `experiments/EXP_G15/` as gate evidence because attempt46 became physically unstable during settling. The valid scientific retry uses cardinal directions only and writes to `experiments/EXP_G15_retry1/`; no preliminary metrics count toward the EXP result.

## Executed comparisons

On every validated snapshot, execute:

1. complete outcome-latent F3 plus two-proposal causal state-value F2;
2. matched state-only F3;
3. F2-disabled single π0.5 sampling;
4. F3-disabled future-visible prompting;
5. restart-after-switch instead of realized-state continuation;
6. open-loop actions generated from repeated copies of the perturbed initial observation;
7. same-attempt unperturbed G14 action replay.

All methods use the same 300-step horizon and exact perturbed snapshot. Save each intervention, chosen or fixed action, realized state/feedback, success, endpoint error, lift/composition state, switch trace, and causal execution counts.

## Metrics and decision

Primary metric is official success count on the valid ten-case cohort. The complete system establishes an integrated contribution only if it strictly beats every ablation in success. Switch error, endpoint error, action jerk, and candidate/committed computation are descriptive secondary metrics and cannot overturn a success tie.

If the complete method ties state or the single-proposal baseline, the latent/F2 contribution is not supported and EXP_G16 must change the mechanism rather than tune one coefficient. If feedback methods beat open-loop/replay but complex variants tie, simplify the candidate final system around the strongest causal baseline.

## Required artifacts

Save the exact perturbation manifest and snapshots, all 70 rollout traces, case metrics, aggregate metrics, command/environment metadata, and an independent audit. Preserve the invalid mixed-pattern trial separately as failed construction evidence, report only the valid retry as EXP_G15, then write the next executable experiment.
