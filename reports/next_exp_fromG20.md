# Next Experiment from EXP_G20: EXP_G21 Adaptive Causal-F2 Trigger

## Hypothesis

G19--G20 show that checkpoint return is unnecessary in the tested regime, while G15--G18 repeatedly show that causal multi-proposal F2 sometimes changes success. A learned latent may contribute by recognizing decision states where the first native pi0.5 proposal has high realized regret and triggering matched-state branch execution only there. This tests a new resource-allocation control role rather than candidate ranking or recovery.

## Causal-regret representation

Build decision groups from audited G15--G18 artifacts. Each sample must contain the exact pre-decision physical state, active action, recent executed-action window, first untouched pi0.5 proposal, all candidates' physically realized scores/successes, and the realized regret of committing candidate zero instead of the best candidate. Split by source attempt.

Fit and compare:

1. a bottleneck causal-regret latent with regret and branch-needed heads;
2. a deterministic autoencoding regret latent;
3. a direct MLP classifier/regressor without a latent bottleneck.

Select the latent family on held-out branch-needed AUROC and realized regret, then freeze its trigger threshold using training/validation only. The trigger objective should penalize candidate execution cost rather than tune a threshold on prospective task success.

## Prospective execution

Run fresh matched closed-loop `lift -> place` rollouts on the ten G18 perturbed snapshots. At each decision query the first native pi0.5 proposal and compare:

1. latent adaptive F2: execute a three-proposal matched branch only when latent risk triggers;
2. direct-MLP adaptive F2;
3. always execute three-proposal causal state-value shooting;
4. never branch / single proposal;
5. a matched random trigger with the latent method's frozen trigger rate;
6. open-loop initial-observation control.

Triggered decisions must execute every candidate from the same complete checkpoint and commit from the restored realized state. Non-triggered decisions commit the first untouched proposal. Save trigger inputs/latents/probabilities, proposal bytes, branch outcomes, committed actions, F3 state, and final task outcome.

## Decision

The latent contributes if it strictly beats the direct trigger in ordered success, and either beats always-shooting in candidate-step efficiency at equal or better success or beats never-branching in success with fewer candidate steps than always-shooting. Ties do not support the claim. If direct or fixed policies dominate again, G22 will stop assigning the latent to decision gates and train an intervention-aware temporal F3 directly on counterfactual downstream regret, while keeping the empirically best state/action F2.
