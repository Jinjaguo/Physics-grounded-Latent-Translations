# EXP_R19 report — interface-gated continuation

## Scientific question
Can the next learned residual and ensemble latent-plant families be evaluated without violating the closed-loop causal interface?

## Audit result
**NOT_RUN_INTERFACE_GATE**. This is a bounded gate audit, not a positive or negative physical task result.

## Concrete evidence
- Disk audit: available bytes=911885660160 (floor=300000000000, passed=True).
- The repository's retained complete CALVIN episode schema is action-only (`rel_actions`, `global_frame_indices`); Wave27 observation windows contain `robot_obs` and `scene_obs` but do not contain a full Bullet snapshot.
- The historical closed-loop state audit is preserved at `results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/calvin_closed_loop_state_audit.md` and its not-run manifest at `results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/closed_loop_not_run_manifest.json`.

## Why this EXP cannot claim held-out control
the retained action-only complete episodes contain no action-conditioned robot/simulator snapshot; R9-R16 already exhausted valid latent surrogates. Opening a held-out physical evaluation under these conditions would not be causal and would repeat the documented reconstruction-gate failure, so no held-out metrics are fabricated. Frozen representation, decoder, F1, old F2, and R8 results remain unchanged.

## Required change
acquire exact causal simulator state or a supported action-conditioned dataset before training a residual plant.

## Decision
`SUCCESS=false`; EXP_R20 is the next bounded audit.
