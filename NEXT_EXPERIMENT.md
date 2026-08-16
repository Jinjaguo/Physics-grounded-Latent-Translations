# Post-Wave78 research program: EXP_R1

Wave78 did not meet the success gate, but the old Wave program reached its mandatory upper bound. Wave79 remains forbidden. The project now adopts the separately defined post-Wave78 direction: hierarchical latent path planning in the frozen action-coordinate space.

## Active direction

The new series began at `EXP_R1` (not Wave79). Its central question is whether a multi-step planner can connect ordered atomic action regions while preserving decoded-action continuity and empirical executable support. The initial stack freezes the action representation and decoder, uses frozen F1 as a local dynamics prior, introduces a new multi-step F2 trajectory planner, and uses oracle F3 subgoal boundaries with waypoint memory.

The first experiment must audit the real repository interfaces and data before optimization, then compare interpolation, F1 rollout, the historical F2 refiner, graph planning, trajectory optimization, sampling-based planning, and hybrid global/local planning. It must use oracle start and switch information first, so that path planning is tested independently of learned progress recognition.

Read the full protocol before starting:

- [`prompts/ACTIONS_AS_COORDINATES_POST_WAVE78_RESEARCH_DIRECTION.md`](prompts/ACTIONS_AS_COORDINATES_POST_WAVE78_RESEARCH_DIRECTION.md)
- [`prompts/EXP_R1_AUTONOMOUS_LATENT_HIERARCHICAL_CONTROL_CODEX_PROMPT.md`](prompts/EXP_R1_AUTONOMOUS_LATENT_HIERARCHICAL_CONTROL_CODEX_PROMPT.md)

EXP_R1 through EXP_R7 were **NOT_SUPPORTED**. EXP_R8 is **SUPPORTED** for offline multi-step path construction. The follow-up closed-loop program EXP_R9–EXP_R58 completed without establishing the full physical system: exact simulator/controller snapshots and causal action-conditioned robot state were unavailable, and F3 readiness did not pass. EXP_R58 is the hard upper bound; EXP_R59 is forbidden. See `FINAL_R9_R58_*.md` for the conservative final claims and required data collection.

## Historical Wave28–Wave78 termination

The old pointwise force-field program ended at Wave78, not because the research problem was abandoned, but because its registered upper bound was reached. Its negative intervention results remain part of the paper evidence and motivate the path-planning formulation above.
