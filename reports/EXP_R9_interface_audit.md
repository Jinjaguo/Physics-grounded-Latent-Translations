# EXP_R9 interface audit

## Recovered interfaces

- Frozen representation: 32-D latent with 16 semantic and 16 execution coordinates; frozen decoder maps one latent to a normalized `(16, 7)` CALVIN action window.
- F1 consumes previous/current 16-D execution states, current 16-D semantic state, and projected target text. Historical F2 consumes the same causal context and returns an execution refinement.
- EXP_R8 supplies a four-waypoint path with a late terminal repair; its exact implementation is frozen as a baseline.
- Complete episode files contain `rel_actions (T,7)` and contiguous global frame indices. EXP_R3 encodes consecutive H16 action windows; the four post-boundary latent windows are the only exact next observations available for each case.
- Wave27 files additionally contain `robot_obs (128,15)` and `scene_obs (128,24)`, but the retained data do not contain a full Bullet snapshot, controller target state, contact state, or exact source branch. The historical CALVIN state audit records this reconstruction failure.

## Closed-loop surrogate

R9 therefore uses causal teacher-forced latent replay: plan H waypoints, consume P waypoints, expose only the next recorded post-boundary action window as the resulting observation, re-encode it with the frozen representation, and replan. This is a valid offline feedback/replanning surrogate for latent path tracking, but it is not physical MPC and does not claim that the planned action changed the recorded environment.

## Timing and oracle F3

The representation supplies one latent per 16-frame H16 window. H=2/4 and P=1/2 are supported by four hidden chunks; H=8 is not fabricated. The annotation boundary and target region provide oracle switching. No learned F3, waypoint controller, or return interface is available in R9.
