# Wave-18 reconstruction gate failure

The planned closed-loop causal continuation study could not be executed because the retained public CALVIN artifacts do not permit exact reconstruction of source branch states. `robot_obs` and `scene_obs` omit movable-object velocities, complete joint/controller state, and contact/constraint state; the retained continuous-play ranges are not known-reset episodes. The installed `reset` also advances one physics step.

Independent simulators agree on continuous state after receiving the same approximate observation reset, but the exposed contact sets differ and neither simulator is proven identical to the recorded source trajectory. The measured source replay deviation and missing state make the intended causal branch comparison invalid.

Per `prompts/dynamics_6.md`, wave 18 stopped before representation/F1/F2/DEL inference. Closed-loop refinement did not fail; it was not evaluated. C4, C5, and C6 remain not tested. See `reconstruction_gate.json`, `zero_intervention_twin_replay_report.json`, and `calvin_closed_loop_state_audit.md`.
