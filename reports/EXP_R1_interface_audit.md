# EXP_R1 interface audit

Created: 2026-08-16T03:46:06-04:00

- Representation: `checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt`; manifest SHA-256 `44ae3c4c9810fdfd44e396d44e0f346a57e9e5c0b4349babc0292960c255fb65`.
- Latent interface: 32 dimensions, 16 semantic prefix + 16 execution suffix.
- Action interface: normalized CALVIN `rel_actions`, 16 frames, 7 channels.
- Decoder: frozen representation decoder, output `(16, 7)` normalized actions.
- F1: `results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F1_execution_mlp.pt`, SHA-256 `41f63d173c919cc01d5f5cbfab3af41983813a63ed018c43dbbc28ddd1df9fb0`; execution-only MLP consumes previous/current execution, current semantic, and causal projected text context.
- Historical F2: `results/dynamics/fifteenth_wave/2026-08-12_dynamics_3/checkpoints/F2_matched_refinement.pt`, SHA-256 `9b19c0c3c47994c734eccfae4a8070a29b8ff35020ac1e7ef04e0f7f8d9be308`; matched iterative refinement, frozen baseline only.
- Physical fields available in each window: `robot_obs (128,15)`, `scene_obs (128,24)`, `rel_actions (128,7)`, and contiguous global frame indices.
- Oracle F3 interface: collection boundary frame and source-session split are available; no learned completion model is used.
- Waypoint/return fields: robot observations are present, but EXP_R1 does not claim return; this experiment isolates path planning.

No guessed field names were used. Representation, decoder, F1, and F2 optimizer/update counts are zero by protocol.
