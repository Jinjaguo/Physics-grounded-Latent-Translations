# Wave35 temporal/state-action bridge experiment

This wave resumes the main research program after the Wave34 representation-stop audit.  The audit stops stacking the same frozen-latent adapter; it does not end the program.  Only two conditions may end the autonomous program: a successful method, or completion of Wave78.  Wave79 must never be started.

Keep the action-text VAE, decoder, and F1/F2 behavior backbones frozen as the primary controls.  Introduce a temporally and state-conditioned bridge that maps the ordered current instruction, arriving instruction, current latent/action history, and a small event-time coordinate to a continuous low-dimensional force.  Do not provide future action chunks, future latent targets, or post-event information at inference.

Compare several bridge families in one wave: text-delta only, state-action conditioned, history/contact conditioned, phase-gated, and recurrent/integrated force trajectories.  Compare q dimensions 2/4/8, PCA versus learned low-rank projections, and continuity/action/latent objectives.  Use Wave21 ordered transitions for training/development and evaluate once on the session-disjoint Wave21 test and the independent Wave27 prospective set.  Wave27 missing previous-instruction labels must remain explicitly marked as neutral-anchor evidence, not silently imputed return data.

Record all failed attempts and continue to Wave36 unless the method succeeds.  A representation/data limitation is a reason to change the next method, never a reason to end before Wave78.
