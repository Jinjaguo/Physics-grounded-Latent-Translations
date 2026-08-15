# Wave 33 — Mixture of local intent fields

Wave32 found that one state-conditioned low-rank basis still trades execution
redirect for continuity.  Wave33 trains two small low-rank field experts and a
causal gate from the current latent.  The frozen action-text VAE/decoder and
F2 backbone remain unchanged.  q={2,4}, gate temperature={0.5,1}, and the
mixture is compared with a single-field control on development before one
held-out opening.  No future input or explicit return flag is allowed.

If a mixture improves redirect but not decoded identity/continuity, the evidence
will indicate that the bottleneck is the frozen action projection or missing
ordered retarget data, not insufficient field expressivity.
