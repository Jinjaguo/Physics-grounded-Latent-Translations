# Wave 30 — Decoder-Jacobian-aware low-rank projection

Wave29 showed that damping improves redirect but leaves decoded action jumps
far above the observed continuity scale.  Wave30 freezes the Wave28 adapter
and Wave29 alpha selection, then scales each causal latent residual using the
frozen decoder's local decoded-action displacement.  This tests whether the
latent-to-action projection, rather than q-direction alone, causes the jump.

The development sweep preregisters decoded-action caps {0.01,0.02,0.05,0.1,
0.2,none}; the scaling uses only the current base latent and the frozen decoder.
No future trajectory, contact, success, or physical return label is available.
Selection occurs on development, then Wave21/Wave27 held-out opens once.  If
continuity still fails, Wave31 must test a genuinely trainable Jacobian-aware
low-rank B with a matched frozen-decoder loss or conclude that the frozen
representation projection is the bottleneck.
