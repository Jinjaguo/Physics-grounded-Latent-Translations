# Wave 29 — Damped latent force-field composition

Wave28 found positive but small low-dimensional redirection and a large
continuity/identity cost.  Wave29 freezes the Wave28 action-text VAE, decoder,
F1/F2 backbones, data splits, and selected checkpoints.  It tests whether the
failure is caused by residual amplitude rather than by the intention direction.

The only new operation is causal residual damping and norm clipping applied to
the Wave28 residual.  Development selects from q={2,4,8}, alpha={0.1,0.25,0.5,
0.75,1.0}, and caps={0.05,0.1,0.2,0.5,none}; held-out is opened once after
selection.  No future latent, action, contact, or success signal is used.
Wave21 ordered events provide return-direction diagnostics; Wave27 remains a
neutral→target prospective set because previous labels are unavailable.

Success requires lower continuity without losing positive execution redirect.
If that fails, Wave30 must test decoder-Jacobian-aware projection or identify
ordered-data insufficiency; it must not enlarge the frozen VAE or append a
complete future trajectory.
