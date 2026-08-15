# Wave 31 — Zero-initialized trainable low-rank gate

Wave30 showed that post-hoc decoder caps do not repair continuity.  Wave31
keeps the frozen action-text VAE/decoder and F1/F2 backbone but trains only a
small low-rank intent adapter plus a zero-initialized per-horizon gate.  The
gate is learned with frozen-decoder action loss, continuity loss, anchor
distillation, and latent direction loss.  q={2,4,8} and continuity weights
{0.3,1,3} are selected on development; held-out opens once after freeze.

No explicit return flag or future target is supplied.  If the gate collapses
to zero, that is evidence that the frozen action representation cannot support
the desired editable path under the current data; Wave32 should then test a
state-conditioned or piecewise field rather than increasing the VAE.
