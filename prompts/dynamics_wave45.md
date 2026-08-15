# Wave45 decoder-tangent basis alignment

Wave44's contrastive objective did not yield action alignment.  Wave45 derives
the low-rank force basis from the frozen decoder's local action Jacobian at
current latents, then compares decoder-tangent, residual-PCA, and random bases
with delta/state/integrated bridges and q=2/4/8.  The decoder and behavior
backbones remain frozen; no future action or latent is an input.  Continue
unless success or Wave78.
