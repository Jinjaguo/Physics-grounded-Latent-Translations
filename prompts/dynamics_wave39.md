# Wave39 semantic/action hard-negative anchors

Wave38's gate reduced disturbance but did not create target identity.  Wave39
adds explicit frozen semantic anchors and hard negatives: the adapted latent's
semantic slice should approach the arriving instruction embedding and move away
from the current instruction embedding, while the decoder action loss keeps the
force executable.  Compare delta/state/integrated causal inputs, q=2/4/8,
PCA/random bases, anchor weights, and margins.  No target action or future
latent is an inference input.  Continue unless success or Wave78.
