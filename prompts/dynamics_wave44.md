# Wave44 matched-state contrastive bridge

Wave43 found that task/source reweighting does not fix prospective retargeting.
Wave44 adds a batch contrastive objective: at matched current physical/latent
states, the arriving instruction prediction must align with its target future
latent and repel other-task hard negatives.  Compare temperatures, contrastive
weights, delta/state/integrated inputs, and q=2/4/8.  Keep VAE/decoder/F1/F2
frozen, do not use future targets as inputs, and continue unless success or
Wave78.
