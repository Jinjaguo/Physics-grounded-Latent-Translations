# Wave 24 paired transition inventory

Inventory rows: **560** across **31** sessions; split counts `{'train': 257, 'development': 139, 'test': 164}`. Train/development paired arrays were reconstructed exactly; 164 test rows contain metadata with null latent/action fields so held-out arrays remain unopened.

Every row is physically contiguous, crosses no reset, uses the next annotation's true start, and retains the original annotation gap without synthetic labels. H1/H2/H4 are 16/32/64 frames.
