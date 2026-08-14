# Wave 21 execution log

- Restored only the official CALVIN training `auto_lang_ann.npy` by HTTP byte range using the pre-existing ZIP manifest; CRC matched.
- Found sparse annotation intervals with no exact end+1/start label changes. Continued with prospectively frozen next-annotation onset boundaries while preserving contiguous source frames and recording every gap/overlap.
- Prepared train/development latents and train-only action regions before training; held-out actions were not serialized.
- Trained all 18 preregistered models on GPU; no seed was added or removed. Development directional gate passed and final preregistration was frozen before test serialization.
- Initial final export exposed an inconsistent decode/re-encode implementation: decoder gripper logits had been passed directly to an encoder trained on -1/+1 gripper values. Corrected only this metric by thresholding logits under the frozen historical convention, then deterministically recomputed the same test predictions/metrics. No model, seed, split, threshold, or claim gate changed.
- Corrected cycle result failed G5; C7/C8 remained rejected.
