# Wave 28 recommended experiment

Wave27's best prospective model is `Core_LN100_PH0_RAT-C`, but readiness is `False`. Failed criteria: H2_full, H4_continuity, H4_decoded.

Stay on the actions-as-coordinates main line: preserve the frozen representation and prospective session discipline, then target only the failed causal bottleneck. If readiness is false, collect phase-balanced independent episodes for the weakest goals, add a compact query-time phase estimator from measured robot state, and train retrieval-conditioned rectified flow with calibrated causal candidate scoring. If readiness is true, run the preregistered offline retargeting/interrupt-return pilot without changing the selected model.

Do not claim contact, cross-collector behavior, or recoverability until those signals and evaluation units exist. A useful method direction is retrieval-augmented action flow with trajectory-level consistency, retaining RAT-C and Phys-F2C controls. Two current primary references motivate this without changing our claim: WorldScape Policy 2.0 (causal short/long event memory, https://arxiv.org/abs/2607.18840) and LaWAM (compact latent dynamics-aware action generation, https://arxiv.org/abs/2606.15768). Flow Policy Gradients (https://arxiv.org/abs/2602.02481) is relevant only after an execution reward exists, so it is not the immediate offline method.
