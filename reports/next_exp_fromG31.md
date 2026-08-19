# Next Experiment from EXP_G31: EXP_G32 Direct-Utility Sequential Recovery

## Hypothesis

G31 shows that learned representation verifiers add error to an otherwise useful recovery mechanism. The hypothesis is that a transparent sequential controller using directly realized current-action utility can preserve exhaustive shooting's success while requiring materially fewer rejected simulator steps. This explicitly simplifies F2 and tests whether learned candidate scoring should be removed from the best system.

## Cross-fitted stopping policies

Use the audited G24 matched-state branch utilities to choose, before testing, several genuinely different sequential policies:

1. absolute-utility stopping: accept the first native candidate whose realized task utility exceeds a cross-fitted threshold;
2. improvement stopping: accept candidate zero if its realized improvement is positive, otherwise restore and accept the first candidate that improves over candidate zero by a cross-fitted margin;
3. stage-specific stopping: separately calibrate lift and place utility thresholds using the explicit stage in the physical state;
4. G31 learned physical verifier as a comparison.

Threshold/margin grids belong to this single EXP. Save cross-fitted choices, mean realized utility, oracle regret, candidate trials and acceptance position. No latent features are allowed in the three direct policies.

## Prospective causal protocol

Run three common-noise repeats over all ten canonical starts with autonomous F3. Compare:

1. absolute direct-utility sequential recovery;
2. improvement-based sequential recovery;
3. stage-specific direct-utility recovery;
4. G31 learned physical verifier recovery;
5. exhaustive physical shooting;
6. G28 point ranking;
7. single pi0.5;
8. initial-observation open loop.

Every direct policy must execute native proposals, measure realized physical effect, accept the already-reached state or restore the exact full checkpoint, and replan from actual feedback. Save the same trial/checkpoint evidence as G31 plus the direct utility decomposition and chosen threshold/margin.

## Decision rule

The simplified controller is supported if at least one direct sequential policy matches or exceeds exhaustive shooting's pooled ordered successes while using strictly fewer rejected candidate steps, and strictly exceeds single pi0.5. A tie among direct policies is resolved first by fewer rejected steps, then endpoint error. If learned physical verification is not better, remove it from the best F2. G33 must then use the selected non-latent F2 as a fixed backbone and test a separate latent-memory contribution under controlled perturbations; it must not return to latent proposal scoring, latent CEM, or action/effect verifier correspondence.
