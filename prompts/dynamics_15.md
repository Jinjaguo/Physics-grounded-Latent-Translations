# Wave 27 Codex Prompt
# Prospective Data Expansion + Physical-State-Enriched Retrieval/Flow Tournament

## 0. Mission

Wave 27 is a broad implementation and data study, not a single-intervention rescue.

Wave 26 compared 79 development entries and froze three held-out candidates: Prior-CFM, History-CFM, and RAT-C. RAT-C was the strongest held-out model overall, with H2 full MSE 0.9052, H4 decoded MSE 0.04850, endpoint identity 0.4238, RedirectGain 0.3604 [0.2955, 0.4633], and Execution RedirectGain 0.1878 [0.1521, 0.2442]. READY_FOR_RETARGETING_TEST remained false because endpoint/decode-reencode identity were still too low.

The next scientific question is:

> Does genuinely new, source-session-independent paired transition data with synchronized physical state make language-conditioned latent dynamics more learnable, and which implementation best uses that information: retrieval, retrieval-initialized flow, state-selected-prior flow, temporally coherent flow, or compact deterministic controls?

Do not organize this wave around one binary gate. Use multiple independent claim decisions, a development scorecard, Pareto analysis, and a final prospective evaluation.

---

## 1. Preserve the central story

The strongest scientific result remains:

> Changing only the next-goal language causally redirects predicted latent dynamics, including execution-space coordinates.

Historical Wave21:

- Full RedirectGain = 0.250126, 95% CI [0.136495, 0.370798].
- Execution RedirectGain = 0.183855, 95% CI [0.100917, 0.263777].

Wave26 also preserved a strong language effect in RAT-C.

Do not weaken this because later transition implementations were imperfect.

Wave27 investigates realization of the language-conditioned vector field, not whether language has an effect.

---

## 2. Wave26 interpretation to preserve

Freeze the following facts:

- Development entries: 79.
- Best state configuration: S0.
- Higher-dimensional latent/action histories did not stably improve matched models.
- S5 was only a causal contact proxy, not true contact.
- S7 true proprioception was unavailable.
- D0->D1->D2 H2 performance improved monotonically for all three representative models.
- D3 was unavailable because there were no remaining source-session-independent compact sessions.
- C19 continuous flow strongest family = NOT_SUPPORTED.
- C20 identity-continuity trade-off reduction = NOT_SUPPORTED.
- C21 more transition data helps = MIXED.
- C22 language and state shape transition distribution = MIXED.
- READY_FOR_RETARGETING_TEST = false.

Interpretation:

> The evidence does not support simply adding more latent-history dimensions. The next missing variables are genuine independent transition coverage and synchronized physical state.

---

## 3. Relevant implementation motivation

Use recent methods only as design motivation:

- LAFM / arXiv:2606.23420 motivates state-selected source priors for fragmented, heteroscedastic action distributions.
- CoLA-Flow / arXiv:2601.23087 motivates temporally coherent flow in continuous latent action spaces.
- 3D FlowMatch Actor / arXiv:2508.11002 motivates efficient flow architectures for robot trajectory prediction.

Do not reproduce these systems end-to-end. Wave27 remains a latent-transition study.

---

## 4. Wave27 has six study axes

A. Prospective independent transition collection.
B. Synchronized physical-state recording.
C. Data-scale and coverage analysis.
D. Retrieval / transition-memory methods.
E. Retrieval-initialized and state-selected flow methods.
F. Simple factorized / physical-state controls.

All axes run on TRAIN/DEVELOPMENT first.

The new prospective test remains sealed until final candidate selection is frozen.

---

## 5. Frozen assets

Freeze and hash:

- CALVIN action encoder.
- CALVIN decoder.
- semantic projection.
- text encoder.
- normalization.
- Wave21 B0/B1.
- Wave24 paired-transition dataset.
- Wave26 valid implementations and exact seeds.
- Wave26 corrected indexing and initialization protocol.

No representation/decoder/text updates.

Write `wave27_frozen_manifest.json`.

---

## 6. Prospective independent transition collection

Wave27 must create genuinely new source-session-independent paired transitions.

Do not reuse development/test sessions.

Do not manufacture independence by overlapping windows.

Each source session gets a unique prospective session ID.

Audit available collector routes before collection:

1. Existing official/human CALVIN play infrastructure.
2. Existing trained policy/agent as source-policy collector.
3. Verified scripted/primitive controller.
4. Human keyboard/SpaceMouse/manual teleoperation if available.

Never assume a collector exists. Verify first.

Tag every record with:

- collector_type.
- collector_version.
- policy/checkpoint ID if applicable.
- environment seed.
- policy/operator seed.
- source_session_id.

Write `wave27_collection_capability_audit.md`.

---

## 7. Required synchronized signals

For every new transition record, save as many of the following as truly available:

- timestamp / simulator control step.
- 7-D robot action.
- joint positions q.
- joint velocities dq.
- TCP/end-effector pose.
- TCP linear velocity.
- TCP angular velocity.
- gripper width/finger positions.
- gripper command.
- object poses if causally observable.
- true simulator contact events.
- contact body IDs.
- contact count.
- contact normal/force if available.
- external atomic language goal.
- annotation/task predicate.
- success indicator if available.

Do not call a heuristic contact proxy "true contact".

Do not reconstruct missing proprioception after the fact.

---

## 8. Collection targets

Preferred:

- >=600 new certified transitions.
- >=80 per next-goal class where feasible.

Minimum useful:

- >=300 new certified transitions.
- >=40 per goal where feasible.

If fewer than 150 genuinely new transitions are obtained, still run diagnostics but classify the data-expansion evidence as limited.

Do not stop the whole wave merely because the preferred target is missed.

---

## 9. Transition certification

Each new transition must satisfy:

- physically contiguous source stream.
- no reset between current and future chunks.
- valid H1.
- valid H2.
- preferred valid H4.
- no NaN/Inf.
- synchronized action/state/goal.
- unique source session.
- traceable source metadata.

If contact is recorded, audit synchronization between contact data and simulator state.

Write `wave27_collection_report.md`.

---

## 10. New source-session split

Split NEW sessions before model training.

Preferred:

- 70% NEW-train.
- 15% NEW-development.
- 15% NEW-prospective-test.

Use source-session separation.

Stratify by next goal and collector type when possible.

Write `wave27_new_data_split_manifest.json`.

---

## 11. Data conditions

Define:

- DATA-L0 = legacy 257 train transitions only.
- DATA-LN25 = legacy + 25% of NEW-train sessions.
- DATA-LN50 = legacy + 50%.
- DATA-LN100 = legacy + all NEW-train.
- DATA-NEW = NEW-train only.

Freeze the nested session subsets.

This explicitly tests whether improvement comes from sample count versus a new collector/domain distribution.

---

## 12. Coverage analysis

For each data condition report:

- transition count.
- source sessions.
- per-goal counts.
- current-latent coverage.
- execution-latent coverage.
- nearest-neighbor radius to dev/test.
- displacement coverage.
- contact-state distribution.
- gripper-state distribution.
- TCP/joint velocity distributions.
- collector-type distribution.

Do not describe data only by raw count.

---

## 13. Physical-state conditions

Construct causal state conditions.

PH0:
- historical S0 latent state only.

PH1:
- z_current + compact physical vector.

PH2:
- z_previous/z_current + compact physical vector.

PH3:
- three recent compact physical states.

PH4:
- event-coded phase from measured causal state, only if reliably definable.

PH5:
- learned small GRU/TCN phase encoder over recent latent/action/physical state.

Compact physical vector can include:

- gripper width and velocity.
- TCP linear/angular velocity.
- joint velocity norm.
- contact flag/count.
- contact onset/persistence/loss indicators.

Avoid huge raw state vectors first.

---

## 14. Retrieval family

RAT-C was Wave26's strongest held-out model, so retrieval is a primary family.

R0: exact Wave26 RAT-C reproduction.

R1: RAT-C with larger transition memory only.

R2: retrieval metric includes normalized physical state.

R3: learned retrieval metric trained on TRAIN transition correspondence.

R4: two-stage retrieval:
- goal/phase filter.
- latent+physical-state nearest neighbor.

R5: attention over retrieved candidate transitions instead of early vector averaging.

R6: retrieve K full candidate trajectories and score them causally.

Test K in {4, 8, 16, 32} on development.

Do not use future target at inference.

---

## 15. Learned transition retrieval metric

For R3, train a small metric encoder.

Inputs:
- z_current.
- compact physical state.
- recent displacement/history.
- language/horizon.

Positive training pairs:
- transitions with similar future displacement under compatible goals/phase.

Negative pairs:
- wrong goal or strongly incompatible displacement.

All labels derive from TRAIN future trajectories only.

Never use test future trajectories to train retrieval.

---

## 16. Transition-quality candidate scorer

For R6 / flow candidate selection, train a compact causal transition-quality scorer.

Do not call it a task-success Q-function unless true success/failure rollout labels exist.

The scorer can use:

- current latent.
- physical state.
- language.
- candidate latent trajectory.
- candidate decoded action.
- continuity features.
- train-retrieval support.

Training targets can be trajectory-quality metrics computed on TRAIN:
- future latent error.
- decoded action error.
- target-transition compatibility.
- continuity.

At inference no ground truth is available.

---

## 17. Retrieval-Initialized Flow (RIF)

Primary new hybrid family:

current coordinate + language + physical state
-> retrieve compatible historical transitions
-> build local source prior
-> conditional flow adapts prior to current query.

Variants:

RIF-A: nearest-retrieved displacement as source center.

RIF-B: weighted local Gaussian from retrieved transitions.

RIF-C: small mixture over top retrieved transitions.

RIF-D: learned prior encoder over retrieved candidates.

This is one of Wave27's highest-priority families.

---

## 18. State-Selected Prior CFM

Revisit Prior-CFM with real physical state and new data.

Compare:

- Prior-CFM PH0.
- Prior-CFM PH1/PH2.
- Prior-CFM learned phase.
- Prior-CFM with DATA-L0.
- Prior-CFM with DATA-LN100.

Source priors may be:
- goal-conditioned.
- goal+horizon-conditioned.
- physical-phase-conditioned.
- retrieval-conditioned.

No future state in prior selection.

---

## 19. Streaming / Warm-Start CFM

Future retargeting happens while the robot is already moving.

Test flow sources centered around:

- previous displacement.
- recent latent velocity extrapolation.
- retrieved local displacement.
- physical-phase-conditioned previous displacement.

Compare to global Gaussian source.

---

## 20. Temporally Coherent CFM

Predict H1/H2/H4 as one coherent trajectory rather than unrelated endpoints.

Use multi-horizon latent supervision.

Evaluate optional temporal losses calibrated from TRAIN ground-truth transition statistics.

Do not force zero acceleration or arbitrary smoothness.

Compare:
- endpoint flow.
- trajectory-level flow.

---

## 21. State-dependent uncertainty flow

Predict source variance / uncertainty from current physical phase.

Compare:
- fixed covariance.
- phase-dependent covariance.
- retrieval-conditioned covariance.

Test whether contact/release transitions need larger/different uncertainty.

---

## 22. Multi-candidate flow + causal selection

Generate N in {4,8} candidates.

Do NOT use future target to select.

Compare causal selection using:
- model likelihood/probability.
- transition-quality critic.
- retrieval support.
- continuity.
- physical compatibility.
- combined score.

Report the gap to the historical best-of-8 oracle, but oracle is descriptive only.

---

## 23. True differentiable frozen-decoder loss

Wave26's Objective_decoded was not a true differentiable frozen-decoder trajectory loss.

Wave27 must implement this correctly.

Freeze decoder parameters:

`decoder.requires_grad = False`

but allow gradient through:

`z_pred -> D(z_pred) -> action loss`

Define:

`L_decode = MSE(D(z_pred), ground_truth_action_chunk)`

for H1/H2/H4.

Required gradient unit test:
- transition model receives nonzero gradients.
- decoder parameters receive none/zero.

Compare matched models with and without L_decode.

Do not add cycle projection.

---

## 24. Dynamic transition contrastive objective

Do not return to static endpoint attraction.

For query `(current state, language)`:

positive:
- its true future transition.

negatives:
- wrong-language transitions from similar states.
- incompatible-phase transitions.

Contrast full transition/path, not just endpoint class.

Use as an optional auxiliary for top retrieval/flow models.

---

## 25. Simple matched controls

Run:

- Wave26 RAT-C.
- F2-C direction/log-magnitude model.
- weighted affine D4.
- Phys-MLP: z_current + language + physical state -> delta.
- Phys-F2C: physical-state-conditioned direction + log magnitude.
- small physical-state-gated MoE.
- learned VQ transition head as a discrete control.

These ensure gains are not merely from more parameters.

---

## 26. Factorial core experiment

For representative models:

- RAT-C.
- RIF.
- Prior-CFM.
- TC-CFM.
- Phys-F2C.

Run data:
- L0.
- LN25.
- LN50.
- LN100.
- NEW-only.

Run state:
- PH0.
- best compact measured-physical-state condition.

Do not run a huge uncontrolled Cartesian product.

This core answers:
- more data?
- physical state?
- interaction?

---

## 27. Development scorecard

No one all-or-nothing gate.

For every model compute:

PRED:
- H2 full MSE.

DECODE:
- H4 decoded MSE.

IDENTITY:
- endpoint macro accuracy.

RECODE:
- decode/reencode accuracy.

CONT:
- continuity error.

LANG:
- RedirectGain.

EXEC:
- Execution RedirectGain.

CONTACT:
- contact-stratified robustness.

EFF:
- inference latency.

Compare to:
- Wave21 B1.
- Wave26 RAT-C.
- Wave26 Prior-CFM.

Write `wave27_development_scorecard.csv`.

---

## 28. Pareto analysis

Construct Pareto fronts over:

- H2 full MSE.
- H4 decoded MSE.
- endpoint identity.
- decode/reencode identity.
- continuity.
- full RedirectGain.
- execution RedirectGain.
- latency.

Select at most FOUR final models, preferably from distinct mechanisms:

1. one retrieval model.
2. one retrieval/state-selected flow.
3. one temporally coherent flow.
4. one simple physical-state control.

Do not select four near-identical CFM variants.

Write `wave27_final_candidate_selection.json`.

---

## 29. Two test domains

Test A: legacy compatibility.

Only use models whose inputs can be computed from legacy held-out fields.

Do not impute missing true physical state.

Test B: new prospective test.

This is the primary Wave27 test and contains synchronized physical state.

No selection/tuning on Test B.

---

## 30. Independent claim matrix

Do not force a single PASS/FAIL.

C23_MORE_DATA:
Does genuinely new independent paired transition data improve matched models?

C24_PHYSICAL_STATE:
Does true synchronized physical state improve matched transition prediction?

C25_RETRIEVAL_MEMORY:
Is retrieval memory an effective action-coordinate transition mechanism?

C26_LOCAL_PRIOR_FLOW:
Does retrieval-initialized/state-selected-prior flow improve over global-source flow?

C27_TEMPORAL_FLOW:
Does trajectory-level flow reduce the endpoint-identity/continuity trade-off?

C28_DECODER_SUPERVISION:
Does true differentiable frozen-decoder supervision improve executable consistency?

C29_LANGUAGE_PHYSICS_JOINT:
Do language and physical phase jointly modulate the local transition distribution?

Each is:
- SUPPORTED.
- NOT_SUPPORTED.
- MIXED.
- NOT_TESTED.

---

## 31. System-readiness flag

Keep a separate engineering flag:

`READY_FOR_RETARGETING_TEST`

Set true only if at least one prospective-test model satisfies approximately:

- Full RedirectGain lower95 > 0.
- Execution RedirectGain lower95 > 0.
- H2 full MSE <= 0.85.
- H4 decoded MSE <= 0.045.
- endpoint identity >= 0.55.
- decode/reencode identity >= 0.50.
- continuity <= 0.195.

These are readiness targets, not the only scientific-success criteria.

Wave27 can be scientifically useful with readiness=false.

---

## 32. Offline retargeting diagnostic

For final candidates:

1. start at current z_t.
2. condition on goal A.
3. predict one local latent step.
4. from that new current coordinate switch language to goal B.
5. predict next step.

Compare with continuing A.

Only language changes at the switch.

Report:
- post-switch RedirectGain.
- execution RedirectGain.
- continuity.
- decoded action jump.
- retrieval-neighbor change.
- source-prior change.

No simulator closed-loop yet unless readiness becomes true after the frozen prospective test.

---

## 33. Return-to-history preparation

For all newly collected trajectories store:

- latent waypoint.
- decoded/source action.
- physical state.
- contact state.
- gripper state.
- timestamp.

Create future recoverability metadata:
- recoverable.
- unknown/nonrecoverable.

Do not claim physical time reversal.

Examples requiring caution:
- release.
- switch toggles.
- drawer transitions.
- collisions.

This is data/interface preparation only.

---

## 34. Canonical lift -> place collection

Intentionally collect diverse:

`lift_blue_block_slider -> place_in_slider`

transitions.

Preferred:
- >=40 new transitions.
- multiple source sessions.
- synchronized gripper/contact/TCP/joint velocity.

Where feasible, collect similar post-lift states followed by alternative goals.

This is high-value for later online retargeting.

---

## 35. Optional exact matched-state branching

If exact simulator state snapshot/restoration is available:

from one identical current physical state, branch to different next goals.

Only perform if exact restoration is validated.

Never approximate source state.

This can create true behavioral counterfactual transition data.

It is optional but high value.

---

## 36. Contact-stratified evaluation

On new prospective test, stratify by measured causal phase:

- no contact.
- contact onset.
- persistent contact/grasp.
- transport.
- contact loss/release.

Only use categories supported by real synchronized signals.

Report whether retrieval/flow/physical-state gains depend on contact phase.

---

## 37. Collector generalization

If data come from multiple collectors:

- human.
- policy.
- scripted.

Report within-collector and cross-collector performance.

Do not silently pool motion styles.

---

## 38. Data-limited vs state-limited decision

Use the factorial results.

Case A:
physical state >> extra data
=> state observability bottleneck.

Case B:
extra data >> physical state
=> data coverage bottleneck.

Case C:
both improve
=> both matter.

Case D:
neither improves
=> transition head or frozen representation bottleneck.

This is a major required conclusion.

---

## 39. Efficiency

Record:
- parameters.
- retrieval latency.
- flow steps.
- inference latency.
- GPU memory.
- samples per query.

Future retargeting needs incremental low-latency inference.

Practical target: <20 ms/query if hardware permits, but do not use as a scientific gate.

---

## 40. Statistics

Independent unit:
`source_session`

Use:
- 10,000 source-session bootstrap replicates.
- seed = 270827.

For new prospective test, cluster by NEW source session.

Never bootstrap overlapping windows as independent episodes.

Use paired model comparisons on identical transitions.

---

## 41. Required tests

At minimum:

- frozen encoder/decoder/text hashes unchanged.
- legacy split unchanged.
- new source sessions unique.
- new split frozen before training.
- train/dev/test session disjoint.
- physical timestamps synchronized.
- true-contact audit.
- no future physical state in inputs.
- retrieval library TRAIN only.
- retrieval normalization TRAIN only.
- transition-quality critic TRAIN/DEV only.
- decoder frozen.
- decoder-through-transition gradient test passes.
- flow prior selection causal.
- candidate sampling reproducible.
- same-state language swap changes only language.
- bootstrap by source session, 10k reps, seed 270827.
- all JSON valid and outputs finite.

---

## 42. Outcome labels

Allow multiple labels:

- DATA_EXPANSION_SUPPORTED.
- DATA_EXPANSION_WEAK.
- PHYSICAL_STATE_SUPPORTED.
- PHYSICAL_STATE_WEAK.
- RETRIEVAL_SUPPORTED.
- RETRIEVAL_FLOW_SUPPORTED.
- STATE_SELECTED_PRIOR_SUPPORTED.
- TEMPORAL_FLOW_SUPPORTED.
- DECODED_SUPERVISION_SUPPORTED.
- CONTACT_SPECIFIC_GAIN.
- IDENTITY_CONTINUITY_TRADEOFF_REDUCED.
- IDENTITY_CONTINUITY_TRADEOFF_PERSISTS.
- READY_FOR_RETARGETING.
- NOT_READY_FOR_RETARGETING.
- MIXED.

Do not force one overall PASS/FAIL.

---

## 43. If retrieval wins

Do not treat this as a weak result.

It may strengthen the Actions-as-Coordinates framing:

`current coordinate + language -> retrieve compatible transition memory -> adapt`

This is naturally:
- memory-aware.
- editable.
- compatible with future return-to-history.

---

## 44. If retrieval-initialized flow wins

This is an especially attractive final mechanism:

`current action coordinate + physical phase + language -> retrieve local transition prior -> flow adapts prior -> continuous latent trajectory`

Story:

> Memory supplies where similar trajectories have gone, language selects the next transition family, physical phase resolves local feasibility, and flow adapts the retrieved prior to the current state.

---

## 45. If true physical state wins

Potential mechanism:

> Language specifies how action coordinates should evolve, while physical phase determines which locally executable transition is available.

Do not abandon action coordinates.

Physical state acts as a local disambiguator.

---

## 46. If more data wins

Potential mechanism:

> Language-conditioned dynamics are transition-coverage limited; larger independent transition memory improves the local model while preserving language redirection.

Prioritize:
- more sessions.
- retrieval library.
- retrieval-initialized flow.

---

## 47. If all branches plateau

Only after testing:
- genuinely new independent data.
- true physical state.
- retrieval.
- retrieval-flow.
- temporal flow.
- differentiable decoder loss.

If all plateau, then Wave28 should revisit the frozen representation:
- temporally structured latent.
- contact-aware latent.
- state-action latent.

Do not make this conclusion earlier.

---

## 48. Required figures

1. Legacy vs new coverage.
2. Data-scale learning curves.
3. Physical-state matched ablations.
4. Retrieval neighborhood examples.
5. Retrieval-initialized flow schematic.
6. Development/prospective Pareto fronts.
7. Contact-stratified results.
8. Lift->place synchronized case.
9. Offline language-switch retarget diagnostic.

---

## 49. Required deliverables

Produce:

- `twenty_seventh_wave_results.md`
- `twenty_seventh_wave_next_experiment.md`
- `wave27_frozen_manifest.json`
- `wave27_collection_capability_audit.md`
- `wave27_collection_preregistration.json`
- `wave27_new_transition_inventory.parquet`
- `wave27_collection_report.md`
- `wave27_physical_state_completeness.md`
- `wave27_new_data_split_manifest.json`
- `wave27_data_scale_manifest.json`
- `wave27_coverage_report.md`
- `wave27_retrieval_results.md`
- `wave27_retrieval_metric_results.md`
- `wave27_candidate_scorer_results.md`
- `wave27_rif_results.md`
- `wave27_prior_cfm_results.md`
- `wave27_streaming_cfm_results.md`
- `wave27_temporal_flow_results.md`
- `wave27_uncertainty_flow_results.md`
- `wave27_physical_state_ablation.md`
- `wave27_phase_state_results.md`
- `wave27_factorized_controls.md`
- `wave27_decoder_loss_audit.md`
- `wave27_decoder_loss_results.md`
- `wave27_transition_contrast_results.md`
- `wave27_data_model_factorial.csv`
- `wave27_development_scorecard.csv`
- `wave27_development_pareto.csv`
- `wave27_final_candidate_selection.json`
- `wave27_final_test_preregistration.json`
- `wave27_legacy_heldout_results.md`
- `wave27_prospective_test_results.md`
- `wave27_contact_stratified_results.md`
- `wave27_collector_generalization.md`
- `wave27_offline_retargeting.md`
- `wave27_return_history_compatibility.md`
- `wave27_lift_to_place_case.md`
- `wave27_claim_matrix.json`
- `wave27_system_readiness.json`
- `wave27_statistical_report.md`
- `wave27_efficiency_report.md`
- `wave27_failure_taxonomy.md`
- `exact_commands.sh`
- `environment_freeze.txt`
- `files_changed.txt`
- `tests_report.txt`
- `updated_RESEARCH_LOG.md`
- `updated_NEXT_EXPERIMENT.md`

---

## 50. Claim matrix JSON

Write `wave27_claim_matrix.json` with:

- C23_more_independent_paired_data_improves_dynamics.
- C24_true_physical_state_improves_transition_prediction.
- C25_retrieval_memory_effective.
- C26_retrieval_or_state_selected_flow_improves_global_flow.
- C27_temporal_trajectory_modeling_reduces_identity_continuity_tradeoff.
- C28_true_frozen_decoder_supervision_improves_executable_consistency.
- C29_language_and_physical_state_jointly_modulate_transition_distribution.
- READY_FOR_RETARGETING_TEST.
- best_data_condition.
- best_physical_state_condition.
- best_retrieval_model.
- best_flow_model.
- best_overall_model.
- data_scaling_effect.
- physical_state_effect.
- contact_specific_effect.
- retrieval_effect.
- flow_prior_effect.
- decoder_loss_effect.
- language_redirect_preserved.
- execution_redirect_preserved.
- endpoint_identity_improved.
- decode_reencode_improved.
- continuity_improved.
- recommended_wave28_direction.

Each claim: SUPPORTED / NOT_SUPPORTED / MIXED / NOT_TESTED.

---

## 51. Final report questions

The report must answer:

1. Which collection routes were available?
2. How many genuinely new source sessions?
3. How many certified paired transitions?
4. Counts per goal?
5. How much true contact coverage?
6. Gripper-width coverage?
7. TCP-velocity coverage?
8. Joint-velocity coverage?
9. Was session independence verified?
10. Did latent/state coverage expand?
11. Did LN25 improve over L0?
12. Did LN50 improve further?
13. Did LN100 improve further?
14. Does NEW-only generalize?
15. Is gain sample-count-driven or collector/domain-driven?
16. Does true physical state beat PH0?
17. Which physical fields matter most?
18. Does true contact help beyond proxy?
19. Does explicit/learned phase help?
20. Does larger memory improve RAT-C?
21. Does physical-state retrieval help?
22. Does learned retrieval metric help?
23. Does candidate scoring help?
24. Does RIF beat RAT-C?
25. Does RIF beat global-source CFM?
26. Does Prior-CFM improve with measured physical state?
27. Does Streaming-CFM help?
28. Does temporal flow improve identity and continuity jointly?
29. Does state-dependent uncertainty help?
30. Can causal sample selection close oracle gap?
31. Was true differentiable decoder supervision implemented correctly?
32. Does decoder supervision help?
33. Does transition contrast help?
34. Which models form the dev Pareto front?
35. Which models were frozen for prospective test?
36. Which model wins prospective test?
37. Is C23 supported?
38. C24?
39. C25?
40. C26?
41. C27?
42. C28?
43. C29?
44. Is READY_FOR_RETARGETING_TEST true?
45. Does lift->place improve?
46. Is there contact-phase heterogeneity?
47. Does cross-collector generalization hold?
48. Is the project now data-limited, state-limited, both, or neither?
49. What should Wave28 implement?
50. What exact paper claim is defensible?

---

## 52. Strategic interpretation

Wave27 should move the project from:

`many transition heads on a small incomplete dataset`

to:

`prospective evidence about data coverage, physical observability, transition memory, and local flow priors`.

The intended long-term system remains:

`current action coordinate + physical phase + next language goal + transition memory -> local latent trajectory -> decoded action`

Then, after readiness:

`new language -> retarget`

and:

`stored trajectory history -> interrupt / return to a previously visited recoverable state`

Wave27's job is to determine whether the transition model and data are finally strong enough to justify that interactive experiment.
