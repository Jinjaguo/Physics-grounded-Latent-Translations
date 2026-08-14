# Wave 19 results — independent official LIBERO-10 replication

## Decision

Wave 19 resolved the project name `LIBERO-long` to the installed official `libero_10` suite and completed a
prospective π0.5 source-data collection with exact branch restoration. The source/snapshot/data gates passed, but
the independent six-seed representation R-gate failed its preregistered continuous-motor condition by a narrow,
nonzero margin. Per protocol, F1/F2 and the final held-out closed-loop evaluation were not run.

This is a valid stopped experiment, not a negative closed-loop result:

- snapshot certification: **PASS**;
- representation semantic diagnostics: **PASS**;
- representation gripper fidelity: **PASS**;
- representation continuous motor fidelity: **FAIL** (`1.200444393 > 1.2`);
- F1/F2, O1–O8, B0–B5, perturbation recovery: **NOT TESTED**;
- final 50-episode test split: **unopened**.

## Official suite and source policy

`requested_internal_name = LIBERO-Long` resolves to `resolved_official_suite_name = libero_10`, with these tasks:

1. put both the alphabet soup and the tomato sauce in the basket
2. put both the cream cheese box and the butter in the basket
3. turn on the stove and put the moka pot on it
4. put the black bowl in the bottom drawer of the cabinet and close it
5. put the white mug on the left plate and put the yellow and white mug on the right plate
6. pick up the book and place it in the back compartment of the caddy
7. put the white mug on the plate and put the chocolate pudding to the right of the plate
8. put both the alphabet soup and the cream cheese box in the basket
9. put both moka pots on the stove
10. put the yellow and white mug in the microwave and close it

LIBERO commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`. OpenPI commit:
`981483dca0fd9acba698fea00aa6e52d56a66c58`. Source generation used the fixed official `pi05_libero`
checkpoint at `/home/jinjaguo/.cache/openpi/pytorch_checkpoints/pi05_libero`, with URI
`gs://openpi-assets/checkpoints/pi05_libero/`; it was not fine-tuned and no π0.5 hidden state was used.

The action interface was 7-D `OSC_POSE` at 20 Hz over 500 Hz physics: six relative normalized OSC inputs plus a
Panda sign-based gripper command. H_action was 16 controls = 0.8 s; H8 was 128 controls = 6.4 s. The first six
inputs use `[-1,1]` OSC bounds; the OpenPI gripper value was passed without client clipping and interpreted by sign.

## Collection, restoration, and frozen data

| task | attempts | raw success | raw failure | certified success |
|---:|---:|---:|---:|---:|
| 0 | 31 | 31 | 0 | 31 |
| 1 | 30 | 30 | 0 | 30 |
| 2 | 30 | 30 | 0 | 30 |
| 3 | 30 | 30 | 0 | 30 |
| 4 | 30 | 30 | 0 | 30 |
| 5 | 40 | 39 | 1 | 30 |
| 6 | 32 | 31 | 1 | 31 |
| 7 | 30 | 30 | 0 | 30 |
| 8 | 50 | 24 | 26 | 24 |
| 9 | 32 | 31 | 1 | 31 |
| **total** | **335** | **306** | **29** | **297** |

Nine successful task-5 episodes were shorter than the frozen causal/future-support rule and therefore stayed raw
but uncertified. All 415 eligible branches (297 at 25%, 118 at 50%; no 75% branch retained 128 future steps)
passed twin/source replay. Integration, controller, and object maximum discrepancies were exactly zero; all values
were finite; restored predicates and source terminal success agreed. The balanced primary dataset contains 24
episodes/task = 240 episodes, split by source episode into 140 train / 50 development / 50 final test. SHA256 source
episode hashes cover every finalized raw file, and branch metadata hashes are frozen.

Every-step MuJoCo contacts were deterministically materialized outside immutable raw folders from the saved
`mjSTATE_INTEGRATION` states: 3,104,618 active contacts over 97,269 control boundaries in all 335 attempts,
including geometry pairs, distance, position, frame, and 6-D force.

## Independent representation R-gate

The representation was trained from scratch on LIBERO actions only: 32-D total latent (16 semantic + 16
execution), H=16, frozen OpenCLIP ViT-L/14 DataComp text features, EMA 0.999, six registered seeds, and
correct/shuffled/reconstruction-only conditions. No CALVIN checkpoint or statistic was loaded.

| condition | result |
|---|---:|
| mean action-to-text semantic delta | 0.940000 |
| clustered action-to-text lower 95% | 0.916667 |
| mean text-to-action semantic delta | 0.900000 |
| clustered text-to-action lower 95% | 0.866667 |
| correct-language continuous MSE | 0.00205037349 |
| reconstruction-only continuous MSE | 0.00170801205 |
| continuous MSE ratio | 1.200444393 |
| frozen maximum ratio | 1.200000000 |
| absolute MSE excess over threshold | 0.000000759028 |
| correct / reconstruction gripper accuracy | 0.987560 / 0.989337 |

The semantic and gripper conditions passed, but `continuous_motor_fidelity_preserved=false`. The numerical miss is
small, yet it is not zero and the decision was frozen; reporting it as a pass by rounding would violate the gate.
No correct-language seed was selected.

## Required final questions

1. Official target: `libero_10`, not the eight custom OOD BDDL tasks.
2. Exact tasks: the ten instructions listed above, with audited official BDDL/init files and predicates.
3. Source provenance: OpenPI `981483d…`, fixed official `pi05_libero` checkpoint and assets.
4. Raw outcomes: 306 successes and 29 failures; per-task counts are in the table.
5. Certified source episodes: 297 collected; 240 balanced primary episodes.
6. Restored twins deterministic: yes, exact zero discrepancy on all 415 admitted branches.
7. Source continuation success reproduced: yes, with official predicates.
8. Interface/frequency: 7-D relative OSC pose + sign gripper, 20 Hz control / 500 Hz physics.
9. Frozen timebase: H_action=16, 0.8 s latent step; H8=6.4 s.
10. Independent from CALVIN: yes; no CALVIN representation, normalization, F1, or F2 was loaded.
11. Language improves bidirectional retrieval: yes as a diagnostic, with large positive clustered deltas.
12. Motor reconstruction preserved: no under the frozen 1.2 ratio requirement; gripper fidelity did pass.
13. LIBERO R-gate: **FAIL**.
14. F2 vs F1 at H1/H2/H4/H8: not tested.
15. Source-episode clustered dynamics AUC/CI: not available.
16. F2 H8 decoded error: not tested.
17. F2 H8 execution kNN radius: not tested.
18. Correction-target cosine: not tested.
19. Exact-state F2 vs F1 success: not tested.
20. Paired closed-loop success difference/CI: not available.
21. Cross-task embodied breadth: not tested.
22. F2 vs norm-matched random: not tested.
23. F2 vs shuffled learned directions: not tested.
24. Negative refinement: not tested.
25. Proposal perturbation recovery: not tested.
26. Off-manifold drift/behavior association: not tested.
27. F1 failure modes repaired by F2: not evaluated.
28. F1 failure modes remaining: not evaluated.
29. CALVIN claims replicated on LIBERO: no full claim is authorized. Exact data/snapshot infrastructure and strong
    semantic diagnostics succeeded, but the overall representation gate stopped the scientific replication.
30. Defensible paper story: Wave 19 establishes a branchable official-LIBERO dataset and shows language-addressable
    action structure, but provides no LIBERO dynamics or closed-loop evidence for refinement.
31. Additional experiment before a cross-domain submission: yes; a prospectively motor-margin representation
    adjudication must pass before the already-frozen downstream protocol can run.

## Bottom line

The official LIBERO environment and π0.5 source pipeline can run and produced a high-quality exact-state dataset.
The experiment correctly stopped at a narrowly failed but mandatory motor-fidelity gate. No claim about LIBERO F2
closed-loop benefit—positive or negative—is warranted from Wave 19.
