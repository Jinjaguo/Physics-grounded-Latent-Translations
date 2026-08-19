# EXP_G15 Report: Physically Valid Perturbation-Cohort Integrated Ablations

## Conclusion

**PARTIALLY SUPPORTED for causal F2, NOT SUPPORTED for the outcome-latent integrated system.** On ten validated 4 cm cardinal object shifts, outcome-latent F3 plus two-proposal causal F2 succeeded 10/10, while F2-disabled single sampling succeeded 9/10. This is a real one-case success contribution from multi-proposal branch execution. However, state-only F3, F3-disabled future-visible control, and restart-after-switch also reached 10/10. Outcome latent therefore contributed no success gain, autonomous F3 did not beat unrestricted future prompting, and realized-state continuation did not beat restart on this cohort. The complete-system acceptance test is false.

Closed-loop feedback itself is strongly supported on these perturbations: repeated-initial-observation open loop achieved 0/10 and same-attempt unperturbed action replay achieved 2/10. This result supports replanning from realized observations, but it does not rescue the proposed latent or module decomposition.

## Cohort construction and repaired gate

The first predeclared mixed cardinal/diagonal construction in `experiments/EXP_G15/` exposed a real simulator gate before a valid ten-case cohort existed. For attempt46, the requested `(0.04, -0.04)` m free-joint shift changed the settled book position from approximately `(-0.209, 0.129, 0.883)` to `(-1.338, 1.499, 2.174)` after only five dummy controls. The resulting common 7–8 m endpoint errors were caused by an invalid pre-policy state, not controller behavior. Those 70 rollout artifacts remain preserved as failure evidence and are excluded from the scientific result.

The repaired runner first constructs all snapshots, checks that realized XY is within 5 mm of the requested offset and Z drift is below 5 mm, and only then begins any evaluated rollout. The formal retry in `experiments/EXP_G15_retry1/` used the same frozen G14 attempts 40–49 with predeclared cardinal-only ±4 cm shifts. All ten snapshots passed before execution; attempt46 remained finite and all feedback controllers completed it, directly confirming the repair.

## Actual interventions

Every method started from the same exact perturbed snapshot and executed up to 300 actual LIBERO controller steps. The comparison included:

1. outcome-latent temporal F3 plus two-proposal causal state-value F2;
2. matched state-only temporal F3;
3. F2-disabled single π0.5 proposal control;
4. F3-disabled future-visible full prompting;
5. restart-after-switch instead of continuation from the realized controller history;
6. actions repeatedly generated from the perturbed initial observation and then run open loop;
7. replay of the corresponding unperturbed G14 outcome-latent action sequence.

For causal methods, every proposal prefix was physically branch-executed from a restored decision snapshot, scored from its realized state, and the selected prefix was restored and committed. Thus the F2 comparison is an intervention comparison rather than teacher-forced prediction.

## Results

| Method | Success | Switch error ↓ | Endpoint error ↓ | Candidate steps | Committed steps |
|---|---:|---:|---:|---:|---:|
| outcome-latent full | 10/10 | 9.5 | 0.045269 | 3,750 | 1,873 |
| state-only F3 | 10/10 | **8.0** | 0.044172 | 3,716 | 1,856 |
| F2-disabled single | 9/10 | 9.5 | 0.047866 | 0 | 2,098 |
| F3-disabled future-visible | 10/10 | n/a | **0.039264** | 3,459 | 1,729 |
| restart after switch | 10/10 | 9.5 | 0.041487 | 5,225 | 2,611 |
| initial-observation open loop | 0/10 | n/a | 1.071203 | 0 | 3,000 |
| unperturbed action replay | 2/10 | n/a | 0.275238 | 0 | 1,852 |

The full method strictly beat single F2, open loop, and replay in success, but tied the three key state/switch/restart ablations. State F3 also switched closer to the retrospective boundary than outcome-latent F3, while future-visible control had the lowest endpoint error. The hypothesis that outcome grounding made the latent a necessary integrated component is therefore falsified on this cohort. G10/G13 evidence that restart often hurts remains valid for those distributions, but G15 proves it is not a universal distinction.

## Audit, reproducibility, and storage

The independent audit passed with no failures. It checked ten realized perturbations and free-joint mappings, all 70 rollout files, 50 proposal-to-commit chains, ten exact unperturbed replay prefixes, ten open-loop raw proposal sources, finite traces, success/lift state, endpoint error, action jerk, strict-gain decisions, and every aggregate.

Formal command:

```bash
PYTHONPATH=src:/home/jinjaguo/LIBERO:/home/jinjaguo/openpi/packages/openpi-client/src \
MUJOCO_GL=egl NUMBA_CACHE_DIR=/tmp/pglt_numba_cache MPLCONFIGDIR=/tmp/pglt_matplotlib \
/home/jinjaguo/anaconda3/envs/libero/bin/python \
scripts/experiments/run_exp_g15_perturbed_ablations.py \
--output experiments/EXP_G15_retry1 --g14 experiments/EXP_G14 \
--host localhost --port 8000 --shift 0.04 --offset-pattern cardinal \
--horizon 300 --proposals 2 --replan-steps 5 --device cpu
```

- `scripts/experiments/run_exp_g15_perturbed_ablations.py`
- `scripts/experiments/audit_exp_g15.py`
- `experiments/EXP_G15/` — preserved invalid mixed-pattern gate evidence
- `experiments/EXP_G15_retry1/perturbation_snapshot_manifest.json`
- `experiments/EXP_G15_retry1/snapshots/`, `rollouts/`, `case_metrics.jsonl`
- `experiments/EXP_G15_retry1/metrics.json`, `audit.json`, `run_metadata.json`

The post-EXP disk check showed 849 GB free. The invalid trial and formal retry each occupy approximately 17 MB.

## Next decision

Using a history latent only as an extra F3 input is rejected: state F3 is at least as good and simpler. EXP_G16 will redefine the representation as a decodable action-chunk coordinate used directly by F2. It will train several chunk representation families on training demonstrations, use them to generate executable local proposals around a π0.5 F1 proposal, physically evaluate every candidate from matched snapshots, and compare against equal-budget raw proposal and raw-action search on a new frozen harder perturbation distribution.
