# EXP_G19 Report: Executed Checkpoint Recovery

## Conclusion

**NOT SUPPORTED: latent checkpoint memory did not improve recovery, and explicit return did not improve the task.** The selected sequence-autoencoder latent and direct-state checkpoint return each completed 8/10 ordered tasks, while no-return closed-loop continuation completed 9/10 with the lowest endpoint error. Latest-checkpoint return reached 6/10 and restart-from-initial reached 6/10. All 30 physical return rollouts did reduce both EEF and book error and reached the stored checkpoint tolerance, so this is not a controller-interface failure: returning reliably to these checkpoints simply was not the best response to the tested disturbance.

## New data and mechanisms

G19 rebuilt 2,110 sequence windows from the audited G18 causal trajectories and fit three matched memory families: supervised contrastive progress latent, sequence autoencoder with progress/phase/success heads, and a direct sequence predictor. On 572 held-out windows, the autoencoder was the preregistered latent winner (progress MAE 0.07022, phase accuracy 1.0), but the non-latent direct model was better overall (progress MAE 0.06717, success accuracy 0.9738 versus 0.9213).

For every prospective case, G19 replayed the successful controller to the autonomous lift-to-place switch, saved four complete simulator/controller checkpoints, then executed a 3.5--4.1 cm OSC disturbance while the book remained grasped. Latent, direct-state, and latest selectors chose a checkpoint. Each physical method then executed proportional OSC actions until both EEF error was at most 1.5 cm and book error at most 3 cm for three consecutive steps, followed by fresh closed-loop pi0.5 continuation from the realized recovered state.

## Results

| Method | Task success | Physical recoveries | Endpoint error down | EEF reduction | Book reduction |
|---|---:|---:|---:|---:|---:|
| latent checkpoint return | 8/10 | 10/10 | 0.074020 | 0.102255 | 0.100739 |
| direct-state return | 8/10 | 10/10 | 0.075290 | 0.059691 | 0.060336 |
| latest checkpoint return | 6/10 | 10/10 | 0.135517 | 0.059691 | 0.060336 |
| no return, direct continuation | **9/10** | n/a | **0.054215** | n/a | n/a |
| restart from initial state | 6/10 | n/a | 0.064181 | n/a | n/a |
| simulator restore upper bound | **9/10** | non-physical | 0.062337 | n/a | n/a |

The latent strictly beat latest and restart, but tied direct-state in success and lost endpoint error. It also lost to no-return and simulator restore. Therefore the latent contribution criterion is false. The executed-recovery criterion is also false because the best physical return did not strictly beat no-return. A central scientific result is that physical checkpoint proximity was not a sufficient recovery objective: all return controllers reached their targets, yet returning could place the policy on a worse continuation distribution than simply replanning from the disturbed state.

## Audit-discovered invalid runs

`EXP_G19` and `retry1` stopped after a result-key logging mismatch. `retry2` completed but its first return action read a stale robosuite controller EEF cache immediately after snapshot restore; the audit detected method-order contamination. `retry3` fixed that controller path but lacked the exact fresh-replay window needed to recompute the current latent. `retry4` was interrupted after 32/60 methods, and `retry5` found the pi0.5 server unavailable before any case. `retry6` added exact selector inputs plus incremental resumable artifacts, used the protocol-matched Wave19 server, and is the sole formal result. Every prior directory is preserved and excluded from the metrics above.

## Audit and artifacts

The independent audit passed. It rebuilt all 2,110 dataset samples and memberships, reloaded all three models, restored 50 checkpoint/disturbed states, recomputed ten selectors from their exact saved inputs, validated 60 disturbance/continuation chains and 30 physical returns, and exactly reconstructed all six aggregate rows.

- `scripts/experiments/run_exp_g19_checkpoint_recovery.py`
- `scripts/experiments/audit_exp_g19.py`
- `experiments/EXP_G19_retry6/memory_dataset.npz`
- `experiments/EXP_G19_retry6/memory_{contrastive,autoencoder,direct}.pt`
- `experiments/EXP_G19_retry6/checkpoint_memory_manifest.json`
- `experiments/EXP_G19_retry6/checkpoints/`, `disturbances/`, `disturbed_snapshots/`
- `experiments/EXP_G19_retry6/rollouts/`, `case_metrics.jsonl`, `metrics.json`, `audit.json`

The post-EXP disk check left 850 GB free. All G19 attempts together occupy about 168 MB.

## Next decision

G20 will not tune return distance or checkpoint choice. It will test a recovery-trigger policy: predict whether direct continuation is at risk and invoke direct-state return only when warranted. Latent and direct-state risk gates will be compared with always-return, never-return, and oracle-trigger controls through new executed continuation rollouts. This asks whether the latent can contribute to the discrete intervention decision even though it failed as a checkpoint metric.
