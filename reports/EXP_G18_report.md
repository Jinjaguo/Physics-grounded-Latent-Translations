# EXP_G18 Report: Realized-Transition Latent Selection

## Conclusion

**NOT SUPPORTED.** The learned transition latent did not improve causal proposal selection. On ten validated 7.5 cm x-axis perturbations it completed 8/10 ordered tasks, while the direct transition MLP and the existing state-value selector each completed 9/10. The latent beat open loop and unperturbed replay, but it did not beat first-candidate or single-proposal control in success. Therefore neither the preregistered latent-contribution criterion nor integrated acceptance is satisfied.

## New scientific content

G18 stopped modifying the action chunks themselves. It rebuilt 5,423 candidate-level, action-conditioned transitions in 2,351 matched candidate groups from the audited G15--G17 branch artifacts. Every sample contains the pre-state, untouched native pi0.5 action chunk, physically realized EEF/object displacement, active phase, and retrospective progress. Three genuinely different predictors were trained and held out by source attempt: a supervised contrastive transition encoder, a transition autoencoder with progress head, and a direct transition MLP without a latent bottleneck.

The contrastive encoder was selected before the causal test because it had the best validation regret (0.01145) and ranking accuracy (0.5560), compared with 0.01301/0.5420 for the autoencoder and 0.01813/0.5504 for the direct MLP. The formal controller queried three raw pi0.5 proposals at each decision, executed all of them from the exact same simulator/controller checkpoint, and then committed one untouched proposal according to the relevant selector.

## Results

| Method | Success | Switch error down | Endpoint error down | Candidate steps |
|---|---:|---:|---:|---:|
| transition latent | 8/10 | **5.5** | 0.071737 | 6,534 |
| direct transition MLP | **9/10** | 7.1 | **0.052234** | 5,804 |
| state-value | **9/10** | 6.5 | 0.059846 | 5,728 |
| first candidate | 8/10 | 6.1 | 0.064283 | 6,418 |
| single raw pi0.5 | 8/10 | 7.7 | 0.074935 | 0 |
| initial-observation open loop | 0/10 | n/a | 1.054414 | 0 |
| unperturbed replay | 0/10 | n/a | 0.466900 | 0 |

The latent's lower switch error is real but does not compensate for its lower ordered success. The decisive result is that direct/state-space selection remained stronger. This rejects the hypothesis that compressing realized candidate transitions supplied a control advantage in the proposal-selection role. The broader causal F2 result remains strong: both open loop and replay failed all ten cases, while selectors using actual branch feedback reached 8--9 successes.

## Invalid preliminary cohorts

The planned 8 cm cardinal cohort (`experiments/EXP_G18`) and its 7.5 cm cardinal retry (`experiments/EXP_G18_retry1`) each exposed invalid pre-rollout physics on attempt 47. They are preserved as infrastructure evidence and are not counted as experiments. The formal run changed the dataset construction to a contact-stable alternating x-axis cohort while retaining 7.5 cm displacement (`experiments/EXP_G18_retry2`). Only retry2 supports the scientific result above.

## Audit and artifacts

The independent audit passed. It rebuilt all 5,423 transition samples and 2,351 groups; validated ten snapshots, 70 rollouts, and 50 intervention chains; recomputed 829 transition-model decisions and 431 first-candidate selections; and checked the exact saved pre-state, action, realized displacement, latent, and score fields.

- `scripts/experiments/run_exp_g18_transition_latent_selector.py`
- `scripts/experiments/audit_exp_g18.py`
- `experiments/EXP_G18/`, `experiments/EXP_G18_retry1/` -- preserved invalid-cohort evidence
- `experiments/EXP_G18_retry2/transition_dataset.npz`
- `experiments/EXP_G18_retry2/contrastive_transition.pt`
- `experiments/EXP_G18_retry2/autoencoder_transition.pt`
- `experiments/EXP_G18_retry2/direct_transition.pt`
- `experiments/EXP_G18_retry2/test_snapshots/`, `test_rollouts/`, `metrics.json`, `audit.json`

The post-EXP disk check left 849 GB free. The three G18 directories occupy about 34 MB in total.

## Next decision

Repeated latent proposal selectors have now failed to beat direct state/action-space baselines. G19 therefore moves the latent to a different control function: selecting a physically meaningful checkpoint after a controlled disturbance, followed by an actually executed return controller and continuation from the realized recovered state. It will distinguish latent memory from direct state memory, latest-checkpoint recovery, no recovery, and simulator-restore upper bounds.
