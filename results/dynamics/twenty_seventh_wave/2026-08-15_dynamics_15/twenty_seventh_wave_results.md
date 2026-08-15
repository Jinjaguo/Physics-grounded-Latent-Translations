# Wave 27: prospective transition memory and physical observability

## Outcome

Collected 407 certified transitions from 52 independent sessions. The frozen prospective winner is `Core_LN100_PH0_RAT-C`. H2 full MSE=1.368115, H4 decoded MSE=0.058471, endpoint=0.5952, recode=0.5272, continuity=0.200456, RedirectGain=1.039290 with session-bootstrap CI [0.9068370827876964, 1.2223196245876013].

`READY_FOR_RETARGETING_TEST=False`; criteria: `{"H2_full": false, "H4_continuity": false, "H4_decoded": false, "H4_endpoint": true, "H4_recode": true, "redirect_lower_ci": true}`.

## Claims

```json
{
  "C23_more_independent_paired_data_improves_dynamics": "SUPPORTED",
  "C24_true_physical_state_improves_transition_prediction": "SUPPORTED",
  "C25_retrieval_memory_effective": "SUPPORTED",
  "C26_retrieval_or_state_selected_flow_improves_global_flow": "NOT_SUPPORTED",
  "C27_temporal_trajectory_modeling_reduces_identity_continuity_tradeoff": "NOT_SUPPORTED",
  "C28_true_frozen_decoder_supervision_improves_executable_consistency": "SUPPORTED",
  "C29_language_and_physical_state_jointly_modulate_transition_distribution": "SUPPORTED",
  "READY_FOR_RETARGETING_TEST": "NOT_SUPPORTED",
  "best_data_condition": "LN100",
  "best_physical_state_condition": "PH1",
  "best_retrieval_model": "Retrieval_R4_factored_K8",
  "best_flow_model": "Flow_PH1_RIF-A_random",
  "best_overall_model": "Core_LN100_PH0_RAT-C",
  "data_scaling_effect": "SUPPORTED",
  "physical_state_effect": "SUPPORTED",
  "contact_specific_effect": "NOT_TESTED",
  "retrieval_effect": "SUPPORTED",
  "flow_prior_effect": "NOT_SUPPORTED",
  "decoder_loss_effect": "SUPPORTED",
  "language_redirect_preserved": "SUPPORTED",
  "execution_redirect_preserved": "SUPPORTED",
  "endpoint_identity_improved": "SUPPORTED",
  "decode_reencode_improved": "SUPPORTED",
  "continuity_improved": "NOT_SUPPORTED",
  "recommended_wave28_direction": "target the failed readiness criteria with phase-balanced independent collection and compact physical retrieval",
  "outcome_labels": [
    "DATA_EXPANSION_SUPPORTED",
    "PHYSICAL_STATE_SUPPORTED",
    "RETRIEVAL_SUPPORTED",
    "MIXED",
    "DECODED_SUPERVISION_SUPPORTED",
    "IDENTITY_CONTINUITY_TRADEOFF_PERSISTS",
    "NOT_READY_FOR_RETARGETING"
  ]
}
```

## Scope

Prospective inference is clustered by source session. True contact, measured velocity, cross-collector generalization, execution success, recoverability, and lift→place chain performance remain untested. Legacy physical fields were never imputed.

See `wave27_final_report_questions.md` for all 50 required answers.
