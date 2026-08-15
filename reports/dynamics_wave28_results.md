# Twenty-eighth wave: low-dimensional intent force field

## Outcome

Wave28 froze the action-text representation, decoder, semantic predictor, F1, and F2. It evaluated 65 development candidates and froze 8 before opening the Wave21 session-disjoint and Wave27 prospective tests. The best frozen candidate was `BACKBONE_F2_q2` with q=2, encoding=E0_linear, field=FF3_attractor, subspace=C2_learned, composition=COMP0_additive, and backbone=F2.

Wave27 prospective held-out RedirectGain=0.016181, execution RedirectGain=0.009274, H4 decoded MSE=1.594116, endpoint=0.1686, continuity=2.552528. Wave27 records have no previous annotation, so they are neutral→target tests and are not used to claim return symmetry.

## Claims

```json
{
  "C30_low_dim_adapter_preserves_frozen_behavior": "SUPPORTED",
  "C31_low_dim_intention_field_improves_retargeting": "MIXED",
  "C32_dynamic_field_beats_static_residual": "NOT_SUPPORTED",
  "C33_learned_low_rank_subspace_beats_random_or_pca": "NOT_SUPPORTED",
  "C34_intention_return_symmetry_supported": "NOT_TESTED",
  "C35_continuity_anchor_improve_editability": "NOT_SUPPORTED",
  "C36_adapter_generalizes_across_backbones": "MIXED",
  "READY_FOR_CLOSED_LOOP_RETARGET": "NOT_SUPPORTED",
  "best_q_dimension": 2,
  "best_language_encoding": "E0_linear",
  "best_field_form": "FF3_attractor",
  "best_subspace_form": "C2_learned",
  "best_composition": "COMP0_additive",
  "best_loss_group": "D",
  "best_backbone": "F2",
  "best_overall_adapter": "BACKBONE_F2_q2",
  "adapter_parameter_count": 96,
  "adapter_fraction_of_total_parameters": 8.727272727272727e-05,
  "next_wave_required": true,
  "next_wave_number": 29
}
```

## Interpretation

The method preserves the main line: F1/F2 determine local behavior and the adapter supplies a small continuous residual steering path. The q field is not treated as physical energy, and no future trajectory/action/contact state enters inference. If readiness is false, the next wave must target the diagnosed field/data bottleneck rather than enlarging the frozen VAE.
