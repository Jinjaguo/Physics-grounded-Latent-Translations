# Wave 28 backbone interface audit

{
  "created_at": "2026-08-15T03:15:37.635440-04:00",
  "device": "cpu",
  "cuda_available": false,
  "representation": {
    "checkpoint": "checkpoints/representation/seed_810/correct_language/checkpoint_ema.pt",
    "latent_dim": 32,
    "semantic_dim": 16,
    "decoder_output": [
      16,
      7
    ],
    "decoder_parameters_trainable": false
  },
  "F1_inputs": [
    "execution_previous(16)",
    "execution_current(16)",
    "semantic_current(16)",
    "frozen_text(16)"
  ],
  "F2_inputs": [
    "same as F1",
    "four frozen refinement iterations"
  ],
  "F1_output": "next_execution(16)",
  "semantic_output": "next_semantic(16)",
  "z_base": "recursive frozen F1/F2 semantic+execution prediction from current z and query-time current language",
  "adapter_input": [
    "z_base",
    "z_current",
    "current_language",
    "target_language"
  ],
  "adapter_output": "low-rank residual in original 32-D latent",
  "future_inputs": [],
  "ordered_wave21_events": 257,
  "wave27_neutral_events": 234,
  "decoder_gradient_norm": 2.2828123569488525,
  "B_rank": 2,
  "q_zero_no_adapter_test": "PASS",
  "frozen_backbone_parameters": "PASS",
  "F1_checkpoint_sha256": "41f63d173c919cc01d5f5cbfab3af41983813a63ed018c43dbbc28ddd1df9fb0",
  "F2_checkpoint_sha256": "9b19c0c3c47994c734eccfae4a8070a29b8ff35020ac1e7ef04e0f7f8d9be308",
  "semantic_checkpoint_sha256": "9b85b797cf9a1df90074efc7e73d862968a73e94c44df61c7fc5f781b10f0617",
  "limitation": "Wave27 records do not contain previous annotation labels; they are neutral->target prospective events and are excluded from return claims."
}
