# Wave62: decoder_action_calibration

Wave62 运行了 decoder_action_calibration，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `decoder_action_calibration_state_q2_pca_w0.4`，execution redirect 约 0.0036，continuity 约 2.76，endpoint 约 0.21。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，因此下一 wave 必须继续。

```json
{
  "wave": 62,
  "method": "decoder_action_calibration",
  "best": "decoder_action_calibration_state_q2_pca_w0.4",
  "SUCCESS": false,
  "READY_FOR_CLOSED_LOOP_RETARGET": "NOT_SUPPORTED",
  "termination_rule": "success or Wave78 only; continue otherwise"
}
```
