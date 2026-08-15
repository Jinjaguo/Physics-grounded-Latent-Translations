# Wave78: final_registered_tournament

Wave78 运行了 final_registered_tournament，在多个输入分支、q 维度、低秩基和损失权重中选择候选；Wave27 held-out 最好是 `final_registered_tournament_delta_q2_pca_w0.1`，execution redirect 约 0.0015，continuity 约 2.76，endpoint 约 0.24。结果说明该方向仍然没有解决 latent 到动作的连续迁移问题；没有达到成功门槛，但 Wave78 上限已经完成，因此研究程序在此结束，Wave79 禁止启动。

```json
{
  "wave": 78,
  "method": "final_registered_tournament",
  "best": "final_registered_tournament_delta_q2_pca_w0.1",
  "SUCCESS": false,
  "READY_FOR_CLOSED_LOOP_RETARGET": "NOT_SUPPORTED",
  "termination_rule": "success or Wave78 only; continue otherwise"
}
```
