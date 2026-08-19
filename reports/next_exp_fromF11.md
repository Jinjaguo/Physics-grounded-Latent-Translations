# Next Experiment from EXP_F11：EXP_F12 Closed-Loop Factor Feedback Recovery

## F11后的主线状态

F11有效formal中，outcome viability对6个no-local states recall 1.0且无unsafe false-local，但false-fallback一个recoverable task09 state；F10 geometry gate在该cohort反而完美。factor/scalar/prior在6个viable states上都出现同一个ranking miss：task09 instance24只有`semantic_full_task`成功，但静态模型都选择`recovery_prompt_feedback`。继续调0.635阈值或tree参数不会回答新问题。

## 新科学假设

> 静态decision-time consequence prediction不足以区分动作文本相似但执行结果不同的recovery proposals；在confirmed failure后执行唯一selected proposal的短prefix、观察真实object/retention/articulation/progress/safety factors，再决定继续、切换proposal或fallback，能修复静态ranking miss。factorized consequences的价值应体现在不同feedback触发不同控制分支，而不是把多个heads再压成一个静态分数。

这是post-failure recovery intervention，不是nominal active probe、failure prediction或two-probe confirmation。每个branch只执行自己选择的proposal prefix，不能试完所有candidates再选择。

## 两阶段selected-only controller

每个confirmed failure先用F11/F10 hybrid support作state gate；local-supported时冻结并执行一个initial proposal的短prefix（例如80 control steps，具体长度只由F8–F11 development traces决定）。随后从同一真实world state重新观察：

- object acquisition与retention；
- task-specific target/joint progress及regression；
- current official success；
- gripper/contact stability与finite/done；
- base-policy handoff readiness。

Full factor controller按显式语义决定：已完成则结束；object稳定且正确task progress则继续当前proposal/base handoff；object到位但articulation无进展则切换`semantic_full_task`；未获取但无regression则继续/切换recovery；出现regression或不稳定则进入factor-conditioned fallback。不得使用其他candidate outcome或未来oracle。

## Mechanism ablations

在同一新state、同一initial proposal和相同prefix budget下实际执行：

1. full factor-feedback loop；
2. no-object feedback（删除acquisition/retention）；
3. no-progress feedback（删除target/joint progress）；
4. static F11 factor selector，不re-observe；
5. matched scalar feedback controller，只使用success probability/official binary reward。

删除某factor必须导致可定位的控制分支变化才算机制证据。若所有ablation仍做同一动作，则该factor在当前controller中没有价值，应删除或重新定义。

## Development与防泄漏

- 从F8–F11保存的真实rollout traces构建prefix→remaining outcome dataset，按full-state group split；不得把同一trace的多个prefix拆到train/test两边。
- prefix cutoff、continue/switch/fallback rules和任何learned feedback model只由development traces冻结。
- F12 formal使用未见official instances，所有stage-1 decisions先写入；stage-2只能读取该branch实际执行产生的feedback。
- 完成所有selected dynamic branches后才运行full oracle和static-proposal counterfactual诊断。

## 必须回答

- closed-loop feedback是否修复F11 task09类型的static ranking miss；
- object/progress factor各自删除时具体改变哪些continue/switch/fallback决策与outcomes；
- failure是initial proposal coverage、prefix controller、feedback selector、handoff还是fallback controller；
- factor feedback相对matched scalar是否改善selected success、harm或fallback calibration；
- fallback执行自身是否安全，不能把正确fallback decision等同于safe fallback outcome。

## Pivot规则

- 若prefix后factors能区分但rule不改善，F13改为learned receding-horizon controller；
- 若prefix factors本身没有信息，F13扩大proposal mechanism（goal-conditioned subtask proposals或action diversity），不调静态阈值；
- 若主要失败来自fallback regression，F13优先实现object-contact-conditioned safe placement/hold再replan；
- 若full factor feedback不优于no-progress/no-object/scalar，收缩对应factor claim并删除无用接口。

## 完成artifacts

```text
experiments/EXP_F12/{frozen_protocol.json,prefix_training_manifest.jsonl,feedback_schema.json,confirmed_failures.jsonl,stage1_decisions.jsonl,stage1_prefix_traces/,stage2_decisions.jsonl,selected_rollouts/,selected_outcomes.jsonl,oracle_rollouts/,oracle_outcomes.jsonl,metrics.json,run_metadata.json,audit.json}
reports/EXP_F12_report.md
reports/next_exp_fromF12.md
```

F12只有在新的confirmed-failure states上真实执行proposal prefix、记录feedback、做stage-2控制并完成后置oracle与审计后才占用EXP_F12。
