# Next Experiment from EXP_F7：EXP_F8 Factor-Gated Regrasp–Place–Close Proposal

## F7后的主线状态

F7把local coverage提高到8/12，并用matched interventions证明object-stage、joint-progress和handoff具有不同机制语义；但4个新states仍无local recovery，根因是在placement prompt下没有先可靠获得物体，甚至出现关空fixture和target regression。`contact_ready≤6cm`没有决策价值，F8删除它，不训练该head。

## 新科学假设

> 把可独立观察的`object_acquired`作为proposal controller的阶段门控，形成`regrasp → verify acquisition → place → verify object goal → close → handoff`，能够避免place policy在未持物时提前操纵fixture，并在未见confirmed-failure states上提高proposal coverage、减少target regression；相同base policy但删除acquisition gate会产生可解释的关空fixture或无物placement失败。

这是新的closed-loop proposal/state-machine formulation，不是改budget。object acquisition用于已确认失败后的recovery control，不是failure detector。

## Development与untouched confirmation

- F6/F7的12 states只用于定义controller、factor阈值和debug，不再作为最终confirmation claim。
- 新生成task03/task05/task09各2个full-state failures，优先official init instance10/11，共6个untouched confirmation states。
- 新snapshot必须保存model placement、integration和controller-observable state，并做双fresh-env exact restore。
- 所有methods从同一snapshot开始；先冻结protocol和proposal definition，再产生outcomes。

## Proposal families

1. `factor_gated_regrasp_place_close`：明确pick/regrasp prompt，只有`object_acquired=true`才切换place；只有官方`object_goal=true`才切换aligned closure（task03/09）；task05在object goal后进入handoff。
2. `ungated_place_close`：F7的place→factor-gated closure，作为删除acquisition factor的matched ablation。
3. `recovery_prompt_feedback`：现有通用恢复prompt。
4. `semantic_full_task`：同budget frozen base policy baseline。
5. `fallback_disengage_replan`：当前真实world state上的safe disengage+replan。

如果纯语言pick prompt仍无法可靠获得物体，F8内实现privileged object-space analytic regrasp oracle并明确标注；它必须通过robot actions执行，禁止teleport object。该oracle用于区分perception/policy不足与物理不可恢复，不得伪装成deployment结果。

## Factor interface与阈值

- `object_acquired`：物体相对failure/table高度提升与EEF-object关系持续满足，必须保存连续lift、distance和last-k stability；阈值在formal outcome前根据object geometry/controller tolerance冻结。
- `object_goal/retained`：官方In与last-10-step retention。
- `joint_progress`：只在task03/09、且object goal已经满足后评估；关空fixture产生的joint motion明确mask掉，避免虚假正例。
- `handoff_compatible`：exact release/lift/base continuation结果。
- `harm_or_worsening`：target regression、object drop、wrong-direction joint regression、done-without-success或异常。
- 删除F7的瞬时`contact_ready≤6cm` factor。

F8需要报告每个state在哪个gate停止：没有acquire、acquired但未place、placed但未close、recovery成功但handoff失败、或fallback。proposal generator、factor gate、handoff必须分别诊断。

## Evaluation与支持/否定标准

- 首要指标：新6 states上factor-gated与ungated的local success、object acquisition、object-goal、harm及handoff差异；
- offline oracle matrix可以执行所有proposal，但任何selected-only confirmation必须先冻结choice，不能用同state outcomes选择；
- 支持：gated方法提高acquisition→placement转化或减少关空fixture/harm，并新增state coverage；
- 部分支持：gate改善物理factor但handoff仍失败，保留acquisition factor并转向handoff；
- 否定：gate不改变阶段结果或ungated同样可靠，则不把acquisition factor加入最终representation；
- 若新cohort仍coverage低，下一实验继续改proposal generator，不训练selector。

## 完成artifacts

```text
experiments/EXP_F8/frozen_protocol.json
experiments/EXP_F8/factor_schema.json
experiments/EXP_F8/confirmed_failures.jsonl
experiments/EXP_F8/restore_cross_env.jsonl
experiments/EXP_F8/proposal_inputs/*.npz
experiments/EXP_F8/rollouts/*.npz
experiments/EXP_F8/outcomes.jsonl
experiments/EXP_F8/factor_matrix.csv
experiments/EXP_F8/metrics.json
experiments/EXP_F8/run_metadata.json
experiments/EXP_F8/audit.json
reports/EXP_F8_report.md
reports/next_exp_fromF8.md
```

F8只有在新confirmed failures、真实factor-gated interventions、feedback和measurable outcomes全部落盘后才完成。现在立即执行，不等待确认。
