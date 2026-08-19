# Next Experiment from EXP_F6：EXP_F7 Goal-Conditioned Analytic Coverage and Factor Intervention

## F6后的主线状态

F6在12个full-state failures上执行72个short-horizon outcomes，local oracle coverage只有2/12，新states为0/36；所有proposal-boundary object/articulation labels为负。按照Research North Star的顺序，当前瓶颈是proposal set，而不是selector。F7不得训练一个退化articulation head，也不得通过再改几十步budget来占用新编号。

## 新科学假设

> 对articulated tasks，显式利用当前物体、handle、target joint和末端执行器几何关系构造的goal-conditioned task-space proposals，可以比语言policy proposals显著提高local coverage；而“接近handle/建立接触/产生正确方向关节进展/安全释放形成有效handoff”是可分别测量、可通过matched intervention操纵、并能解释最终continuation成败的物理consequences。

F7首先允许读取simulator state构造analytic proposals，但必须标为`privileged proposal oracle`，不能作为最终deployment方法或selector输入。其作用是区分“控制上根本不可恢复”和“现有可部署proposal generator覆盖不足”，并验证候选factor是否有独立机制含义。若oracle coverage成立，后续实验再用视觉/可观测状态替换privileged perception。

## Cohort与proposal families

沿用F6已审计的12个full-state canonical failures，所有choices从完全相同snapshot开始。对task03/task09构造至少以下matched proposals：

1. `aligned_contact_close`：预接近handle、建立接触、沿目标关节的正确切向/轴向推进、受控释放；
2. `misaligned_contact_control`：相同路径长度和动作幅度，但接触点或推进方向错配；
3. `approach_only_control`：达到handle附近但不施加闭合推进；
4. `semantic_policy_baseline`：F6中最相关的closure-specific proposal；
5. `fallback_disengage_replan`：从当前真实world state安全释放后replan。

task05不是articulation任务，使用object recovery/place analytic oracle、matched wrong-target/approach-only controls、semantic baseline和fallback，以验证object-progress factor。不得把env reset或home pose当fallback。

## Consequence interface

不再直接把最终官方`closed` predicate当唯一articulation factor。每个proposal边界保存连续物理量及冻结阈值：

- `target_proximity/contact_ready`：EEF/handle或object/target相对几何是否进入可执行接触域；
- `object_or_joint_progress`：目标物体位置误差或目标joint到goal的距离相对failure state是否改善；
- `contact_retention/stability`：proposal后段是否保持所需接触/物体关系，而非瞬时穿越；
- `handoff_compatible`：统一release/lift后由base policy在固定budget内是否成功；
- `harm_or_worsening`：错误方向joint motion、object目标误差显著增大、official done-without-success、非有限状态或既有goal回退。

阈值必须在查看formal outcomes前由MuJoCo几何尺度、task success predicate和controller tolerance确定并写入`factor_schema.json`。连续量必须保存，避免只留下阈值后的bit。

## F7需要区分的解释

- aligned oracle成功且controls失败：现有proposal generator是coverage瓶颈，physical factors有因果区分力；
- aligned和misaligned都成功：factor/control定义不够区分，不能据此声称factor必要；
- contact/joint progress改善但handoff失败：handoff是独立factor，而非scalar success的改名；
- factor改善与最终结果无关联：删除或重定义该factor；
- privileged aligned proposals仍普遍失败：analytic controller/control horizon有结构性问题，F7内修复到真实干预完成，或如实否定该proposal hypothesis后在F8换新proposal family。

## Evaluation

- 每state报告proposal oracle coverage、fallback coverage、method成功矩阵和新旧proposal的coverage增量；
- 报告各factor的正例数、连续margin分布、proposal family间matched effect和与handoff/final success的条件关系；
- 单独列出“无有效proposal”“有效proposal存在但factor判错”“factor满足但handoff失败”；
- mechanism ablation必须真实执行aligned、wrong-direction、wrong-contact/approach-only interventions，不能只离线改label；
- oracle state只能进入proposal generator，不能被伪装成部署selector特征；所有decision-time inputs、raw analytic commands、feedback traces和outcomes落盘。

## 完成artifacts

```text
experiments/EXP_F7/frozen_protocol.json
experiments/EXP_F7/factor_schema.json
experiments/EXP_F7/confirmed_failures.jsonl
experiments/EXP_F7/proposals.jsonl
experiments/EXP_F7/proposal_inputs/*.npz
experiments/EXP_F7/rollouts/*.npz
experiments/EXP_F7/outcomes.jsonl
experiments/EXP_F7/factor_matrix.csv
experiments/EXP_F7/metrics.json
experiments/EXP_F7/run_metadata.json
experiments/EXP_F7/audit.json
reports/EXP_F7_report.md
reports/next_exp_fromF7.md
```

F7只有在新proposal/control formulations真实执行并得到`intervention → feedback → measurable factor/handoff metric`后才完成。现在立即实现并运行，不等待确认。
