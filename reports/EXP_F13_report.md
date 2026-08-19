# EXP_F13：Factor-terminal recovery option graph

## 研究问题与假设

F12 表明固定时长的 closed-loop feedback 不能可靠表达 recovery 的真实完成条件，而且 segmented controller 虽然减少了 harmful recovery，却没有提高 task success。F13 因而检验一个更结构性的假设：如果将 consequence factors 直接绑定到可观测的 option terminal，并让控制器按 `recover → place → articulate → handoff` 的事件图推进，factorized interface 是否会比单段 recovery 更可靠，并让删除某个 factor 产生可解释的失败。

本实验不研究 failure prediction。所有 episode 都从已经确认失败的真实 simulator state 开始，先冻结部署决策，再执行 selected-only intervention，最后才运行 oracle branches。

## 相对 F12 的实质性推进

- 将 factor 从“多个 outcome heads”提升为实际控制接口：每个 factor 决定 option 是否终止以及下一 option。
- 新增 event-terminal option graph，而非继续调整 probe length、threshold 或网络宽度。
- 将 object-terminal 和 articulation-terminal 分别做机制级删除。
- 与 monolithic recovery 及 scalar option policy 做同状态、同 handoff 的真实 intervention 对照。
- 保存逐次 option transition，使“哪个 factor 导致了什么控制转移”可机器复核。

## 数据、方法与冻结协议

正式 cohort 使用新的 simulator instances 28 和 29，覆盖 6 个任务/严重度条件，共构造 12 个 confirmed-failure states。构造器在写入 cohort 前验证 `env.check_success() == false`。每个 state 上预先冻结 5 个部署分支：

1. `factor_options`：完整 factor-terminal graph；
2. `no_object_terminal`：删除 object-goal terminal；
3. `no_articulation_option`：删除 articulation option；
4. `monolithic_recovery`：单段 recovery；
5. `scalar_option`：以 scalar option score 决定后继 option。

部署分支全部先写入 `stage1_decisions.jsonl`，再执行 intervention。oracle 仅在全部 selected-only rollout 完成后运行，用于 proposal coverage 与 selector headroom 诊断，不进入部署决策。

主要 artifact：

- 冻结协议：`experiments/EXP_F13/frozen_protocol.json`
- confirmed failures：`experiments/EXP_F13/confirmed_failures.jsonl`
- option/factor 定义：`experiments/EXP_F13/option_schema.json`
- 部署前决策：`experiments/EXP_F13/stage1_decisions.jsonl`
- selected-only 结果：`experiments/EXP_F13/selected_outcomes.jsonl`
- event transitions：`experiments/EXP_F13/option_transitions.jsonl`
- oracle 结果：`experiments/EXP_F13/oracle_outcomes.jsonl`
- restore 交叉环境记录：`experiments/EXP_F13/restore_cross_env.jsonl`
- 汇总：`experiments/EXP_F13/metrics.json`
- 运行元数据：`experiments/EXP_F13/run_metadata.json`
- 独立审计：`experiments/EXP_F13/audit.json`

实现脚本：

- `scripts/experiments/run_exp_f13_factor_options.py`
- `scripts/experiments/resume_exp_f13_oracle.py`
- `scripts/experiments/audit_exp_f13.py`

## 实际执行量与有效性

- confirmed-failure states：12
- selected-only interventions：60
- selected rollout steps：11,593
- post-selection oracle interventions：48
- oracle rollout steps：14,686
- option transitions：86
- oracle local coverage：6/12 = 0.500
- 审计结果：PASS，0 discrepancy

审计重新计算了 cohort 数量、决策先于 outcome、decision/outcome 对应、transition 数组与 selected outcome 一致、selected/oracle step 总数、oracle local coverage 和 support confusion。oracle 的 6/12 coverage 只表示 proposal set 中至少存在成功的 local proposal，不作为部署输入。

## Quantitative results

| 部署方法 | final task success | harmful recovery | mean option count |
|---|---:|---:|---:|
| factor_options | 4/12 | 2/12 | 1.750 |
| no_object_terminal | 4/12 | 2/12 | 1.417 |
| no_articulation_option | 4/12 | 2/12 | 1.500 |
| monolithic_recovery | 5/12 | 2/12 | 1.000 |
| scalar_option | 5/12 | 2/12 | 1.500 |

完整 factor graph 的 transition counts 为：`articulate=3, fallback=1, handoff=10, place=5, terminal=2`。这证明 factors 确实改变了实际控制序列；但它没有改善最终 outcome。

support gate 的 oracle confusion 为：

- true fallback：5
- false fallback：1
- false local：1

因此总体失败不能归为单一 selector error：既有 proposal coverage 缺失，也有 support prediction 错误，还有 proposal 已存在却被 option decomposition 执行坏的情况。

## 成功案例与机制证据

在 task03 instance29 上，完整图实际完成了 `recover → place → articulate` 的多阶段序列，并最终成功。这说明 event-terminal factors 不是日志装饰：它们能够驱动真实的阶段转移，并与后续 handoff 形成可执行接口。

但相同 state 上所有主要 controller 都成功，因此该案例只支持“factor graph 可执行”，不支持“factor graph 是成功所必需”。`no_object_terminal` 与 `no_articulation_option` 虽改变了 transition 路径，却没有改变总体成功数，当前数据不能为这两个 factor 的独立 outcome value 提供正证据。

## 失败案例与分解诊断

### 1. Option decomposition 造成的 selector/control loss

task09 instance28 上，完整 graph 执行 `recover → place → articulate` 后失败；`monolithic_recovery` 与 scalar 的 monolithic 路径成功。oracle 中 semantic/ungated proposal 成功，而对应 recovery branch 失败。这里 proposal set 中存在有效行为，但 factor graph 将它拆成不稳定的阶段序列，造成了真实 outcome loss。这直接否定“更细 factor transition 必然更可靠”。

### 2. Local proposal 本身有害

task03 instance28 moderate state 被 support gate 判为 local，但 oracle 中没有成功 local recovery。所有部署 controller 的 recovery/monolithic phase 都使 task margin 明显恶化；后续 fallback 或 continuation 只增加较小的额外损失。这是 proposal coverage/control failure，不应归咎于 handoff。

### 3. Support gate 错误拒绝可行 local recovery

task09 instance29 severe state 被 gate 送往 fallback，但 post-selection oracle 的三个 local branches 都成功。严重度标签并不等价于 no-local，F13 因而继续坚持由 intervention oracle 诊断 coverage，而不是预设 severe 必须 fallback。

### 4. Fallback primitive 本身伤害 world state

task05 instance28 severe state 中，local oracle 与现有 fallback 都失败。现有 release/lift fallback 在 proposal phase 内将 margin 从约 -64 降至约 -448，随后 continuation 进一步降至约 -541。这里“选择 fallback”并没有构成安全行为；fallback controller 自身必须成为可诊断、可替换的 control formulation。

## Hypothesis judgment

核心假设被否定：当前 factor-terminal graph 没有优于 monolithic/scalar controller，且 success 由 5/12 降到 4/12。factor 删除虽然改变 transition count，却没有带来 outcome 差异，所以还不能声称这些 factors 具有独立必要性。

F13 支持的较窄结论是：factorized consequences 可以形成 machine-verifiable 的控制接口，且能把失败定位到 proposal coverage、support gate、option decomposition、fallback primitive 或 handoff；但“可诊断”尚未转化为更好的恢复性能。

## Bug、无效 run 与恢复说明

- 初始 smoke run 保存在 `experiments/EXP_F13_smoke`。其中 scalar branch 在 stage-1 已选择 fallback 后仍进入 semantic rerank，绕过 support gate。该 run 不进入正式指标。
- 修复后在 `experiments/EXP_F13_smoke_v2` 验证所有 severe smoke branches 保持 `fallback`，再开始正式 cohort。
- 正式 oracle 在 39/48 时因外部进程中断。以相同 checkpoint、normalization assets 与 seed 重启 policy server，`resume_exp_f13_oracle.py` 只执行缺失的 9 个唯一 oracle keys；没有重复记录。
- 初版 audit 将严格 mtime 小于作为顺序条件，而部分 decision/outcome 文件的纳秒 mtime 完全相等。核对数组及记录级 decision ordering 后，将文件级条件修正为 `decision_mtime <= outcome_mtime`。这不是科学数据差异，最终审计为 PASS。

以上异常及修复均保留；无效 smoke 没有被删除，也没有混入正式 12-state 指标。

## 对研究主线的新信息

F13 把问题从“factor heads 是否预测得更准”推进到了“factor 是否构成有用的 deployment handoff/control interface”。结果显示，单纯用 factor 切分 option 不够：错误的阶段边界会破坏本来有效的 monolithic trajectory；同时现有 fallback 不是天然安全的。下一实验不能继续微调 option threshold，而应把 fallback primitive 本身当作 intervention variable，并把 immediate harm、replan 后 harm 与最终 task success 分开评估。
