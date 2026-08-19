# Next Experiment from EXP_F2：EXP_F3 Articulation-Closure Recovery Skills

## F2 后主线状态

Priority A 的 confirmed-failure snapshot与matched execution已经成立；Priority B 从F1的2/6扩大到F2在articulated subset上的6/8，但仍不够稳定。F2 的32/40 `In` 对19/40 `Close`直接定位了新的结构瓶颈：很多 proposal能完成对象placement，却没有可靠关闭 drawer/microwave。

F3 继续解决 proposal coverage，不训练 selector，不回到 failure detection，也不把 predicate accuracy当研究目标。

## 应放弃的 formulation

- 不再通过增加 full-task horizon解决 microwave closure；F2给了240步且仍失败。
- 不再重复 fixed/predicate prompt duration微调；predicate upper bound没有胜过fixed。
- 不再用仅按 object XYZ offset适配的 raw retrieval；它不编码 handle/contact geometry。
- 不把“达到 In”当完整 recovery success；deployment goal明确还要求 Close。

## 新科学假设

> `In=true, Close=false` 是一个独立、可执行且可监督的 recovery mode。显式的 articulation-closure skill——使用真实 fixture joint/handle state与闭环 task-space control，或从成功 closure segments学习的 feedback skill——能够为F2未覆盖 states增加新的proposal headroom；若它只关闭空机构却破坏已放置对象，则还需要独立的 placement-retention consequence。

这个 hypothesis直接源于F2的机制数据，不是新超参数。

## F3 implementation directions

### A. Analytic handle/joint feedback skill

读取实际 MuJoCo model中 drawer/microwave的 hinge/slide joint、handle geom/site和EEF pose；实现短时 waypoint controller：approach handle → establish contact →沿关节允许方向push/slide → disengage。每5步从真实joint/EEF/contact反馈修正，不恢复world。

该方向只有在实际模型暴露稳定handle/site且OSC action语义支持时执行；必须先从代码/runtime读取精确名字与joint axis，禁止猜ID。

### B. Demonstration-derived closure feedback skill

如果analytic handle接口不稳定，则从Wave-19 train successes中通过重放当前goal predicates提取 `In首次成立 -> Close成立` 的真实 closure segments。训练或检索一个以当前EEF、fixture joint、object retention与最近action为输入的短时 feedback controller；必须按source episode分组，不能把同一轨迹窗口分到train/test两边。

### C. Mechanism controls

- F2 fixed semantic milestone（强prompt baseline）；
- equal-budget full-task replan；
- close-prompt-only；
- analytic或learned closure skill；
- shuffled/wrong-joint direction control，用于证明joint/handle alignment而非额外动作预算；
- structured fallback使用独立closure skill，而不是复用失败的local proposal。

## 新数据与实际 intervention

- 从F2中所有`In=true, final=false` states重新生成/保存可恢复的closure-boundary snapshots；如果F2 rollout没有保存完整中间snapshot，必须在同一failure/seed下重执行至首次`In`并保存controller state，这只是F3基础设施，不单占EXP。
- 在每个closure-boundary snapshot上执行全部closure proposals并记录object retention、joint motion、Close、final official success。
- 再生成至少两个新的task09 confirmed failures并通过F2最强placement controller到达closure boundary，作为新execution evidence。
- F3完成必须包含新的实际closure interventions和新的oracle coverage matrix。

## consequence decomposition

每条closure intervention至少记录：

```text
placement_retained (In after closure)
fixture_joint_progress / Close
unsafe_or_invalid under supported predicates
handoff_needed and handoff_success
final official success
```

这样可区分“没有closure proposal”“closure动作有效但把物体弄出容器”“closure成功但handoff失败”。这也开始为后续factor definitions提供实测依据，但F3本身仍不训练factor predictor。

## 支持/否定

支持：新closure family在F2 extended/fixed/predicate都失败的checkpoint上产生`Close && In`，对union coverage有正marginal contribution，并在新states复现。

否定：closure skill不能改变fixture joint、只在旧state成功、关闭时系统性破坏`In`，或wrong-direction control同样有效。否定后F4必须转向learned contact-rich recovery policy/short-horizon planner，而不是调整一个push magnitude。

## F3完成 artifacts

```text
experiments/EXP_F3/frozen_protocol.json
experiments/EXP_F3/closure_checkpoints.jsonl
experiments/EXP_F3/checkpoints/*.pkl
experiments/EXP_F3/rollouts/*.npz
experiments/EXP_F3/outcomes.jsonl
experiments/EXP_F3/coverage_matrix.csv
experiments/EXP_F3/metrics.json
experiments/EXP_F3/run_metadata.json
experiments/EXP_F3/audit.json
reports/EXP_F3_report.md
reports/next_exp_fromF3.md
```

F3审计必须从step traces重算joint progress、In retention、Close、final success与marginal coverage。完成后依据coverage决定是否进入deployment-consistent consequence labels；现在立即执行F3。
