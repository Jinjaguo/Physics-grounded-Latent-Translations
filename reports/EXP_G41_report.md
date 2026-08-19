# EXP_G41 Report: Oracle-Boundary Lift-to-Place Control through a Frozen Action Coordinate

## 结论

**SUPPORTED：改变 frozen latent trajectory 会因果性地改变真实机器人运动。NOT SUPPORTED：当前 frozen action coordinate 尚不能作为 goal-correct、可导航的 lift→place 控制坐标。**

在 7 个完全 held-out 的真实 CALVIN `lift_blue_block_slider -> place_in_slider` 边界上，记录的 source continuation 为 7/7 官方 place success，证明每个 matched 起点都有可执行的 place transition。冻结的 Wave21 language-conditioned dynamics 与 pointwise residual 各为 2/7，linear target-region path 为 0/7，新的 constrained multi-step latent path 仅为 1/7；同一 planner 换成 lift goal 或 wrong-language goal 也都是 1/7。因此新 path 没有比简单 baseline 提供 physical place 控制价值，也没有表现出 goal-specific success advantage。

负结果不是因为 path 完全没有改变 latent 或动作。相对 frozen LCT，constrained place path 把 mean decode→re-encode residual 从 2.026 降到 0.750，并把 place-region endpoint margin 从 -2.419 改善到 -0.498。相对 matched lift/wrong-language controls，它分别造成 action RMS 0.381/0.371、TCP endpoint 0.182/0.284 m、object endpoint 0.106/0.130 m 的真实差异。问题是这些变化没有沿着 goal-correct physical effect：place path 对两个语言 controls 都是 paired 1 win / 1 loss；两个正 place-region margin case 均未完成物理 place。

因此 G41 首次把历史 Wave21 的“language redirects predicted latent dynamics”推进到真实因果执行，但同时否定了更强 claim：更接近 language action region、更加 decode-consistent 的 latent path 并不自动构成可用的 physical control coordinate。

## 研究假设与必要性

F15 recovery-proposal 分支已暂停。G41 回到 Actions as Coordinates 的核心问题：机器人已经位于 lift action latent trajectory 上时，仅给出下一原子意图 place，并冻结 action representation、decoder、text projection 与 language dynamics，是否可以只修改 latent trajectory，让 decoded physical behavior 连续转入 place。

这个实验直接填补了 Wave21–Wave78 与 G 系列之间的证据断层：

- Wave21 证明同一 latent state 上换 next-language 会改变 predicted future latent，但没有执行这些动作；
- Wave28–78 广泛否定 pointwise/low-rank steering 同时满足 retarget、continuity 与 endpoint identity；
- post-Wave78 宪章提出 multi-step latent path planning，但尚未执行真实 lift→place path；
- G39/G40 的 latent 是 LIBERO failure-correction generator，并不是 atomic-language-conditioned latent dynamics。

## 新数据与基础设施

G41 从 CALVIN 官方 `task_D_D.zip` 中定点提取 21 个 Wave21 已冻结 pair 的 boundary physical frames，只下载对应 ZIP members，不下载 177 GB 全包。split 保持 Wave21 source-session independence：8 train、6 development、7 held-out test。

旧 compact archive 只含 `rel_actions` 与 `global_frame_indices`，缺失 `robot_obs/scene_obs`。新增 extractor 从官方 ZIP64 central directory 取得 member offsets，恢复每个 boundary 的完整 recorded NPZ state。14/14 train/development source continuations和7/7 held-out source continuations均在真实 CALVIN simulator 中完成官方 `place_in_slider`。

历史 snapshot gate 也得到明确修复：单个 env 中只恢复 Bullet state 会受 Python-side runtime cache影响；为每个 `(boundary, method)` 新建 fresh env 并从同一 recorded `robot_obs/scene_obs` reset 后，64-step twin replay所有物理组件与contact signature误差为0。正式比较使用fresh-env matched starts，不使用不完整的long-horizon restore。

## 冻结表示与方法

所有 deployable 方法均不能读取真实 future endpoint或intermediate future actions。冻结组件为：

- 32-D CALVIN action representation，16-D semantic + 16-D execution；
- frozen decoder，输出16×7 CALVIN action chunk；
- 6-member Wave21 `B1_correct_language` LCT ensemble；
- frozen text projection与goal embeddings；
- train-only action regions。

当前 latent 是 frozen action encoder 对最近16个实际执行controls的编码，不是 observation encoder。每执行一个 decoded 16-step chunk 后，系统观察真实 CALVIN state，保存物理反馈，并只用实际执行的 action window重新编码；下一次 latent planning使用该 re-encoded state。报告不把 action-only re-encoding描述成 observation-conditioned encoding。

比较七种机制：

1. `source_future_oracle`：记录的真实未来动作，仅用于proposal headroom；
2. `frozen_lct_place_free_rollout`：直接以place language递推LCT；
3. `linear_nearest_place_region_path`：线性走向train-only最近place coordinate；
4. `pointwise_place_residual`：历史pointwise steering代表；
5. `constrained_latent_path_place`：优化4-step latent path；
6. 同一planner但target为当前lift；
7. 同一planner但target为错误的`turn_on_lightbulb`。

constrained objective在development前登记为 terminal region、frozen LCT dynamics、decoded boundary continuity与empirical support四项，权重分别2.0、1.0、1.0、0.2，60个optimizer steps。development结果为7/14，低于LCT与pointwise的9/14；没有为救结果改权重，原配置一次性打开held-out。

## Held-out quantitative results

| Method | place success | lift retained before release | place margin ↑ | decode/re-encode ↓ | action boundary jump ↓ | mean max TCP step |
|---|---:|---:|---:|---:|---:|---:|
| source future oracle | **7/7** | 6/7 | -0.745 | **0.000** | **0.126** | 0.0152 m |
| frozen LCT place | 2/7 | 6/7 | -2.419 | 2.026 | 0.345 | 0.0136 m |
| linear place-region path | 0/7 | **7/7** | -0.449 | 0.881 | 0.646 | **0.0089 m** |
| pointwise place residual | 2/7 | 6/7 | -2.248 | 1.801 | 0.336 | 0.0117 m |
| constrained place path | 1/7 | 5/7 | **-0.498** | 0.750 | 0.377 | 0.0115 m |
| constrained lift control | 1/7 | **7/7** | -2.840 | 0.732 | 0.373 | 0.0122 m |
| constrained wrong-language control | 1/7 | **7/7** | -2.289 | **0.719** | 0.411 | 0.0105 m |

所有 learned/path方法action saturation为0，因此失败不是简单的command clipping。constrained place path 的连续性没有显著异常，但它只在5/7 case保持release前lift，弱于两个matched language controls的7/7。

## 机制分析

### 1. 语言可以因果改变真实trajectory，但当前方向不可靠

同一boundary、同一planner只换goal，会产生大幅不同的decoded action与physical endpoint。这把Wave21的offline causal redirection推进到了真实执行层。可是place、lift、wrong-language三者success完全相同，place相对每个control都是1胜1负，说明这种causal authority没有变成goal correctness。

### 2. Semantic region entry不是physical place的可靠coordinate

constrained path显著改善place-region margin，但7个case中两个margin为正的case全部physical failure；唯一physical success的margin为-0.889。source oracle也只有部分endpoint被nearest-region rule识别为place。train-only semantic region描述action identity，但不是足够的control terminal set。

### 3. Decode consistency必要但不充分

constrained path的cycle residual只有LCT的约37%，说明multi-step optimization确实让path更接近frozen encoder/decoder支持。然而它的physical success更低。这直接复现并扩展Wave22的教训：global/local decoder consistency不能单独决定goal-correct execution。

### 4. Oracle headroom很大，瓶颈不是状态不可place

每个held-out state上的source future都成功，排除了“这些lift state没有place continuation”。当前瓶颈属于 latent consequence/control semantics：optimizer不知道哪些latent displacement会在当前physical state产生所需object/contact effect。

## 成功、失败与无效runs

- 成功基础设施：官方physical boundary extraction、14/14 development与7/7 held-out source viability、fresh-env exact twin、49条held-out causal execution、independent replay audit。
- 被否定假设：constrained multi-step latent path优于LCT/pointwise并实现goal-specific place。
- 部分支持：language-conditioned latent path能够产生显著、可审计的真实physical redirection；低cycle residual与semantic target proximity可以同时取得。
- 初始viability命令两次在任何intervention前因config路径和compact archive文件假设失败；均未产生科研结果，消耗0个EXP ID。
- 单env Bullet restore在64步后有3.51的contact-signature最大误差；该路径被弃用，正式run采用经验证为0误差的fresh-env twins。
- 两个smoke目录只验证接口/梯度，不进入正式aggregate。
- 上述问题不削弱负结论；相反，source oracle与独立fresh-env replay使物理比较有效。

## Machine-verifiable artifacts

- Preregistration: `experiments/EXP_G41_preregistration.json`
- Boundary extractor: `scripts/experiments/extract_exp_g41_calvin_boundaries.py`
- Runner: `scripts/experiments/run_exp_g41_latent_path_transition.py`
- Independent auditor: `scripts/experiments/audit_exp_g41.py`
- Development: `experiments/EXP_G41_development/boundary_frames/`, `run_default/`
- Held-out source manifest: `experiments/EXP_G41/boundary_frames/source_manifest.json`
- Held-out protocol/rollouts: `experiments/EXP_G41/heldout/frozen_protocol.json`, `rollouts/`
- Per-case/aggregate metrics: `experiments/EXP_G41/heldout/case_metrics.jsonl`, `metrics.json`
- Independent audit: `experiments/EXP_G41/heldout/audit.json`

审计重新加载196个action chunks、重新编码latent，并从recorded boundary重放全部49条rollout共3,136 simulator steps；0 discrepancy。实验完成后磁盘剩余约845 GB，高于200 GB下限。

## 科学判断

G41没有证明 learned action latent 已经是 programmable coordinate system。它证明了更窄但真实的新事实：冻结表示上language-conditioned latent intervention拥有physical causal authority；同时它也证明当前semantic region/dynamics/support/continuity objective没有把这份authority校准成goal-correct local control。下一步必须直接识别“哪些latent方向在当前physical state产生哪些可重复action/object effects”，而不是继续优化endpoint threshold、loss weight、path horizon或恢复系统。
