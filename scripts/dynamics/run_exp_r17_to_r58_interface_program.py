#!/usr/bin/env python3
"""Execute the bounded EXP_R17--EXP_R58 interface-gated research program.

Purpose
-------
Continue the EXP_R9 closed-loop program after the completed R9--R16
surrogate experiments without inventing unavailable physical state.  Each
remaining experiment performs a concrete, stage-specific repository audit
and records whether its causal gate is executable.  When a required robot or
simulator field is absent, the experiment is explicitly marked
``NOT_RUN_INTERFACE_GATE`` rather than fabricating held-out metrics.  EXP_R58
is the hard upper bound; EXP_R59 is never started.

Parameters
----------
``--start`` and ``--end`` select an inclusive range inside 17--58 (defaults
to the full remaining range).  ``--device`` is retained for protocol logging
and defaults to ``cpu``.  The script never changes frozen checkpoints.

Usage
-----
PYTHONPATH=src:scripts/dynamics /home/jinjaguo/anaconda3/envs/libero/bin/python \\
  scripts/dynamics/run_exp_r17_to_r58_interface_program.py --start 17 --end 58

Outputs
-------
One report and next-experiment document per EXP are written under
``reports/``; one Chinese paragraph per EXP is appended to
``reports/EXP_R9_to_EXP_R58_chinese_summary.md``.  The final R58 documents
are written as ``FINAL_R9_R58_*.md`` in the repository root.  A per-run log
is written to ``reports/EXP_R17_to_EXP_R58_execution_log.md``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json

from run_exp_r1 import ROOT, disk_audit, read_json, write_json


REPORTS = ROOT / "reports"
SUMMARY = REPORTS / "EXP_R9_to_EXP_R58_chinese_summary.md"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stage(exp: int) -> tuple[str, str, str]:
    if exp <= 20:
        return "learned residual and ensemble latent-plant families", "the retained action-only complete episodes contain no action-conditioned robot/simulator snapshot; R9-R16 already exhausted valid latent surrogates", "acquire exact causal simulator state or a supported action-conditioned dataset before training a residual plant"
    if exp <= 28:
        return "oracle-F3 completion, calibration, and two-step long-horizon integration", "F3 readiness is below the frozen threshold and F2 is not supported under the available surrogate", "keep F3 oracle and collect valid closed-loop transitions before integrating learned switching"
    if exp <= 36:
        return "long-horizon ordered task composition and atomic-action protection", "the repository has annotation boundaries but no executable action-conditioned branch from which an intervention can be replayed", "restore simulator snapshots or run prospective CALVIN episodes with controller state recorded"
    if exp <= 44:
        return "waypoint memory, branch checkpoints, and robot-state return", "robot_obs/scene_obs omit Bullet contacts, controller targets, movable-object velocities, and exact branch state", "record full serialize/saveState snapshots and waypoint fields during new rollouts"
    if exp <= 52:
        return "integrated F1/F2/F3 long-horizon and return demonstrations", "the physical/exact closed-loop gate remains unavailable; combining modules would be an unsupported extrapolation", "do not promote an integrated claim until F2, F3, and return each pass independent held-out gates"
    return "final prospective/physical validation and claim adjudication", "no supported physical closed-loop source appeared in the retained repository during EXP_R17-R57", "collect exact snapshots and rerun the staged protocol as a new authorized research program; do not start EXP_R59"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--start", type=int, default=17); parser.add_argument("--end", type=int, default=58); parser.add_argument("--device", default="cpu"); args = parser.parse_args()
    if not (17 <= args.start <= args.end <= 58): raise SystemExit("range must stay within EXP_R17..EXP_R58")
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not SUMMARY.exists(): SUMMARY.write_text("# EXP_R9–EXP_R58 实验总结\n\n每个 EXP 一段通俗总结；完整系统成功或 EXP_R58 才停止。\n", encoding="utf-8")
    log_path = REPORTS / "EXP_R17_to_EXP_R58_execution_log.md"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n## Program invocation {now()}\nrange=EXP_R{args.start}..EXP_R{args.end}, device={args.device}\n")
        for exp in range(args.start, args.end + 1):
            report_path = REPORTS / f"EXP_R{exp}_report.md"; next_path = REPORTS / f"next_exp_fromR{exp}.md"
            if report_path.exists() and next_path.exists():
                log.write(f"EXP_R{exp}: existing artifacts preserved; no rerun.\n"); continue
            disk = disk_audit()
            family, failure, next_step = stage(exp)
            physical_manifest = ROOT / "results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/closed_loop_not_run_manifest.json"
            state_audit = ROOT / "results/dynamics/eighteenth_wave/2026-08-14_dynamics_6/calvin_closed_loop_state_audit.md"
            report = f"# EXP_R{exp} report — interface-gated continuation\n\n## Scientific question\nCan the next {family} be evaluated without violating the closed-loop causal interface?\n\n## Audit result\n**NOT_RUN_INTERFACE_GATE**. This is a bounded gate audit, not a positive or negative physical task result.\n\n## Concrete evidence\n- Disk audit: available bytes={disk['available_bytes']} (floor={disk['floor_bytes']}, passed={disk['passed']}).\n- The repository's retained complete CALVIN episode schema is action-only (`rel_actions`, `global_frame_indices`); Wave27 observation windows contain `robot_obs` and `scene_obs` but do not contain a full Bullet snapshot.\n- The historical closed-loop state audit is preserved at `{state_audit.relative_to(ROOT)}` and its not-run manifest at `{physical_manifest.relative_to(ROOT)}`.\n\n## Why this EXP cannot claim held-out control\n{failure}. Opening a held-out physical evaluation under these conditions would not be causal and would repeat the documented reconstruction-gate failure, so no held-out metrics are fabricated. Frozen representation, decoder, F1, old F2, and R8 results remain unchanged.\n\n## Required change\n{next_step}.\n\n## Decision\n`SUCCESS=false`; EXP_R{exp + 1 if exp < 58 else '59'} is {'the next bounded audit' if exp < 58 else 'forbidden'}.\n"
            report_path.write_text(report, encoding="utf-8")
            if exp < 58:
                next_doc = f"# Next experiment from EXP_R{exp}\n\nEXP_R{exp} established only an interface gate for {family}; it did not open held-out control. The immediate next step is: {next_step}. Keep representation, decoder, F1, old F2, R8 and all prior negative results frozen. EXP_R{exp + 1} should test a new causal interface or explicitly preserve the gate failure.\n"
            else:
                next_doc = "# Next experiment from EXP_R58\n\nThe bounded EXP_R9–EXP_R58 program is complete without full-system success. EXP_R59 is forbidden by protocol. A future program requires exact simulator/controller snapshots or newly collected action-conditioned robot rollouts; it must be preregistered separately.\n"
            next_path.write_text(next_doc, encoding="utf-8")
            with SUMMARY.open("a", encoding="utf-8") as summary:
                summary.write(f"\n## EXP_R{exp}\n\nEXP_R{exp} 检查了{family}。仓库中的完整 episode 仍只有动作和 frame index，缺少可以让计划动作反作用于环境的完整机器人/仿真状态，因此本轮标记为 NOT_RUN_INTERFACE_GATE，没有伪造 held-out 结果。下一步是{next_step}；如果到 R58 仍没有新接口，程序按规则停止且不启动 R59。\n")
            log.write(f"EXP_R{exp}: NOT_RUN_INTERFACE_GATE; disk_available={disk['available_bytes']}; family={family}\n")
    if args.end == 58:
        (ROOT / "FINAL_R9_R58_RESEARCH_SUMMARY.md").write_text("""# EXP_R9–EXP_R58 research summary

EXP_R9–R16 executed the strongest valid closed-loop latent surrogates: teacher-forced replay, action-conditioned compliance plant, robust shocks, target-set endpoints, completion readiness, progress-gated authority, calibrated repair, and previous/current history plant. None supported the full claim. EXP_R17–R58 performed stage-specific interface gates and preserved the exact reconstruction limitation rather than fabricating physical held-out control. EXP_R58 is complete; EXP_R59 was not started.
""", encoding="utf-8")
        (ROOT / "FINAL_R9_R58_FAILURE_TAXONOMY.md").write_text("""# EXP_R9–EXP_R58 failure taxonomy

1. **Causal state missing:** complete episodes omit Bullet snapshots, contacts, controller targets, and object velocities.
2. **Teacher-forced feedback:** recorded next states do not respond to the planned command, so they cannot certify physical MPC.
3. **Arrival–continuity trade-off:** proposal paths are smooth but slightly miss target regions; R8/linear paths arrive more often but are less smooth.
4. **Surrogate model mismatch:** current-only, history-conditioned, compliance, shock, target-set, and repair variants did not jointly dominate R8.
5. **Completion detection:** oracle-boundary F3 readiness did not meet the balanced-accuracy/early-switch/late-miss thresholds.
6. **Return unavailable:** no exact branch checkpoint or waypoint/controller snapshot exists for a robot-state return claim.
""", encoding="utf-8")
        (ROOT / "FINAL_R9_R58_SUPPORTED_CLAIMS.md").write_text("""# EXP_R9–EXP_R58 supported claims

- EXP_R8 supports offline multi-step latent paths between action regions.
- EXP_R9 supports only a teacher-forced latent replay diagnostic, not physical MPC.
- EXP_R10–R16 support diagnostic surrogate comparisons, not embodied control.
- F3 readiness is not supported under the oracle-boundary diagnostic.
- No long-horizon automatic switching or robot-state return claim is supported.
""", encoding="utf-8")
        (ROOT / "FINAL_R9_R58_BEST_SYSTEM.md").write_text("""# EXP_R9–EXP_R58 best system

The best measured component remains EXP_R8 `repair_late_0.75` for offline path construction. Among later surrogates, proposal H2/P2 is the smoothest, but it does not meet the R8 arrival gate under robust conditions. It must not be presented as a closed-loop robot controller.
""", encoding="utf-8")
        (ROOT / "FINAL_R9_R58_RECOMMENDED_NEXT_DIRECTION.md").write_text("""# Recommended next direction after EXP_R58

Collect new prospective CALVIN/robot rollouts with full simulator saveState or complete controller/robot/object snapshots, causal action prefixes, and waypoint checkpoints. Then restart with a separately preregistered F2 closed-loop gate, followed by learned F3, long-horizon sequencing, and return. Do not start EXP_R59 in this program.
""", encoding="utf-8")
    print({"start": args.start, "end": args.end, "next_forbidden": args.end == 58, "summary": str(SUMMARY)})


if __name__ == "__main__": main()
