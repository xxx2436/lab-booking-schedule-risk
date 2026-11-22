"""风险缓解情景的蒙特卡洛仿真：量化接口契约冻结与自动化回归的收益。

与主实验（src/simulate.py）相比，本脚本仅将 T14 的悲观工期由 16 收敛至 12、
T16 的悲观工期由 20 收敛至 17，其余输入、随机种子与模拟次数保持不变，
用于在报告中提供"缓解前后"的对比（见报告第 7 章表 9）。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulate import (  # noqa: E402
    calculate_schedule,
    dependency_indexes,
    load_tasks,
    sample_beta_pert,
    topological_order,
)

SEED = 20261115
RUNS = 10_000
DEADLINES = (80.0, 90.0)
# 缓解情景参数：T14、T16 悲观工期收敛
MITIGATION = {"T14": 12.0, "T16": 17.0}


def main() -> int:
    tasks = load_tasks(ROOT / "data" / "tasks.csv")
    order = topological_order(tasks)

    mitigated = [
        replace(task, b=MITIGATION[task.task_id]) if task.task_id in MITIGATION else task
        for task in tasks
    ]

    rng = np.random.default_rng(SEED)
    sampled = sample_beta_pert(mitigated, RUNS, rng)
    _, ef, _, _, _ = calculate_schedule(sampled, mitigated, order)
    sinks = dependency_indexes(mitigated)[2]
    durations = np.max(ef[:, sinks], axis=1)

    rows = [
        {"指标": "蒙特卡洛平均总工期", "数值": float(np.mean(durations)), "单位": "工作日"},
        {"指标": "P50总工期", "数值": float(np.quantile(durations, 0.50)), "单位": "工作日"},
        {"指标": "P80总工期", "数值": float(np.quantile(durations, 0.80)), "单位": "工作日"},
        {"指标": "P90总工期", "数值": float(np.quantile(durations, 0.90)), "单位": "工作日"},
    ]
    for deadline in DEADLINES:
        rows.append({
            "指标": f"{deadline:g}工作日按期概率",
            "数值": float(np.mean(durations <= deadline)),
            "单位": "概率",
        })

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "mitigation_scenario_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["指标", "数值", "单位"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已完成缓解情景 {RUNS:,} 次模拟（T14 b→12、T16 b→17，种子 {SEED}）。")
    for row in rows:
        print(f"{row['指标']}：{row['数值']:.4f} {row['单位']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
