"""风险缓解情景的蒙特卡洛仿真：量化接口契约冻结与自动化回归的潜在收益。

脚本保留与正式实验相同的固定种子结果，并以 20 个连续随机种子重复原始与
缓解情景，从而避免仅凭单一种子判断参数收敛效果。该结果是条件性模拟，不是
对真实项目必然缩短工期的承诺。
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
REPLICATION_SEEDS = tuple(range(SEED, SEED + 20))
# 缓解情景参数：T14、T16 悲观工期收敛
MITIGATION = {"T14": 12.0, "T16": 17.0}


def simulate_project(tasks, order, seed: int) -> np.ndarray:
    """在给定种子下返回项目总工期样本。"""
    rng = np.random.default_rng(seed)
    sampled = sample_beta_pert(tasks, RUNS, rng)
    _, ef, _, _, _ = calculate_schedule(sampled, tasks, order)
    sinks = dependency_indexes(tasks)[2]
    return np.max(ef[:, sinks], axis=1)


def metric_row(prefix: str, durations: np.ndarray) -> dict[str, float]:
    """将一次模拟的关键统计量整理为可比较字段。"""
    return {
        f"{prefix}平均总工期": float(np.mean(durations)),
        f"{prefix}P50总工期": float(np.quantile(durations, 0.50)),
        f"{prefix}P80总工期": float(np.quantile(durations, 0.80)),
        f"{prefix}P90总工期": float(np.quantile(durations, 0.90)),
        f"{prefix}80工作日按期概率": float(np.mean(durations <= 80.0)),
        f"{prefix}90工作日按期概率": float(np.mean(durations <= 90.0)),
    }


def main() -> int:
    tasks = load_tasks(ROOT / "data" / "tasks.csv")
    order = topological_order(tasks)
    mitigated = [
        replace(task, b=MITIGATION[task.task_id]) if task.task_id in MITIGATION else task
        for task in tasks
    ]

    durations = simulate_project(mitigated, order, SEED)

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

    replications = []
    for seed in REPLICATION_SEEDS:
        baseline = metric_row("原始", simulate_project(tasks, order, seed))
        scenario = metric_row("缓解", simulate_project(mitigated, order, seed))
        replications.append({
            "随机种子": seed,
            **baseline,
            **scenario,
            "平均总工期变化": scenario["缓解平均总工期"] - baseline["原始平均总工期"],
            "P80总工期变化": scenario["缓解P80总工期"] - baseline["原始P80总工期"],
            "P90总工期变化": scenario["缓解P90总工期"] - baseline["原始P90总工期"],
            "80工作日按期概率变化": scenario["缓解80工作日按期概率"] - baseline["原始80工作日按期概率"],
            "90工作日按期概率变化": scenario["缓解90工作日按期概率"] - baseline["原始90工作日按期概率"],
        })
    with (out_dir / "mitigation_scenario_replications.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replications[0]))
        writer.writeheader()
        writer.writerows(replications)

    print(f"已完成缓解情景 {RUNS:,} 次模拟（T14 b→12、T16 b→17，种子 {SEED}）。")
    for row in rows:
        print(f"{row['指标']}：{row['数值']:.4f} {row['单位']}")
    print(f"已完成 {len(REPLICATION_SEEDS)} 个随机种子的重复实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
