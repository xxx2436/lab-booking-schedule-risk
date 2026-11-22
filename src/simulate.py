"""基于项目 WBS 的 CPM 与 Beta-PERT 蒙特卡洛仿真程序。

输入 CSV 是案例任务、依赖关系和三点估算的唯一来源；所有结果均在运行时计算。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


LAMBDA_PERT = 4.0
CRITICAL_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Task:
    """从已确认 CSV 输入中读取的一项 WBS 任务。"""

    task_id: str
    name: str
    description: str
    predecessors: tuple[str, ...]
    a: float
    m: float
    b: float
    rationale: str

    @property
    def pert_mean(self) -> float:
        """PERT 期望工期，用作 CPM 的确定性工期。"""
        return (self.a + LAMBDA_PERT * self.m + self.b) / (LAMBDA_PERT + 2.0)


def load_tasks(csv_path: Path) -> list[Task]:
    """读取并校验唯一的案例输入文件。"""
    required = {
        "任务ID", "任务名称", "任务说明", "前置任务",
        "乐观工期a", "最可能工期m", "悲观工期b", "三点估算依据",
    }
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("tasks.csv 缺少一个或多个必需字段。")
        rows = list(reader)

    if not rows:
        raise ValueError("tasks.csv 至少应包含一个任务。")

    tasks: list[Task] = []
    ids: set[str] = set()
    for row in rows:
        task_id = row["任务ID"].strip()
        if not task_id or task_id in ids:
            raise ValueError(f"任务 ID 必须非空且唯一：{task_id!r}")
        ids.add(task_id)
        try:
            a = float(row["乐观工期a"])
            m = float(row["最可能工期m"])
            b = float(row["悲观工期b"])
        except ValueError as exc:
            raise ValueError(f"任务 {task_id} 的工期参数不是数值。") from exc
        if a < 0 or not (a <= m <= b):
            raise ValueError(f"任务 {task_id} 必须满足 0 <= a <= m <= b。")
        predecessor_text = row["前置任务"].strip()
        predecessors = () if predecessor_text == "无" else tuple(
            item.strip() for item in predecessor_text.split("|") if item.strip()
        )
        tasks.append(Task(
            task_id=task_id,
            name=row["任务名称"].strip(),
            description=row["任务说明"].strip(),
            predecessors=predecessors,
            a=a,
            m=m,
            b=b,
            rationale=row["三点估算依据"].strip(),
        ))

    for task in tasks:
        unknown = set(task.predecessors) - ids
        if unknown:
            raise ValueError(f"任务 {task.task_id} 引用了不存在的前置任务：{sorted(unknown)}")
        if task.task_id in task.predecessors:
            raise ValueError(f"任务 {task.task_id} 不能以自身为前置任务。")
    return tasks


def topological_order(tasks: list[Task]) -> list[int]:
    """返回稳定的拓扑顺序，并拒绝循环依赖。"""
    index = {task.task_id: idx for idx, task in enumerate(tasks)}
    indegree = [len(task.predecessors) for task in tasks]
    successors: list[list[int]] = [[] for _ in tasks]
    for child, task in enumerate(tasks):
        for predecessor in task.predecessors:
            successors[index[predecessor]].append(child)

    ready = [idx for idx, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for successor in successors[current]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(tasks):
        raise ValueError("任务依赖图存在循环依赖。")
    return order


def dependency_indexes(tasks: list[Task]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """构建前置任务、后续任务和终止任务的索引列表。"""
    index = {task.task_id: idx for idx, task in enumerate(tasks)}
    predecessors = [[index[item] for item in task.predecessors] for task in tasks]
    successors: list[list[int]] = [[] for _ in tasks]
    for child, parent_indexes in enumerate(predecessors):
        for parent in parent_indexes:
            successors[parent].append(child)
    sinks = [idx for idx, child_list in enumerate(successors) if not child_list]
    return predecessors, successors, sinks


def sample_beta_pert(tasks: list[Task], n_runs: int, rng: np.random.Generator) -> np.ndarray:
    """以缩放后的 Beta-PERT 分布抽样；常数工期任务保持不变。"""
    sampled = np.empty((n_runs, len(tasks)), dtype=float)
    for column, task in enumerate(tasks):
        if task.a == task.b:
            sampled[:, column] = task.a
            continue
        span = task.b - task.a
        alpha = 1.0 + LAMBDA_PERT * (task.m - task.a) / span
        beta = 1.0 + LAMBDA_PERT * (task.b - task.m) / span
        sampled[:, column] = task.a + span * rng.beta(alpha, beta, size=n_runs)
    return sampled


def calculate_schedule(
    durations: np.ndarray,
    tasks: list[Task],
    order: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """为单次或多次工期输入计算 ES、EF、LS、LF 与总时差。"""
    values = np.asarray(durations, dtype=float)
    is_one_run = values.ndim == 1
    if is_one_run:
        values = values.reshape(1, -1)
    if values.shape[1] != len(tasks):
        raise ValueError("工期数组的任务数量与 tasks.csv 不一致。")

    predecessors, successors, sinks = dependency_indexes(tasks)
    n_runs, n_tasks = values.shape
    es = np.zeros((n_runs, n_tasks), dtype=float)
    ef = np.zeros((n_runs, n_tasks), dtype=float)
    for task_index in order:
        parent_indexes = predecessors[task_index]
        if parent_indexes:
            es[:, task_index] = np.max(ef[:, parent_indexes], axis=1)
        ef[:, task_index] = es[:, task_index] + values[:, task_index]

    project_duration = np.max(ef[:, sinks], axis=1)
    lf = np.zeros((n_runs, n_tasks), dtype=float)
    ls = np.zeros((n_runs, n_tasks), dtype=float)
    for task_index in reversed(order):
        child_indexes = successors[task_index]
        if child_indexes:
            lf[:, task_index] = np.min(ls[:, child_indexes], axis=1)
        else:
            lf[:, task_index] = project_duration
        ls[:, task_index] = lf[:, task_index] - values[:, task_index]
    total_float = ls - es

    if is_one_run:
        return es[0], ef[0], ls[0], lf[0], total_float[0]
    return es, ef, ls, lf, total_float


def average_rank(values: np.ndarray) -> np.ndarray:
    """在不依赖 SciPy 的情况下，对向量进行含并列值的平均秩排序。"""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """返回 Pearson 相关系数；若有向量为常量则返回 NaN。"""
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_sensitivity(
    sampled_durations: np.ndarray,
    project_durations: np.ndarray,
    total_float: np.ndarray,
    tasks: list[Task],
) -> list[dict[str, float | int | str]]:
    """计算 Pearson 主排序、Spearman 交叉验证和关键路径出现频率。"""
    total_ranks = average_rank(project_durations)
    records: list[dict[str, float | int | str]] = []
    for index, task in enumerate(tasks):
        samples = sampled_durations[:, index]
        pearson = correlation(samples, project_durations)
        spearman = correlation(average_rank(samples), total_ranks)
        critical_count = int(np.count_nonzero(np.abs(total_float[:, index]) <= CRITICAL_TOLERANCE))
        records.append({
            "任务ID": task.task_id,
            "任务名称": task.name,
            "Pearson相关系数": pearson,
            "Spearman秩相关系数": spearman,
            "关键路径出现次数": critical_count,
            "关键路径出现频率": critical_count / sampled_durations.shape[0],
        })

    def rank_key(record: dict[str, float | int | str]) -> float:
        value = float(record["Pearson相关系数"])
        return -abs(value) if np.isfinite(value) else float("inf")

    records.sort(key=rank_key)
    for rank, record in enumerate(records, start=1):
        record["敏感度排序"] = rank
    return records


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    """写入可由 Excel 正确识别的 UTF-8 BOM CSV 输出。"""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    tasks: list[Task],
    cpm_schedule: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sampled: np.ndarray,
    project_durations: np.ndarray,
    sensitivity: list[dict[str, float | int | str]],
    deadlines: list[float],
    seed: int,
    order: list[int],
) -> None:
    """生成全部结果表、可复现配置和所需图表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cpm_es, cpm_ef, cpm_ls, cpm_lf, cpm_float = cpm_schedule

    def clean_schedule_value(value: float) -> float:
        """清除浮点计算在进度表中产生的近零残差。"""
        return 0.0 if abs(value) <= CRITICAL_TOLERANCE else float(value)

    cpm_rows = []
    for index, task in enumerate(tasks):
        cpm_rows.append({
            "任务ID": task.task_id,
            "任务名称": task.name,
            "CPM确定性工期_PERT期望值": task.pert_mean,
            "最早开始ES": clean_schedule_value(cpm_es[index]),
            "最早完成EF": clean_schedule_value(cpm_ef[index]),
            "最迟开始LS": clean_schedule_value(cpm_ls[index]),
            "最迟完成LF": clean_schedule_value(cpm_lf[index]),
            "总时差TF": clean_schedule_value(cpm_float[index]),
            "CPM关键任务": "是" if abs(cpm_float[index]) <= CRITICAL_TOLERANCE else "否",
        })
    write_csv(output_dir / "cpm_schedule.csv", cpm_rows, list(cpm_rows[0]))

    simulation_rows = []
    for run_index, values in enumerate(sampled, start=1):
        row = {"仿真编号": run_index}
        row.update({task.task_id: values[index] for index, task in enumerate(tasks)})
        row["项目总工期"] = project_durations[run_index - 1]
        simulation_rows.append(row)
    write_csv(
        output_dir / "simulation_runs.csv",
        simulation_rows,
        ["仿真编号", *[task.task_id for task in tasks], "项目总工期"],
    )

    cpm_total = float(max(cpm_ef[index] for index in dependency_indexes(tasks)[2]))
    summary_rows = [
        {"指标": "CPM总工期_PERT期望工期", "数值": cpm_total, "单位": "工作日"},
        {"指标": "蒙特卡洛期望总工期", "数值": float(np.mean(project_durations)), "单位": "工作日"},
        {"指标": "蒙特卡洛总工期标准差", "数值": float(np.std(project_durations, ddof=1)), "单位": "工作日"},
        {"指标": "P50总工期", "数值": float(np.quantile(project_durations, 0.50)), "单位": "工作日"},
        {"指标": "P80总工期", "数值": float(np.quantile(project_durations, 0.80)), "单位": "工作日"},
        {"指标": "P90总工期", "数值": float(np.quantile(project_durations, 0.90)), "单位": "工作日"},
        {"指标": "模拟次数", "数值": int(project_durations.size), "单位": "次"},
        {"指标": "随机种子", "数值": seed, "单位": "无"},
    ]
    write_csv(output_dir / "project_summary.csv", summary_rows, ["指标", "数值", "单位"])

    deadline_rows = []
    for deadline in deadlines:
        count = int(np.count_nonzero(project_durations <= deadline))
        deadline_rows.append({
            "截止日期_工作日": deadline,
            "按期完成次数": count,
            "模拟次数": int(project_durations.size),
            "按期完工概率": count / project_durations.size,
        })
    write_csv(
        output_dir / "deadline_probabilities.csv",
        deadline_rows,
        ["截止日期_工作日", "按期完成次数", "模拟次数", "按期完工概率"],
    )

    sensitivity_fields = [
        "任务ID", "任务名称", "Pearson相关系数", "Spearman秩相关系数",
        "关键路径出现次数", "关键路径出现频率", "敏感度排序",
    ]
    write_csv(output_dir / "task_sensitivity.csv", sensitivity, sensitivity_fields)

    duration_rows = []
    for index, task in enumerate(tasks):
        values = sampled[:, index]
        duration_rows.append({
            "任务ID": task.task_id,
            "任务名称": task.name,
            "均值": float(np.mean(values)),
            "标准差": float(np.std(values, ddof=1)),
            "P10": float(np.quantile(values, 0.10)),
            "P50": float(np.quantile(values, 0.50)),
            "P90": float(np.quantile(values, 0.90)),
            "最小值": float(np.min(values)),
            "最大值": float(np.max(values)),
        })
    write_csv(output_dir / "task_duration_summary.csv", duration_rows, list(duration_rows[0]))

    comparison_rows = [
        {"指标": "CPM基准工期（PERT期望值）", "工期_工作日": cpm_total},
        {"指标": "蒙特卡洛平均总工期", "工期_工作日": float(np.mean(project_durations))},
        {"指标": "蒙特卡洛P50总工期", "工期_工作日": float(np.quantile(project_durations, 0.50))},
        {"指标": "蒙特卡洛P80总工期", "工期_工作日": float(np.quantile(project_durations, 0.80))},
        {"指标": "蒙特卡洛P90总工期", "工期_工作日": float(np.quantile(project_durations, 0.90))},
    ]
    write_csv(output_dir / "cpm_vs_monte_carlo.csv", comparison_rows, ["指标", "工期_工作日"])

    tornado_rows = _calculate_tornado(tasks, order)
    write_csv(
        output_dir / "tornado_analysis.csv", tornado_rows,
        ["任务ID", "任务名称", "乐观情景项目总工期", "悲观情景项目总工期", "工期影响幅度"],
    )

    config = {
        "输入文件": "data/tasks.csv",
        "模拟次数": int(project_durations.size),
        "随机种子": seed,
        "Beta-PERT形状参数lambda": LAMBDA_PERT,
        "关键路径判定容差": CRITICAL_TOLERANCE,
        "截止日期_工作日": deadlines,
        "CPM确定性工期定义": "(a + 4m + b) / 6",
        "拓扑顺序": [tasks[index].task_id for index in order],
        "NumPy版本": np.__version__,
        "Matplotlib版本": matplotlib.__version__,
    }
    with (output_dir / "config_used.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    _plot_histogram(output_dir, project_durations, cpm_total, deadlines)
    _plot_sensitivity(output_dir, sensitivity)
    _plot_task_network(output_dir, tasks, cpm_float)
    _plot_duration_boxplot(output_dir, sampled, tasks)
    _plot_duration_cdf(output_dir, project_durations)
    _plot_deadline_probability_curve(output_dir, project_durations, deadlines)
    _plot_critical_path_frequency(output_dir, sensitivity)
    _plot_tornado_chart(output_dir, tornado_rows)
    _plot_cpm_vs_monte_carlo(output_dir, comparison_rows)


def _project_duration_for_durations(durations: np.ndarray, tasks: list[Task], order: list[int]) -> float:
    """计算给定一组确定性任务工期时的项目总工期。"""
    _, ef, _, _, _ = calculate_schedule(durations, tasks, order)
    return float(np.max(ef[dependency_indexes(tasks)[2]]))


def _calculate_tornado(tasks: list[Task], order: list[int]) -> list[dict[str, float | str]]:
    """固定其他任务为最可能工期，计算单任务乐观与悲观情景的影响。"""
    base = np.array([task.m for task in tasks], dtype=float)
    rows: list[dict[str, float | str]] = []
    for index, task in enumerate(tasks):
        optimistic = base.copy()
        pessimistic = base.copy()
        optimistic[index] = task.a
        pessimistic[index] = task.b
        low = _project_duration_for_durations(optimistic, tasks, order)
        high = _project_duration_for_durations(pessimistic, tasks, order)
        rows.append({
            "任务ID": task.task_id,
            "任务名称": task.name,
            "乐观情景项目总工期": low,
            "悲观情景项目总工期": high,
            "工期影响幅度": high - low,
        })
    return sorted(rows, key=lambda item: float(item["工期影响幅度"]), reverse=True)


def _plot_histogram(output_dir: Path, project_durations: np.ndarray, cpm_total: float, deadlines: list[float]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    axis.hist(project_durations, bins=40, color="#4C78A8", edgecolor="white", alpha=0.9)
    axis.axvline(cpm_total, color="#E45756", linewidth=2, label="CPM 基准工期")
    axis.axvline(np.mean(project_durations), color="#54A24B", linewidth=2, label="蒙特卡洛平均工期")
    for deadline in deadlines:
        axis.axvline(deadline, color="#F2CF5B", linestyle="--", linewidth=1.6, label=f"截止日期 {deadline:g}")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys())
    axis.set_title("项目总工期分布直方图")
    axis.set_xlabel("项目总工期（工作日）")
    axis.set_ylabel("模拟频数")
    fig.savefig(output_dir / "project_duration_histogram.png", dpi=180)
    plt.close(fig)


def _plot_sensitivity(output_dir: Path, sensitivity: list[dict[str, float | int | str]]) -> None:
    ordered = list(reversed(sensitivity))
    task_ids = [str(item["任务ID"]) for item in ordered]
    values = [float(item["Pearson相关系数"]) for item in ordered]
    fig, axis = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    axis.barh(task_ids, values, color="#4C78A8")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("任务工期 Pearson 敏感度分析")
    axis.set_xlabel("与项目总工期的 Pearson 相关系数")
    axis.set_ylabel("任务 ID")
    fig.savefig(output_dir / "sensitivity_pearson.png", dpi=180)
    plt.close(fig)


def _plot_task_network(output_dir: Path, tasks: list[Task], cpm_float: np.ndarray) -> None:
    """绘制任务依赖网络，并以颜色区分 CPM 关键任务。"""
    task_index = {task.task_id: index for index, task in enumerate(tasks)}
    levels: list[int] = []
    for task in tasks:
        levels.append(0 if not task.predecessors else 1 + max(levels[task_index[pred]] for pred in task.predecessors))
    positions: dict[int, tuple[float, float]] = {}
    for level in sorted(set(levels)):
        members = [index for index, item_level in enumerate(levels) if item_level == level]
        for row, index in enumerate(members):
            positions[index] = (float(level), -float(row))

    fig, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    for child, task in enumerate(tasks):
        for predecessor in task.predecessors:
            parent = task_index[predecessor]
            is_critical = (
                abs(cpm_float[parent]) <= CRITICAL_TOLERANCE
                and abs(cpm_float[child]) <= CRITICAL_TOLERANCE
            )
            axis.annotate(
                "", xy=positions[child], xytext=positions[parent],
                arrowprops={"arrowstyle": "->", "color": "#E45756" if is_critical else "#8C8C8C", "lw": 1.7},
            )
    for index, task in enumerate(tasks):
        color = "#E45756" if abs(cpm_float[index]) <= CRITICAL_TOLERANCE else "#4C78A8"
        x, y = positions[index]
        axis.scatter(x, y, s=1050, color=color, edgecolors="white", linewidths=1.5, zorder=3)
        axis.text(x, y, task.task_id, ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=4)
    axis.scatter([], [], s=120, color="#E45756", label="CPM 关键任务")
    axis.scatter([], [], s=120, color="#4C78A8", label="非关键任务")
    axis.legend(loc="upper left")
    axis.set_title("任务依赖网络图（节点以任务 ID 表示）")
    axis.set_xlabel("任务阶段层级")
    axis.set_yticks([])
    axis.set_xticks(sorted(set(levels)))
    axis.set_xlim(-0.8, max(levels) + 0.8)
    axis.set_ylim(min(y for _, y in positions.values()) - 0.8, 0.8)
    axis.set_frame_on(False)
    fig.savefig(output_dir / "task_network.png", dpi=180)
    plt.close(fig)


def _plot_duration_boxplot(output_dir: Path, sampled: np.ndarray, tasks: list[Task]) -> None:
    """绘制各任务抽样工期的箱线图。"""
    fig, axis = plt.subplots(figsize=(13, 6.2), constrained_layout=True)
    axis.boxplot(sampled, tick_labels=[task.task_id for task in tasks], showfliers=False)
    axis.set_title("各任务 Beta-PERT 抽样工期箱线图")
    axis.set_xlabel("任务 ID")
    axis.set_ylabel("工期（工作日）")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "task_duration_boxplot.png", dpi=180)
    plt.close(fig)


def _plot_duration_cdf(output_dir: Path, project_durations: np.ndarray) -> None:
    """绘制项目总工期经验累积分布，并标记主要分位数。"""
    values = np.sort(project_durations)
    probabilities = np.arange(1, values.size + 1) / values.size
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    axis.plot(values, probabilities, color="#4C78A8", linewidth=2)
    for quantile, color in ((0.50, "#54A24B"), (0.80, "#F2CF5B"), (0.90, "#E45756")):
        duration = float(np.quantile(project_durations, quantile))
        axis.axvline(duration, color=color, linestyle="--", linewidth=1.6, label=f"P{int(quantile * 100)}：{duration:.2f} 天")
    axis.set_title("项目总工期经验累积分布函数")
    axis.set_xlabel("项目总工期（工作日）")
    axis.set_ylabel("累计概率")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(output_dir / "project_duration_cdf.png", dpi=180)
    plt.close(fig)


def _plot_deadline_probability_curve(
    output_dir: Path, project_durations: np.ndarray, deadlines: list[float],
) -> None:
    """绘制截止日期与经验按期完工概率之间的关系。"""
    lower = float(np.floor(np.min(project_durations)))
    upper = float(np.ceil(np.max(project_durations)))
    dates = np.linspace(lower, upper, 120)
    probabilities = [float(np.mean(project_durations <= date)) for date in dates]
    fig, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    axis.plot(dates, probabilities, color="#4C78A8", linewidth=2, label="按期完工概率")
    for deadline in deadlines:
        probability = float(np.mean(project_durations <= deadline))
        axis.scatter([deadline], [probability], color="#E45756", zorder=3)
        axis.annotate(f"{deadline:g} 天：{probability:.1%}", (deadline, probability), xytext=(6, 7), textcoords="offset points")
    axis.set_title("截止日期与按期完工概率曲线")
    axis.set_xlabel("截止日期（工作日）")
    axis.set_ylabel("按期完工概率")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.savefig(output_dir / "deadline_probability_curve.png", dpi=180)
    plt.close(fig)


def _plot_critical_path_frequency(output_dir: Path, sensitivity: list[dict[str, float | int | str]]) -> None:
    """绘制每项任务在模拟关键路径中的出现频率。"""
    ordered = sorted(sensitivity, key=lambda item: float(item["关键路径出现频率"]))
    fig, axis = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    axis.barh(
        [str(item["任务ID"]) for item in ordered],
        [float(item["关键路径出现频率"]) for item in ordered],
        color="#59A14F",
    )
    axis.set_title("任务进入关键路径的频率")
    axis.set_xlabel("关键路径出现频率")
    axis.set_ylabel("任务 ID")
    axis.set_xlim(0, 1.05)
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "critical_path_frequency.png", dpi=180)
    plt.close(fig)


def _plot_tornado_chart(output_dir: Path, tornado_rows: list[dict[str, float | str]]) -> None:
    """绘制单因素乐观至悲观工期影响的龙卷风图。"""
    ordered = list(reversed(tornado_rows))
    lows = [float(row["乐观情景项目总工期"]) for row in ordered]
    highs = [float(row["悲观情景项目总工期"]) for row in ordered]
    labels = [str(row["任务ID"]) for row in ordered]
    fig, axis = plt.subplots(figsize=(9, 6.4), constrained_layout=True)
    for row, (low, high) in enumerate(zip(lows, highs)):
        axis.barh(row, high - low, left=low, color="#F28E2B", alpha=0.85)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("单因素工期影响龙卷风图")
    axis.set_xlabel("仅改变该任务时的项目总工期范围（工作日）")
    axis.set_ylabel("任务 ID")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(output_dir / "tornado_chart.png", dpi=180)
    plt.close(fig)


def _plot_cpm_vs_monte_carlo(output_dir: Path, comparison_rows: list[dict[str, float | str]]) -> None:
    """绘制 CPM 基准与蒙特卡洛工期统计量的对比图。"""
    labels = [str(row["指标"]) for row in comparison_rows]
    values = [float(row["工期_工作日"]) for row in comparison_rows]
    fig, axis = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    bars = axis.bar(range(len(labels)), values, color=["#E45756", "#4C78A8", "#59A14F", "#F2CF5B", "#B07AA1"])
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    axis.set_title("CPM 基准与蒙特卡洛项目总工期对比")
    axis.set_ylabel("工期（工作日）")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "cpm_vs_monte_carlo.png", dpi=180)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 CPM 与 Beta-PERT 蒙特卡洛工期风险分析。")
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks.csv"), help="WBS 输入 CSV 文件路径。")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="结果文件输出目录。")
    parser.add_argument("--runs", type=int, default=10_000, help="蒙特卡洛模拟次数（默认：10000）。")
    parser.add_argument("--seed", type=int, default=20261115, help="随机种子（默认：20261115）。")
    parser.add_argument(
        "--deadline", type=float, action="append", default=None,
        help="截止日期（工作日）；可重复指定多个情景。默认使用 CPM 基准工期。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.runs < 1:
        raise ValueError("--runs 必须至少为 1。")
    tasks = load_tasks(args.tasks)
    order = topological_order(tasks)
    cpm_durations = np.array([task.pert_mean for task in tasks], dtype=float)
    cpm_schedule = calculate_schedule(cpm_durations, tasks, order)
    _, cpm_ef, _, _, _ = cpm_schedule
    sinks = dependency_indexes(tasks)[2]
    cpm_total = float(np.max(cpm_ef[sinks]))
    deadlines = sorted(set(args.deadline if args.deadline is not None else [cpm_total]))

    rng = np.random.default_rng(args.seed)
    sampled = sample_beta_pert(tasks, args.runs, rng)
    es, ef, _, _, total_float = calculate_schedule(sampled, tasks, order)
    del es
    project_durations = np.max(ef[:, sinks], axis=1)
    sensitivity = compute_sensitivity(sampled, project_durations, total_float, tasks)
    write_outputs(
        args.output_dir, tasks, cpm_schedule, sampled, project_durations,
        sensitivity, deadlines, args.seed, order,
    )

    print(f"已完成 {args.runs:,} 次模拟，随机种子为 {args.seed}。")
    print(f"CPM 基准工期（PERT 期望工期）：{cpm_total:.2f} 工作日")
    print(f"蒙特卡洛平均总工期：{np.mean(project_durations):.2f} 工作日")
    for deadline in deadlines:
        probability = np.mean(project_durations <= deadline)
        print(f"截止日期为 {deadline:g} 工作日时的按期完工概率：{probability:.2%}")
    print(f"结果已写入：{args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        raise SystemExit(2)
