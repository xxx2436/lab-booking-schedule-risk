from __future__ import annotations

import numpy as np
import pytest

from src.simulate import Task, calculate_schedule, sample_beta_pert, topological_order


def make_task(task_id: str, predecessors: tuple[str, ...], a: float, m: float, b: float) -> Task:
    return Task(task_id, task_id, "test", predecessors, a, m, b, "test")


def test_pert_mean_uses_a_4m_b_over_6() -> None:
    task = make_task("A", (), 2, 3, 5)
    assert task.pert_mean == pytest.approx((2 + 4 * 3 + 5) / 6)


def test_constant_duration_uses_no_random_sampling() -> None:
    task = make_task("A", (), 4, 4, 4)
    sampled = sample_beta_pert([task], 20, np.random.default_rng(7))
    assert sampled.shape == (20, 1)
    assert np.all(sampled == 4)


def test_schedule_honours_parallel_predecessors() -> None:
    tasks = [
        make_task("A", (), 2, 2, 2),
        make_task("B", (), 3, 3, 3),
        make_task("C", ("A", "B"), 4, 4, 4),
    ]
    order = topological_order(tasks)
    es, ef, ls, lf, total_float = calculate_schedule(np.array([2.0, 3.0, 4.0]), tasks, order)
    assert es.tolist() == [0.0, 0.0, 3.0]
    assert ef.tolist() == [2.0, 3.0, 7.0]
    assert total_float.tolist() == [1.0, 0.0, 0.0]
    assert ls.tolist() == [1.0, 0.0, 3.0]
    assert lf.tolist() == [3.0, 3.0, 7.0]


def test_cycle_is_rejected() -> None:
    tasks = [
        make_task("A", ("B",), 1, 1, 1),
        make_task("B", ("A",), 1, 1, 1),
    ]
    with pytest.raises(ValueError, match="循环依赖"):
        topological_order(tasks)


def test_beta_pert_samples_stay_in_bounds() -> None:
    """验证 Beta-PERT 抽样始终落在 [a, b] 区间内。"""
    task = make_task("A", (), 2, 3, 5)
    samples = sample_beta_pert([task], 10000, np.random.default_rng(7))
    assert samples.shape == (10000, 1)
    assert np.all(samples >= 2)
    assert np.all(samples <= 5)
