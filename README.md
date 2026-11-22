# 基于蒙特卡洛模拟的校园实验室预约管理系统工期风险评估

本仓库为《软件工程管理》课程作业题目一的可复现代码与结果包。案例为“校园实验室预约管理系统”，研究对象是该软件项目的工期风险；本仓库不实现实际 Web 系统。

## 研究内容

- 以 18 项 WBS、任务依赖关系和 PERT 三点工期为案例输入。
- 以 PERT 期望工期 `(a + 4m + b) / 6` 计算 CPM 确定性基准。
- 使用 Beta-PERT 分布开展默认 10,000 次蒙特卡洛模拟。
- 输出项目总工期分布、截止日期按期概率、Pearson/Spearman 敏感度、关键路径出现频率和单因素龙卷风分析。
- 随附缓解情景脚本，用于量化接口契约冻结与自动化回归对工期风险的预期影响。

## 仓库结构

```text
.
├─ data/tasks.csv          # 唯一案例输入：WBS、依赖和 a/m/b 工期
├─ src/simulate.py         # CPM 与蒙特卡洛仿真主程序
├─ tests/test_simulate.py  # 单元测试
├─ outputs/                # 已生成的结果表、图表和 10,000 次仿真明细
├─ scripts/                # 补充分析脚本（含缓解情景）
├─ docs/                   # 运行复现说明与提交前检查清单
├─ requirements.txt        # Python 依赖
└─ .github/workflows/      # 自动测试工作流
```

## 环境与安装

建议使用 Python 3.11 或更高版本。在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
```

## 运行仿真

默认运行 10,000 次模拟，未指定截止日期时自动使用 CPM 基准工期：

```powershell
python -m src.simulate
```

复现仓库中的主要结果：

```powershell
python -m src.simulate --runs 10000 --seed 20261115 --deadline 70 --deadline 80 --deadline 90 --output-dir outputs
```

运行单元测试：

```powershell
python -m pytest -q
```

测试覆盖 PERT 期望工期、常数工期抽样、Beta-PERT 抽样边界、网络计划计算与循环依赖拒绝。

运行后将覆盖 `outputs/` 中的同名结果文件。相同输入、相同参数与随机种子下，结果可复现；改变随机种子后，抽样结果会产生合理波动。

## 已生成结果概览

仓库随附的结果基于 10,000 次模拟、随机种子 `20261115`：

| 指标 | 结果 |
|---|---:|
| CPM 基准工期 | 81.00 工作日 |
| 蒙特卡洛平均总工期 | 82.54 工作日 |
| P80 总工期 | 87.07 工作日 |
| P90 总工期 | 89.51 工作日 |
| 80 工作日按期概率 | 32.58% |
| 90 工作日按期概率 | 91.53% |

详细字段、图表说明和文件对应关系见 [docs/运行与复现说明.md](docs/运行与复现说明.md)。

如需复现报告中的缓解情景结果，可运行：

```powershell
python scripts/run_mitigation_scenario.py
```

该脚本仅将 T14、T16 的悲观工期分别从 16、20 收敛至 12、17 工作日；其余输入、模拟次数和随机种子不变，并将结果写入 `outputs/mitigation_scenario_summary.csv`。

## 代码与结果说明

- `data/tasks.csv` 是任务、依赖和三点工期的唯一案例输入，不在代码中硬编码案例参数。
- `outputs/simulation_runs.csv` 保存完整的 10,000 次仿真明细。
- 本仓库仅包含代码复现材料，不包含课程报告、参考文献清单或课程作业 PDF。

## 参考文献与使用边界

报告使用的参考文献已列在报告末尾。所有三点工期和案例业务规则均为课程作业案例假设，不代表真实校园系统的实测项目数据。
