# 输出结果

本目录中的结果均由 `data/tasks.csv` 和程序运行时的参数计算生成，不在代码中写死。

- `cpm_schedule.csv`：使用 PERT 期望工期的 CPM 进度表。
- `simulation_runs.csv`：每次模拟的抽样工期及项目总工期。
- `project_summary.csv`：项目总工期汇总统计。
- `deadline_probabilities.csv`：截止日期的按期完工概率。
- `task_sensitivity.csv`：Pearson、Spearman 与关键路径出现频率。
- `task_duration_summary.csv`：各任务抽样工期统计。
- `cpm_vs_monte_carlo.csv`：CPM 与蒙特卡洛主要工期统计量对比。
- `tornado_analysis.csv`：单因素工期影响分析。
- `config_used.json`：可复现实验配置。
- `task_network.png`：标注 CPM 关键任务的任务依赖网络图。
- `task_duration_boxplot.png`：各任务工期箱线图。
- `project_duration_histogram.png`：项目工期分布直方图。
- `project_duration_cdf.png`：项目总工期经验累积分布函数图。
- `deadline_probability_curve.png`：截止日期与按期完工概率曲线。
- `sensitivity_pearson.png`：Pearson 敏感度条形图。
- `critical_path_frequency.png`：任务进入关键路径的频率图。
- `tornado_chart.png`：单因素工期影响龙卷风图。
- `cpm_vs_monte_carlo.png`：CPM 与蒙特卡洛工期对比图。
