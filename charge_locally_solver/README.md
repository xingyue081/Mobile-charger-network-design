# 本地补能约束生成

这个目录只实现 charge-locally operation 的约束生成。合成 2 区用来和整数穷举对照。`run_20zone.py` 读取福田数据包。它不修改集中补能的求解代码。

建议阅读顺序：

1. `patterns.py`：三段利润的斜率和常数。
2. `instance.py`：合成数据、样本权重、局部盒子。
3. `support.py`：盒子与椭球交集上的支持函数。
4. `master.py`：投资、部署、repositioning 和割。
5. `separation.py`：给定部署后，一个样本的最坏利润。
6. `constraint_generation.py`：加割的循环，以及什么情况下才算收敛。
7. `enumeration.py`、`validate.py`、`check_acceleration.py`：2 区对照。
8. `real_instance.py`、`run_20zone.py`：福田数据包和限时求解。

注释里的“核对”是需要人看口径的地方，不是已经判定的错误。

模块对应关系：

- `instance.py`：合成实例和事件概率口径。
- `patterns.py`：三段利润的斜率和 beta。
- `support.py`：椭球与截断盒子上的支持函数。
- `master.py`：受限主问题。初始只放第 0 段割，随后逐条加入。
- `separation.py`：给定当前解，求一个样本的最坏三段利润。
- `constraint_generation.py`：主问题与分离之间的循环。
- `enumeration.py`：独立穷举全部小规模整数方案和全部 pattern。
- `validate.py`：比较两边的目标和决策。

在本目录运行：

```bash
python validate.py
```

结果写到 `validation_result.json`。当前已对照的范围只覆盖合成的 2 个 zone、2 个日期类型。分离在这个规模上可以求解到 Gurobi 的最优状态；这不是福田规模上的最优性证书，也不是与集中补能的运行时间比较。

`run_20zone.py` 默认读取 `/Users/xingyueyi/Desktop/CG_solver/futian`。
支持函数只对已经加入的 pattern 计算。分离模型按样本缓存，椭球仍是凸二次不等式。上界连续两轮不动时，主问题再找几个同值解一起分离。结果写在 `outputs/`，到时停止不等于收敛。
