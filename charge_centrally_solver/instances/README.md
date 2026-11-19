# 实例目录

算法代码留在上一级。这里只放数据入口，全部是指向 `mobile charger_new` 的符号链接，没有复制、也没有改原始文件。

换实例时只换目录，不改 `main.py` / `master.py` / `oracle.py`。`CG_alrogithm/solver_interface.py` 的入口是：

```text
build_solver_inputs(module2, module3)
```

半径目录必须和 `module3` 同级，名字固定为 `wasserstein_radius_calibration_outputs`。

| 目录 | 用途 |
|---|---|
| `hetero_10zone/` | 异质 10 区，区域 102, 105, 106, 107, 108, 111, 1119, 1120, 1121, 1126，kappa=180 |
| `canonical_10zone/` | 冻结 canonical 10 区，区域 708, 712, 729, 731, 737, 741, 744, 751, 775, 802 |

每个 10 区目录里还有 `solver_inputs.pkl`，那是旧求解器已经打好的包。当前 CG 接口直接读 `module2` 和 `module3`，不必再经过这个 pickle。

换规模只改 `--instance`，不改模型代码：

```text
python main.py --instance hetero_10zone
python main.py --instance canonical_10zone
python main.py --instance futian
python lagrangian_loop.py --instance hetero_10zone
```

`lagrangian_loop.py --instance hetero_10zone` 是已冻结的 10 区流程。`canonical_10zone` 目前只从 `main.py` 跑列生成。福田数据包在 `CG_solver/futian`，由 `python main.py --instance futian` 读取。
