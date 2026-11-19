# 4-hub 参数探针（目标：移动补残余、利润≥5%、2h 内 gap≤1%）

## 快筛结论（样本盒代理，非完整 DRO）
- 利润 lift≥5% 时，解几乎总是把 4×30 车队买满。
- 车队落到内部后，lift 通常掉到约 2%。
- 临界附近较接近目标的一点：`cf=210, cm=170, hub=240` → 约 m=90（3 个 hub×30）、lift≈4.9%。

## DRO 探针
脚本 `run_dro_probes.sh`：每个组合先 fixed-only（默认 20min）再 centrally（默认 40min，gap 目标 1%）。
