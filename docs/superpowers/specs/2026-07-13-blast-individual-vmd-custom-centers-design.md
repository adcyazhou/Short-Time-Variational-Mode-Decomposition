# 爆破三实验九通道自定义中心频率 VMD 设计

## 目标

新建一个独立、自包含的 Jupyter Notebook，读取仓库根目录下的
`5m.TXT`、`10m.TXT` 和 `15m.TXT`，分别对每个文件的 Tran、Vert、Long
通道执行单通道 VMD，共完成九次独立分解。每条信号允许单独指定模态数
`K` 和全部 K 个模态的初始中心频率；VMD 惩罚参数固定为 2000。

初始中心频率只用于第一次迭代的热启动。后续迭代中，每个模态的中心频率
都必须按该模态频谱的能量重心继续更新，不得固定在用户指定值。

## 交付物

- 新 Notebook：`blast_individual_vmd_custom_centers.ipynb`。
- 确定性 Notebook 构建器：
  `tools/build_blast_individual_vmd_custom_centers_notebook.py`。
- 自动测试：
  `tests/test_blast_individual_vmd_custom_centers_notebook.py`。

不修改或混入 `figure_experiment_STVMD_ssvep_singlechannel.ipynb` 的 SSVEP
实验内容。新 Notebook 以其中的 `VMD` 类为算法来源，并明确标注唯一的行为
调整：传入的中心频率由“固定中心”改为“可继续迭代的初始中心”。

## 数据读取

Notebook 直接读取三个 Instantel ASCII 文件。解析流程必须：

1. 从文件头读取采样率、预触发长度和单位；
2. 定位数值表头并提取 Tran、Vert、Long 三列；
3. 保持每个文件自身的完整有效长度，不对 5m、10m、15m 做拼接、对齐或
   共同长度截断；
4. 为每条信号构造以秒为单位的时间轴；
5. 拒绝缺列、空数据、非有限值和无效采样率。

每个文件和方向形成一个独立分析键，例如 `("5m", "Tran")`。

## 参数配置

Notebook 顶部提供单一、醒目的配置单元。九条信号分别配置：

```python
VMD_CONFIG = {
    "5m": {
        "Tran": {"K": None, "centers_hz": []},
        "Vert": {"K": None, "centers_hz": []},
        "Long": {"K": None, "centers_hz": []},
    },
    "10m": {
        "Tran": {"K": None, "centers_hz": []},
        "Vert": {"K": None, "centers_hz": []},
        "Long": {"K": None, "centers_hz": []},
    },
    "15m": {
        "Tran": {"K": None, "centers_hz": []},
        "Vert": {"K": None, "centers_hz": []},
        "Long": {"K": None, "centers_hz": []},
    },
}
```

九组参数默认留空，正式 Notebook 在用户填写前必须明确报错，不得自行选择
参数或执行分析。每个 `centers_hz` 必须明确列出全部 K 个初始中心；程序不
自动添加 0 Hz 残余模态。若需要 0 Hz 初始模态，用户应把 `0.0` 明确写入该
列表并相应计入 K。自动测试可在专用环境变量下使用隔离的测试参数，但不得
改变正式配置单元的留空状态。

全局参数为：

```python
ALPHA = 2000.0
N_FFT = 64
TAU = 1e-5
TOL = 1e-9
MAX_ITERS = 10000
```

`ALPHA` 固定为 2000，不允许九条信号分别覆盖。其余全局数值参数集中在同一
单元，便于必要时统一调整，但不改变本设计的中心频率语义。

## 参数校验

每次分解前必须验证：

- K 是正整数；
- `len(centers_hz) == K`；
- 所有中心频率均为有限实数；
- 中心频率按严格升序排列且互不重复；
- 每个中心满足 `0 <= f < fs / 2`；
- `ALPHA == 2000.0`；
- `N_FFT >= 2`、`TAU >= 0`、`TOL > 0`、`MAX_ITERS >= 2`。

错误信息必须指出具体的距离、方向和无效字段，使用户能直接修改对应配置。

## VMD 算法

算法沿用 `figure_experiment_STVMD_ssvep_singlechannel.ipynb` 中的单通道
`VMD`：反射填充、单边实 FFT、ADMM 模态更新、拉格朗日乘子更新、能量重心
中心更新、收敛判断和逆 FFT 后处理均保持一致。

源实现虽然按奇偶索引只访问当前与下一迭代状态，却按 `MAX_ITERS` 分配完整
历史数组。新 Notebook 将 `u_hat_plus`、`lambda_hat` 和 `omega_plus` 改为两个
循环缓冲区，并用明确的 `current_index`、`next_index` 计算收敛差异和返回最终
状态。该调整不改变更新方程或停止准则，但避免完整爆破记录因迭代历史数组
耗尽内存，并确保返回的是最后一次已完成迭代。

为实现热启动，需要对原类的中心更新语义做一个明确调整。用户输入的 Hz
中心转换为源实现使用的归一化频率：

```python
omega_init = np.asarray(centers_hz, dtype=float) / (fs / 2.0)
```

这是由源 VMD 的内部频率轴决定的：内部值 1.0 对应奈奎斯特频率
`fs / 2`。最终中心换回 Hz 时使用相反变换
`center_hz = omega * (fs / 2.0)`。

它只赋给：

```python
omega_plus[0, :] = omega_init
```

从第一次 ADMM 更新完成后起，包括第 0 个模态在内的全部模态都执行：

```python
omega_plus[next_index, k] = (
    np.sum(freqs * np.sum(np.abs(u_hat_next[:, :, k]) ** 2, axis=0))
    / np.sum(np.abs(u_hat_next[:, :, k]) ** 2)
)
```

因此初始中心只是搜索起点，最终中心由数据决定。若某模态能量分母为零或
非有限值，分解必须以包含数据键和模态编号的异常停止，不能静默产生 NaN。

最终模态按最终中心频率升序排序。对应的初始中心也用相同最终排序索引重排，
以便图标题正确展示每个输出模态的初始值和最终值。

## 批量分析流程

Notebook 按 5m、10m、15m，再按 Tran、Vert、Long 的固定顺序执行九次
单通道 VMD。每次分析：

1. 读取当前完整波形；
2. 读取并验证该波形的 K 和初始中心；
3. 使用固定 `alpha=2000` 运行 VMD；
4. 逆变换得到形状为 `(K, N)` 的时域模态；
5. 计算所有模态之和与原始波形之间的重构 RMSE；
6. 保存初始中心、最终中心、模态、重构信号、RMSE、迭代次数和收敛状态；
7. 生成该波形的模态图。

九条信号互相独立，不共享模态中心或 VMD 状态。

## 图形输出

每条信号生成一张独立图，共九张图。图形包含 `K + 1` 个共享时间轴的子图：

- 第 1 行：原始波形；
- 第 2 至 `K + 1` 行：Mode 1 至 Mode K。

每个模态子图标题或左侧标签必须包含：

```text
Mode k: init=xx.xx Hz, final=yy.yy Hz
```

整图标题包含距离、方向、K 和 `alpha=2000`。横轴为相对时间（s），纵轴
沿用源文件单位（mm/s）。所有子图使用一致的线宽和可读网格，不叠加不同
模态，以便逐个检查波形。

Notebook 在每张图前打印简洁摘要：数据键、样本数、采样率、初始中心、
最终中心、迭代次数、是否收敛和重构 RMSE。

## 测试策略

自动测试覆盖以下行为：

1. 构建器可重复生成结构一致的 Notebook；
2. Notebook 包含九条独立配置且 α 固定为 2000；
3. 参数校验拒绝 K 与中心数量不匹配、未排序、重复、越过奈奎斯特频率等输入；
4. Instantel 解析器正确读取三个方向和采样率；
5. 用两个已知正弦分量的短合成信号运行 VMD，验证输出形状、有限值和低重构
   误差；
6. 使用故意偏离真实频率的初始中心，验证最终中心与初始值不同，从而证明
   中心在迭代而非固定；
7. 验证 0 Hz 只在用户明确配置时出现，不会自动添加；
8. 绘图函数为 K 个模态创建 K+1 个坐标轴，并显示初始与最终中心；
9. 以缩短信号和较小迭代数执行 Notebook 快速路径，验证九条分析均能完成，
   不在测试中运行完整长记录的 10000 次上限。

## 非目标

- 不执行 STVMD、DAMD、Meanshift 或 HVR；
- 不自动估计 K、中心频率或 α；
- 不把三个距离作为多通道联合 VMD；
- 不自动筛选噪声模态、趋势模态或重构“去噪信号”；
- 不覆盖现有 Notebook、图片或用户当前未提交的修改；
- 不把用户指定中心固定在整个迭代过程中。
