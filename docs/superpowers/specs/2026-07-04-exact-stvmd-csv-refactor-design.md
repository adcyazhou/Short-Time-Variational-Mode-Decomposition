# 原仓库 STVMD 与 CSV 数据流重构设计

## 目标

重构 `blast_multichannel_stvmd.ipynb`，不再使用自定义分批动态 STVMD 内核。Notebook 改为：

1. 使用 pandas 将 `5m.TXT`、`10m.TXT`、`15m.TXT` 的 ASCII 数值段转换为 CSV；
2. 从 CSV 读取速度数据并截取三个测点的共同有效长度；
3. 原样使用 `main_STVMD.ipynb` 中的 Numba 辅助函数和 `STVMD` 类；
4. 使用原实现中的 `tqdm` 进度条执行 Tran、Vert、Long 三次动态多通道 STVMD；
5. 保留当前已确认的模态、瞬时中心频率、重构和频谱—IF对应图。

## 方案选择

采用“构建时提取原始Notebook源码”的方式。构建器读取 `main_STVMD.ipynb`：

- 单元0：NumPy、Numba、SciPy、Matplotlib、tqdm导入；
- 单元1：`_buffer`、`buffer`、`_unbuffer`、`unbuffer`、`_window_norm`、`window_norm`；
- 单元3：完整 `STVMD` 类。

这三段源码不手工改写。生成后的分析Notebook仍是自包含文件；运行时不依赖构建器或原Notebook。

未采用的方案：

- 手工复制并维护STVMD类：容易与原仓库产生无意差异；
- 从外部Python模块导入：会使交付的Notebook不再自包含。

测试将逐字符比较生成Notebook中的上述源码与 `main_STVMD.ipynb` 对应单元，防止重构过程中修改原算法。

## CSV 数据格式

每个ASCII文件生成一个CSV：

- `data_csv/5m.csv`
- `data_csv/10m.csv`
- `data_csv/15m.csv`

CSV列固定为：

```text
Sample,Time_s,Tran,Vert,Long
```

其中：

- `Sample` 从0开始；
- `Time_s = Sample / fs - pretrigger_seconds`；
- `Tran`、`Vert`、`Long` 保持原始 `mm/s` 数值；
- CSV不包含仪器文本头。

Notebook仍从TXT头部读取采样率、预触发长度和事件时间，用于构造时间列并验证三个文件。数值表使用 `pandas.read_csv(..., sep=r"\s+")` 解析，CSV使用 `DataFrame.to_csv(index=False)` 写出。STVMD输入随后使用 `pandas.read_csv` 从CSV加载。

`data_csv/`为派生数据目录，不提交到Git，但Notebook运行转换单元后会实际生成文件。

## 数据截取

三个测点均从各自CSV的第0行开始，截取到共同最短长度14336点。每个方向组成：

```python
Tran = np.vstack([csv_5m.Tran, csv_10m.Tran, csv_15m.Tran])
Vert = np.vstack([csv_5m.Vert, csv_10m.Vert, csv_15m.Vert])
Long = np.vstack([csv_5m.Long, csv_10m.Long, csv_15m.Long])
```

不执行前部补零、尾部补零、峰值对齐、微分、滤波或重采样。

## 原始 STVMD 调用

每个方向使用原类的标准流程：

```python
window_func = scipy.signal.windows.hamming(WINDOW_LENGTH, sym=False)
stvmd = STVMD(
    num_channel=3,
    window_func=window_func,
    alpha=ALPHA,
    n_fft=WINDOW_LENGTH,
    K=K,
    tol=TOL,
    tau=TAU,
    maxiters=MAX_ITERS,
)
f_hat_s, windowed_signal = stvmd.prepare_offline(direction_signal)
u_hat, center_frequency = stvmd.apply(f_hat_s, dynamic=True)
modes = stvmd.postprocess(u_hat)
```

步长由原类固定为 `self.hop_len = 1`。动态迭代使用原代码中的 `tqdm`。Numba只加速原文件中已经用 `@jit(nopython=True, cache=True)` 标记的缓冲、反缓冲和窗归一化函数；不对 `STVMD.apply_dynamic` 追加新的Numba装饰器。

参数继续集中在配置单元：

- `K`：默认4，可调3–5；
- `ALPHA`：默认50；
- `WINDOW_LENGTH`：仓库候选值8、16、32、64、128、256，默认64；
- `TOL`：默认 `1e-9`；
- `TAU`：默认 `1e-5`；
- `MAX_ITERS`：默认2000。

## 运行成本

完全复制原算法意味着不再按时窗分批。原实现会同时分配全部时窗的模态频谱、拉格朗日变量和所有迭代的中心频率数组。对14336点数据，内存和运行时间会明显高于当前分批实现。

Notebook在执行前显示基于 `C、F、K、T、MAX_ITERS` 的粗略内存估算，并明确说明：

- 调小 `MAX_ITERS` 可减少中心频率历史数组内存；
- 增大时窗会增加频率维度；
- 完整三方向分析可能耗时较长；
- 自动测试只使用短合成信号和真实数据片段，不冒充完整运行结果。

这些提示位于算法外部，不修改原始STVMD源码。

## 结果适配与图形

原始类返回：

- `modes`：`(K, C, N)`；
- `u_hat`：`(C, F, K, T)`；
- `center_frequency`：`(K, T)`，为归一化中心频率。

外部适配层将中心频率换算为Hz：

```python
center_freq_hz = center_frequency * (fs / 2)
```

平均时频功率直接由 `f_hat_s` 计算：

```python
mean_tf_power = np.mean(np.abs(f_hat_s) ** 2, axis=0)
```

后续重构误差、能量占比、5%–95%频带及四类图形继续使用现有定义。适配层可以重构，但不得修改原始辅助函数或 `STVMD` 类。

## 验证

重构采用测试优先：

1. 测试pandas转换实际生成三份CSV，列名、行数、时间列和速度值正确；
2. 测试生成Notebook中的原始导入、Numba辅助函数和 `STVMD` 类与源Notebook逐字符一致；
3. 测试原类包含 `hop_len = 1`、Numba装饰器和 `tqdm`调用；
4. 用短三通道合成信号运行原始动态STVMD，验证输出形状和有限数值；
5. 测试三个方向从CSV按5m、10m、15m顺序组装并截取共同长度；
6. 快速执行完整Notebook，验证CSV转换、三方向调用、绘图和可选保存单元；
7. 重复构建Notebook，确认文件内容稳定。

完整14336点三方向分解不作为自动测试，因为原算法的时间和内存成本较高。
