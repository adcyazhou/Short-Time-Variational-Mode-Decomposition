# 5m/Long 多方法去噪时频对比图设计

## 目标

将同一条 5m/Long 原始爆破振动信号与三种去噪结果放在统一坐标系中比较：CEEMDAN、VMD-SSA，以及本项目采用平均多尺度排列熵阈值 0.60 得到的 VMD-MPE 重构信号。输出时域叠加图和频域叠加图，并保存用于复核的对齐数据。

## 输入

- 原始信号：`G:/我的云端硬盘/单孔漏斗爆破/5m.TXT` 的 Long 通道；若云盘占位文件当前不可读，则使用工作区内内容相同的 `5m.TXT`。
- CEEMDAN：`C:/Users/admin/Documents/多种适应vmd/output/ceemdan_paper_denoising/results/denoised_signal.csv` 中的 `ceemdan_denoised`。
- VMD-SSA：`D:/vmd_k8_output/results/algorithm_SSA_denoised_signal.csv` 中的 `denoised_SSA_mm_s`。
- VMD-MPE：`output/5m_long_mpe_denoising/5m_Long_denoised_signal.csv` 中阈值 0.60 对应的 `denoised`。

## 数据对齐

四份数据均应包含 33,493 个样本，采样频率为 4096 Hz。以原始 `5m.TXT` 的 Long 通道为基准，通过各 CSV 内保存的原始列逐点校验数据身份。CEEMDAN 和 VMD-SSA 的时间从 0 s 开始，而原始记录含 0.5 s 预触发；绘图统一采用原始记录时间轴，即 -0.5 s 至 7.6767578125 s。任何长度不一致、非有限值或原始列不一致均应终止处理并报告错误，不进行静默截断或插值。

## 时域图

在同一坐标轴叠加四条完整波形：原始信号、CEEMDAN、VMD-SSA、VMD-MPE（MPE阈值0.60）。采用色盲友好的不同颜色和不同线型，横轴为时间（s），纵轴为振动速度（mm/s）。原始曲线使用较细黑线，三条去噪曲线使用不同颜色，以降低遮挡。

## 频域图

对四条未经加窗、未经额外滤波的完整时域序列直接计算实数单边 FFT。采用标准单边幅值缩放：除直流与奈奎斯特分量外，正频率幅值乘以 2/N。频率图只显示 0 至 250 Hz，纵轴为单边幅值（mm/s）。四条频谱使用与时域图一致的颜色和线型。

## 输出

输出目录为 `output/5m_long_four_method_comparison_mpe_0_60/`，包含：

- 时域对比图 PNG 与 PDF；
- 0–250 Hz 频域对比图 PNG 与 PDF；
- 时域与频域组合双面板图 PNG 与 PDF；
- 四条已对齐时域信号 CSV；
- 0–250 Hz 单边频谱 CSV；
- 数据核对与主要幅值统计 CSV；
- 可重复运行的 Python 脚本。

PNG 使用 300 dpi，PDF 保留矢量线条。所有图使用一致的字体、图例、颜色和单位。

## 验证

运行前验证输入存在、样本数量一致、数值有限、采样间隔一致，并验证三份 CSV 中的原始信号与 `5m.TXT` Long 通道逐点相同。运行后验证输出文件非空、频率范围为 0–250 Hz、频率分辨率为 `4096/33493` Hz，并对四条信号的峰值、均值、RMS 和频域峰值进行独立复算。最终以图像检查确认图例、坐标、单位和曲线可辨识。
