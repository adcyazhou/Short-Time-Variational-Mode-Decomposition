# 5m/Long 多尺度排列熵噪声模态分析

## 文件与数据概况

- 输入文件：`vmd_all_modes.xlsx`（Excel OOXML表格）。
- 工作表：`5m_Long`。
- 样本数：33493；列数：11；缺失值：0。
- 采样频率：4096 Hz；时间范围：-0.5 至 7.67676 s。
- 振速单位：mm/s；VMD模态数：8。

## 方法

- 依据：*Applied Sciences* 2023, 13, 3322，第3.2、3.3和4.1节。
- MPE参数：嵌入维数 m=6，延迟 tau=1，尺度1-5。
- 模态MPE：尺度1-5归一化排列熵的算术平均。
- 噪声判据：平均MPE > 0.6。
- 判为噪声并删除的模态：[8]。
- 去噪信号：所有非噪声模态求和。
- 效果指标：SNR与RMSE分别按论文公式(19)、(20)计算。

## 各模态计算结果

| mode | column | initial_center_hz | final_center_hz | pe_scale_1 | pe_scale_2 | pe_scale_3 | pe_scale_4 | pe_scale_5 | mean_mpe | mpe_threshold | is_noise | decision | mode_energy | mode_energy_percent | mode_rms | mode_peak_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mode_1 | 15.1 | 14.5782 | 0.409278 | 0.481797 | 0.494095 | 0.479018 | 0.453114 | 0.46346 | 0.6 | False | retained | 29594.8 | 73.4208 | 0.940006 | 15.3336 |
| 2 | mode_2 | 36.2 | 34.8958 | 0.306322 | 0.366518 | 0.396501 | 0.412235 | 0.425475 | 0.38141 | 0.6 | False | retained | 8431.97 | 20.9186 | 0.50175 | 8.49371 |
| 3 | mode_3 | 61.5 | 61.3027 | 0.274283 | 0.359518 | 0.415217 | 0.45667 | 0.488535 | 0.398845 | 0.6 | False | retained | 1520.72 | 3.7727 | 0.213082 | 3.63565 |
| 4 | mode_4 | 111.5 | 101.563 | 0.285253 | 0.392916 | 0.473612 | 0.525447 | 0.554782 | 0.446402 | 0.6 | False | retained | 151.786 | 0.376561 | 0.0673192 | 1.06866 |
| 5 | mode_5 | 139.3 | 136.583 | 0.314562 | 0.441623 | 0.51998 | 0.556529 | 0.588363 | 0.484212 | 0.6 | False | retained | 187.162 | 0.464324 | 0.0747535 | 1.61167 |
| 6 | mode_6 | 146.2 | 166.847 | 0.334795 | 0.46693 | 0.532675 | 0.578231 | 0.654857 | 0.513497 | 0.6 | False | retained | 154.494 | 0.38328 | 0.0679171 | 1.31158 |
| 7 | mode_7 | 208.8 | 218.017 | 0.368741 | 0.499045 | 0.562799 | 0.660203 | 0.701243 | 0.558406 | 0.6 | False | retained | 123.395 | 0.306125 | 0.0606975 | 1.32598 |
| 8 | mode_8 | 244.6 | 266.827 | 0.408913 | 0.536688 | 0.665597 | 0.731391 | 0.762681 | 0.621054 | 0.6 | True | noise | 144.174 | 0.357677 | 0.0656096 | 1.63315 |

## 去噪效果数据

| metric | value | unit | interpretation |
| --- | --- | --- | --- |
| SNR | 23.8112 | dB | larger is better (paper Eq. 19) |
| RMSE | 0.0760889 | mm/s | smaller is better (paper Eq. 20) |
| Original peak absolute velocity | 22.21 | mm/s | reference |
| Denoised peak absolute velocity | 21.8155 | mm/s | small change preferred |
| Peak change | -1.77642 | % | absolute change near zero preferred |
| Original squared-amplitude energy | 46635.7 | (mm/s)^2 | reference |
| Denoised squared-amplitude energy | 45593.1 | (mm/s)^2 | high retention preferred |
| Energy retained | 97.7644 | % | high retention preferred |
| Removed component energy | 144.174 | (mm/s)^2 | reported for transparency |
| Correlation original-denoised | 0.997962 | - | closer to 1 indicates waveform preservation |
| Original global dominant frequency | 1.22294 | Hz | supplementary FFT metric, not paper AOK |
| Denoised global dominant frequency | 1.22294 | Hz | supplementary FFT metric, not paper AOK |

## 主要结论

- 仅Mode 8的平均MPE=0.621054超过0.6，故按论文阈值将其判为噪声模态。
- 删除Mode 8后，SNR=23.811 dB，RMSE=0.076089 mm/s。
- 峰值变化=-1.776%，能量保留=97.764%，原始与去噪波形相关系数=0.997962。
- 按论文“较大SNR、较小RMSE、峰值和主频变化较小”的方向性标准，结果显示波形保真度较高；但没有干净真值，不能把这些指标解释为真实噪声误差。

## 建议与限制

- 论文对实测信号的SNR和RMSE本质上比较去噪结果与含噪原始测量值，主要反映保真度，而非真实去噪误差。
- 本报告的全局主频采用Hann窗FFT，只作为补充检查；论文主频保持性采用AOK时频分析，二者不可完全等同。
- Mode 8最终中心频率约266.83 Hz，略高于论文现场案例中主要能量0-250 Hz的范围；这一事实支持噪声判断，但正式判别仍以平均MPE阈值0.6为准。
- 建议在其他炮次或通道上复核阈值稳定性，并保存人工检查结果，避免把短时高频有效冲击误删。
