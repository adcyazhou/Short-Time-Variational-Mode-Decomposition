"""Build the self-contained blast multichannel STVMD notebook."""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "blast_multichannel_stvmd.ipynb"


def md(source):
    return new_markdown_cell(source)


def core(source):
    return new_code_cell(source, metadata={"tags": ["core"]})


IMPORTS = r'''
from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import warnings

import numpy as np
import scipy.fft
import scipy.signal
import matplotlib.pyplot as plt
'''.strip()


LOADER = r'''
@dataclass(frozen=True)
class BlastRecord:
    path: Path
    metadata: dict
    fs: float
    pretrigger_seconds: float
    columns: tuple
    data: np.ndarray


def _metadata_number(metadata, key):
    if key not in metadata:
        raise ValueError(f"缺少元数据字段: {key}")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", metadata[key])
    if match is None:
        raise ValueError(f"无法解析元数据字段 {key}: {metadata[key]!r}")
    return float(match.group())


def load_instantel_txt(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入文件: {path}")
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    metadata = {}
    header_index = None
    for index, raw in enumerate(lines):
        stripped = raw.strip().strip('"')
        if all(name in stripped for name in ("Tran", "Vert", "Long")):
            header_index = index
            break
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()
    if header_index is None:
        raise ValueError(f"{path.name}: 未找到 Tran/Vert/Long 数据表头")
    data = np.loadtxt(lines[header_index + 1 :], dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] != 3:
        raise ValueError(f"{path.name}: 期望3列数据，实际为{data.shape[1]}列")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: 数据包含 NaN 或无穷值")
    fs = _metadata_number(metadata, "Sample Rate")
    pretrigger = abs(_metadata_number(metadata, "Pre-trigger Length"))
    if fs <= 0:
        raise ValueError(f"{path.name}: 采样率必须为正数")
    return BlastRecord(
        path=path,
        metadata=metadata,
        fs=fs,
        pretrigger_seconds=pretrigger,
        columns=("Tran", "Vert", "Long"),
        data=data,
    )


def prepare_direction_inputs(records):
    order = ("5m", "10m", "15m")
    missing = [key for key in order if key not in records]
    if missing:
        raise ValueError(f"缺少测点记录: {missing}")
    fs_values = np.array([records[key].fs for key in order], dtype=float)
    pre_values = np.array(
        [records[key].pretrigger_seconds for key in order], dtype=float
    )
    if not np.allclose(fs_values, fs_values[0]):
        raise ValueError(f"采样率不一致: {fs_values.tolist()}")
    if not np.allclose(pre_values, pre_values[0]):
        raise ValueError(f"预触发长度不一致: {pre_values.tolist()}")
    common_n = min(records[key].data.shape[0] for key in order)
    signals = {}
    for column_index, direction in enumerate(("Tran", "Vert", "Long")):
        signals[direction] = np.vstack(
            [records[key].data[:common_n, column_index] for key in order]
        )
    time_s = np.arange(common_n) / fs_values[0] - pre_values[0]
    return signals, time_s
'''.strip()


def build():
    cells = [
        md(
            "# 单孔漏斗爆破：多通道动态 STVMD 分析\n\n"
            "本 Notebook 以仓库算法及其对应论文为主，按步长1对三个方向分别分析。"
        ),
        md("## 1. 参数配置"),
        core(IMPORTS),
        md("## 2. 数据读取与校验"),
        core(LOADER),
        md("## 3. 动态多通道 STVMD"),
        md("## 4. Tran 方向"),
        md("## 5. Vert 方向"),
        md("## 6. Long 方向"),
        md("## 7. 结果保存"),
    ]
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbformat.write(notebook, TARGET)
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    build()
