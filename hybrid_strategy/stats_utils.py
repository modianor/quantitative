# -*- coding: utf-8 -*-
"""统计检验工具。"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def sharpe_significance_test(returns: Iterable[float], benchmark_sharpe: float = 0.0) -> Tuple[float, float]:
    """使用近似 Jobson-Korkie 标准误检验 Sharpe 是否显著高于基准。

    Args:
        returns: 日收益率序列（小数形式）。
        benchmark_sharpe: 对比基准 Sharpe（默认 0）。

    Returns:
        tuple[float, float]: (annual_sharpe, 双侧p值)
    """
    arr = np.asarray(list(returns), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0, 1.0

    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0, 1.0

    mean = float(np.mean(arr))
    sr = mean / std * math.sqrt(252.0)

    n = float(arr.size)
    se = math.sqrt(max((1.0 + 0.5 * sr * sr) / n, 1e-12))
    z = (sr - float(benchmark_sharpe)) / se
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return sr, float(min(max(p_value, 0.0), 1.0))

