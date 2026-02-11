# -*- coding: utf-8 -*-
"""进阶量化组件：交易成本、波动率、过拟合检验、风险平价。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import math

import numpy as np


@dataclass
class AlmgrenChrissConfig:
    """Almgren-Chriss 简化参数。"""

    temporary_impact: float = 0.08
    permanent_impact: float = 0.02
    risk_aversion: float = 1e-6
    min_slippage_bps: float = 2.0


class AlmgrenChrissSlippageModel:
    """使用参与率和波动率估计执行滑点。"""

    def __init__(self, config: AlmgrenChrissConfig | None = None):
        self.config = config or AlmgrenChrissConfig()

    def estimate_slippage_bps(self, participation_rate: float, annual_vol: float, horizon_days: float = 1.0) -> float:
        part = max(float(participation_rate), 0.0)
        sigma = max(float(annual_vol), 1e-6)
        horizon = max(float(horizon_days), 1e-6)

        temp = self.config.temporary_impact * sigma * math.sqrt(part)
        perm = self.config.permanent_impact * sigma * part
        risk_term = self.config.risk_aversion * sigma * math.sqrt(horizon)

        bps = (temp + perm + risk_term) * 10000.0
        return max(float(bps), float(self.config.min_slippage_bps))

    def apply_to_price(self, price: float, side: str, participation_rate: float, annual_vol: float) -> float:
        bps = self.estimate_slippage_bps(participation_rate, annual_vol)
        slip = bps / 10000.0
        px = float(price)
        if side.upper() == "BUY":
            return px * (1.0 + slip)
        return px * (1.0 - slip)


class RealizedVolatilityEstimator:
    """Realized Volatility（支持 close-to-close / Parkinson）。"""

    def __init__(self, lookback: int = 20, annualization: int = 252, method: str = "close"):
        self.lookback = int(max(lookback, 2))
        self.annualization = int(max(annualization, 1))
        self.method = method

    def estimate_from_series(self, closes: Iterable[float], highs: Iterable[float] | None = None,
                             lows: Iterable[float] | None = None) -> float:
        c = np.asarray(list(closes), dtype=float)
        if c.size < 3:
            return 0.0

        c = c[-(self.lookback + 1):]
        if self.method == "parkinson" and highs is not None and lows is not None:
            h = np.asarray(list(highs), dtype=float)[-self.lookback:]
            l = np.asarray(list(lows), dtype=float)[-self.lookback:]
            valid = (h > 0) & (l > 0) & (h >= l)
            if np.sum(valid) >= 2:
                rs = np.log(h[valid] / l[valid]) ** 2
                var = float(np.mean(rs)) / (4.0 * math.log(2.0))
                return math.sqrt(max(var, 0.0)) * math.sqrt(self.annualization)

        ret = np.diff(np.log(np.where(c > 0, c, np.nan)))
        ret = ret[np.isfinite(ret)]
        if ret.size < 2:
            return 0.0
        return float(np.std(ret, ddof=1)) * math.sqrt(self.annualization)


class DeflatedSharpeRatio:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado) 的简化实现。"""

    @staticmethod
    def estimate(sharpe: float, n_returns: int, n_trials: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
        n = max(int(n_returns), 2)
        m = max(int(n_trials), 1)
        sr = float(sharpe)

        # Expected max SR under multiple testing (Gumbel approximation)
        ems = math.sqrt(2.0 * math.log(m))
        ems -= (math.log(math.log(m)) + math.log(4.0 * math.pi)) / (2.0 * max(ems, 1e-9)) if m > 1 else 0.0

        var_sr = (1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2)) / max(n - 1, 1)
        std_sr = math.sqrt(max(var_sr, 1e-9))
        z = (sr - ems * std_sr) / std_sr

        # 近似标准正态CDF
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class RiskParityAllocator:
    """多资产风险平价权重分配。"""

    def __init__(self, min_weight: float = 0.0, max_weight: float = 1.0):
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def inverse_vol_weights(self, vol_by_asset: Dict[str, float]) -> Dict[str, float]:
        safe = {k: max(float(v), 1e-6) for k, v in vol_by_asset.items()}
        if not safe:
            return {}
        inv = {k: 1.0 / v for k, v in safe.items()}
        total = sum(inv.values())
        raw = {k: v / total for k, v in inv.items()}

        clipped = {k: min(max(w, self.min_weight), self.max_weight) for k, w in raw.items()}
        norm = sum(clipped.values())
        if norm <= 0:
            n = len(clipped)
            return {k: 1.0 / n for k in clipped}
        return {k: w / norm for k, w in clipped.items()}


class TransitionAdaptiveHMMMixin:
    """按后验概率动态更新转移矩阵，适配不同市场状态持久性。"""

    def adapt_transition(self, posterior: List[float], learning_rate: float = 0.03) -> List[List[float]]:
        lr = min(max(float(learning_rate), 0.0), 1.0)
        post = np.asarray(posterior, dtype=float)
        if post.size == 0:
            return []

        base = np.asarray(self.transition, dtype=float)
        identity_push = np.eye(base.shape[0])
        mixed_target = (1.0 - post[:, None]) * base + post[:, None] * identity_push
        updated = (1.0 - lr) * base + lr * mixed_target

        # 行归一化
        row_sum = np.sum(updated, axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 1e-9, 1.0, row_sum)
        updated = updated / row_sum

        self.transition = updated.tolist()
        return self.transition
