# -*- coding: utf-8 -*-
"""风险管理组件（滑点、仓位、回撤、风险平价）。"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import minimize


class RealisticSlippageModel:
    def __init__(self):
        self.base_slippage = 0.001
        self.vol_coef = 5.0
        self.urgency_penalty = {"limit": 0.0005, "market": 0.0015, "stop": 0.0025}

    def estimate(self, price: float, order_type: str, volatility: float, order_size: float, adv: float, side: str = "buy") -> float:
        slippage = self.base_slippage

        if volatility > 0.02:
            slippage += (volatility - 0.02) * self.vol_coef

        slippage += self.urgency_penalty.get(order_type, 0.002)

        if adv > 0:
            participation = order_size / adv
            if participation > 0.01:
                slippage += participation ** 1.5 * 0.01

        direction = 1 if side.lower() == "buy" else -1
        return float(price) * (1 + direction * slippage)


class VolatilityTargeting:
    def __init__(self, target_vol: float = 0.15, vol_floor: float = 0.08, vol_cap: float = 0.60):
        self.target_vol = target_vol
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap

    def estimate_volatility(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return self.target_vol

        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weights /= weights.sum()
        weighted_var = np.average(returns ** 2, weights=weights)
        annual_vol = np.sqrt(weighted_var) * np.sqrt(252)
        return float(np.clip(annual_vol, self.vol_floor, self.vol_cap))

    def calculate_position_size(self, capital: float, price: float, recent_returns: np.ndarray) -> int:
        vol = self.estimate_volatility(recent_returns)
        vol_scalar = np.clip(self.target_vol / max(vol, 1e-9), 0.2, 2.0)
        target_value = capital * vol_scalar * 0.5
        return max(int(target_value / max(price, 1e-9)), 0)


class KellyPositionSizer:
    def __init__(self, fraction: float = 0.25, max_position: float = 0.50):
        self.kelly_fraction = fraction
        self.max_position = max_position

    def calculate(self, win_rate: float, avg_win: float, avg_loss: float, capital: float, price: float) -> int:
        if win_rate <= 0 or avg_loss <= 0:
            return 0

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss
        kelly = np.clip(((p * b - q) / max(b, 1e-9)) * self.kelly_fraction, 0, self.max_position)
        target_value = capital * kelly
        return max(int(target_value / max(price, 1e-9)), 0)


class RiskParity:
    def calculate_weights(self, returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        symbols = list(returns.keys())
        n_assets = len(symbols)
        if n_assets == 0:
            return {}

        ret_matrix = np.column_stack([returns[s] for s in symbols])
        cov_matrix = np.cov(ret_matrix.T) * 252

        def objective(weights):
            weights = np.abs(weights)
            weights /= max(weights.sum(), 1e-9)
            marginal = cov_matrix @ weights
            risk_contrib = weights * marginal
            return np.var(risk_contrib)

        constraints = {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1}
        bounds = [(0, 0.5) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            return {s: 1 / n_assets for s in symbols}

        weights = np.abs(result.x)
        weights /= weights.sum()
        return {s: float(w) for s, w in zip(symbols, weights)}


class DrawdownController:
    def __init__(self, max_drawdown: float = 0.15, reduce_at: float = 0.10, stop_at: float = 0.15):
        self.max_dd = max_drawdown
        self.reduce_threshold = reduce_at
        self.stop_threshold = stop_at
        self.equity_peak = 0.0

    def update(self, equity: float):
        self.equity_peak = max(self.equity_peak, equity)

    def get_drawdown(self, equity: float) -> float:
        if self.equity_peak == 0:
            return 0.0
        return 1 - equity / self.equity_peak

    def get_position_scalar(self, equity: float) -> float:
        dd = self.get_drawdown(equity)
        if dd >= self.stop_threshold:
            return 0.0
        if dd >= self.reduce_threshold:
            range_dd = self.stop_threshold - self.reduce_threshold
            return float(np.clip(1 - (dd - self.reduce_threshold) / max(range_dd, 1e-9), 0, 1))
        return 1.0

    def should_stop_trading(self, equity: float) -> bool:
        return self.get_drawdown(equity) >= self.stop_threshold
