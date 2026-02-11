# -*- coding: utf-8 -*-
"""回测统计显著性检验工具。"""

from __future__ import annotations

import numpy as np
from scipy import stats


class PerformanceTests:
    @staticmethod
    def sharpe_ratio_test(returns: np.ndarray, benchmark: float = 0, confidence: float = 0.95) -> dict:
        n = len(returns)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        se = np.sqrt((1 + 0.5 * sharpe ** 2 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / n)
        t_stat = (sharpe - benchmark) / se
        p_value = 1 - stats.t.cdf(t_stat, df=n - 1)

        t_crit = stats.t.ppf(confidence, df=n - 1)
        ci_lower = sharpe - t_crit * se
        ci_upper = sharpe + t_crit * se

        return {
            "sharpe": float(sharpe),
            "se": float(se),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "is_significant": bool(p_value < (1 - confidence)),
        }

    @staticmethod
    def deflated_sharpe(observed_sharpe: float, n_returns: int, n_trials: int, skew: float = 0, kurt: float = 3) -> dict:
        expected_max_sr = np.sqrt(2 * np.log(n_trials))
        sr_var = (1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe ** 2) / (n_returns - 1)
        sr_std = np.sqrt(max(sr_var, 1e-9))
        dsr = (observed_sharpe - expected_max_sr * sr_std) / sr_std
        p_value = 1 - stats.norm.cdf(dsr)

        return {
            "observed_sharpe": float(observed_sharpe),
            "deflated_sharpe": float(dsr),
            "expected_max_sr": float(expected_max_sr),
            "p_value": float(p_value),
            "is_significant": bool(p_value < 0.05),
        }

    @staticmethod
    def drawdown_test(equity_curve: np.ndarray, threshold: float = -0.20) -> dict:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak

        max_dd = drawdown.min()
        mean_dd = drawdown[drawdown < 0].mean() if np.any(drawdown < 0) else 0.0

        in_drawdown = drawdown < -0.01
        dd_duration = []
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            elif current_duration > 0:
                dd_duration.append(current_duration)
                current_duration = 0

        avg_duration = np.mean(dd_duration) if dd_duration else 0
        max_duration = max(dd_duration) if dd_duration else 0

        recovery_time = None
        if max_dd < threshold:
            dd_start = np.argmin(drawdown)
            recovery_idx = np.where(drawdown[dd_start:] >= -0.01)[0]
            if len(recovery_idx) > 0:
                recovery_time = int(recovery_idx[0])

        return {
            "max_drawdown": float(max_dd),
            "mean_drawdown": float(mean_dd),
            "avg_duration": float(avg_duration),
            "max_duration": int(max_duration),
            "recovery_time": recovery_time,
            "exceeds_threshold": bool(max_dd < threshold),
        }

    @staticmethod
    def monte_carlo_simulation(returns: np.ndarray, n_simulations: int = 1000, confidence: float = 0.95) -> dict:
        n_periods = len(returns)
        simulated_sharpes = []
        simulated_max_dds = []

        for _ in range(n_simulations):
            sim_returns = np.random.choice(returns, size=n_periods, replace=True)
            sharpe = sim_returns.mean() / sim_returns.std() * np.sqrt(252)
            equity = np.cumprod(1 + sim_returns)
            peak = np.maximum.accumulate(equity)
            dd = ((equity - peak) / peak).min()
            simulated_sharpes.append(sharpe)
            simulated_max_dds.append(dd)

        alpha = 1 - confidence
        sharpe_ci = np.percentile(simulated_sharpes, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        dd_ci = np.percentile(simulated_max_dds, [alpha / 2 * 100, (1 - alpha / 2) * 100])

        return {
            "sharpe_mean": float(np.mean(simulated_sharpes)),
            "sharpe_std": float(np.std(simulated_sharpes)),
            "sharpe_ci": sharpe_ci,
            "max_dd_mean": float(np.mean(simulated_max_dds)),
            "max_dd_std": float(np.std(simulated_max_dds)),
            "max_dd_ci": dd_ci,
        }
