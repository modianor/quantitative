# -*- coding: utf-8 -*-
"""参数优化模块（贝叶斯优化 + Walk-forward）。"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args


class BayesianOptimizer:
    def __init__(self):
        self.param_space = [
            Real(0.01, 0.05, name="stop_loss_pct"),
            Real(0.02, 0.08, name="profit_take_pct"),
            Real(1.5, 4.0, name="chand_atr_mult"),
            Real(0.8, 1.8, name="vol_ratio_min"),
            Real(-0.30, -0.10, name="dd_drawdown_th"),
            Real(0.10, 0.30, name="target_vol"),
            Integer(10, 40, name="lookback"),
        ]
        self.best_params = None
        self.best_score = -np.inf

    def objective_function(self, params_values: list, backtest_func: Callable, data: pd.DataFrame) -> float:
        params = {
            "stop_loss_pct": params_values[0] * 100,
            "profit_take_pct": params_values[1] * 100,
            "chand_atr_mult": params_values[2],
            "vol_ratio_min": params_values[3],
            "dd_drawdown_th": params_values[4],
            "target_vol": params_values[5],
            "lookback": params_values[6],
        }
        try:
            result = backtest_func(data, params)
            sharpe = result.get("sharpe", 0)
            returns = result.get("return", 0)
            max_dd = result.get("max_dd", 100)
            trades = result.get("trades", 0)
            trade_penalty = (5 - trades) * 5 if trades < 5 else 0
            score = returns - max_dd + sharpe * 10 - trade_penalty
            return -score
        except Exception:
            return 1e6

    def optimize(self, backtest_func: Callable, train_data: pd.DataFrame, n_calls: int = 50) -> Dict:
        @use_named_args(self.param_space)
        def objective(**params_dict):
            params_list = [params_dict[p.name] for p in self.param_space]
            return self.objective_function(params_list, backtest_func, train_data)

        result = gp_minimize(objective, self.param_space, n_calls=n_calls, random_state=42, verbose=False, n_jobs=-1)
        self.best_params = {
            "stop_loss_pct": result.x[0] * 100,
            "profit_take_pct": result.x[1] * 100,
            "chand_atr_mult": result.x[2],
            "vol_ratio_min": result.x[3],
            "dd_drawdown_th": result.x[4],
            "target_vol": result.x[5],
            "lookback": int(result.x[6]),
        }
        self.best_score = -result.fun
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_iterations": len(result.func_vals),
            "convergence": result.func_vals,
        }


class WalkForwardAnalysis:
    def __init__(self, train_period: int = 252 * 2, test_period: int = 252, step: int = 126):
        self.train_period = train_period
        self.test_period = test_period
        self.step = step

    def run(self, data: pd.DataFrame, backtest_func: Callable, optimizer: BayesianOptimizer) -> pd.DataFrame:
        results = []
        total_bars = len(data)
        n_windows = (total_bars - self.train_period - self.test_period) // self.step + 1

        for i in range(n_windows):
            train_start = i * self.step
            train_end = train_start + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period
            if test_end > total_bars:
                break

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            opt_result = optimizer.optimize(backtest_func=backtest_func, train_data=train_data, n_calls=30)
            test_result = backtest_func(test_data, opt_result["best_params"])

            results.append(
                {
                    "window": i + 1,
                    "train_start": train_data.index[0],
                    "train_end": train_data.index[-1],
                    "test_start": test_data.index[0],
                    "test_end": test_data.index[-1],
                    "train_score": opt_result["best_score"],
                    "test_return": test_result["return"],
                    "test_sharpe": test_result["sharpe"],
                    "test_max_dd": test_result["max_dd"],
                    "test_trades": test_result["trades"],
                    **opt_result["best_params"],
                }
            )

        return pd.DataFrame(results)
