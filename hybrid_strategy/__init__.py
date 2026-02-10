# -*- coding: utf-8 -*-
"""优化版混合策略包。"""

from .data_utils import (
    PandasWithSignals,
    load_from_yfinance,
    load_from_csv,
    ema,
    atr,
    rolling_slope,
    clv,
    detect_main_uptrend,
)
from .managers import RegimeDetector, PositionManager, ExitManager
from .strategy import OptimizedHybrid4ModeV2
from .backtest import plot_mode_report, run_backtest, batch_backtest, CONFIG_LOADED

__all__ = [
    "PandasWithSignals",
    "load_from_yfinance",
    "load_from_csv",
    "ema",
    "atr",
    "rolling_slope",
    "clv",
    "detect_main_uptrend",
    "RegimeDetector",
    "PositionManager",
    "ExitManager",
    "OptimizedHybrid4ModeV2",
    "plot_mode_report",
    "run_backtest",
    "batch_backtest",
    "CONFIG_LOADED",
]
