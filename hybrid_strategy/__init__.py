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
from .managers import RegimeDetector, HMMRegimeDetector, PositionManager, ExitManager
from .strategy import OptimizedHybrid4ModeV2
from .meta_labeling import TripleBarrierConfig, TripleBarrierLabeler, ExitEventMapper, LogisticMetaModel, MetaLabelingFilter, TradeMetaRecorder
from .advanced_models import (
    AlmgrenChrissConfig,
    AlmgrenChrissSlippageModel,
    RealizedVolatilityEstimator,
    DeflatedSharpeRatio,
    RiskParityAllocator,
)
from .signals import OnlineSignalCalculator
from .hmm_regime import GaussianHMM, RegimeDetectorHMM
from .meta_label import MetaLabeler, AdaptiveMetaThreshold
from .risk_manager import RealisticSlippageModel, VolatilityTargeting, KellyPositionSizer, RiskParity, DrawdownController
from .statistical_tests import PerformanceTests
from .main_strategy_v2 import ProductionStrategy
from .backtest import plot_mode_report, run_backtest, batch_backtest, CONFIG_LOADED
from .walk_forward import walk_forward_validation

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
    "HMMRegimeDetector",
    "PositionManager",
    "ExitManager",
    "OptimizedHybrid4ModeV2",
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "ExitEventMapper",
    "LogisticMetaModel",
    "MetaLabelingFilter",
    "TradeMetaRecorder",
    "AlmgrenChrissConfig",
    "AlmgrenChrissSlippageModel",
    "RealizedVolatilityEstimator",
    "DeflatedSharpeRatio",
    "RiskParityAllocator",
    "OnlineSignalCalculator",
    "GaussianHMM",
    "RegimeDetectorHMM",
    "MetaLabeler",
    "AdaptiveMetaThreshold",
    "RealisticSlippageModel",
    "VolatilityTargeting",
    "KellyPositionSizer",
    "RiskParity",
    "DrawdownController",
    "PerformanceTests",
    "ProductionStrategy",
    "plot_mode_report",
    "run_backtest",
    "batch_backtest",
    "CONFIG_LOADED",
    "walk_forward_validation",
]
