# -*- coding: utf-8 -*-
"""生产级整合策略示例。"""

from __future__ import annotations

import backtrader as bt
import numpy as np

from .hmm_regime import RegimeDetectorHMM
from .meta_label import AdaptiveMetaThreshold, MetaLabeler
from .risk_manager import DrawdownController, VolatilityTargeting
from .signals import OnlineSignalCalculator


class ProductionStrategy(bt.Strategy):
    params = dict(
        stop_loss_atr_mult=2.0,
        profit_target_mult=3.0,
        vol_target=0.15,
        lookback=20,
        max_drawdown=0.15,
    )

    def __init__(self):
        super().__init__()
        self.atr = bt.ind.ATR(period=14)
        self.ema20 = bt.ind.EMA(period=20)
        self.ema50 = bt.ind.EMA(period=50)
        self.ema200 = bt.ind.EMA(period=200)

        self.signal_calc = OnlineSignalCalculator(self)
        self.regime = RegimeDetectorHMM(self, lookback=120)
        self.meta_labeler = MetaLabeler()
        self.adaptive_threshold = AdaptiveMetaThreshold()
        self.vol_target_sizer = VolatilityTargeting(target_vol=self.p.vol_target)
        self.dd_controller = DrawdownController(reduce_at=0.10, stop_at=self.p.max_drawdown)

        self.entry_features = None
        self.entry_price = None
        self.stop_loss_price = None
        self.profit_target = None
        self.equity_peak = 0.0

    def next(self):
        equity = self.broker.getvalue()
        self.equity_peak = max(self.equity_peak, equity)
        self.regime.update()
        self.dd_controller.update(equity)

        if self.dd_controller.should_stop_trading(equity):
            if self.position:
                self.close()
            return

        mode_id, mode_name = self.regime.get_mode()
        confidence = self.regime.get_confidence()
        if confidence < 0.6:
            return

        if self.position:
            close = float(self.data.close[0])
            if close <= self.stop_loss_price or close >= self.profit_target:
                self._exit_trade(close)
            return

        if mode_name != "TREND_RUN":
            return

        if self.signal_calc.calculate_trend_score() < 4 or self.signal_calc.calculate_momentum_score() < 2:
            return

        features = self.meta_labeler.extract_features(self, mode_id)
        drawdown = 1 - equity / self.equity_peak if self.equity_peak > 0 else 0.0
        volatility = self._estimate_volatility()
        threshold = self.adaptive_threshold.get_threshold(drawdown, volatility, mode_name)
        should_enter, _ = self.meta_labeler.should_take_signal(features, threshold)
        if not should_enter:
            return

        returns = self._get_recent_returns(self.p.lookback)
        target_size = self.vol_target_sizer.calculate_position_size(equity, float(self.data.close[0]), returns)
        final_size = int(target_size * self.dd_controller.get_position_scalar(equity))
        if final_size > 0:
            self._enter_trade(final_size, features)

    def _enter_trade(self, size: int, features: np.ndarray):
        price = float(self.data.close[0])
        atr = float(self.atr[0])
        self.stop_loss_price = price - self.p.stop_loss_atr_mult * atr
        self.profit_target = price + self.p.profit_target_mult * atr
        self.entry_features = features
        self.entry_price = price
        self.buy(size=size)

    def _exit_trade(self, exit_price: float):
        is_profit = self.entry_price is not None and exit_price > self.entry_price
        if self.entry_features is not None:
            self.meta_labeler.register_trade(self.entry_features, bool(is_profit))
        self.close()
        self.entry_features = None
        self.entry_price = None
        self.stop_loss_price = None
        self.profit_target = None

    def _get_recent_returns(self, n: int) -> np.ndarray:
        if len(self) < n + 1:
            return np.array([0.0])
        return np.array([float(self.data.close[-i + 1]) / float(self.data.close[-i]) - 1 for i in range(1, n + 1)])

    def _estimate_volatility(self) -> float:
        returns = self._get_recent_returns(20)
        return float(np.std(returns) * np.sqrt(252))
