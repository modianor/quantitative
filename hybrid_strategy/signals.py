# -*- coding: utf-8 -*-
"""在线信号计算模块（逐 bar 计算，避免前视偏差）。"""

from __future__ import annotations

import numpy as np


class OnlineSignalCalculator:
    """逐 bar 计算趋势/动量/回撤健康度评分。"""

    def __init__(self, strategy):
        self.strat = strategy

    def calculate_trend_score(self) -> int:
        d = self.strat.data
        if len(self.strat) < 200:
            return 0

        score = 0
        ema20 = float(self.strat.ema20[0])
        ema50 = float(self.strat.ema50[0])
        ema200 = float(self.strat.ema200[0])

        if ema20 > ema50:
            score += 1
        if ema50 > ema200:
            score += 1
        if float(d.close[0]) > ema20:
            score += 1

        if len(self.strat.ema20) >= 10:
            slope = (ema20 / float(self.strat.ema20[-10]) - 1) * 100
            if slope > 0:
                score += 1

        if len(self.strat) >= 55:
            highest = max(float(d.close[-i]) for i in range(1, 56))
            if float(d.close[0]) >= highest:
                score += 1

        if len(self.strat) >= 2:
            today_below = float(d.close[0]) < ema20
            yesterday_below = float(d.close[-1]) < float(self.strat.ema20[-1])
            if not (today_below and yesterday_below):
                score += 1

        return score

    def calculate_momentum_score(self) -> int:
        d = self.strat.data
        if len(self.strat) < 30:
            return 0

        score = 0
        close = float(d.close[0])

        if len(self.strat) >= 20:
            high20 = max(float(d.close[-i]) for i in range(1, 21))
            if close > high20:
                score += 1

        if len(self.strat) >= 30:
            ret_10d = close / float(d.close[-10]) - 1
            ret_30d = close / float(d.close[-30]) - 1
            if ret_10d > ret_30d:
                score += 1

        day_range = float(d.high[0]) - float(d.low[0])
        if day_range > 1e-9:
            clv = (close - float(d.low[0])) / day_range
            if clv > 0.7:
                score += 1

        if len(self.strat) >= 20:
            vol_ma = np.mean([float(d.volume[-i]) for i in range(20)])
            if float(d.volume[0]) > vol_ma * 1.2:
                score += 1

        return score

    def calculate_pullback_score(self) -> int:
        d = self.strat.data
        if len(self.strat) < 60:
            return 0

        score = 0
        close = float(d.close[0])
        ema20 = float(self.strat.ema20[0])

        high60 = max(float(d.close[-i]) for i in range(60))
        drawdown = close / high60 - 1
        if drawdown > -0.08:
            score += 1

        days_below_ema = 0
        for i in range(min(10, len(self.strat))):
            if float(d.close[-i]) <= float(self.strat.ema20[-i]):
                days_below_ema += 1
            else:
                break
        if days_below_ema <= 5:
            score += 1

        if len(self.strat) >= 20:
            vol_ma = np.mean([float(d.volume[-i]) for i in range(20)])
            if float(d.volume[0]) < vol_ma:
                score += 1

        if len(self.strat) >= 30:
            low_now = min(float(d.low[-i]) for i in range(20))
            low_prev = min(float(d.low[-i]) for i in range(20, 30))
            if low_now > low_prev:
                score += 1

        return score

    def is_main_uptrend(
        self,
        trend_threshold: int = 4,
        momentum_threshold: int = 2,
        pullback_threshold: int = 2,
    ) -> bool:
        trend = self.calculate_trend_score()
        momentum = self.calculate_momentum_score()
        pullback = self.calculate_pullback_score()

        return (
            trend >= trend_threshold
            and momentum >= momentum_threshold
            and pullback >= pullback_threshold
        )
