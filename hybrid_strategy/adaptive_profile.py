# -*- coding: utf-8 -*-
"""股票结构画像与参数自适应。"""

from __future__ import annotations

import math


class StockProfileLearner:
    """根据近端行情学习股票结构类型与阶段，并输出参数偏移。"""

    def __init__(self, strategy):
        self.strat = strategy
        self.p = strategy.params
        self.archetype = "UNKNOWN"
        self.stage = "UNKNOWN"
        self.confidence = 0.0
        self.last_update_len = -1

    def update(self) -> None:
        if not bool(getattr(self.p, "adaptive_profile_enabled", True)):
            return

        current_len = len(self.strat)
        if current_len == self.last_update_len:
            return

        self.last_update_len = current_len
        lookback = int(max(getattr(self.p, "adaptive_profile_lookback", 80), 20))
        if current_len <= lookback + 2:
            return

        d = self.strat.datas[0]
        closes = [float(d.close[-i]) for i in range(lookback, -1, -1)]
        highs = [float(d.high[-i]) for i in range(lookback, -1, -1)]
        lows = [float(d.low[-i]) for i in range(lookback, -1, -1)]
        vols = [float(d.volume[-i]) for i in range(lookback, -1, -1)]
        if min(closes) <= 0:
            return

        returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
        net = closes[-1] / closes[0] - 1.0
        annual_vol = self._annualized_vol(returns)

        ema20 = float(self.strat.ema20[0])
        ema50 = float(self.strat.ema50[0])
        ema200 = float(self.strat.ema200[0])
        slope20 = self._slope([float(self.strat.ema20[-i]) for i in range(min(20, len(self.strat.ema20)-1), -1, -1)])

        flip_ratio = self._flip_ratio(returns)
        true_range_pct = (max(highs) - min(lows)) / max(closes[-1], 1e-9)
        avg_vol = sum(vols) / max(len(vols), 1)
        recent_vol = sum(vols[-10:]) / max(min(len(vols), 10), 1)
        volume_ratio = recent_vol / max(avg_vol, 1e-9)

        trend_like = net > 0.08 and ema20 > ema50 > ema200 and slope20 > 0 and flip_ratio < 0.42
        range_like = abs(net) < 0.05 and flip_ratio > 0.52 and true_range_pct < 0.22
        markdown_like = net < -0.10 and ema20 < ema50 and slope20 < 0
        volatile_like = annual_vol >= float(getattr(self.p, "adaptive_high_vol_threshold", 0.45))

        if trend_like and not volatile_like:
            self.archetype = "TREND_LEADER"
        elif markdown_like and volume_ratio > 1.05:
            self.archetype = "DISTRIBUTION"
        elif range_like:
            self.archetype = "RANGE_BOUND"
        elif volatile_like:
            self.archetype = "HIGH_BETA"
        else:
            self.archetype = "BALANCED"

        self.stage = self._infer_stage(net, slope20, flip_ratio, volume_ratio, ema20, ema50)
        self.confidence = self._compute_confidence(net, flip_ratio, annual_vol)

    def _infer_stage(
        self,
        net: float,
        slope20: float,
        flip_ratio: float,
        volume_ratio: float,
        ema20: float,
        ema50: float,
    ) -> str:
        if net < -0.10 and slope20 < 0:
            return "MARKDOWN"
        if abs(net) < 0.04 and flip_ratio > 0.50 and volume_ratio < 1.1:
            return "BASE"
        if net > 0.08 and slope20 > 0 and ema20 > ema50:
            return "MARKUP"
        if flip_ratio > 0.55 and abs(net) <= 0.08:
            return "SIDEWAYS"
        if net > 0.05 and slope20 <= 0 and volume_ratio > 1.1:
            return "DISTRIBUTION"
        return "TRANSITION"

    @staticmethod
    def _annualized_vol(returns) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
        return math.sqrt(max(var, 0.0)) * math.sqrt(252.0)

    @staticmethod
    def _flip_ratio(returns) -> float:
        flips = 0
        pairs = 0
        for i in range(1, len(returns)):
            p = returns[i - 1]
            c = returns[i]
            if abs(p) < 1e-8 or abs(c) < 1e-8:
                continue
            pairs += 1
            if (p > 0 > c) or (p < 0 < c):
                flips += 1
        return flips / max(pairs, 1)

    @staticmethod
    def _slope(series) -> float:
        n = len(series)
        if n < 3:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(series) / n
        num = sum((i - x_mean) * (series[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if den <= 0:
            return 0.0
        return (num / den) / max(abs(y_mean), 1e-9)

    def _compute_confidence(self, net: float, flip_ratio: float, annual_vol: float) -> float:
        net_score = min(abs(net) / 0.15, 1.0)
        stability = 1.0 - min(abs(flip_ratio - 0.5) / 0.5, 1.0)
        vol_score = min(annual_vol / max(float(getattr(self.p, "adaptive_high_vol_threshold", 0.45)), 1e-6), 1.0)
        return max(0.15, min(0.95, 0.45 * net_score + 0.35 * stability + 0.20 * vol_score))

    def get_adjustment(self, param_name: str, base_value: float) -> float:
        """返回参数自适应后的值。"""
        if not bool(getattr(self.p, "adaptive_profile_enabled", True)):
            return float(base_value)

        conf = float(self.confidence)
        if conf < float(getattr(self.p, "adaptive_confidence_min", 0.30)):
            return float(base_value)

        value = float(base_value)

        if param_name == "vol_ratio_min":
            if self.archetype in {"RANGE_BOUND", "DISTRIBUTION"}:
                value += 0.15 * conf
            elif self.archetype == "TREND_LEADER":
                value -= 0.08 * conf
        elif param_name == "add_vol_ratio_min":
            if self.stage in {"SIDEWAYS", "DISTRIBUTION"}:
                value += 0.20 * conf
            elif self.stage == "MARKUP":
                value -= 0.05 * conf
        elif param_name == "target_vol_annual":
            if self.archetype == "TREND_LEADER" and self.stage == "MARKUP":
                value *= 1.0 + 0.30 * conf
            elif self.archetype in {"RANGE_BOUND", "DISTRIBUTION"}:
                value *= 1.0 - 0.22 * conf
            elif self.archetype == "HIGH_BETA":
                value *= 1.0 - 0.10 * conf
        elif param_name == "max_exposure":
            if self.stage == "MARKUP":
                value *= 1.0 + 0.20 * conf
            elif self.stage in {"MARKDOWN", "DISTRIBUTION"}:
                value *= 1.0 - 0.25 * conf
        elif param_name == "hmm_min_confidence":
            if self.stage in {"SIDEWAYS", "DISTRIBUTION"}:
                value += 0.08 * conf
            elif self.stage == "MARKUP":
                value -= 0.05 * conf
        elif param_name == "hmm_trend_prob_threshold":
            if self.stage == "MARKUP":
                value -= 0.08 * conf
            elif self.stage in {"SIDEWAYS", "DISTRIBUTION"}:
                value += 0.10 * conf
        elif param_name == "profit_take_pct":
            if self.stage == "MARKUP":
                value *= 1.0 + 0.12 * conf
            elif self.stage in {"SIDEWAYS", "DISTRIBUTION"}:
                value *= 1.0 - 0.18 * conf
        elif param_name == "stop_loss_pct":
            if self.stage == "MARKUP":
                value *= 1.0 + 0.10 * conf
            elif self.stage in {"MARKDOWN", "DISTRIBUTION"}:
                value *= 1.0 - 0.12 * conf

        return self._clip_param(param_name, value)

    @staticmethod
    def _clip_param(param_name: str, value: float) -> float:
        bounds = {
            "vol_ratio_min": (0.6, 2.0),
            "add_vol_ratio_min": (0.5, 1.8),
            "target_vol_annual": (0.10, 0.45),
            "max_exposure": (0.15, 1.0),
            "hmm_min_confidence": (0.25, 0.75),
            "hmm_trend_prob_threshold": (0.45, 0.90),
            "profit_take_pct": (6.0, 60.0),
            "stop_loss_pct": (3.0, 20.0),
        }
        lo, hi = bounds.get(param_name, (-1e12, 1e12))
        return min(max(float(value), lo), hi)
