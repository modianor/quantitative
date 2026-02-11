# -*- coding: utf-8 -*-
"""股票结构画像与参数自适应。"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class StockProfileLearner:
    """根据行情学习股票结构类型、阶段，并输出参数自适应值。"""

    def __init__(self, strategy):
        self.strat = strategy
        self.p = strategy.params
        self.archetype = "UNKNOWN"
        self.stage = "UNKNOWN"
        self.confidence = 0.0
        self.last_update_len = -1
        self.metrics: Dict[str, float] = {}
        # 在线学习状态：每个(票型, 阶段, 参数)都维护一组可学习偏移量。
        # value 表示相对基础参数的偏移（mul型表示比例偏移，add型表示绝对偏移）。
        self.learned_state: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self.param_meta: Dict[str, Dict[str, float | str]] = {
            "vol_ratio_min": {"op": "add", "lr": 0.04, "floor": -0.35, "cap": 0.35},
            "add_vol_ratio_min": {"op": "add", "lr": 0.04, "floor": -0.30, "cap": 0.40},
            "target_vol_annual": {"op": "mul", "lr": 0.05, "floor": -0.30, "cap": 0.30},
            "max_exposure": {"op": "mul", "lr": 0.05, "floor": -0.35, "cap": 0.20},
            "hmm_min_confidence": {"op": "add", "lr": 0.04, "floor": -0.10, "cap": 0.20},
            "hmm_trend_prob_threshold": {"op": "add", "lr": 0.04, "floor": -0.10, "cap": 0.20},
            "profit_take_pct": {"op": "mul", "lr": 0.05, "floor": -0.35, "cap": 0.35},
            "stop_loss_pct": {"op": "mul", "lr": 0.05, "floor": -0.35, "cap": 0.20},
            "chand_atr_mult": {"op": "mul", "lr": 0.04, "floor": -0.25, "cap": 0.25},
            "breakout_n": {"op": "mul", "lr": 0.03, "floor": -0.50, "cap": 0.80},
        }

    def update(self) -> None:
        if not bool(getattr(self.p, "adaptive_profile_enabled", True)):
            return

        current_len = len(self.strat)
        if current_len == self.last_update_len:
            return

        self.last_update_len = current_len
        lookback = int(max(getattr(self.p, "adaptive_profile_lookback", 120), 60))
        if current_len <= lookback + 5:
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
        beta = self._estimate_beta(returns)
        trend_slope_120 = self._slope(closes[-120:]) if len(closes) >= 120 else self._slope(closes)
        trend_persistence = sum(1 for r in returns if r > 0) / max(len(returns), 1)
        max_dd_mean = self._rolling_max_drawdown_mean(closes, window=30)
        breakout_success = self._breakout_success_rate(closes, lookback=20, horizon=5, threshold=0.015)

        ema20 = float(self.strat.ema20[0])
        ema50 = float(self.strat.ema50[0])
        ema200 = float(self.strat.ema200[0])
        slope20 = self._slope([
            float(self.strat.ema20[-i]) for i in range(min(20, len(self.strat.ema20) - 1), -1, -1)
        ])

        atr_series = self._atr_series(highs, lows, closes)
        atr30_recent = sum(atr_series[-30:]) / max(min(30, len(atr_series)), 1)
        atr30_prev = sum(atr_series[-60:-30]) / max(min(30, len(atr_series[-60:-30])), 1) if len(atr_series) >= 60 else atr30_recent
        atr_declining = atr30_recent <= atr30_prev * 0.92
        range_compress = (max(highs[-30:]) - min(lows[-30:])) / max(closes[-1], 1e-9)

        avg_vol = sum(vols) / max(len(vols), 1)
        recent_vol = sum(vols[-10:]) / max(min(len(vols), 10), 1)
        recent_vol_30 = sum(vols[-30:]) / max(min(len(vols), 30), 1)
        prev_vol_30 = sum(vols[-60:-30]) / max(min(len(vols[-60:-30]), 30), 1) if len(vols) >= 60 else recent_vol_30
        volume_ratio = recent_vol / max(avg_vol, 1e-9)
        volume_trend = recent_vol_30 / max(prev_vol_30, 1e-9)

        self.metrics = {
            "annual_vol": annual_vol,
            "beta": beta,
            "max_dd_mean": max_dd_mean,
            "trend_slope_120": trend_slope_120,
            "trend_persistence": trend_persistence,
            "breakout_success": breakout_success,
            "volume_ratio": volume_ratio,
            "range_compress": range_compress,
            "atr30_recent": atr30_recent,
            "atr30_prev": atr30_prev,
        }

        self.archetype = self._classify_archetype(
            annual_vol=annual_vol,
            beta=beta,
            trend_slope_120=trend_slope_120,
            trend_persistence=trend_persistence,
            breakout_success=breakout_success,
            max_dd_mean=max_dd_mean,
        )
        self.stage = self._infer_stage(
            net=net,
            slope20=slope20,
            ema20=ema20,
            ema50=ema50,
            ema200=ema200,
            atr_declining=atr_declining,
            range_compress=range_compress,
            volume_trend=volume_trend,
            breakout_success=breakout_success,
        )
        self.confidence = self._compute_confidence(
            net=net,
            annual_vol=annual_vol,
            trend_persistence=trend_persistence,
            breakout_success=breakout_success,
            range_compress=range_compress,
        )

    def _classify_archetype(
        self,
        annual_vol: float,
        beta: float,
        trend_slope_120: float,
        trend_persistence: float,
        breakout_success: float,
        max_dd_mean: float,
    ) -> str:
        if annual_vol >= 0.38 and beta >= 1.2 and trend_slope_120 > 0 and trend_persistence >= 0.54:
            return "HIGH_BETA_GROWTH"
        if annual_vol >= 0.32 and abs(trend_slope_120) < 0.0012 and breakout_success < 0.40:
            return "CYCLICAL"
        if annual_vol <= 0.22 and max_dd_mean <= 0.10 and trend_persistence >= 0.50:
            return "DEFENSIVE"
        if abs(trend_slope_120) < 0.0010 and breakout_success <= 0.35:
            return "CHOPPY"
        return "BALANCED"

    def _infer_stage(
        self,
        net: float,
        slope20: float,
        ema20: float,
        ema50: float,
        ema200: float,
        atr_declining: bool,
        range_compress: float,
        volume_trend: float,
        breakout_success: float,
    ) -> str:
        # Wyckoff风格阶段判断
        if atr_declining and abs(slope20) < 0.0012 and range_compress < 0.10 and volume_trend <= 1.02:
            return "ACCUMULATION"
        if ema50 > ema200 and ema20 > ema50 and slope20 > 0 and net > 0.06 and volume_trend >= 1.03:
            return "MARKUP"
        if net > 0.04 and slope20 <= 0 and volume_trend < 0.98 and breakout_success < 0.45:
            return "DISTRIBUTION"
        if ema20 < ema50 and slope20 < 0 and net < -0.08:
            return "MARKDOWN"
        if abs(net) <= 0.08 and range_compress < 0.16:
            return "RANGE"
        return "TRANSITION"

    @staticmethod
    def _annualized_vol(returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)
        return math.sqrt(max(var, 0.0)) * math.sqrt(252.0)

    def _estimate_beta(self, stock_returns: List[float]) -> float:
        d = self.strat.datas[0]
        if not hasattr(d, "benchmark_close"):
            return 1.0

        lb = min(len(stock_returns), 120)
        if lb < 20:
            return 1.0

        bench = [float(getattr(d, "benchmark_close")[-i]) for i in range(lb, -1, -1)]
        if min(bench) <= 0:
            return 1.0
        bench_returns = [bench[i] / bench[i - 1] - 1.0 for i in range(1, len(bench))]
        stock_slice = stock_returns[-len(bench_returns):]
        if len(stock_slice) < 10 or len(stock_slice) != len(bench_returns):
            return 1.0

        s_mean = sum(stock_slice) / len(stock_slice)
        b_mean = sum(bench_returns) / len(bench_returns)
        cov = sum((s - s_mean) * (b - b_mean) for s, b in zip(stock_slice, bench_returns)) / max(len(stock_slice) - 1, 1)
        b_var = sum((b - b_mean) ** 2 for b in bench_returns) / max(len(bench_returns) - 1, 1)
        if b_var <= 1e-12:
            return 1.0
        return cov / b_var

    @staticmethod
    def _rolling_max_drawdown_mean(closes: List[float], window: int = 30) -> float:
        if len(closes) <= window + 1:
            return 0.0
        dds = []
        for i in range(window, len(closes)):
            seg = closes[i - window:i + 1]
            peak = seg[0]
            mdd = 0.0
            for px in seg:
                peak = max(peak, px)
                mdd = max(mdd, 1.0 - px / max(peak, 1e-9))
            dds.append(mdd)
        return sum(dds) / max(len(dds), 1)

    @staticmethod
    def _breakout_success_rate(
        closes: List[float],
        lookback: int = 20,
        horizon: int = 5,
        threshold: float = 0.015,
    ) -> float:
        if len(closes) <= lookback + horizon + 2:
            return 0.0
        trials = 0
        wins = 0
        for i in range(lookback, len(closes) - horizon):
            prev_high = max(closes[i - lookback:i])
            if closes[i] >= prev_high:
                trials += 1
                fwd_ret = closes[i + horizon] / max(closes[i], 1e-9) - 1.0
                if fwd_ret >= threshold:
                    wins += 1
        return wins / max(trials, 1)

    @staticmethod
    def _atr_series(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
        if len(closes) < 2:
            return []
        tr_vals = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_vals.append(tr / max(closes[i], 1e-9))
        return tr_vals

    @staticmethod
    def _slope(series: List[float]) -> float:
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

    @staticmethod
    def _legacy_stage_alias(stage: str) -> str:
        alias = {
            "ACCUMULATION": "BASE",
            "RANGE": "SIDEWAYS",
            "MARKUP": "MARKUP",
            "DISTRIBUTION": "DISTRIBUTION",
            "MARKDOWN": "MARKDOWN",
            "TRANSITION": "TRANSITION",
        }
        return alias.get(stage, stage)

    def _compute_confidence(
        self,
        net: float,
        annual_vol: float,
        trend_persistence: float,
        breakout_success: float,
        range_compress: float,
    ) -> float:
        trend_score = min(abs(net) / 0.15, 1.0)
        vol_score = min(annual_vol / max(float(getattr(self.p, "adaptive_high_vol_threshold", 0.45)), 1e-6), 1.0)
        persistence_score = 1.0 - min(abs(trend_persistence - 0.5) / 0.5, 1.0)
        breakout_score = min(max(breakout_success, 0.0), 1.0)
        compress_score = 1.0 - min(range_compress / 0.25, 1.0)
        raw = 0.30 * trend_score + 0.20 * vol_score + 0.20 * persistence_score + 0.15 * breakout_score + 0.15 * compress_score
        return max(0.15, min(0.95, raw))

    def context_key(self) -> Tuple[str, str]:
        return self.archetype, self.stage

    def _state_for(self, context: Tuple[str, str], param_name: str) -> Dict[str, float]:
        key = (context[0], context[1], param_name)
        st = self.learned_state.get(key)
        if st is None:
            st = {"value": 0.0, "count": 0.0}
            self.learned_state[key] = st
        return st

    def _heuristic_bias(self, param_name: str) -> float:
        vol = float(self.metrics.get("annual_vol", 0.0))
        breakout_success = float(self.metrics.get("breakout_success", 0.0))
        trend_persistence = float(self.metrics.get("trend_persistence", 0.5))

        if param_name == "max_exposure":
            if self.stage in {"MARKDOWN", "DISTRIBUTION"}:
                return -0.15
            if self.stage == "MARKUP":
                return 0.08
        if param_name == "target_vol_annual":
            return 0.10 if self.stage == "MARKUP" and breakout_success > 0.45 else -0.10 if vol > 0.45 else 0.0
        if param_name in {"hmm_min_confidence", "hmm_trend_prob_threshold"}:
            return 0.05 if self.stage in {"MARKDOWN", "DISTRIBUTION", "RANGE"} else -0.03 if self.stage == "MARKUP" else 0.0
        if param_name in {"vol_ratio_min", "add_vol_ratio_min"}:
            return 0.10 if self.stage in {"RANGE", "DISTRIBUTION"} else -0.06 if self.stage == "MARKUP" else 0.0
        if param_name == "breakout_n":
            return 0.50 if self.stage in {"RANGE", "ACCUMULATION"} else -0.20 if self.stage == "MARKUP" else 0.0
        if param_name == "stop_loss_pct":
            return -0.10 if self.stage in {"MARKDOWN", "DISTRIBUTION"} else 0.05 if trend_persistence > 0.55 else 0.0
        if param_name == "profit_take_pct":
            return 0.12 if self.stage == "MARKUP" else -0.10 if self.stage in {"RANGE", "DISTRIBUTION"} else 0.0
        if param_name == "chand_atr_mult":
            return 0.10 if self.stage == "MARKUP" else -0.10 if self.stage in {"RANGE", "MARKDOWN"} else 0.0
        return 0.0

    def get_adjustment(self, param_name: str, base_value: float) -> float:
        """返回参数自适应后的值。"""
        if not bool(getattr(self.p, "adaptive_profile_enabled", True)):
            return float(base_value)

        conf = float(self.confidence)
        if conf < float(getattr(self.p, "adaptive_confidence_min", 0.30)):
            return float(base_value)

        context = self.context_key()
        meta = self.param_meta.get(param_name)
        if not meta:
            return self._clip_param(param_name, float(base_value))

        state = self._state_for(context, param_name)
        learned = float(state.get("value", 0.0))
        heuristic = self._heuristic_bias(param_name)
        signal = (0.65 * heuristic + 0.35 * learned) * conf

        if str(meta.get("op")) == "mul":
            value = float(base_value) * (1.0 + signal)
        else:
            value = float(base_value) + signal

        return self._clip_param(param_name, value)

    def observe_trade(self, pnl_pct: float, context: Optional[Tuple[str, str]] = None) -> None:
        """根据平仓结果更新在线学习参数。"""
        if context is None:
            context = self.context_key()

        # reward压缩到[-1, 1]，提升稳健性
        reward = math.tanh(float(pnl_pct) / 8.0)
        conf = max(0.15, float(self.confidence))
        for param_name, meta in self.param_meta.items():
            st = self._state_for(context, param_name)
            n = float(st.get("count", 0.0))
            lr = float(meta.get("lr", 0.03)) / (1.0 + 0.05 * n)
            updated = float(st.get("value", 0.0)) + lr * reward * conf
            lo = float(meta.get("floor", -1.0))
            hi = float(meta.get("cap", 1.0))
            st["value"] = min(max(updated, lo), hi)
            st["count"] = n + 1.0

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
            "chand_atr_mult": (1.2, 4.0),
            "breakout_n": (5.0, 999.0),
        }
        lo, hi = bounds.get(param_name, (-1e12, 1e12))
        return min(max(float(value), lo), hi)
