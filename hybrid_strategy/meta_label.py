# -*- coding: utf-8 -*-
"""Meta-labeling 模块（L2 正则 + 时序交叉验证）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class MetaLabeler:
    def __init__(
        self,
        take_profit_pct: float = 3.0,
        stop_loss_pct: float = 2.0,
        max_holding_bars: int = 20,
        min_samples: int = 50,
        cv_folds: int = 3,
    ):
        self.tp_pct = take_profit_pct / 100
        self.sl_pct = stop_loss_pct / 100
        self.max_hold = max_holding_bars
        self.min_samples = min_samples
        self.cv_folds = cv_folds

        self.features = []
        self.labels = []

        self.scaler = StandardScaler()
        self.model = LogisticRegressionCV(
            Cs=10,
            cv=TimeSeriesSplit(n_splits=cv_folds),
            penalty="l2",
            scoring="roc_auc",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def label_signal(self, entry_price: float, future_highs: np.ndarray, future_lows: np.ndarray) -> int:
        tp_level = entry_price * (1 + self.tp_pct)
        sl_level = entry_price * (1 - self.sl_pct)
        horizon = min(len(future_highs), self.max_hold)

        for i in range(horizon):
            if future_lows[i] <= sl_level:
                return 0
            if future_highs[i] >= tp_level:
                return 1

        final_price = (future_highs[-1] + future_lows[-1]) / 2
        return int(final_price > entry_price)

    def extract_features(self, strategy, mode_id: int) -> np.ndarray:
        d = strategy.data
        feats = [float(mode_id)]

        atrp = float(strategy.atr[0]) / max(float(d.close[0]), 1e-9)
        feats.append(atrp)

        if len(strategy) >= 20:
            vol_ma = np.mean([float(d.volume[-i]) for i in range(20)])
            vol_ratio = float(d.volume[0]) / max(vol_ma, 1e-9)
        else:
            vol_ratio = 1.0
        feats.append(vol_ratio)

        if hasattr(d, "trend_score"):
            trend_score = float(getattr(d, "trend_score")[0]) / 6.0
        else:
            trend_score = 1.0 if float(strategy.ema20[0]) > float(strategy.ema50[0]) else 0.0
        feats.append(trend_score)

        momentum = float(d.close[0]) / float(d.close[-10]) - 1 if len(strategy) >= 10 else 0.0
        feats.append(momentum)

        ema20_dist = (float(d.close[0]) / max(float(strategy.ema20[0]), 1e-9) - 1) * 100
        feats.append(ema20_dist)

        position_bias = float(strategy.position.size > 0)
        feats.append(position_bias)
        return np.array(feats, dtype=float)

    def add_sample(self, features: np.ndarray, label: int):
        self.features.append(features)
        self.labels.append(label)
        if len(self.features) > 500:
            self.features.pop(0)
            self.labels.pop(0)

    def train(self) -> Optional[dict]:
        if len(self.features) < self.min_samples:
            return None

        x = np.array(self.features)
        y = np.array(self.labels)
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self.is_trained = True

        return {
            "n_samples": len(y),
            "positive_rate": float(y.mean()),
            "best_C": float(self.model.C_[0]),
            "cv_score": float(self.model.scores_[1].mean()),
        }

    def predict_proba(self, features: np.ndarray) -> float:
        if not self.is_trained:
            return 0.5
        x_scaled = self.scaler.transform(features.reshape(1, -1))
        return float(self.model.predict_proba(x_scaled)[0, 1])

    def should_take_signal(self, features: np.ndarray, threshold: float = 0.55) -> Tuple[bool, float]:
        prob = self.predict_proba(features)
        return prob >= threshold, prob

    def register_trade(self, entry_features: np.ndarray, is_profit: bool):
        self.add_sample(entry_features, int(is_profit))
        if len(self.features) >= self.min_samples and len(self.features) % 20 == 0:
            self.train()


class AdaptiveMetaThreshold:
    def __init__(
        self,
        base_threshold: float = 0.55,
        drawdown_sensitivity: float = 0.1,
        volatility_sensitivity: float = 0.05,
    ):
        self.base = base_threshold
        self.dd_sens = drawdown_sensitivity
        self.vol_sens = volatility_sensitivity

    def get_threshold(self, account_drawdown: float, market_volatility: float, regime: str) -> float:
        threshold = self.base

        if account_drawdown > 0.05:
            dd_penalty = min((account_drawdown - 0.05) / 0.15, 1.0)
            threshold += self.dd_sens * dd_penalty

        if market_volatility > 0.30:
            vol_penalty = min((market_volatility - 0.30) / 0.30, 1.0)
            threshold += self.vol_sens * vol_penalty

        if regime == "TREND_RUN" and account_drawdown < 0.03:
            threshold -= 0.05

        return float(np.clip(threshold, 0.35, 0.75))
