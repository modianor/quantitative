# -*- coding: utf-8 -*-
"""Triple Barrier + Meta-labeling 模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TripleBarrierConfig:
    take_profit_pct: float
    stop_loss_pct: float
    max_holding_bars: int = 20


class TripleBarrierLabeler:
    """将收益路径映射为三重障碍标签。"""

    LABEL_TP = 1
    LABEL_SL = -1
    LABEL_TIMEOUT = 0

    def __init__(self, config: TripleBarrierConfig):
        self.config = config

    def label_forward_path(self, entry_price: float, highs: List[float], lows: List[float]) -> int:
        up = entry_price * (1.0 + self.config.take_profit_pct / 100.0)
        down = entry_price * (1.0 - self.config.stop_loss_pct / 100.0)

        horizon = min(len(highs), len(lows), int(self.config.max_holding_bars))
        for i in range(horizon):
            if lows[i] <= down:
                return self.LABEL_SL
            if highs[i] >= up:
                return self.LABEL_TP
        return self.LABEL_TIMEOUT


class ExitEventMapper:
    """将策略出场事件统一映射到三重障碍标签。"""

    def __init__(self):
        self._map = {
            "STOP_LOSS": TripleBarrierLabeler.LABEL_SL,
            "INTRADAY_STOP": TripleBarrierLabeler.LABEL_SL,
            "PROFIT_TAKE": TripleBarrierLabeler.LABEL_TP,
            "CHANDELIER": TripleBarrierLabeler.LABEL_TIMEOUT,
            "REGIME_CUT": TripleBarrierLabeler.LABEL_TIMEOUT,
        }

    def to_label(self, exit_tag: str, realized_return_pct: float) -> int:
        if exit_tag in self._map:
            return self._map[exit_tag]
        return TripleBarrierLabeler.LABEL_TP if realized_return_pct > 0 else TripleBarrierLabeler.LABEL_SL


class BaseMetaModel:
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        raise NotImplementedError

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LogisticMetaModel(BaseMetaModel):
    """轻量Logit模型（numpy GD实现，避免外部依赖）。"""

    def __init__(self, lr: float = 0.05, n_iter: int = 400):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-x_clip))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        if len(features) == 0:
            return

        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)

        if self.w is None or self.w.shape[0] != x.shape[1]:
            self.w = np.zeros(x.shape[1], dtype=float)
            self.b = 0.0

        for _ in range(self.n_iter):
            logits = x @ self.w + self.b
            pred = self._sigmoid(logits)
            err = pred - y
            grad_w = x.T @ err / float(len(x))
            grad_b = float(np.mean(err))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.w is None:
            return np.full(len(features), 0.5, dtype=float)
        x = np.asarray(features, dtype=float)
        logits = x @ self.w + self.b
        return self._sigmoid(logits)


class MetaLabelingFilter:
    """在线元标签过滤器：训练“是否执行信号”。"""

    def __init__(
        self,
        model: Optional[BaseMetaModel] = None,
        prob_threshold: float = 0.53,
        min_samples: int = 40,
        retrain_interval: int = 10,
    ):
        self.model = model or LogisticMetaModel()
        self.prob_threshold = float(prob_threshold)
        self.min_samples = int(min_samples)
        self.retrain_interval = int(retrain_interval)

        self._features: List[List[float]] = []
        self._labels: List[int] = []
        self._approved = 0

    def register_sample(self, feature: List[float], label: int) -> None:
        # meta-label: 1=可执行(正期望)，0=应过滤
        meta_label = 1 if int(label) == TripleBarrierLabeler.LABEL_TP else 0
        self._features.append([float(v) for v in feature])
        self._labels.append(meta_label)
        self._approved += 1

        if len(self._labels) >= self.min_samples and self._approved >= self.retrain_interval:
            x = np.asarray(self._features, dtype=float)
            y = np.asarray(self._labels, dtype=float)

            # 标准化可提升收敛稳定性
            mu = np.mean(x, axis=0)
            sigma = np.std(x, axis=0)
            sigma = np.where(sigma <= 1e-9, 1.0, sigma)
            x_norm = (x - mu) / sigma

            self.model.fit(x_norm, y)
            self._approved = 0
            self._mu = mu
            self._sigma = sigma

    def allow_signal(self, feature: List[float]) -> Tuple[bool, float]:
        if len(self._labels) < self.min_samples:
            return True, 0.5

        x = np.asarray([feature], dtype=float)
        mu = getattr(self, "_mu", np.zeros(x.shape[1], dtype=float))
        sigma = getattr(self, "_sigma", np.ones(x.shape[1], dtype=float))
        x_norm = (x - mu) / np.where(sigma <= 1e-9, 1.0, sigma)

        proba = float(self.model.predict_proba(x_norm)[0])
        return bool(proba >= self.prob_threshold), proba


class TradeMetaRecorder:
    """记录信号特征与出场标签，用于在线学习。"""

    def __init__(self, exit_mapper: Optional[ExitEventMapper] = None):
        self.exit_mapper = exit_mapper or ExitEventMapper()
        self.active_feature: Optional[List[float]] = None
        self.active_entry_price: Optional[float] = None
        self.active_entry_tag: Optional[str] = None

    def mark_entry(self, feature: List[float], entry_price: float, entry_tag: str) -> None:
        self.active_feature = [float(v) for v in feature]
        self.active_entry_price = float(entry_price)
        self.active_entry_tag = str(entry_tag)

    def close_trade(self, exit_tag: str, exit_price: float) -> Optional[Tuple[List[float], int]]:
        if self.active_feature is None or self.active_entry_price is None:
            return None

        entry = float(self.active_entry_price)
        ret = (float(exit_price) / max(entry, 1e-9) - 1.0) * 100.0
        label = self.exit_mapper.to_label(exit_tag, ret)
        feature = self.active_feature

        self.active_feature = None
        self.active_entry_price = None
        self.active_entry_tag = None
        return feature, label
