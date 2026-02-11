# -*- coding: utf-8 -*-
"""HMM 市场状态识别模块（Gaussian + Baum-Welch）。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal


class GaussianHMM:
    """高斯隐马尔可夫模型（对角协方差）。"""

    def __init__(self, n_states: int = 4, n_features: int = 4):
        self.n_states = n_states
        self.n_features = n_features

        self.start_prob = np.ones(n_states) / n_states
        self.transition = self._init_transition_matrix()
        self.means = self._init_means()
        self.covars = self._init_covariances()

    def _init_transition_matrix(self) -> np.ndarray:
        trans = np.eye(self.n_states) * 0.7
        off_diag = (1 - 0.7) / (self.n_states - 1)
        trans += off_diag * (1 - np.eye(self.n_states))

        if self.n_states == 4:
            trans[0, 1] = 0.15
            trans[0, 2] = 0.08
            trans[0, 3] = 0.02
            trans[0, 0] = 0.75
            trans[0] /= trans[0].sum()

        return trans

    def _init_means(self) -> np.ndarray:
        if self.n_states == 4 and self.n_features == 4:
            return np.array([
                [-0.05, 0.030, 0.15, +0.0012],
                [-0.08, 0.025, 0.45, +0.0002],
                [-0.26, 0.090, 0.35, -0.0010],
                [-0.36, 0.055, 0.20, +0.0005],
            ])
        return np.random.randn(self.n_states, self.n_features) * 0.1

    def _init_covariances(self) -> np.ndarray:
        if self.n_states == 4 and self.n_features == 4:
            return np.array([
                [0.05, 0.015, 0.20, 0.0012],
                [0.06, 0.012, 0.24, 0.0010],
                [0.10, 0.030, 0.20, 0.0018],
                [0.08, 0.020, 0.16, 0.0015],
            ]) ** 2
        return np.ones((self.n_states, self.n_features)) * 0.1

    def _emission_prob(self, obs: np.ndarray, state: int) -> float:
        mean = self.means[state]
        cov = np.diag(self.covars[state])
        prob = multivariate_normal.pdf(obs, mean=mean, cov=cov, allow_singular=True)
        return max(float(prob), 1e-300)

    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        t_size = len(observations)
        alpha = np.zeros((t_size, self.n_states))

        for s in range(self.n_states):
            alpha[0, s] = self.start_prob[s] * self._emission_prob(observations[0], s)

        for t in range(1, t_size):
            for s in range(self.n_states):
                alpha[t, s] = (
                    np.sum(alpha[t - 1] * self.transition[:, s])
                    * self._emission_prob(observations[t], s)
                )
            alpha[t] /= alpha[t].sum() + 1e-300

        log_likelihood = float(np.log(alpha[-1].sum() + 1e-300))
        return alpha, log_likelihood

    def backward(self, observations: np.ndarray) -> np.ndarray:
        t_size = len(observations)
        beta = np.zeros((t_size, self.n_states))
        beta[-1] = 1.0

        for t in range(t_size - 2, -1, -1):
            emission_next = np.array(
                [self._emission_prob(observations[t + 1], s2) for s2 in range(self.n_states)]
            )
            for s in range(self.n_states):
                beta[t, s] = np.sum(self.transition[s] * beta[t + 1] * emission_next)
            beta[t] /= beta[t].sum() + 1e-300

        return beta

    def baum_welch(self, observations: np.ndarray, max_iter: int = 50, tol: float = 1e-4) -> List[float]:
        log_likelihoods: List[float] = []
        t_size = len(observations)

        for _ in range(max_iter):
            alpha, log_like = self.forward(observations)
            beta = self.backward(observations)
            log_likelihoods.append(log_like)

            if len(log_likelihoods) > 1:
                improvement = log_likelihoods[-1] - log_likelihoods[-2]
                if abs(improvement) < tol:
                    break

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((t_size - 1, self.n_states, self.n_states))
            for t in range(t_size - 1):
                emission_next = np.array(
                    [self._emission_prob(observations[t + 1], s2) for s2 in range(self.n_states)]
                )
                for i in range(self.n_states):
                    xi[t, i, :] = (
                        alpha[t, i] * self.transition[i, :] * emission_next * beta[t + 1, :]
                    )
                xi[t] /= xi[t].sum() + 1e-300

            self.start_prob = gamma[0]
            denom = gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300
            self.transition = xi.sum(axis=0) / denom

            for s in range(self.n_states):
                gamma_sum = gamma[:, s].sum() + 1e-300
                self.means[s] = (gamma[:, s] @ observations) / gamma_sum
                diff = observations - self.means[s]
                self.covars[s] = ((gamma[:, s][:, None] * (diff ** 2)).sum(axis=0) / gamma_sum)

        return log_likelihoods

    def predict(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t_size = len(observations)
        delta = np.zeros((t_size, self.n_states))
        psi = np.zeros((t_size, self.n_states), dtype=int)

        for s in range(self.n_states):
            delta[0, s] = self.start_prob[s] * self._emission_prob(observations[0], s)

        for t in range(1, t_size):
            for s in range(self.n_states):
                probs = delta[t - 1] * self.transition[:, s]
                psi[t, s] = int(np.argmax(probs))
                delta[t, s] = probs[psi[t, s]] * self._emission_prob(observations[t], s)

        states = np.zeros(t_size, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(t_size - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        alpha, _ = self.forward(observations)
        beta = self.backward(observations)
        posterior = alpha * beta
        posterior /= posterior.sum(axis=1, keepdims=True) + 1e-300

        return states, posterior


class RegimeDetectorHMM:
    """策略内逐 bar 更新的 HMM 状态检测器。"""

    MODE_TREND = 0
    MODE_CHOP = 1
    MODE_DRAWDOWN = 2
    MODE_BASE = 3

    MODE_NAMES = {
        0: "TREND_RUN",
        1: "TOP_CHOP",
        2: "DRAWDOWN",
        3: "BASE_BUILD",
    }

    def __init__(self, strategy, lookback: int = 120):
        self.strat = strategy
        self.lookback = lookback
        self.model = GaussianHMM(n_states=4, n_features=4)
        self.is_trained = False

        self.feature_buffer: List[np.ndarray] = []
        self.last_state = 0
        self.last_posterior = np.array([1.0, 0.0, 0.0, 0.0])

    def extract_features(self) -> Optional[np.ndarray]:
        d = self.strat.data
        if len(self.strat) < 60:
            return None

        close = float(d.close[0])
        high_60 = max(float(d.close[-i]) for i in range(60))
        drawdown = close / high_60 - 1
        atrp = float(self.strat.atr[0]) / max(close, 1e-9)

        crosses = 0
        upper = min(60, len(self.strat))
        for i in range(1, upper):
            prev_pos = float(d.close[-i]) > float(self.strat.ema20[-i])
            curr_pos = (
                float(d.close[-i + 1]) > float(self.strat.ema20[-i + 1])
                if i > 1
                else (close > float(self.strat.ema20[0]))
            )
            if prev_pos != curr_pos:
                crosses += 1
        cross_rate = crosses / 60.0

        if len(self.strat.ema20) >= 10:
            slope = (float(self.strat.ema20[0]) / float(self.strat.ema20[-10]) - 1) / max(close, 1e-9)
        else:
            slope = 0.0

        return np.array([drawdown, atrp, cross_rate, slope], dtype=float)

    def update(self):
        features = self.extract_features()
        if features is None:
            return

        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.lookback:
            self.feature_buffer.pop(0)

        if len(self.feature_buffer) >= self.lookback and not self.is_trained:
            self.model.baum_welch(np.array(self.feature_buffer), max_iter=30)
            self.is_trained = True

        if self.is_trained and len(self.feature_buffer) % 50 == 0:
            self.model.baum_welch(np.array(self.feature_buffer), max_iter=10)

    def get_mode(self) -> Tuple[int, str]:
        if not self.is_trained or len(self.feature_buffer) < 20:
            return self.last_state, self.MODE_NAMES[self.last_state]

        recent_obs = np.array(self.feature_buffer[-20:])
        states, posterior = self.model.predict(recent_obs)

        self.last_state = int(states[-1])
        self.last_posterior = posterior[-1]
        return self.last_state, self.MODE_NAMES[self.last_state]

    def get_confidence(self) -> float:
        return float(self.last_posterior[self.last_state])
