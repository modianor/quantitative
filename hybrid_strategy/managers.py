# -*- coding: utf-8 -*-
"""状态识别、仓位与出场管理模块。"""

import math

class RegimeDetector:
    MODE_TREND = 0
    MODE_TOPCHOP = 1
    MODE_DRAWDOWN = 2
    MODE_BASE = 3

    MODE_NAMES = {
        0: "TREND_RUN",
        1: "TOP_CHOP",
        2: "DRAWDOWN",
        3: "BASE_BUILD",
    }

    def __init__(self, strategy):
        self.strat = strategy
        self.p = strategy.params
        self.cross_cache = {}
        self.last_calc_len = 0

        # Mode切换缓冲
        self.last_mode = None
        self.mode_counter = 0
        self.mode_buffer_days = 2

    def get_mode(self):
        """返回 (mode_id, mode_name) - 带缓冲机制"""
        raw_mode_id, raw_mode_name = self._calculate_raw_mode()

        if self.last_mode is None:
            self.last_mode = raw_mode_id
            self.mode_counter = self.mode_buffer_days
            return raw_mode_id, raw_mode_name

        if raw_mode_id != self.last_mode:
            self.mode_counter = 1
        else:
            self.mode_counter += 1

        if self.mode_counter >= self.mode_buffer_days:
            self.last_mode = raw_mode_id
            return raw_mode_id, raw_mode_name
        else:
            return self.last_mode, self.MODE_NAMES[self.last_mode]

    def _calculate_raw_mode(self):
        if len(self.strat) < self.p.min_bars_required:
            return self.MODE_TREND, self.MODE_NAMES[self.MODE_TREND]

        close = float(self.strat.data.close[0])
        hh = max(float(self.strat.hh_stage[0]), 1e-9)
        dd = close / hh - 1.0
        atrp = float(self.strat.atr[0]) / max(close, 1e-9)

        if dd <= float(self.p.dd_drawdown_th) or atrp >= float(self.p.atrp_drawdown_th):
            return self.MODE_DRAWDOWN, self.MODE_NAMES[self.MODE_DRAWDOWN]

        in_high_zone = dd >= float(self.p.high_zone_dd_th)
        atr_ma = self._get_atr_ma()
        atr_shrink = float(self.strat.atr[0]) < atr_ma * float(self.p.atr_shrink_ratio)
        cross_cnt = self._count_cross_ema20_cached(self.p.stage_lookback)

        if in_high_zone and (atr_shrink or cross_cnt >= int(self.p.cross_top_min)):
            return self.MODE_TOPCHOP, self.MODE_NAMES[self.MODE_TOPCHOP]

        in_base_zone = dd <= float(self.p.base_zone_dd_th)
        atr_ok = atrp <= float(self.p.base_atrp_th)
        hl_up = self._check_higher_lows(self.p.base_hl_consecutive)

        if in_base_zone and atr_ok and hl_up:
            return self.MODE_BASE, self.MODE_NAMES[self.MODE_BASE]

        return self.MODE_TREND, self.MODE_NAMES[self.MODE_TREND]

    def _get_atr_ma(self):
        if len(self.strat.atr) < 20:
            return float(self.strat.atr[0])
        total = sum(float(self.strat.atr[-i]) for i in range(20))
        return total / 20.0

    def _count_cross_ema20_cached(self, lookback: int) -> int:
        current_len = len(self.strat)
        if current_len == self.last_calc_len and lookback in self.cross_cache:
            return self.cross_cache[lookback]

        if current_len < lookback + 2:
            return 999

        cnt = 0
        prev = None
        for i in range(lookback, 0, -1):
            v = 1 if float(self.strat.data.close[-i]) > float(self.strat.ema20[-i]) else 0
            if prev is None:
                prev = v
            else:
                if v != prev:
                    cnt += 1
                    prev = v

        self.cross_cache[lookback] = cnt
        self.last_calc_len = current_len
        return cnt

    def _check_higher_lows(self, consecutive: int) -> bool:
        if len(self.strat.ll_base) < consecutive + self.p.base_hl_shift:
            return False

        for i in range(consecutive):
            ll_now = float(self.strat.ll_base[-i])
            ll_prev = float(self.strat.ll_base[-i - self.p.base_hl_shift])
            if ll_now <= ll_prev:
                return False
        return True


class HMMRegimeDetector:
    """基于4状态高斯隐马尔可夫的状态识别器。"""

    MODE_TREND = RegimeDetector.MODE_TREND
    MODE_TOPCHOP = RegimeDetector.MODE_TOPCHOP
    MODE_DRAWDOWN = RegimeDetector.MODE_DRAWDOWN
    MODE_BASE = RegimeDetector.MODE_BASE
    MODE_NAMES = RegimeDetector.MODE_NAMES

    def __init__(self, strategy, fallback_detector: RegimeDetector):
        self.strat = strategy
        self.p = strategy.params
        self.fallback = fallback_detector

        self.last_mode = None
        self.mode_counter = 0
        self.mode_buffer_days = int(getattr(self.p, "hmm_mode_buffer_days", 2))

        self.posterior = [0.70, 0.10, 0.10, 0.10]
        self.transition = [
            [0.80, 0.15, 0.03, 0.02],
            [0.25, 0.60, 0.10, 0.05],
            [0.15, 0.10, 0.70, 0.05],
            [0.35, 0.10, 0.10, 0.45],
        ]

        # 发射分布特征顺序：dd, atrp, cross_rate, slope
        self.means = [
            [-0.05, 0.030, 0.15, +0.0012],  # TREND_RUN
            [-0.08, 0.025, 0.45, +0.0002],  # TOP_CHOP
            [-0.26, 0.090, 0.35, -0.0010],  # DRAWDOWN
            [-0.36, 0.055, 0.20, +0.0005],  # BASE_BUILD
        ]
        self.stds = [
            [0.05, 0.015, 0.20, 0.0012],
            [0.06, 0.012, 0.24, 0.0010],
            [0.10, 0.030, 0.20, 0.0018],
            [0.08, 0.020, 0.16, 0.0015],
        ]

    def get_mode(self):
        min_bars = int(max(self.p.min_bars_required, getattr(self.p, "hmm_warmup_bars", 240)))
        if len(self.strat) < min_bars:
            return self.fallback.get_mode()

        features = self._extract_features()
        if features is None:
            return self.fallback.get_mode()

        filtered = self._forward_filter(features)
        mode_id = max(range(4), key=lambda i: filtered[i])
        confidence = float(filtered[mode_id])
        confidence_min = float(getattr(self.p, "hmm_min_confidence", 0.45))

        # 置信度不足时，回退到规则状态机
        if confidence < confidence_min:
            return self.fallback.get_mode()

        if self.last_mode is None:
            self.last_mode = mode_id
            self.mode_counter = self.mode_buffer_days
            return mode_id, self.MODE_NAMES[mode_id]

        if mode_id != self.last_mode:
            self.mode_counter = 1
        else:
            self.mode_counter += 1

        if self.mode_counter >= self.mode_buffer_days:
            self.last_mode = mode_id
            return mode_id, self.MODE_NAMES[mode_id]

        return self.last_mode, self.MODE_NAMES[self.last_mode]

    def _extract_features(self):
        try:
            close = float(self.strat.data.close[0])
            if close <= 0:
                return None

            hh = max(float(self.strat.hh_stage[0]), 1e-9)
            dd = close / hh - 1.0

            atrv = float(self.strat.atr[0])
            atrp = atrv / max(close, 1e-9)

            lookback = int(getattr(self.p, "stage_lookback", 60))
            cross_cnt = self.fallback._count_cross_ema20_cached(lookback)
            cross_rate = min(float(cross_cnt) / max(float(lookback), 1.0), 1.0)

            slope = self._ema20_slope(int(getattr(self.p, "slope_win", 10)))
            return [dd, atrp, cross_rate, slope]
        except Exception:
            return None

    def _forward_filter(self, x_t):
        pred = [0.0, 0.0, 0.0, 0.0]
        for j in range(4):
            pred[j] = sum(self.posterior[i] * self.transition[i][j] for i in range(4))

        emissions = self._gaussian_emissions(x_t)
        unnorm = [pred[i] * emissions[i] for i in range(4)]
        denom = float(sum(unnorm))

        if denom <= 1e-12:
            self.posterior = pred
            return pred

        self.posterior = [v / denom for v in unnorm]
        return self.posterior

    def _gaussian_emissions(self, x_t):
        return [self._diag_gaussian_pdf(x_t, self.means[s], self.stds[s]) for s in range(4)]

    @staticmethod
    def _diag_gaussian_pdf(x, mu, sigma) -> float:
        # 独立高斯，log域防止下溢
        safe_sigma = [max(float(s), 1e-6) for s in sigma]
        logp = 0.0
        for xi, mi, si in zip(x, mu, safe_sigma):
            z = (float(xi) - float(mi)) / si
            logp += -0.5 * z * z - math.log(si) - 0.5 * math.log(2.0 * math.pi)
        return float(math.exp(max(logp, -100.0)))

    def _ema20_slope(self, window: int) -> float:
        if len(self.strat.ema20) < window:
            return 0.0

        x = list(range(window))
        x_mean = sum(x) / float(window)
        y = [float(self.strat.ema20[-i]) for i in range(window - 1, -1, -1)]
        y_mean = sum(y) / float(window)

        denom = sum((xi - x_mean) ** 2 for xi in x)
        if denom <= 0:
            return 0.0

        num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(window))
        slope = float(num / denom)
        return slope / max(float(y_mean), 1e-9)


# =============================
# 仓位管理器
# =============================
class PositionManager:
    def __init__(self, strategy):
        self.strat = strategy
        self.p = strategy.params

    def target_size(self, ratio: float) -> int:
        price = float(self.strat.data.close[0])
        if price <= 0:
            return 0
        value = float(self.strat.broker.getvalue())
        effective_exposure = self._effective_exposure(float(ratio))
        target_value = value * effective_exposure
        return int(target_value // price)

    def _effective_exposure(self, ratio: float) -> float:
        # 兼容旧逻辑：关闭波动目标时仍是固定仓位比例
        if not bool(getattr(self.p, "use_vol_targeting", True)):
            return min(max(float(self.p.max_exposure) * ratio, 0.0), 1.0)

        annual_vol = self._annualized_volatility()
        target_vol = max(float(getattr(self.p, "target_vol_annual", 0.20)), 1e-6)
        vol_scalar = target_vol / annual_vol

        min_scalar = float(getattr(self.p, "min_vol_scalar", 0.30))
        max_scalar = float(getattr(self.p, "max_vol_scalar", 1.00))
        vol_scalar = min(max(vol_scalar, min_scalar), max_scalar)

        exposure = float(self.p.max_exposure) * ratio * vol_scalar
        return min(max(exposure, 0.0), 1.0)

    def _annualized_volatility(self) -> float:
        lookback = int(max(getattr(self.p, "vol_lookback", 20), 2))

        rets = []
        max_hist = min(len(self.strat.data.close) - 1, lookback)
        for i in range(1, max_hist + 1):
            c0 = float(self.strat.data.close[-i])
            c1 = float(self.strat.data.close[-i - 1])
            if c0 > 0 and c1 > 0:
                rets.append(math.log(c0 / c1))

        if len(rets) < 2:
            atrp = float(self.strat.atr[0]) / max(float(self.strat.data.close[0]), 1e-9)
            annual_vol = atrp * math.sqrt(252.0)
        else:
            mean_ret = sum(rets) / float(len(rets))
            variance = sum((r - mean_ret) ** 2 for r in rets) / float(len(rets) - 1)
            annual_vol = math.sqrt(max(variance, 0.0)) * math.sqrt(252.0)

        floor_vol = float(getattr(self.p, "vol_floor_annual", 0.10))
        cap_vol = float(getattr(self.p, "vol_cap_annual", 0.80))
        return min(max(annual_vol, floor_vol), cap_vol)

    def scale_to(self, ratio: float, reason: str, mode_name: str, tag: str):
        target = self.target_size(ratio)
        current = int(self.strat.position.size)
        add = target - current
        if add > 0:
            price = float(self.strat.data.close[0])

            # 买入前日志
            self.strat.log(f"[{mode_name}] {reason} | 目标={target} 当前={current} 加={add}")
            self.strat.order = self.strat.buy(size=add)
            dt = self.strat.data.datetime.date(0)
            self.strat.trade_marks.append((dt, price, "BUY", mode_name, tag))

            # 买入后预估持仓（实际成交价可能略有不同）
            if current > 0:
                old_cost = float(self.strat.position.price)
                new_avg_cost = (old_cost * current + price * add) / target
            else:
                new_avg_cost = price

            new_value = target * price
            cash_after = self.strat.broker.cash - add * price
            total_after = self.strat.broker.getvalue()

            self.strat.log(f"   → 预估持仓: {target}股 @ ${new_avg_cost:.2f} | "
                           f"市值约${new_value:,.0f}")
            self.strat.log(f"   → 预估现金: ${cash_after:,.0f} | 总资产: ${total_after:,.0f}")

    def scale_down_to(self, ratio: float, reason: str, mode_name: str, tag: str):
        target = self.target_size(ratio)
        current = int(self.strat.position.size)
        reduce = current - target
        if reduce > 0:
            price = float(self.strat.data.close[0])
            avg_cost = float(self.strat.position.price)
            pnl = (price / avg_cost - 1) * 100

            # 卖出前日志
            self.strat.log(f"[{mode_name}] {reason} | 目标={target} 当前={current} 减={reduce} | "
                           f"盈亏{pnl:+.2f}%")
            self.strat.order = self.strat.sell(size=reduce)
            dt = self.strat.data.datetime.date(0)
            self.strat.trade_marks.append((dt, price, "SELL", mode_name, tag))

            # 卖出后预估
            cash_gain = reduce * price
            cash_after = self.strat.broker.cash + cash_gain

            if target > 0:
                remaining_value = target * price
                self.strat.log(f"   → 预估持仓: {target}股 @ ${avg_cost:.2f} | "
                               f"市值约${remaining_value:,.0f}")
            else:
                self.strat.log(f"   → 预估持仓: 空仓")

            self.strat.log(f"   → 预估现金: ${cash_after:,.0f} | 总资产: ${self.strat.broker.getvalue():,.0f}")


# =============================
# 出场管理器（票型差异化）
# =============================
class ExitManager:
    def __init__(self, strategy):
        self.strat = strategy
        self.p = strategy.params
        self.pos_mgr = strategy.pos_mgr

    def _mark_exit(self, tag: str, price: float):
        self.strat.last_exit_tag = str(tag)
        self.strat.last_exit_price = float(price)

    def check_stop_loss(self, mode_name: str) -> bool:
        """票型差异化止损（带盘中模拟）"""
        if not self.strat.position:
            return False

        # 高波动票：禁用止损（stop_loss_pct >= 999）
        if self.p.stop_loss_pct >= 999:
            return False

        close = float(self.strat.data.close[0])
        cost = float(self.strat.position.price)
        pos_size = int(self.strat.position.size)

        # 模拟盘中止损：使用当日最低价
        low_today = float(self.strat.data.low[0])

        # 计算收盘价盈亏
        pnl_close = (close / cost - 1.0) * 100

        # 计算盘中最低点盈亏（模拟最坏情况）
        pnl_intraday_low = (low_today / cost - 1.0) * 100

        # 如果盘中最低点触发止损
        if pnl_intraday_low <= -float(self.p.stop_loss_pct):
            # 假设在盘中触发时立即止损，使用止损价附近的价格
            # 估算：止损价 + 0.5%滑点
            estimated_exit_price = cost * (1 - float(self.p.stop_loss_pct) / 100) * 0.995

            self.strat.log(f"[{mode_name}] ⚠️ 盘中止损触发 | 最低={pnl_intraday_low:.2f}% | 收盘={pnl_close:.2f}%")
            self.strat.log(f"   → 模拟盘中卖出价格=${estimated_exit_price:.2f} (含0.5%滑点)")

            self.strat.order = self.strat.close()
            dt = self.strat.data.datetime.date(0)
            self.strat.trade_marks.append((dt, estimated_exit_price, "SELL", mode_name, "INTRADAY_STOP"))
            self._mark_exit("INTRADAY_STOP", estimated_exit_price)

            # 预估卖出后状态
            cash_gain = pos_size * close  # 实际按收盘价成交
            self.strat.log(f"   → 预估现金: ${self.strat.broker.cash + cash_gain:,.0f} | "
                           f"总资产: ${self.strat.broker.getvalue():,.0f}")

            return True

        # 正常收盘止损
        elif pnl_close <= -float(self.p.stop_loss_pct):
            self.strat.log(f"[{mode_name}] 收盘止损触发 PnL={pnl_close:.2f}% 阈值=-{self.p.stop_loss_pct:.2f}%")
            self.strat.order = self.strat.close()
            dt = self.strat.data.datetime.date(0)
            self.strat.trade_marks.append((dt, close, "SELL", mode_name, "STOP_LOSS"))
            self._mark_exit("STOP_LOSS", close)

            # 预估卖出后状态
            cash_gain = pos_size * close
            self.strat.log(f"   → 预估现金: ${self.strat.broker.cash + cash_gain:,.0f} | "
                           f"总资产: ${self.strat.broker.getvalue():,.0f}")

            return True

        return False

    def check_regime_sell(self, mode_id: int, mode_name: str) -> bool:
        if not self.strat.position:
            return False

        # TOP_CHOP：满仓减到二档
        if mode_id == RegimeDetector.MODE_TOPCHOP and self.strat.tranche >= 3:
            target_ratio = self.p.tranche_targets[1]
            self.pos_mgr.scale_down_to(target_ratio, "高位横盘减仓", mode_name, "REGIME_CUT")
            self._mark_exit("REGIME_CUT", float(self.strat.data.close[0]))
            self.strat.tranche = 2
            return True

        return False

    def check_profit_taking(self, mode_name: str) -> bool:
        """分批止盈"""
        if not self.strat.position or self.strat.tranche < 3:
            return False

        close = float(self.strat.data.close[0])
        cost = float(self.strat.position.price)
        pnl_pct = (close / cost - 1.0) * 100

        if pnl_pct >= float(self.p.profit_take_pct) and not self.strat.profit_taken:
            pos_size = int(self.strat.position.size)
            reduce_size = int(pos_size // 3)

            if reduce_size > 0:
                self.strat.log(f"[{mode_name}] 分批止盈 PnL={pnl_pct:.2f}% | 减仓={reduce_size}")
                self.strat.order = self.strat.sell(size=reduce_size)
                dt = self.strat.data.datetime.date(0)
                self.strat.trade_marks.append((dt, close, "SELL", mode_name, "PROFIT_TAKE"))
                self._mark_exit("PROFIT_TAKE", close)

                # 预估止盈后状态
                remaining = pos_size - reduce_size
                remaining_value = remaining * close
                cash_gain = reduce_size * close

                self.strat.log(f"   → 预估持仓: {remaining}股 @ ${cost:.2f} | 市值约${remaining_value:,.0f}")
                self.strat.log(f"   → 预估现金: ${self.strat.broker.cash + cash_gain:,.0f}")

                self.strat.profit_taken = True
                return True
        return False

    def check_chandelier(self, mode_name: str) -> bool:
        """Chandelier Exit"""
        if not self.strat.position:
            return False

        close = float(self.strat.data.close[0])
        cost = float(self.strat.position.price)
        pos_size = int(self.strat.position.size)
        pnl = (close / cost - 1.0) * 100

        chand_line = float(self.strat.hh_chand[0]) - float(self.p.chand_atr_mult) * float(self.strat.atr[0])

        if close < chand_line:
            self.strat.log(f"[{mode_name}] Chandelier Exit | Close=${close:.2f} < ${chand_line:.2f} | "
                           f"持仓盈亏{pnl:+.2f}%")
            self.strat.order = self.strat.close()
            dt = self.strat.data.datetime.date(0)
            self.strat.trade_marks.append((dt, close, "SELL", mode_name, "CHANDELIER"))
            self._mark_exit("CHANDELIER", close)

            # 预估清仓后状态
            cash_gain = pos_size * close
            self.strat.log(f"   → 预估现金: ${self.strat.broker.cash + cash_gain:,.0f} | "
                           f"总资产: ${self.strat.broker.getvalue():,.0f}")

            return True
        return False
