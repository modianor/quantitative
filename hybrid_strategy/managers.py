# -*- coding: utf-8 -*-
"""状态识别、仓位与出场管理模块。"""

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
        target_value = value * float(self.p.max_exposure) * float(ratio)
        return int(target_value // price)

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

            # 预估清仓后状态
            cash_gain = pos_size * close
            self.strat.log(f"   → 预估现金: ${self.strat.broker.cash + cash_gain:,.0f} | "
                           f"总资产: ${self.strat.broker.getvalue():,.0f}")

            return True
        return False
