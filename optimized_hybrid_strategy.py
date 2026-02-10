# -*- coding: utf-8 -*-
"""
优化版四阶段自适应策略 v2.2 - 独立配置文件版
核心改进：
1. 每个股票独立配置文件，方便管理
2. 自动加载stock_configs.py
3. 高波动成长股：禁用止损，只用Chandelier趋势跟踪
4. 低波动大盘股：严格止损 + 快速止盈
5. 垃圾股/困境股：黑名单（不交易）
"""

import math
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt

# 导入股票配置
try:
    from stock_configs import get_stock_config, print_stock_info, list_all_stocks

    CONFIG_LOADED = True
except ImportError:
    print("⚠️ 警告: 未找到stock_configs.py，使用内置配置")
    CONFIG_LOADED = False


# =============================
# 数据加载工具
# =============================
def load_from_yfinance(symbol: str, start=None, end=None, period="5y"):
    import yfinance as yf

    if start or end:
        raw = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False)
    else:
        raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [str(c).strip().title() for c in raw.columns]
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    raw.index.name = "Date"
    return raw


def load_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    miss = [c for c in required if c not in cols]
    if miss:
        raise ValueError(f"CSV 缺列: {miss}")

    df = df.rename(columns={
        cols["date"]: "Date",
        cols["open"]: "Open",
        cols["high"]: "High",
        cols["low"]: "Low",
        cols["close"]: "Close",
        cols["volume"]: "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# =============================
# 自定义 DataFeed
# =============================
class PandasWithSignals(bt.feeds.PandasData):
    lines = ("is_main_uptrend", "main_uptrend_start", "trend_score", "mom_score", "pb_score", "vol_ratio")
    params = (
        ("is_main_uptrend", -1),
        ("main_uptrend_start", -1),
        ("trend_score", -1),
        ("mom_score", -1),
        ("pb_score", -1),
        ("vol_ratio", -1),
    )


# =============================
# 主升浪打分法（pandas离线计算）
# =============================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rolling_slope(s: pd.Series, window: int = 10) -> pd.Series:
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope(y: np.ndarray) -> float:
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return s.rolling(window, min_periods=window).apply(lambda y: slope(np.asarray(y)), raw=False)


def clv(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, pd.NA)
    return (df["Close"] - df["Low"]) / rng


def detect_main_uptrend(df: pd.DataFrame,
                        ma_fast=20, ma_mid=50, ma_slow=200,
                        vol_ma=20, breakout_20=20, breakout_55=55,
                        slope_win=10,
                        pullback_lookback=60,
                        pullback_dd_th=-0.08,
                        pullback_days_th=5,
                        vol_ratio_th=1.2,
                        score_threshold=(4, 2, 2)):
    df = df.copy()

    df["EMA20"] = ema(df["Close"], ma_fast)
    df["EMA50"] = ema(df["Close"], ma_mid)
    df["EMA200"] = ema(df["Close"], ma_slow)

    df["VOL_MA20"] = df["Volume"].rolling(vol_ma, min_periods=vol_ma).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]

    df["ATR14"] = atr(df, 14)
    df["CLV"] = clv(df)

    df["HHV_CLOSE_20"] = df["Close"].rolling(breakout_20, min_periods=breakout_20).max()
    df["HHV_CLOSE_55"] = df["Close"].rolling(breakout_55, min_periods=breakout_55).max()

    df["RET_10"] = df["Close"].pct_change(10)
    df["RET_30"] = df["Close"].pct_change(30)
    df["EMA20_SLOPE"] = rolling_slope(df["EMA20"], slope_win)

    rolling_high = df["Close"].rolling(pullback_lookback, min_periods=pullback_lookback).max()
    df["DD_60"] = df["Close"] / rolling_high - 1.0

    below = (df["Close"] <= df["EMA20"]).astype(int)
    run = [0] * len(below)
    for i in range(len(below)):
        if i == 0:
            run[i] = int(below.iat[i])
        else:
            run[i] = run[i - 1] + 1 if below.iat[i] == 1 else 0
    df["PB_DAYS"] = pd.Series(run, index=df.index).clip(upper=10)

    df["LL20"] = df["Low"].rolling(20, min_periods=20).min()
    df["HL_FLAG"] = (df["LL20"] > df["LL20"].shift(10)).astype(int)

    # TrendScore (0~6)
    t1 = (df["EMA20"] > df["EMA50"]).astype(int)
    t2 = (df["EMA50"] > df["EMA200"]).astype(int)
    t3 = (df["Close"] > df["EMA20"]).astype(int)
    t4 = (df["EMA20_SLOPE"] > 0).astype(int)
    t5 = (df["Close"] >= df["HHV_CLOSE_55"]).astype(int)
    two_days_below = ((df["Close"] < df["EMA20"]) & (df["Close"].shift(1) < df["EMA20"].shift(1))).astype(int)
    t6 = (1 - two_days_below).clip(0, 1).astype(int)
    df["TrendScore"] = t1 + t2 + t3 + t4 + t5 + t6

    # MomentumScore (0~4)
    m1 = (df["Close"] > df["HHV_CLOSE_20"].shift(1)).astype(int)
    m2 = (df["RET_10"] > df["RET_30"]).astype(int)
    m3 = (df["CLV"] > 0.7).astype(int)
    m4 = (df["VOL_RATIO"] > vol_ratio_th).astype(int)
    df["MomScore"] = m1 + m2 + m3 + m4

    # PullbackScore (0~4)
    p1 = (df["DD_60"] > pullback_dd_th).astype(int)
    p2 = (df["PB_DAYS"] <= pullback_days_th).astype(int)
    p3 = (df["Volume"] < df["VOL_MA20"]).astype(int)
    p4 = df["HL_FLAG"].astype(int)
    df["PbScore"] = p1 + p2 + p3 + p4

    ts_th, ms_th, ps_th = score_threshold
    df["is_main_uptrend"] = (
            (df["TrendScore"] >= ts_th) &
            (df["MomScore"] >= ms_th) &
            (df["PbScore"] >= ps_th)
    ).astype(int)

    df["main_uptrend_start"] = ((df["is_main_uptrend"] == 1) & (df["is_main_uptrend"].shift(1) == 0)).astype(int)

    return df


# =============================
# Mode识别器（带缓冲机制）
# =============================
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


# =============================
# 优化版策略主体 v2.1
# =============================
class OptimizedHybrid4ModeV2(bt.Strategy):
    params = dict(
        max_exposure=0.60,
        tranche_targets=(0.30, 0.60, 1.00),
        probe_ratio=0.15,
        breakout_n=20,
        vol_ratio_min=1.2,
        ema_pullback=20,
        pullback_atr_band=1.0,
        rebound_confirm=True,
        add_breakout_n=10,
        add_vol_ratio_min=1.0,
        drawdown_tolerance=0.08,
        chand_period=22,
        chand_atr_mult=2.8,
        atr_period=14,
        stop_loss_pct=8.0,
        profit_take_pct=30.0,
        min_bars_required=210,
        stage_lookback=60,
        slope_win=10,
        high_zone_dd_th=-0.10,
        cross_top_min=12,
        atr_shrink_ratio=0.7,
        dd_drawdown_th=-0.18,
        atrp_drawdown_th=0.09,
        base_zone_dd_th=-0.35,
        base_atrp_th=0.09,
        base_hl_win=20,
        base_hl_shift=10,
        base_hl_consecutive=3,
        base_probe_cooldown=10,
        base_pyramid_profit_th=5.0,
        cooldown_bars=3,
        require_main_uptrend=True,
        allow_entry_in_top_chop=False,
        print_log=False,
    )

    def log(self, txt, show_position=False):
        if self.p.print_log:
            dt = self.datas[0].datetime.date(0)

            # 基本信息
            print(f"{dt} {txt}")

            # 持仓详情（可选）
            if show_position:
                pos_size = int(self.position.size)
                cash = self.broker.cash
                value = self.broker.getvalue()

                if pos_size > 0:
                    avg_price = float(self.position.price)
                    current_price = float(self.data.close[0])
                    position_value = pos_size * current_price
                    pnl = (current_price / avg_price - 1) * 100

                    print(f"   📊 持仓: {pos_size}股 @ 均价${avg_price:.2f} | "
                          f"市值=${position_value:,.0f} | 盈亏{pnl:+.2f}%")
                    print(f"   💰 现金: ${cash:,.0f} | 总资产: ${value:,.0f}")
                else:
                    print(f"   📊 空仓")
                    print(f"   💰 现金: ${cash:,.0f} | 总资产: ${value:,.0f}")

    def __init__(self):
        d = self.datas[0]

        self.atr = bt.ind.ATR(d, period=self.p.atr_period)
        self.ema20 = bt.ind.EMA(d.close, period=20)
        self.ema50 = bt.ind.EMA(d.close, period=50)
        self.ema200 = bt.ind.EMA(d.close, period=200)

        self.hh_chand = bt.ind.Highest(d.high, period=self.p.chand_period)
        self.hhv_entry = bt.ind.Highest(d.close, period=self.p.breakout_n)
        self.hhv_add = bt.ind.Highest(d.close, period=self.p.add_breakout_n)
        self.hh_stage = bt.ind.Highest(d.close, period=self.p.stage_lookback)
        self.ll_base = bt.ind.Lowest(d.low, period=self.p.base_hl_win)

        self.order = None
        self.cooldown = 0
        self.tranche = 0
        self.pb_touched = False
        self.profit_taken = False
        self.base_probe_counter = 0
        self.base_pyramid_count = 0

        self.regime = RegimeDetector(self)
        self.pos_mgr = PositionManager(self)
        self.exit_mgr = ExitManager(self)

        self.rec_dates = []
        self.rec_close = []
        self.rec_equity = []
        self.rec_regime = []
        self.rec_mode_name = []
        self.trade_marks = []

    def next(self):
        d = self.datas[0]
        dt = d.datetime.date(0)

        mode_id, mode_name = self.regime.get_mode()

        self.rec_dates.append(dt)
        self.rec_close.append(float(d.close[0]))
        self.rec_equity.append(float(self.broker.getvalue()))
        self.rec_regime.append(int(mode_id))
        self.rec_mode_name.append(mode_name)

        if self.order:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.base_probe_counter > 0:
            self.base_probe_counter -= 1

        # 持仓：出场责任链
        if self.position:
            # 1) 止损（高波动票会跳过）
            if self.exit_mgr.check_stop_loss(mode_name):
                self.cooldown = self.p.cooldown_bars
                self._reset_state()
                return

            # 2) Regime减仓
            if self.exit_mgr.check_regime_sell(mode_id, mode_name):
                return

            # 3) 分批止盈
            if self.exit_mgr.check_profit_taking(mode_name):
                return

            # 4) Chandelier
            if self.exit_mgr.check_chandelier(mode_name):
                self.cooldown = self.p.cooldown_bars
                self._reset_state()
                return

            # 持仓：加仓
            if mode_name == "DRAWDOWN":
                return

            # BASE金字塔加仓
            if mode_name == "BASE_BUILD":
                if self.base_pyramid_count >= 2:
                    return

                close = float(d.close[0])
                cost = float(self.position.price)
                profit_pct = (close / cost - 1.0) * 100

                if profit_pct >= float(self.p.base_pyramid_profit_th):
                    if self.base_probe_counter == 0:
                        new_ratio = float(self.p.probe_ratio) * (1 + self.base_pyramid_count + 1)
                        self.pos_mgr.scale_to(new_ratio, f"BASE金字塔加仓{self.base_pyramid_count + 1}", mode_name,
                                              "PYRAMID")
                        self.base_pyramid_count += 1
                        self.base_probe_counter = self.p.base_probe_cooldown
                return

            if mode_name != "TREND_RUN":
                return

            if self.p.require_main_uptrend and getattr(d, "is_main_uptrend")[0] < 1:
                return

            if self.tranche >= len(self.p.tranche_targets):
                return

            close = float(d.close[0])
            cost = float(self.position.price)
            if close <= cost * (1.0 - float(self.p.drawdown_tolerance)):
                return

            # 第2档
            if self.tranche == 1:
                ema20 = float(self.ema20[0])
                atrv = float(self.atr[0])
                band = float(self.p.pullback_atr_band) * atrv
                lower, upper = ema20 - band, ema20 + band

                if lower <= float(d.low[0]) <= upper:
                    self.pb_touched = True

                if self.pb_touched:
                    if (not self.p.rebound_confirm) or (close > ema20):
                        if getattr(d, "vol_ratio")[0] >= 1.0:
                            self.pos_mgr.scale_to(self.p.tranche_targets[1], "第2档回踩确认", mode_name, "TRANCHE2")
                            self.tranche = 2
                            self.pb_touched = False
                return

            # 第3档
            if self.tranche == 2:
                if getattr(d, "trend_score")[0] < 4:
                    return

                if getattr(d, "vol_ratio")[0] >= float(self.p.add_vol_ratio_min):
                    if close > float(self.hhv_add[-1]):
                        self.pos_mgr.scale_to(self.p.tranche_targets[2], "第3档再突破", mode_name, "TRANCHE3")
                        self.tranche = 3
                return

            return

        # 空仓：开仓
        if self.cooldown > 0:
            return

        if mode_name == "DRAWDOWN":
            return

        if mode_name == "TOP_CHOP" and (not self.p.allow_entry_in_top_chop):
            return

        # BASE_BUILD试探仓
        if mode_name == "BASE_BUILD":
            if self.base_probe_counter > 0:
                return

            if float(d.close[0]) <= float(self.ema20[0]):
                return
            if getattr(d, "vol_ratio")[0] < 1.0:
                return

            self.tranche = 0
            self.pb_touched = False
            self.profit_taken = False
            self.base_pyramid_count = 0
            self.base_probe_counter = self.p.base_probe_cooldown
            self.pos_mgr.scale_to(float(self.p.probe_ratio), "BASE试探仓", mode_name, "PROBE")
            return

        # TREND_RUN首仓
        if mode_name == "TREND_RUN":
            if self.p.require_main_uptrend and getattr(d, "is_main_uptrend")[0] < 1:
                return

            if getattr(d, "vol_ratio")[0] < float(self.p.vol_ratio_min):
                return

            if float(d.close[0]) <= float(self.hhv_entry[-1]):
                return

            self.tranche = 1
            self.pb_touched = False
            self.profit_taken = False
            self.base_pyramid_count = 0
            self.pos_mgr.scale_to(self.p.tranche_targets[0], "第1档突破首仓", mode_name, "TRANCHE1")
            return

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Completed and order.issell():
                if self.position.size == 0:
                    self._reset_state()
            self.order = None

    def _reset_state(self):
        self.tranche = 0
        self.pb_touched = False
        self.profit_taken = False
        self.base_pyramid_count = 0


# =============================
# 可视化
# =============================
def plot_mode_report(strat, symbol=""):
    dates = pd.to_datetime(strat.rec_dates)
    close = pd.Series(strat.rec_close, index=dates)
    equity = pd.Series(strat.rec_equity, index=dates)
    mode = pd.Series(strat.rec_regime, index=dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(close.index, close.values, label="Close", color='black', linewidth=1)

    colors = {0: 'green', 1: 'orange', 2: 'red', 3: 'blue'}
    labels = {0: 'TREND_RUN', 1: 'TOP_CHOP', 2: 'DRAWDOWN', 3: 'BASE_BUILD'}

    for mode_id in [0, 1, 2, 3]:
        in_block = False
        start = None
        for i in range(len(mode)):
            if mode.iat[i] == mode_id and not in_block:
                in_block = True
                start = mode.index[i]
            if in_block and (i == len(mode) - 1 or mode.iat[i] != mode_id):
                end = mode.index[i]
                ax1.axvspan(start, end, alpha=0.15, color=colors[mode_id], label=labels[mode_id])
                in_block = False

    marker_cfg = {
        ("BUY", "TREND_RUN", "TRANCHE1"): ("^", "green", "T1"),
        ("BUY", "TREND_RUN", "TRANCHE2"): ("^", "lime", "T2"),
        ("BUY", "TREND_RUN", "TRANCHE3"): ("^", "yellow", "T3"),
        ("BUY", "BASE_BUILD", "PROBE"): ("P", "blue", "PROBE"),
        ("BUY", "BASE_BUILD", "PYRAMID"): ("*", "cyan", "PYRA"),
        ("SELL", "STOP_LOSS"): ("v", "purple", "STOP"),
        ("SELL", "PROFIT_TAKE"): ("v", "gold", "PROFIT"),
        ("SELL", "REGIME_CUT"): ("v", "orange", "REGIME"),
        ("SELL", "CHANDELIER"): ("v", "red", "CHAND"),
    }

    groups = {}
    for dt, price, side, mode_name, tag in strat.trade_marks:
        if tag in ["STOP_LOSS", "PROFIT_TAKE", "REGIME_CUT", "CHANDELIER"]:
            key = (side, tag)
        else:
            key = (side, mode_name, tag)
        groups.setdefault(key, {"x": [], "y": []})
        groups[key]["x"].append(pd.to_datetime(dt))
        groups[key]["y"].append(price)

    for key, xy in groups.items():
        cfg = marker_cfg.get(key, ("o", "gray", str(key)))
        mk, color, lbl = cfg
        ax1.scatter(xy["x"], xy["y"], marker=mk, color=color, s=80, label=lbl, zorder=5)

    ax1.set_title(f"{symbol} Price + Mode + Trades (v2.1 票型差异化)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")


# =============================
# 回测入口（使用stock_configs.py）
# =============================
def run_backtest(
        symbol="NVDA",
        use_yfinance=True,
        csv_path=None,
        cash=100000,
        commission=0.0008,
        slippage=0.0005,
        custom_params=None,
        show_config=True
):
    """
    运行回测

    参数:
        symbol: 股票代码
        custom_params: 自定义参数（覆盖配置文件）
        show_config: 是否显示配置信息
    """
    # 1. 加载股票配置
    if CONFIG_LOADED:
        config = get_stock_config(symbol)

        # 显示配置信息
        if show_config:
            print_stock_info(symbol)

        # 检查黑名单
        if config["status"] == "blacklisted":
            print(f"⛔ {symbol} 在黑名单中，停止回测")
            return None, None

        # 获取参数
        params = config.get("params", {})
        category = config.get("category", "medium_vol")

    else:
        # 未加载配置文件，使用默认参数
        print(f"⚠️ 使用默认参数测试 {symbol}")
        params = {
            "stop_loss_pct": 10.0,
            "profit_take_pct": 25.0,
            "vol_ratio_min": 1.2,
            "chand_atr_mult": 2.8,
        }
        category = "unknown"

    # 2. 加载数据
    if use_yfinance:
        df = load_from_yfinance(symbol, start="2020-01-01", end="2026-02-11")
    else:
        if not csv_path:
            raise ValueError("use_yfinance=False 时必须提供 csv_path")
        df = load_from_csv(csv_path)

    # 3. 计算主升浪信号
    df2 = detect_main_uptrend(df, vol_ratio_th=1.2, score_threshold=(4, 2, 2))

    # 4. 准备feed
    df2["is_main_uptrend"] = df2["is_main_uptrend"].fillna(0).astype(int)
    df2["main_uptrend_start"] = df2["main_uptrend_start"].fillna(0).astype(int)
    df2["trend_score"] = df2["TrendScore"].fillna(0).astype(int)
    df2["mom_score"] = df2["MomScore"].fillna(0).astype(int)
    df2["pb_score"] = df2["PbScore"].fillna(0).astype(int)
    df2["vol_ratio"] = df2["VOL_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data = PandasWithSignals(dataname=df2)

    # 5. Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    if slippage and slippage > 0:
        cerebro.broker.set_slippage_perc(slippage)

    # 6. 策略参数（基础参数）
    strategy_params = dict(
        max_exposure=0.60,
        tranche_targets=(0.30, 0.60, 1.00),
        probe_ratio=0.15,
        drawdown_tolerance=0.08,
        stop_loss_pct=10.0,  # 默认值
        profit_take_pct=25.0,  # 默认值
        high_zone_dd_th=-0.10,
        cross_top_min=12,
        atr_shrink_ratio=0.7,
        base_zone_dd_th=-0.35,
        base_atrp_th=0.09,
        base_hl_consecutive=3,
        base_probe_cooldown=10,
        base_pyramid_profit_th=5.0,
        require_main_uptrend=True,
        print_log=True,
    )

    # 7. 应用股票配置
    strategy_params.update(params)

    # 8. 应用自定义参数（最高优先级）
    if custom_params:
        strategy_params.update(custom_params)
        print(f"\n⚙️  应用自定义参数: {custom_params}")

    # 9. 显示最终配置
    print(f"\n{'=' * 60}")
    print(f"📋 {symbol} 回测配置 ({category.upper()})")
    print(f"{'=' * 60}")
    print(f"止损: {strategy_params['stop_loss_pct']}%")
    print(f"止盈: {strategy_params['profit_take_pct']}%")
    print(f"Chandelier: {strategy_params.get('chand_atr_mult', 2.8)}")
    print(f"量能要求: {strategy_params.get('vol_ratio_min', 1.2)}x")
    print(f"{'=' * 60}\n")

    cerebro.addstrategy(OptimizedHybrid4ModeV2, **strategy_params)

    # 10. 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="rets")

    # 11. 运行
    results = cerebro.run()
    strat = results[0]

    # 12. 打印统计
    start = cash
    end = cerebro.broker.getvalue()
    total_return = (end / start - 1) * 100

    dd = strat.analyzers.dd.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    total_closed = trades.get("total", {}).get("closed", 0)
    won = trades.get("won", {}).get("total", 0)
    lost = trades.get("lost", {}).get("total", 0)

    pnl_net = trades.get("pnl", {}).get("net", {}).get("total", 0.0)
    pnl_won = trades.get("won", {}).get("pnl", {}).get("total", 0.0)
    pnl_lost = trades.get("lost", {}).get("pnl", {}).get("total", 0.0)

    winrate = (won / total_closed * 100) if total_closed else 0.0
    profit_factor = (pnl_won / abs(pnl_lost)) if pnl_lost else float("inf")

    print("\n" + "=" * 60)
    print("回测结果 v2.2 (独立配置文件)")
    print("=" * 60)
    print(f"标的: {symbol}")
    print(f"初始资金: ${start:,.2f}")
    print(f"最终资金: ${end:,.2f}")
    print(f"总收益: {total_return:.2f}%")
    print(f"最大回撤: {dd.get('max', {}).get('drawdown', 0.0):.2f}%")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', None)}")
    print(f"总交易次数: {total_closed} | 盈利: {won} | 亏损: {lost} | 胜率: {winrate:.2f}%")
    print(f"净盈亏: ${pnl_net:.2f} | 盈亏比: {profit_factor:.2f}")
    print("=" * 60 + "\n")

    return strat, df2


# =============================
# 批量回测工具
# =============================
def batch_backtest(symbols=None, tier=None, show_details=False):
    """
    批量回测多个股票

    参数:
        symbols: 股票列表，如 ["NVDA", "AAPL"]
        tier: 按评级筛选，如 "S" 表示只测试Tier S
        show_details: 是否显示详细日志
    """
    if not CONFIG_LOADED:
        print("❌ 未加载stock_configs.py，无法批量回测")
        return

    # 确定要测试的股票列表
    if symbols:
        test_symbols = symbols
    elif tier:
        stocks = list_all_stocks(tier=tier)
        test_symbols = list(stocks.keys())
    else:
        stocks = list_all_stocks()
        test_symbols = list(stocks.keys())

    print(f"\n{'=' * 60}")
    print(f"批量回测 - 共{len(test_symbols)}只股票")
    print(f"{'=' * 60}\n")

    results = []

    for symbol in test_symbols:
        print(f"\n{'🔄' * 30}")
        print(f"测试: {symbol}")
        print(f"{'🔄' * 30}\n")

        try:
            # 临时关闭详细日志
            import sys
            from io import StringIO

            if not show_details:
                old_stdout = sys.stdout
                sys.stdout = StringIO()

            strat, _ = run_backtest(symbol, show_config=False)

            if not show_details:
                sys.stdout = old_stdout

            if strat:
                # 提取结果
                trades = strat.analyzers.trades.get_analysis()
                total_closed = trades.get("total", {}).get("closed", 0)
                won = trades.get("won", {}).get("total", 0)

                pnl_won = trades.get("won", {}).get("pnl", {}).get("total", 0.0)
                pnl_lost = trades.get("lost", {}).get("pnl", {}).get("total", 0.0)

                winrate = (won / total_closed * 100) if total_closed else 0.0
                profit_factor = (pnl_won / abs(pnl_lost)) if pnl_lost else 0.0

                final_value = strat.broker.getvalue()
                total_return = (final_value / 100000 - 1) * 100

                dd = strat.analyzers.dd.get_analysis()
                max_dd = dd.get('max', {}).get('drawdown', 0.0)

                results.append({
                    "symbol": symbol,
                    "return": total_return,
                    "win_rate": winrate,
                    "profit_factor": profit_factor,
                    "max_dd": max_dd,
                    "trades": total_closed,
                })

                print(f"✅ {symbol}: 收益{total_return:+.2f}% | 胜率{winrate:.1f}% | 盈亏比{profit_factor:.2f}")

        except Exception as e:
            print(f"❌ {symbol} 测试失败: {e}")

    # 汇总结果
    print(f"\n{'=' * 80}")
    print(f"批量回测汇总")
    print(f"{'=' * 80}")
    print(f"{'股票':<8} {'收益':>8} {'胜率':>8} {'盈亏比':>8} {'回撤':>8} {'交易次数':>10}")
    print(f"{'-' * 80}")

    for r in sorted(results, key=lambda x: x['return'], reverse=True):
        print(f"{r['symbol']:<8} {r['return']:>7.2f}% {r['win_rate']:>7.1f}% "
              f"{r['profit_factor']:>8.2f} {r['max_dd']:>7.2f}% {r['trades']:>10}")

    avg_return = sum(r['return'] for r in results) / len(results) if results else 0
    print(f"{'-' * 80}")
    print(f"平均收益: {avg_return:.2f}%")
    print(f"{'=' * 80}\n")

    return results


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    # 示例1: 测试单个股票
    print("\n" + "🚀" * 30)
    print("示例1: 测试单个股票 (NVDA)")
    print("🚀" * 30)
    run_backtest("NVDA")

    # 示例2: 测试黑名单股票
    # print("\n" + "⛔" * 30)
    # print("示例2: 测试黑名单股票 (WMT)")
    # print("⛔" * 30)
    # run_backtest("WMT")

    # 示例3: 自定义参数
    # print("\n" + "⚙️" * 30)
    # print("示例3: 自定义参数测试 (AAPL)")
    # print("⚙️" * 30)
    # run_backtest("AAPL", custom_params={"stop_loss_pct": 8.0})

    # 示例4: 批量测试Tier S股票
    # print("\n" + "📊" * 30)
    # print("示例4: 批量测试Tier S股票")
    # print("📊" * 30)
    # batch_backtest(tier="S")

    # 示例5: 批量测试指定股票
    # print("\n" + "📊" * 30)
    # print("示例5: 批量测试指定股票")
    # print("📊" * 30)
    # batch_backtest(symbols=["NVDA", "GOOGL", "AAPL"])