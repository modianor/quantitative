# -*- coding: utf-8 -*-
"""优化版四阶段自适应策略 v2.2（重构版）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt

from stock_configs import get_stock_config, print_stock_info, list_all_stocks


class PandasWithSignals(bt.feeds.PandasData):
    lines = ("is_main_uptrend", "main_uptrend_start", "trend_score", "mom_score", "pb_score", "vol_ratio")
    params = tuple((name, -1) for name in lines)


def load_from_yfinance(symbol: str, start=None, end=None, period="5y"):
    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, period=None if (start or end) else period, interval="1d", auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).strip().title() for c in raw.columns]
    return raw[["Open", "High", "Low", "Close", "Volume"]].dropna().rename_axis("Date")


def load_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]
    miss = [c for c in required if c not in cols]
    if miss:
        raise ValueError(f"CSV 缺列: {miss}")
    out = df.rename(columns={cols[k]: k.title() if k != "date" else "Date" for k in required})
    out["Date"] = pd.to_datetime(out["Date"])
    return out.set_index("Date").sort_index()[["Open", "High", "Low", "Close", "Volume"]].dropna()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rolling_slope(s: pd.Series, window: int = 10) -> pd.Series:
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return s.rolling(window, min_periods=window).apply(lambda y: _slope(np.asarray(y)), raw=False)


def detect_main_uptrend(df: pd.DataFrame, vol_ratio_th=1.2, score_threshold=(4, 2, 2)):
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["EMA200"] = ema(out["Close"], 200)
    out["VOL_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["VOL_RATIO"] = out["Volume"] / out["VOL_MA20"]
    out["ATR14"] = atr(out, 14)
    out["CLV"] = (out["Close"] - out["Low"]) / (out["High"] - out["Low"]).replace(0, pd.NA)
    out["HHV_CLOSE_20"] = out["Close"].rolling(20, min_periods=20).max()
    out["HHV_CLOSE_55"] = out["Close"].rolling(55, min_periods=55).max()
    out["RET_10"] = out["Close"].pct_change(10)
    out["RET_30"] = out["Close"].pct_change(30)
    out["EMA20_SLOPE"] = rolling_slope(out["EMA20"], 10)
    out["DD_60"] = out["Close"] / out["Close"].rolling(60, min_periods=60).max() - 1.0

    below = (out["Close"] <= out["EMA20"]).astype(int)
    pb_days = [0] * len(below)
    for i in range(len(below)):
        pb_days[i] = int(below.iat[i]) if i == 0 else pb_days[i - 1] + 1 if below.iat[i] else 0
    out["PB_DAYS"] = pd.Series(pb_days, index=out.index).clip(upper=10)

    out["LL20"] = out["Low"].rolling(20, min_periods=20).min()
    out["HL_FLAG"] = (out["LL20"] > out["LL20"].shift(10)).astype(int)

    t = [
        (out["EMA20"] > out["EMA50"]).astype(int),
        (out["EMA50"] > out["EMA200"]).astype(int),
        (out["Close"] > out["EMA20"]).astype(int),
        (out["EMA20_SLOPE"] > 0).astype(int),
        (out["Close"] >= out["HHV_CLOSE_55"]).astype(int),
        (1 - ((out["Close"] < out["EMA20"]) & (out["Close"].shift(1) < out["EMA20"].shift(1))).astype(int)).clip(0, 1).astype(int),
    ]
    out["TrendScore"] = sum(t)
    out["MomScore"] = (
        (out["Close"] > out["HHV_CLOSE_20"].shift(1)).astype(int)
        + (out["RET_10"] > out["RET_30"]).astype(int)
        + (out["CLV"] > 0.7).astype(int)
        + (out["VOL_RATIO"] > vol_ratio_th).astype(int)
    )
    out["PbScore"] = (
        (out["DD_60"] > -0.08).astype(int)
        + (out["PB_DAYS"] <= 5).astype(int)
        + (out["Volume"] < out["VOL_MA20"]).astype(int)
        + out["HL_FLAG"].astype(int)
    )

    ts_th, ms_th, ps_th = score_threshold
    out["is_main_uptrend"] = ((out["TrendScore"] >= ts_th) & (out["MomScore"] >= ms_th) & (out["PbScore"] >= ps_th)).astype(int)
    out["main_uptrend_start"] = ((out["is_main_uptrend"] == 1) & (out["is_main_uptrend"].shift(1) == 0)).astype(int)
    return out


class RegimeDetector:
    MODE_NAMES = {0: "TREND_RUN", 1: "TOP_CHOP", 2: "DRAWDOWN", 3: "BASE_BUILD"}

    def __init__(self, strategy):
        self.strat = strategy
        self.last_mode = None
        self.mode_counter = 0
        self.mode_buffer_days = 2

    def get_mode(self):
        raw = self._raw_mode()
        if self.last_mode is None:
            self.last_mode = raw
            self.mode_counter = self.mode_buffer_days
            return raw, self.MODE_NAMES[raw]
        self.mode_counter = self.mode_counter + 1 if raw == self.last_mode else 1
        if self.mode_counter >= self.mode_buffer_days:
            self.last_mode = raw
        return self.last_mode, self.MODE_NAMES[self.last_mode]

    def _raw_mode(self) -> int:
        p, s = self.strat.p, self.strat
        if len(s) < p.min_bars_required:
            return 0
        close = float(s.data.close[0])
        hh = max(float(s.hh_stage[0]), 1e-9)
        dd = close / hh - 1.0
        atrp = float(s.atr[0]) / max(close, 1e-9)
        if dd <= p.dd_drawdown_th or atrp >= p.atrp_drawdown_th:
            return 2
        in_high_zone = dd >= p.high_zone_dd_th
        if in_high_zone and (float(s.atr[0]) < self._atr_ma20() * p.atr_shrink_ratio or self._cross_ema20(p.stage_lookback) >= p.cross_top_min):
            return 1
        if dd <= p.base_zone_dd_th and atrp <= p.base_atrp_th and self._higher_lows(p.base_hl_consecutive):
            return 3
        return 0

    def _atr_ma20(self):
        if len(self.strat.atr) < 20:
            return float(self.strat.atr[0])
        return sum(float(self.strat.atr[-i]) for i in range(20)) / 20.0

    def _cross_ema20(self, lookback):
        if len(self.strat) < lookback + 2:
            return 999
        cnt, prev = 0, None
        for i in range(lookback, 0, -1):
            v = 1 if float(self.strat.data.close[-i]) > float(self.strat.ema20[-i]) else 0
            if prev is None:
                prev = v
            elif v != prev:
                cnt += 1
                prev = v
        return cnt

    def _higher_lows(self, consecutive):
        p = self.strat.p
        if len(self.strat.ll_base) < consecutive + p.base_hl_shift:
            return False
        return all(float(self.strat.ll_base[-i]) > float(self.strat.ll_base[-i - p.base_hl_shift]) for i in range(consecutive))


class OptimizedHybrid4ModeV2(bt.Strategy):
    params = dict(
        max_exposure=0.60, tranche_targets=(0.30, 0.60, 1.00), probe_ratio=0.15,
        breakout_n=20, vol_ratio_min=1.2, pullback_atr_band=1.0, rebound_confirm=True,
        add_breakout_n=10, add_vol_ratio_min=1.0, drawdown_tolerance=0.08,
        chand_period=22, chand_atr_mult=2.8, atr_period=14, stop_loss_pct=8.0,
        profit_take_pct=30.0, min_bars_required=210, stage_lookback=60,
        high_zone_dd_th=-0.10, cross_top_min=12, atr_shrink_ratio=0.7,
        dd_drawdown_th=-0.18, atrp_drawdown_th=0.09, base_zone_dd_th=-0.35,
        base_atrp_th=0.09, base_hl_win=20, base_hl_shift=10, base_hl_consecutive=3,
        base_probe_cooldown=10, base_pyramid_profit_th=5.0, cooldown_bars=3,
        require_main_uptrend=True, allow_entry_in_top_chop=False, print_log=False,
    )

    def log(self, text):
        if self.p.print_log:
            print(f"{self.datas[0].datetime.date(0)} {text}")

    def __init__(self):
        d = self.datas[0]
        self.atr = bt.ind.ATR(d, period=self.p.atr_period)
        self.ema20 = bt.ind.EMA(d.close, period=20)
        self.hhv_entry = bt.ind.Highest(d.close, period=self.p.breakout_n)
        self.hhv_add = bt.ind.Highest(d.close, period=self.p.add_breakout_n)
        self.hh_stage = bt.ind.Highest(d.close, period=self.p.stage_lookback)
        self.hh_chand = bt.ind.Highest(d.high, period=self.p.chand_period)
        self.ll_base = bt.ind.Lowest(d.low, period=self.p.base_hl_win)

        self.order = None
        self.cooldown = 0
        self.tranche = 0
        self.pb_touched = False
        self.profit_taken = False
        self.base_probe_counter = 0
        self.base_pyramid_count = 0

        self.regime = RegimeDetector(self)
        self.rec_dates, self.rec_close, self.rec_equity, self.rec_regime, self.rec_mode_name, self.trade_marks = [], [], [], [], [], []

    def _target_size(self, ratio):
        price = float(self.data.close[0])
        if price <= 0:
            return 0
        return int((self.broker.getvalue() * self.p.max_exposure * ratio) // price)

    def _scale_to(self, ratio, reason, mode_name, tag):
        target, current = self._target_size(ratio), int(self.position.size)
        size = target - current
        if size > 0:
            self.log(f"[{mode_name}] {reason} 加仓{size}")
            self.order = self.buy(size=size)
            self.trade_marks.append((self.data.datetime.date(0), float(self.data.close[0]), "BUY", mode_name, tag))

    def _scale_down_to(self, ratio, reason, mode_name, tag):
        target, current = self._target_size(ratio), int(self.position.size)
        size = current - target
        if size > 0:
            self.log(f"[{mode_name}] {reason} 减仓{size}")
            self.order = self.sell(size=size)
            self.trade_marks.append((self.data.datetime.date(0), float(self.data.close[0]), "SELL", mode_name, tag))

    def next(self):
        d = self.datas[0]
        mode_id, mode_name = self.regime.get_mode()

        self.rec_dates.append(d.datetime.date(0)); self.rec_close.append(float(d.close[0])); self.rec_equity.append(float(self.broker.getvalue())); self.rec_regime.append(mode_id); self.rec_mode_name.append(mode_name)
        if self.order:
            return
        self.cooldown = max(0, self.cooldown - 1)
        self.base_probe_counter = max(0, self.base_probe_counter - 1)

        if self.position:
            if self.p.stop_loss_pct < 999:
                low_pnl = (float(d.low[0]) / float(self.position.price) - 1) * 100
                if low_pnl <= -float(self.p.stop_loss_pct):
                    self.order = self.close(); self.cooldown = self.p.cooldown_bars; self._reset_state(); return
            if mode_id == 1 and self.tranche >= 3:
                self._scale_down_to(self.p.tranche_targets[1], "高位横盘减仓", mode_name, "REGIME_CUT"); self.tranche = 2; return
            pnl = (float(d.close[0]) / float(self.position.price) - 1) * 100
            if self.tranche >= 3 and pnl >= float(self.p.profit_take_pct) and not self.profit_taken:
                self.order = self.sell(size=int(self.position.size // 3)); self.profit_taken = True; return
            chand_line = float(self.hh_chand[0]) - float(self.p.chand_atr_mult) * float(self.atr[0])
            if float(d.close[0]) < chand_line:
                self.order = self.close(); self.cooldown = self.p.cooldown_bars; self._reset_state(); return

            if mode_name != "TREND_RUN":
                return
            if self.p.require_main_uptrend and d.is_main_uptrend[0] < 1:
                return
            if self.tranche == 1:
                band = float(self.p.pullback_atr_band) * float(self.atr[0]); ema20 = float(self.ema20[0])
                if ema20 - band <= float(d.low[0]) <= ema20 + band:
                    self.pb_touched = True
                if self.pb_touched and ((not self.p.rebound_confirm) or float(d.close[0]) > ema20) and d.vol_ratio[0] >= 1.0:
                    self._scale_to(self.p.tranche_targets[1], "第2档回踩确认", mode_name, "TRANCHE2"); self.tranche = 2; self.pb_touched = False
            elif self.tranche == 2 and d.trend_score[0] >= 4 and d.vol_ratio[0] >= self.p.add_vol_ratio_min and float(d.close[0]) > float(self.hhv_add[-1]):
                self._scale_to(self.p.tranche_targets[2], "第3档再突破", mode_name, "TRANCHE3"); self.tranche = 3
            return

        if self.cooldown > 0 or mode_name == "DRAWDOWN" or (mode_name == "TOP_CHOP" and not self.p.allow_entry_in_top_chop):
            return

        if mode_name == "BASE_BUILD":
            if self.base_probe_counter == 0 and float(d.close[0]) > float(self.ema20[0]) and d.vol_ratio[0] >= 1.0:
                self._reset_state(); self.base_probe_counter = self.p.base_probe_cooldown
                self._scale_to(self.p.probe_ratio, "BASE试探仓", mode_name, "PROBE")
            return

        if mode_name == "TREND_RUN" and (not self.p.require_main_uptrend or d.is_main_uptrend[0] >= 1) and d.vol_ratio[0] >= self.p.vol_ratio_min and float(d.close[0]) > float(self.hhv_entry[-1]):
            self.tranche = 1
            self._scale_to(self.p.tranche_targets[0], "第1档突破首仓", mode_name, "TRANCHE1")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Completed and order.issell() and self.position.size == 0:
                self._reset_state()
            self.order = None

    def _reset_state(self):
        self.tranche = 0
        self.pb_touched = False
        self.profit_taken = False
        self.base_pyramid_count = 0


def plot_mode_report(strat, symbol=""):
    dates = pd.to_datetime(strat.rec_dates)
    close = pd.Series(strat.rec_close, index=dates)
    mode = pd.Series(strat.rec_regime, index=dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(close.index, close.values, label="Close", color="black", linewidth=1)

    colors = {0: "green", 1: "orange", 2: "red", 3: "blue"}
    labels = {0: "TREND_RUN", 1: "TOP_CHOP", 2: "DRAWDOWN", 3: "BASE_BUILD"}
    for mode_id in [0, 1, 2, 3]:
        in_block, start = False, None
        for i in range(len(mode)):
            if mode.iat[i] == mode_id and not in_block:
                in_block, start = True, mode.index[i]
            if in_block and (i == len(mode) - 1 or mode.iat[i] != mode_id):
                ax1.axvspan(start, mode.index[i], alpha=0.15, color=colors[mode_id], label=labels[mode_id]); in_block = False

    equity = pd.Series(strat.rec_equity, index=dates)
    ax2.plot(equity.index, equity.values, label="Equity", color="purple")
    ax1.set_title(f"{symbol} Price + Mode + Trades")
    ax2.set_title("Equity Curve")
    ax1.legend(loc="upper left", ncol=3)
    ax2.legend(loc="upper left")
    plt.tight_layout()
    return fig


def run_backtest(symbol="NVDA", use_yfinance=True, csv_path=None, cash=100000, commission=0.0008, slippage=0.0005, custom_params=None, show_config=True):
    config = get_stock_config(symbol)
    if show_config:
        print_stock_info(symbol)
    if config["status"] == "blacklisted":
        print(f"⛔ {symbol} 在黑名单中，停止回测")
        return None, None

    params = config.get("params", {}).copy()
    if custom_params:
        params.update(custom_params)

    df = load_from_yfinance(symbol, start="2020-01-01", end="2026-02-11") if use_yfinance else load_from_csv(csv_path)
    df2 = detect_main_uptrend(df)
    for col, src in [("is_main_uptrend", "is_main_uptrend"), ("main_uptrend_start", "main_uptrend_start"), ("trend_score", "TrendScore"), ("mom_score", "MomScore"), ("pb_score", "PbScore")]:
        df2[col] = df2[src].fillna(0).astype(int)
    df2["vol_ratio"] = df2["VOL_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cerebro = bt.Cerebro(); cerebro.adddata(PandasWithSignals(dataname=df2))
    cerebro.broker.setcash(cash); cerebro.broker.setcommission(commission=commission)
    if slippage and slippage > 0:
        cerebro.broker.set_slippage_perc(slippage)

    base_params = dict(max_exposure=0.60, tranche_targets=(0.30, 0.60, 1.00), probe_ratio=0.15, drawdown_tolerance=0.08, stop_loss_pct=10.0, profit_take_pct=25.0, require_main_uptrend=True, print_log=True)
    base_params.update(params)
    cerebro.addstrategy(OptimizedHybrid4ModeV2, **base_params)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]
    start, end = cash, cerebro.broker.getvalue()
    print(f"{symbol} 回测完成: 收益 {(end / start - 1) * 100:.2f}%")
    return strat, df2


def batch_backtest(symbols=None, tier=None, show_details=False):
    test_symbols = symbols or list(list_all_stocks(tier=tier).keys())
    results = []
    for symbol in test_symbols:
        try:
            strat, _ = run_backtest(symbol, show_config=False)
            if strat is None:
                continue
            trades = strat.analyzers.trades.get_analysis()
            total_closed = trades.get("total", {}).get("closed", 0)
            won = trades.get("won", {}).get("total", 0)
            pnl_won = trades.get("won", {}).get("pnl", {}).get("total", 0.0)
            pnl_lost = trades.get("lost", {}).get("pnl", {}).get("total", 0.0)
            winrate = (won / total_closed * 100) if total_closed else 0.0
            profit_factor = (pnl_won / abs(pnl_lost)) if pnl_lost else 0.0
            final_value = strat.broker.getvalue()
            results.append({"symbol": symbol, "return": (final_value / 100000 - 1) * 100, "win_rate": winrate, "profit_factor": profit_factor, "trades": total_closed})
        except Exception as exc:
            print(f"❌ {symbol} 测试失败: {exc}")
    return sorted(results, key=lambda x: x["return"], reverse=True)


if __name__ == "__main__":
    run_backtest("NVDA")
