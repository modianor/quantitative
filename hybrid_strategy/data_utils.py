# -*- coding: utf-8 -*-
"""数据加载与信号计算模块。"""

import numpy as np
import pandas as pd
import backtrader as bt

def load_from_yfinance(symbol: str, start=None, end=None, period="5y"):
    import yfinance as yf

    try:
        if start or end:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        else:
            raw = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
    except Exception as exc:
        raise ValueError(
            f"yfinance 下载失败: symbol={symbol}, start={start}, end={end}, period={period}, error={exc}"
        ) from exc

    if raw is None or raw.empty:
        # 有些环境下指定 start/end 会触发 yfinance 的临时异常，这里自动退回 period 再试一次
        if start or end:
            retry = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if retry is not None and not retry.empty:
                raw = retry

    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance 未返回有效数据: symbol={symbol}, start={start}, end={end}, period={period}"
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [str(c).strip().title() for c in raw.columns]
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

    if raw.empty:
        raise ValueError(
            f"yfinance 数据清洗后为空: symbol={symbol}, start={start}, end={end}, period={period}"
        )

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
    lines = ("is_main_uptrend", "main_uptrend_start", "trend_score", "mom_score", "pb_score", "vol_ratio", "vol_zscore")
    params = (
        ("is_main_uptrend", -1),
        ("main_uptrend_start", -1),
        ("trend_score", -1),
        ("mom_score", -1),
        ("pb_score", -1),
        ("vol_ratio", -1),
        ("vol_zscore", -1),
    )
    # 👇 添加这两行
    def __init__(self):
        super(PandasWithSignals, self).__init__()

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
    df["VOL_STD20"] = df["Volume"].rolling(vol_ma, min_periods=vol_ma).std()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20"]
    df["VOL_ZSCORE"] = (df["Volume"] - df["VOL_MA20"]) / df["VOL_STD20"].replace(0, pd.NA)

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
