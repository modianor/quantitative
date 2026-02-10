# -*- coding: utf-8 -*-
"""Walk-forward 稳健性验证工具。"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import backtrader as bt

from .data_utils import load_from_yfinance, detect_main_uptrend, PandasWithSignals
from .backtest import _validate_backtest_data, CONFIG_LOADED
from .strategy import OptimizedHybrid4ModeV2

try:
    from stock_configs import get_stock_config
except ImportError:  # pragma: no cover
    get_stock_config = None


@dataclass
class FoldResult:
    symbol: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_year: int
    train_best_score: float
    val_best_score: float
    test_return: float
    test_max_dd: float
    test_sharpe: float
    trades: int
    win_rate: float
    best_params: Dict[str, float]


def _default_params(symbol: str) -> Dict[str, float]:
    params = {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    }
    if CONFIG_LOADED and get_stock_config:
        cfg = get_stock_config(symbol)
        if cfg.get("status") == "approved":
            params.update(cfg.get("params", {}))
    return params


def _activity_profile_overrides(profile: str) -> Dict[str, float]:
    """通过档位控制交易活跃度。"""
    if profile == "active":
        return {
            "vol_ratio_min": 0.9,
            "add_vol_ratio_min": 0.9,
            "base_probe_cooldown": 5,
            "cooldown_bars": 1,
            "cross_top_min": 10,
            "allow_entry_in_top_chop": True,
            "hmm_min_confidence": 0.40,
            "meta_prob_threshold": 0.50,
            "meta_min_samples": 20,
        }
    return {}


def _strategy_base_params(custom_params: Optional[Dict[str, float]] = None) -> Dict:
    strategy_params = dict(
        max_exposure=0.60,
        use_vol_targeting=True,
        target_vol_annual=0.20,
        vol_lookback=20,
        vol_floor_annual=0.10,
        vol_cap_annual=0.80,
        min_vol_scalar=0.30,
        max_vol_scalar=1.00,
        tranche_targets=(0.30, 0.60, 1.00),
        probe_ratio=0.15,
        drawdown_tolerance=0.08,
        stop_loss_pct=10.0,
        profit_take_pct=25.0,
        high_zone_dd_th=-0.10,
        cross_top_min=12,
        atr_shrink_ratio=0.7,
        base_zone_dd_th=-0.35,
        base_atrp_th=0.09,
        base_hl_consecutive=3,
        base_probe_cooldown=10,
        base_pyramid_profit_th=5.0,
        require_main_uptrend=True,
        use_hmm_regime=True,
        hmm_warmup_bars=240,
        hmm_min_confidence=0.45,
        hmm_mode_buffer_days=2,
        print_log=False,
    )
    if custom_params:
        strategy_params.update(custom_params)
    return strategy_params


def _prepare_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = load_from_yfinance(symbol, start=start, end=end)
    df = detect_main_uptrend(raw, vol_ratio_th=1.2, score_threshold=(4, 2, 2))
    df["is_main_uptrend"] = df["is_main_uptrend"].fillna(0).astype(int)
    df["main_uptrend_start"] = df["main_uptrend_start"].fillna(0).astype(int)
    df["trend_score"] = df["TrendScore"].fillna(0).astype(int)
    df["mom_score"] = df["MomScore"].fillna(0).astype(int)
    df["pb_score"] = df["PbScore"].fillna(0).astype(int)
    df["vol_ratio"] = df["VOL_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _run_slice_backtest(
    df: pd.DataFrame,
    params: Dict[str, float],
    cash: float = 100000,
    trade_start_date: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    min_required_bars = int(params.get("min_bars_required", 210))
    _validate_backtest_data("slice", df, min_required_bars)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0008)
    cerebro.broker.set_slippage_perc(0.0005)

    data = PandasWithSignals(dataname=df)
    cerebro.adddata(data)
    runtime_params = dict(params)
    runtime_params["trade_start_date"] = trade_start_date.date() if trade_start_date is not None else None
    cerebro.addstrategy(OptimizedHybrid4ModeV2, **_strategy_base_params(runtime_params))

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    strat = cerebro.run()[0]

    end_value = cerebro.broker.getvalue()
    total_return = (end_value / cash - 1) * 100
    dd = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0.0)
    sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
    sharpe = float(sharpe) if sharpe is not None else 0.0

    trades = strat.analyzers.trades.get_analysis()
    total_closed = trades.get("total", {}).get("closed", 0)
    won = trades.get("won", {}).get("total", 0)
    win_rate = (won / total_closed * 100) if total_closed else 0.0

    return {
        "return": total_return,
        "max_dd": dd,
        "sharpe": sharpe,
        "trades": total_closed,
        "win_rate": win_rate,
    }


def _candidate_params(base: Dict[str, float]) -> List[Dict[str, float]]:
    stop_base = float(base.get("stop_loss_pct", 10.0))
    profit_base = float(base.get("profit_take_pct", 25.0))
    chand_base = float(base.get("chand_atr_mult", 2.8))
    vol_ratio_base = float(base.get("vol_ratio_min", 1.2))
    dd_base = float(base.get("dd_drawdown_th", -0.18))

    if stop_base >= 900:
        stop_candidates = [999.0]
    else:
        stop_candidates = [max(1.0, stop_base * k) for k in (0.9, 1.0, 1.1)]

    profit_candidates = [max(5.0, profit_base * k) for k in (0.9, 1.0, 1.1)]
    chand_candidates = [max(1.5, chand_base + d) for d in (-0.2, 0.0, 0.2)]
    vol_ratio_candidates = [min(1.8, max(0.6, vol_ratio_base * k)) for k in (0.9, 1.0, 1.1)]
    dd_candidates = [min(-0.10, max(-0.35, dd_base + d)) for d in (-0.03, 0.0, 0.03)]

    candidates: List[Dict[str, float]] = [dict(base)]
    seen: set[Tuple[float, float, float, float, float]] = set()

    for stop in stop_candidates:
        for profit in profit_candidates:
            for chand in chand_candidates:
                for vol_ratio in vol_ratio_candidates:
                    for dd_th in dd_candidates:
                        key = (round(stop, 2), round(profit, 2), round(chand, 2), round(vol_ratio, 2), round(dd_th, 3))
                        if key in seen:
                            continue
                        seen.add(key)
                        p = dict(base)
                        p.update(
                            {
                                "stop_loss_pct": key[0],
                                "profit_take_pct": key[1],
                                "chand_atr_mult": key[2],
                                "vol_ratio_min": key[3],
                                "dd_drawdown_th": key[4],
                            }
                        )
                        candidates.append(p)
    return candidates


def _score_metrics(metrics: Dict[str, float], min_trades: int) -> float:
    ret = metrics["return"]
    dd = metrics["max_dd"]
    sharpe = metrics["sharpe"]
    trades = metrics["trades"]
    score = ret - 1.0 * dd + 6.0 * sharpe + 0.12 * trades
    if trades < min_trades:
        score -= 3.0 * (min_trades - trades)
    return score


def _build_eval_slice(history_df: pd.DataFrame, eval_df: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    warmup_df = history_df.tail(warmup_bars)
    merged = pd.concat([warmup_df, eval_df], axis=0)
    return merged[~merged.index.duplicated(keep="last")]


def walk_forward_validation(
    symbols: List[str],
    start: str = "2018-01-01",
    end: Optional[str] = None,
    train_years: int = 3,
    test_years: int = 1,
    target_years: Optional[List[int]] = None,
    min_trades_train: int = 2,
    val_months: int = 6,
    top_k_train: int = 8,
    activity_profile: str = "balanced",
) -> List[FoldResult]:
    if end is None:
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[INFO] activity_profile={activity_profile}")

    all_results: List[FoldResult] = []
    target_years = target_years or [2022, 2024]

    for symbol in symbols:
        print(f"\n{'=' * 88}\n[WALK-FORWARD] {symbol}\n{'=' * 88}")
        df = _prepare_data(symbol, start, end)
        df = df.sort_index()

        fold_train_start = pd.Timestamp(start)
        last_date = df.index.max()
        base_params = _default_params(symbol)
        base_params.update(_activity_profile_overrides(activity_profile))

        while True:
            fold_train_end = fold_train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
            fold_test_start = fold_train_end + pd.Timedelta(days=1)
            fold_test_end = fold_test_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1)

            if fold_test_end > last_date:
                break

            train_df = df.loc[(df.index >= fold_train_start) & (df.index <= fold_train_end)].copy()
            test_df = df.loc[(df.index >= fold_test_start) & (df.index <= fold_test_end)].copy()

            if len(train_df) < 260 or len(test_df) < 210:
                fold_train_start = fold_train_start + pd.DateOffset(years=test_years)
                continue

            val_start = max(train_df.index.min(), fold_train_end - pd.DateOffset(months=val_months) + pd.Timedelta(days=1))
            train_core_df = train_df.loc[train_df.index < val_start].copy()
            val_df = train_df.loc[train_df.index >= val_start].copy()

            if len(train_core_df) < 210 or len(val_df) < 80:
                train_core_df = train_df
                val_df = train_df.tail(min(126, len(train_df))).copy()

            ranked_train: List[Tuple[float, Dict[str, float]]] = []
            for params in _candidate_params(base_params):
                try:
                    train_metrics = _run_slice_backtest(train_core_df, params)
                except Exception:
                    continue
                train_score = _score_metrics(train_metrics, min_trades=min_trades_train)
                ranked_train.append((train_score, params))

            ranked_train.sort(key=lambda x: x[0], reverse=True)
            shortlist = ranked_train[: max(1, top_k_train)]

            best_train_score = -1e9
            best_val_score = -1e9
            best_params = dict(base_params)

            for train_score, params in shortlist:
                warmup_bars = int(max(params.get("min_bars_required", 210), params.get("hmm_warmup_bars", 240)))
                val_with_warmup = _build_eval_slice(train_core_df, val_df, warmup_bars)
                try:
                    val_metrics = _run_slice_backtest(val_with_warmup, params, trade_start_date=val_df.index.min())
                except Exception:
                    continue

                val_score = _score_metrics(val_metrics, min_trades=max(1, min_trades_train - 1))
                # 参数选择只依据验证期表现，避免把训练分数当作前瞻信号。
                if val_score > best_val_score:
                    best_train_score = train_score
                    best_val_score = val_score
                    best_params = params

            if best_val_score <= -1e9 and shortlist:
                best_train_score, best_params = shortlist[0]
                best_val_score = best_train_score

            warmup_bars = int(max(best_params.get("min_bars_required", 210), best_params.get("hmm_warmup_bars", 240)))
            test_with_warmup = _build_eval_slice(train_df, test_df, warmup_bars)

            test_metrics = _run_slice_backtest(
                test_with_warmup,
                best_params,
                trade_start_date=fold_test_start,
            )
            fold = FoldResult(
                symbol=symbol,
                train_start=fold_train_start.strftime("%Y-%m-%d"),
                train_end=fold_train_end.strftime("%Y-%m-%d"),
                test_start=fold_test_start.strftime("%Y-%m-%d"),
                test_end=fold_test_end.strftime("%Y-%m-%d"),
                test_year=fold_test_start.year,
                train_best_score=best_train_score,
                val_best_score=best_val_score,
                test_return=test_metrics["return"],
                test_max_dd=test_metrics["max_dd"],
                test_sharpe=test_metrics["sharpe"],
                trades=int(test_metrics["trades"]),
                win_rate=test_metrics["win_rate"],
                best_params={
                    "stop_loss_pct": best_params["stop_loss_pct"],
                    "profit_take_pct": best_params["profit_take_pct"],
                    "chand_atr_mult": best_params["chand_atr_mult"],
                    "vol_ratio_min": best_params.get("vol_ratio_min"),
                    "dd_drawdown_th": best_params.get("dd_drawdown_th"),
                },
            )
            all_results.append(fold)

            print(
                f"{fold.test_year}: OOS收益={fold.test_return:+.2f}% | 回撤={fold.test_max_dd:.2f}% | "
                f"Sharpe={fold.test_sharpe:.3f} | 交易={fold.trades} | "
                f"ValScore={fold.val_best_score:.2f} | TrainScore={fold.train_best_score:.2f}"
            )

            fold_train_start = fold_train_start + pd.DateOffset(years=test_years)

    _print_summary(all_results, target_years)
    return all_results


def _print_summary(results: List[FoldResult], target_years: List[int]):
    print(f"\n{'=' * 88}\nWalk-forward 汇总\n{'=' * 88}")
    if not results:
        print("无结果")
        return

    df = pd.DataFrame([r.__dict__ for r in results])

    grouped = df.groupby("test_year", as_index=False).agg(
        avg_return=("test_return", "mean"),
        avg_dd=("test_max_dd", "mean"),
        avg_sharpe=("test_sharpe", "mean"),
        avg_trades=("trades", "mean"),
        folds=("symbol", "count"),
    )

    print("\n按验证年份统计:")
    for _, row in grouped.iterrows():
        print(
            f"{int(row['test_year'])}: 平均OOS收益={row['avg_return']:.2f}% | 平均回撤={row['avg_dd']:.2f}% | "
            f"平均Sharpe={row['avg_sharpe']:.3f} | 平均交易={row['avg_trades']:.1f} | 样本={int(row['folds'])}"
        )

    print("\n目标年份(>=15%)检查:")
    for year in target_years:
        sub = df[df["test_year"] == year]
        if sub.empty:
            print(f"{year}: 无滚动窗口覆盖")
            continue
        avg_ret = sub["test_return"].mean()
        ok = "✅" if avg_ret >= 15 else "❌"
        print(f"{ok} {year}: 平均OOS收益={avg_ret:.2f}%")

    print("\n按股票统计(目标年份):")
    for symbol in sorted(df["symbol"].unique()):
        sub = df[(df["symbol"] == symbol) & (df["test_year"].isin(target_years))]
        if sub.empty:
            continue
        avg_ret = sub["test_return"].mean()
        avg_dd = sub["test_max_dd"].mean()
        print(f"{symbol:<6} 目标年平均收益={avg_ret:>6.2f}% | 目标年平均回撤={avg_dd:>5.2f}%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward 稳健性验证")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "GOOGL", "AAPL", "MSFT", "TSLA"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--target-years", nargs="+", type=int, default=[2022, 2024])
    parser.add_argument("--min-trades-train", type=int, default=2)
    parser.add_argument("--val-months", type=int, default=6)
    parser.add_argument("--top-k-train", type=int, default=8)
    parser.add_argument(
        "--activity-profile",
        choices=["balanced", "active"],
        default="balanced",
        help="交易活跃度档位：active 会放宽入场过滤并缩短冷却。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    walk_forward_validation(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        train_years=args.train_years,
        test_years=args.test_years,
        target_years=args.target_years,
        min_trades_train=args.min_trades_train,
        val_months=args.val_months,
        top_k_train=args.top_k_train,
        activity_profile=args.activity_profile,
    )


if __name__ == "__main__":
    main()
