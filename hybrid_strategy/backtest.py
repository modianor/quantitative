# -*- coding: utf-8 -*-
"""回测入口与批量测试模块。"""

import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt

from .data_utils import load_from_yfinance, load_from_csv, detect_main_uptrend, PandasWithSignals
from .strategy import OptimizedHybrid4ModeV2

try:
    from stock_configs import get_stock_config, print_stock_info, list_all_stocks

    CONFIG_LOADED = True
except ImportError:
    print("⚠️ 警告: 未找到stock_configs.py，使用内置配置")
    CONFIG_LOADED = False


def _validate_backtest_data(symbol: str, df: pd.DataFrame, min_required_bars: int):
    if df is None or df.empty:
        raise ValueError(f"{symbol} 无可用K线数据，无法回测")

    if len(df) < int(min_required_bars):
        raise ValueError(
            f"{symbol} 数据长度不足: 仅{len(df)}根K线，至少需要{int(min_required_bars)}根"
        )

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
        today = pd.Timestamp.today().normalize()
        end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = load_from_yfinance(symbol, start="2020-01-01", end=end_date)
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

    # 5. Cerebro
    cerebro = bt.Cerebro()
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

    # 回测至少需要足够数据支撑长周期指标（EMA200等）
    min_required_bars = int(strategy_params.get("min_bars_required", 210))
    _validate_backtest_data(symbol, df2, min_required_bars)

    data = PandasWithSignals(dataname=df2)
    cerebro.adddata(data)

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
