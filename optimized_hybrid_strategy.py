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

from hybrid_strategy import (
    CONFIG_LOADED,
    PandasWithSignals,
    RegimeDetector,
    PositionManager,
    ExitManager,
    OptimizedHybrid4ModeV2,
    load_from_yfinance,
    load_from_csv,
    ema,
    atr,
    rolling_slope,
    clv,
    detect_main_uptrend,
    plot_mode_report,
    run_backtest,
    batch_backtest,
)


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
    batch_backtest(symbols=["NVDA", "GOOGL", "AAPL"])
