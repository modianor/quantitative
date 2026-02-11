# -*- coding: utf-8 -*-
"""
优化版四阶段自适应策略 v2.3 - 全自适应学习版
核心改进：
1. 移除手工股票配置文件
2. 参数由策略在运行中在线学习优化
3. 维持统一入口，支持单票和批量回测
"""

from hybrid_strategy import (
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
