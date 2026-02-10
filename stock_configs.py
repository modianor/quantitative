# -*- coding: utf-8 -*-
"""股票配置文件：统一管理标的参数与黑名单。"""

from __future__ import annotations

from typing import Dict, Optional

DEFAULT_PARAMS = {
    "stop_loss_pct": 10.0,
    "profit_take_pct": 25.0,
    "vol_ratio_min": 1.2,
    "chand_atr_mult": 2.8,
    "dd_drawdown_th": -0.18,
}


def _cfg(
    name: str,
    tier: str,
    category: str,
    params: dict,
    performance: dict,
    notes: str,
) -> dict:
    merged = DEFAULT_PARAMS.copy()
    merged.update(params)
    return {
        "name": name,
        "tier": tier,
        "category": category,
        "params": merged,
        "performance": performance,
        "notes": notes,
    }


QUALITY_STOCKS: Dict[str, dict] = {
    "NVDA": _cfg("NVIDIA", "S", "high_vol", {"stop_loss_pct": 999, "profit_take_pct": 40.0, "chand_atr_mult": 2.5, "dd_drawdown_th": -0.22}, {"return": 55.81, "win_rate": 34.78, "profit_factor": 2.59, "max_drawdown": -12.85}, "半导体龙头，AI核心受益股，高波动但趋势明确"),
    "BA": _cfg("Boeing", "S", "high_vol", {"stop_loss_pct": 999, "profit_take_pct": 20.0, "chand_atr_mult": 2.5, "dd_drawdown_th": -0.22}, {"return": 55.81, "win_rate": 34.78, "profit_factor": 2.59, "max_drawdown": -12.85}, "高波动趋势股"),
    "UNH": _cfg("UnitedHealth", "S", "low_vol", {"stop_loss_pct": 5.0, "profit_take_pct": 20.0, "vol_ratio_min": 1.0, "chand_atr_mult": 2.5}, {"return": 55.81, "win_rate": 34.78, "profit_factor": 2.59, "max_drawdown": -12.85}, "低波动大盘医疗"),
    "AMD": _cfg("AMD", "S", "high_vol", {"stop_loss_pct": 999, "profit_take_pct": 40.0, "vol_ratio_min": 0.8, "chand_atr_mult": 2.5, "dd_drawdown_th": -0.22}, {"return": 23.97, "win_rate": 45.83, "profit_factor": 2.03, "max_drawdown": -9.63}, "CPU/GPU双线发展，数据中心增长强劲"),
    "ANET": _cfg("Arista Networks", "S", "high_vol", {"stop_loss_pct": 999, "profit_take_pct": 40.0, "vol_ratio_min": 1.0, "chand_atr_mult": 2.5, "dd_drawdown_th": -0.22}, {"return": 26.60, "win_rate": 32.00, "profit_factor": 1.69, "max_drawdown": -15.50}, "云网络设备，数据中心核心供应商"),
    "GOOGL": _cfg("Alphabet (Class A)", "S", "low_vol", {"stop_loss_pct": 6.0, "profit_take_pct": 20.0, "vol_ratio_min": 1.3, "chand_atr_mult": 3.0, "dd_drawdown_th": -0.15}, {"return": 30.45, "win_rate": 42.86, "profit_factor": 3.79, "max_drawdown": -10.00}, "互联网巨头，搜索+云计算+AI"),
    "GOOG": _cfg("Alphabet (Class C)", "S", "low_vol", {"stop_loss_pct": 6.0, "profit_take_pct": 20.0, "vol_ratio_min": 1.3, "chand_atr_mult": 3.0, "dd_drawdown_th": -0.15}, {"return": 28.84, "win_rate": 53.85, "profit_factor": 4.67, "max_drawdown": -9.50}, "与GOOGL基本一致，无投票权"),
    "ORCL": _cfg("Oracle", "S", "low_vol", {"stop_loss_pct": 6.0, "profit_take_pct": 20.0, "vol_ratio_min": 1.3, "chand_atr_mult": 3.0, "dd_drawdown_th": -0.15}, {"return": 25.34, "win_rate": 50.00, "profit_factor": 3.26, "max_drawdown": -11.00}, "数据库巨头，云转型成功"),
    "AAPL": _cfg("Apple", "S", "medium_vol", {}, {"return": 25.06, "win_rate": 61.54, "profit_factor": 4.23, "max_drawdown": -8.99}, "消费电子+服务，现金流强劲"),
    "AVGO": _cfg("Broadcom", "A", "low_vol", {"stop_loss_pct": 6.0, "profit_take_pct": 20.0, "vol_ratio_min": 1.3, "chand_atr_mult": 3.0, "dd_drawdown_th": -0.15}, {"return": 23.73, "win_rate": 44.44, "profit_factor": 2.47, "max_drawdown": -10.32}, "半导体基础设施，网络+存储芯片"),
    "DELL": _cfg("Dell Technologies", "B", "medium_vol", {}, {"return": 16.17, "win_rate": 35.00, "profit_factor": 1.57, "max_drawdown": -12.00}, "PC+服务器，AI服务器受益"),
    "META": _cfg("Meta Platforms", "C", "medium_vol", {}, {"return": 12.94, "win_rate": 42.11, "profit_factor": 1.73, "max_drawdown": -14.00}, "社交媒体+VR/AR，广告业务稳定"),
    "INTU": _cfg("Intuit", "C", "medium_vol", {}, {"return": 11.62, "win_rate": 37.50, "profit_factor": 1.92, "max_drawdown": -13.00}, "财务软件，小企业服务SaaS"),
    "MSFT": _cfg("Microsoft", "C", "medium_vol", {}, {"return": 10.97, "win_rate": 38.89, "profit_factor": 1.77, "max_drawdown": -12.50}, "云计算+AI，Azure增长强劲"),
    "MU": _cfg("Micron Technology", "C", "medium_vol", {}, {"return": 10.11, "win_rate": 47.37, "profit_factor": 1.53, "max_drawdown": -15.00}, "存储芯片，周期性强但受益AI"),
}

BLACKLIST_CONFIGS: Dict[str, dict] = {
    "INTC": {"reason": "半导体困境股，失去技术领先地位", "category": "cycle"},
    "QCOM": {"reason": "手机芯片周期下行", "category": "cycle"},
    "IBM": {"reason": "传统IT，增长停滞", "category": "legacy"},
    "CSCO": {"reason": "网络设备，增长缓慢", "category": "legacy"},
    "ACN": {"reason": "咨询服务，非产品型", "category": "service"},
    "PANW": {"reason": "网络安全，竞争激烈", "category": "competitive"},
    "AMZN": {"reason": "盈亏比不足，波动大收益低", "category": "low_profit_factor"},
    "ADBE": {"reason": "增长放缓，估值高", "category": "low_profit_factor"},
    "NOW": {"reason": "SaaS估值压力", "category": "low_profit_factor"},
    "SNOW": {"reason": "亏损扩大，估值过高", "category": "unprofitable"},
    "DDOG": {"reason": "竞争加剧，增长放缓", "category": "competitive"},
    "UBER": {"reason": "盈利不稳定", "category": "unprofitable"},
    "PLTR": {"reason": "概念股，基本面不稳", "category": "speculative"},
    "AMAT": {"reason": "半导体设备周期性强", "category": "cycle"},
    "TXN": {"reason": "模拟芯片，增长有限", "category": "mature"},
    "TSM": {"reason": "代工龙头但波动小，不适合策略", "category": "low_volatility"},
    "CRWD": {"reason": "网络安全，竞争激烈", "category": "competitive"},
    "CRM": {"reason": "Salesforce增长放缓", "category": "mature"},
    "ASML": {"reason": "光刻机龙头但波动小", "category": "low_volatility"},
    "LRCX": {"reason": "半导体设备，从+22%崩到+2%", "category": "performance_collapse"},
    "NET": {"reason": "CDN服务，从+20%崩到+1.82%", "category": "performance_collapse"},
    "CDNS": {"reason": "EDA软件，持续亏损-1.87%", "category": "performance_collapse"},
    "WMT": {"reason": "零售防御股，胜率16.67%，盈亏比0.79", "category": "defensive"},
    "NVS": {"reason": "医药防御股，盈亏比0.20（史上最差）", "category": "defensive"},
    "MAR": {"reason": "酒店周期股，胜率28.57%，盈亏比1.10", "category": "cycle"},
    "ONDS": {"reason": "医疗垃圾股，回撤-36.46%，止损率72%", "category": "junk"},
}


def get_stock_config(symbol: str) -> dict:
    symbol = symbol.upper().strip()

    if symbol in BLACKLIST_CONFIGS:
        return {
            "status": "blacklisted",
            "symbol": symbol,
            "reason": BLACKLIST_CONFIGS[symbol]["reason"],
            "category": BLACKLIST_CONFIGS[symbol]["category"],
        }

    if symbol in QUALITY_STOCKS:
        cfg = QUALITY_STOCKS[symbol].copy()
        cfg["status"] = "approved"
        cfg["symbol"] = symbol
        return cfg

    return {
        "status": "unknown",
        "symbol": symbol,
        "category": "medium_vol",
        "params": DEFAULT_PARAMS.copy(),
        "notes": "未测试股票，使用默认MEDIUM_VOL配置",
    }


def list_all_stocks(tier: Optional[str] = None) -> Dict[str, dict]:
    if tier:
        tier = tier.upper().strip()
        return {k: v for k, v in QUALITY_STOCKS.items() if v.get("tier") == tier}
    return QUALITY_STOCKS.copy()


def print_stock_info(symbol: str) -> None:
    config = get_stock_config(symbol)

    print(f"\n{'=' * 60}")
    print(f"股票: {config['symbol']}")
    print(f"{'=' * 60}")

    if config["status"] == "blacklisted":
        print("❌ 状态: 黑名单")
        print(f"原因: {config['reason']}")
        print(f"类别: {config['category']}")
        print("建议: 不建议交易")
    elif config["status"] == "approved":
        print("✅ 状态: 已验证优质股")
        print(f"名称: {config['name']}")
        print(f"评级: Tier {config['tier']}")
        print(f"类别: {config['category'].upper()}")
        print("\n参数配置:")
        for key, value in config["params"].items():
            print(f"  {key}: {value}")
        print("\n历史表现:")
        perf = config["performance"]
        print(f"  收益: {perf['return']:.2f}%")
        print(f"  胜率: {perf['win_rate']:.2f}%")
        print(f"  盈亏比: {perf['profit_factor']:.2f}")
        print(f"  最大回撤: {perf['max_drawdown']:.2f}%")
        print(f"\n备注: {config['notes']}")
    else:
        print("⚠️ 状态: 未测试")
        print(f"类别: 使用默认配置 ({config['category'].upper()})")
        print("\n参数配置:")
        for key, value in config["params"].items():
            print(f"  {key}: {value}")
        print(f"\n备注: {config['notes']}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print_stock_info("NVDA")
    print_stock_info("WMT")
    print_stock_info("MRVL")
    print("Tier S:", ", ".join(sorted(list_all_stocks("S").keys())))
