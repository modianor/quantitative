# -*- coding: utf-8 -*-
"""
股票配置文件 - stock_configs.py
每个股票独立配置，方便管理和调整
"""

# =============================
# 优质股票配置（已验证）
# =============================

# Tier S - 顶级股票（收益>25%）
NVDA_CONFIG = {
    "name": "NVIDIA",
    "tier": "S",
    "category": "high_vol",
    "params": {
        "stop_loss_pct": 999,  # 禁用止损（高波动股）
        "profit_take_pct": 40.0,  # 高止盈目标
        "vol_ratio_min": 1.2,  # 宽松量能
        "chand_atr_mult": 2.5,  # Chandelier倍数
        "dd_drawdown_th": -0.22,  # 危险期判断
    },
    "performance": {
        "return": 55.81,
        "win_rate": 34.78,
        "profit_factor": 2.59,
        "max_drawdown": -12.85,
    },
    "notes": "半导体龙头，AI核心受益股，高波动但趋势明确"
}

BA_CONFIG = {
    "name": "BA",
    "tier": "S",
    "category": "high_vol",
    "params": {
        "stop_loss_pct": 999,  # 禁用止损（高波动股）
        "profit_take_pct": 20.0,  # 高止盈目标
        "vol_ratio_min": 1.2,  # 宽松量能
        "chand_atr_mult": 2.5,  # Chandelier倍数
        "dd_drawdown_th": -0.22,  # 危险期判断
    },
    "performance": {
        "return": 55.81,
        "win_rate": 34.78,
        "profit_factor": 2.59,
        "max_drawdown": -12.85,
    },
    "notes": "半导体龙头，AI核心受益股，高波动但趋势明确"
}
UNH_CONFIG = {
    "name": "UNH",
    "tier": "S",
    "category": "high_vol",
    "params": {
        "stop_loss_pct": 5,  # 禁用止损（高波动股）
        "profit_take_pct": 20.0,  # 高止盈目标
        "vol_ratio_min": 1.0,  # 宽松量能
        "chand_atr_mult": 2.5,  # Chandelier倍数
        "dd_drawdown_th": -0.22,  # 危险期判断
    },
    "performance": {
        "return": 55.81,
        "win_rate": 34.78,
        "profit_factor": 2.59,
        "max_drawdown": -12.85,
    },
    "notes": "半导体龙头，AI核心受益股，高波动但趋势明确"
}

AMD_CONFIG = {
    "name": "AMD",
    "tier": "S",
    "category": "high_vol",
    "params": {
        "stop_loss_pct": 999,
        "profit_take_pct": 40.0,
        "vol_ratio_min": 0.8,
        "chand_atr_mult": 2.5,
        "dd_drawdown_th": -0.22,
    },
    "performance": {
        "return": 23.97,
        "win_rate": 45.83,
        "profit_factor": 2.03,
        "max_drawdown": -9.63,
    },
    "notes": "CPU/GPU双线发展，数据中心增长强劲"
}

ANET_CONFIG = {
    "name": "Arista Networks",
    "tier": "S",
    "category": "high_vol",
    "params": {
        "stop_loss_pct": 999,
        "profit_take_pct": 40.0,
        "vol_ratio_min": 1.0,
        "chand_atr_mult": 2.5,
        "dd_drawdown_th": -0.22,
    },
    "performance": {
        "return": 26.60,
        "win_rate": 32.00,
        "profit_factor": 1.69,
        "max_drawdown": -15.50,
    },
    "notes": "云网络设备，数据中心核心供应商"
}

GOOGL_CONFIG = {
    "name": "Alphabet (Class A)",
    "tier": "S",
    "category": "low_vol",
    "params": {
        "stop_loss_pct": 6.0,  # 严格止损
        "profit_take_pct": 20.0,  # 快速止盈
        "vol_ratio_min": 1.1,  # 放宽量能，增加触发机会
        "chand_atr_mult": 3.0,  # 宽松Chandelier
        "dd_drawdown_th": -0.15,
    },
    "performance": {
        "return": 30.45,
        "win_rate": 42.86,
        "profit_factor": 3.79,
        "max_drawdown": -10.00,
    },
    "notes": "互联网巨头，搜索+云计算+AI"
}

GOOG_CONFIG = {
    "name": "Alphabet (Class C)",
    "tier": "S",
    "category": "low_vol",
    "params": {
        "stop_loss_pct": 6.0,
        "profit_take_pct": 20.0,
        "vol_ratio_min": 1.1,
        "chand_atr_mult": 3.0,
        "dd_drawdown_th": -0.15,
    },
    "performance": {
        "return": 28.84,
        "win_rate": 53.85,
        "profit_factor": 4.67,
        "max_drawdown": -9.50,
    },
    "notes": "与GOOGL基本一致，无投票权"
}

ORCL_CONFIG = {
    "name": "Oracle",
    "tier": "S",
    "category": "low_vol",
    "params": {
        "stop_loss_pct": 6.0,
        "profit_take_pct": 20.0,
        "vol_ratio_min": 1.1,
        "chand_atr_mult": 3.0,
        "dd_drawdown_th": -0.15,
    },
    "performance": {
        "return": 25.34,
        "win_rate": 50.00,
        "profit_factor": 3.26,
        "max_drawdown": -11.00,
    },
    "notes": "数据库巨头，云转型成功"
}

AAPL_CONFIG = {
    "name": "Apple",
    "tier": "S",
    "category": "medium_vol",
    "params": {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    },
    "performance": {
        "return": 25.06,
        "win_rate": 61.54,
        "profit_factor": 4.23,
        "max_drawdown": -8.99,
    },
    "notes": "消费电子+服务，现金流强劲"
}

# Tier A - 优秀股票（收益20-25%）
AVGO_CONFIG = {
    "name": "Broadcom",
    "tier": "A",
    "category": "low_vol",
    "params": {
        "stop_loss_pct": 6.0,
        "profit_take_pct": 20.0,
        "vol_ratio_min": 1.1,
        "chand_atr_mult": 3.0,
        "dd_drawdown_th": -0.15,
    },
    "performance": {
        "return": 23.73,  # 历史回测
        "win_rate": 44.44,
        "profit_factor": 2.47,
        "max_drawdown": -10.32,
    },
    "notes": "半导体基础设施，网络+存储芯片"
}

# Tier B - 良好股票（收益15-20%）
DELL_CONFIG = {
    "name": "Dell Technologies",
    "tier": "B",
    "category": "medium_vol",
    "params": {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    },
    "performance": {
        "return": 16.17,
        "win_rate": 35.00,
        "profit_factor": 1.57,
        "max_drawdown": -12.00,
    },
    "notes": "PC+服务器，AI服务器受益"
}

# Tier C - 可用股票（收益10-15%）
META_CONFIG = {
    "name": "Meta Platforms",
    "tier": "C",
    "category": "medium_vol",
    "params": {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    },
    "performance": {
        "return": 12.94,
        "win_rate": 42.11,
        "profit_factor": 1.73,
        "max_drawdown": -14.00,
    },
    "notes": "社交媒体+VR/AR，广告业务稳定"
}

INTU_CONFIG = {
    "name": "Intuit",
    "tier": "C",
    "category": "medium_vol",
    "params": {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    },
    "performance": {
        "return": 11.62,
        "win_rate": 37.50,
        "profit_factor": 1.92,
        "max_drawdown": -13.00,
    },
    "notes": "财务软件，小企业服务SaaS"
}

MSFT_CONFIG = {
    "name": "Microsoft",
    "tier": "C",
    "category": "medium_vol",
    "params": {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    },
    "performance": {
        "return": 10.97,
        "win_rate": 38.89,
        "profit_factor": 1.77,
        "max_drawdown": -12.50,
    },
    "notes": "云计算+AI，Azure增长强劲"
}

MU_CONFIG = {
    "name": "Micron Technology",
    "tier": "C",
    "category": "medium_vol",
    "params": {
        "stop_loss_pct": 10.0,
        "profit_take_pct": 25.0,
        "vol_ratio_min": 1.2,
        "chand_atr_mult": 2.8,
        "dd_drawdown_th": -0.18,
    },
    "performance": {
        "return": 10.11,
        "win_rate": 47.37,
        "profit_factor": 1.53,
        "max_drawdown": -15.00,
    },
    "notes": "存储芯片，周期性强但受益AI"
}

# =============================
# 黑名单股票配置
# =============================

BLACKLIST_CONFIGS = {
    # 周期股/困境股
    "INTC": {"reason": "半导体困境股，失去技术领先地位", "category": "cycle"},
    "QCOM": {"reason": "手机芯片周期下行", "category": "cycle"},
    "IBM": {"reason": "传统IT，增长停滞", "category": "legacy"},
    "CSCO": {"reason": "网络设备，增长缓慢", "category": "legacy"},
    "ACN": {"reason": "咨询服务，非产品型", "category": "service"},
    "PANW": {"reason": "网络安全，竞争激烈", "category": "competitive"},

    # 低盈亏比/低胜率科技股
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

    # 表现崩盘
    "LRCX": {"reason": "半导体设备，从+22%崩到+2%", "category": "performance_collapse"},
    "NET": {"reason": "CDN服务，从+20%崩到+1.82%", "category": "performance_collapse"},
    "CDNS": {"reason": "EDA软件，持续亏损-1.87%", "category": "performance_collapse"},

    # 防御股
    "WMT": {"reason": "零售防御股，胜率16.67%，盈亏比0.79", "category": "defensive"},
    "NVS": {"reason": "医药防御股，盈亏比0.20（史上最差）", "category": "defensive"},

    # 周期股
    "MAR": {"reason": "酒店周期股，胜率28.57%，盈亏比1.10", "category": "cycle"},

    # 高回撤/假突破
    "ONDS": {"reason": "医疗垃圾股，回撤-36.46%，止损率72%", "category": "junk"},
}


# =============================
# 配置查询函数
# =============================

def get_stock_config(symbol: str):
    """获取股票配置"""
    # 定义所有优质股票配置
    QUALITY_STOCKS = {
        "NVDA": NVDA_CONFIG,
        "UNH": UNH_CONFIG,
        "BA": BA_CONFIG,
        "AMD": AMD_CONFIG,
        "ANET": ANET_CONFIG,
        "GOOGL": GOOGL_CONFIG,
        "GOOG": GOOG_CONFIG,
        "ORCL": ORCL_CONFIG,
        "AAPL": AAPL_CONFIG,
        "AVGO": AVGO_CONFIG,
        "DELL": DELL_CONFIG,
        "META": META_CONFIG,
        "INTU": INTU_CONFIG,
        "MSFT": MSFT_CONFIG,
        "MU": MU_CONFIG,
    }

    # 检查黑名单
    if symbol in BLACKLIST_CONFIGS:
        return {
            "status": "blacklisted",
            "reason": BLACKLIST_CONFIGS[symbol]["reason"],
            "category": BLACKLIST_CONFIGS[symbol]["category"]
        }

    # 检查优质股票
    if symbol in QUALITY_STOCKS:
        config = QUALITY_STOCKS[symbol].copy()
        config["status"] = "approved"
        config["symbol"] = symbol
        return config

    # 未知股票，返回默认配置
    return {
        "status": "unknown",
        "symbol": symbol,
        "category": "medium_vol",
        "params": {
            "stop_loss_pct": 12.0,
            "profit_take_pct": 30.0,
            "vol_ratio_min": 1.0,
            "chand_atr_mult": 3.0,
            "dd_drawdown_th": -0.20,
        },
        "notes": "未测试股票，使用放宽版MEDIUM_VOL配置（提高交易频率）"
    }


def list_all_stocks(tier=None):
    """列出所有已配置的股票"""
    QUALITY_STOCKS = {
        "NVDA": NVDA_CONFIG,
        "AMD": AMD_CONFIG,
        "ANET": ANET_CONFIG,
        "GOOGL": GOOGL_CONFIG,
        "GOOG": GOOG_CONFIG,
        "ORCL": ORCL_CONFIG,
        "AAPL": AAPL_CONFIG,
        "AVGO": AVGO_CONFIG,
        "DELL": DELL_CONFIG,
        "META": META_CONFIG,
        "INTU": INTU_CONFIG,
        "MSFT": MSFT_CONFIG,
        "MU": MU_CONFIG,
    }

    if tier:
        return {k: v for k, v in QUALITY_STOCKS.items() if v.get("tier") == tier}
    return QUALITY_STOCKS


def print_stock_info(symbol: str):
    """打印股票详细信息"""
    config = get_stock_config(symbol)

    print(f"\n{'=' * 60}")
    print(f"股票: {symbol}")
    print(f"{'=' * 60}")

    if config["status"] == "blacklisted":
        print(f"❌ 状态: 黑名单")
        print(f"原因: {config['reason']}")
        print(f"类别: {config['category']}")
        print(f"建议: 不建议交易")

    elif config["status"] == "approved":
        print(f"✅ 状态: 已验证优质股")
        print(f"名称: {config['name']}")
        print(f"评级: Tier {config['tier']}")
        print(f"类别: {config['category'].upper()}")
        print(f"\n参数配置:")
        for key, value in config['params'].items():
            print(f"  {key}: {value}")
        print(f"\n历史表现:")
        perf = config['performance']
        print(f"  收益: {perf['return']:.2f}%")
        print(f"  胜率: {perf['win_rate']:.2f}%")
        print(f"  盈亏比: {perf['profit_factor']:.2f}")
        print(f"  最大回撤: {perf['max_drawdown']:.2f}%")
        print(f"\n备注: {config['notes']}")

    else:
        print(f"⚠️  状态: 未测试")
        print(f"类别: 使用默认配置 ({config['category'].upper()})")
        print(f"\n参数配置:")
        for key, value in config['params'].items():
            print(f"  {key}: {value}")
        print(f"\n备注: {config['notes']}")

    print(f"{'=' * 60}\n")


# =============================
# 测试代码
# =============================
if __name__ == "__main__":
    # 测试查询
    print("=" * 60)
    print("股票配置系统测试")
    print("=" * 60)

    # 测试优质股
    print_stock_info("NVDA")

    # 测试黑名单
    print_stock_info("WMT")

    # 测试未知股票
    print_stock_info("MRVL")

    # 列出所有Tier S股票
    print("\n" + "=" * 60)
    print("Tier S 股票列表:")
    print("=" * 60)
    tier_s = list_all_stocks(tier="S")
    for symbol, config in tier_s.items():
        perf = config['performance']
        print(f"{symbol:6s} | {config['name']:25s} | 收益: {perf['return']:6.2f}% | 胜率: {perf['win_rate']:5.2f}%")
