# -*- coding: utf-8 -*-
"""Deprecated: 系统已切换为全自适应在线学习，不再使用手工股票配置。"""


def get_stock_config(symbol: str):
    _ = symbol
    return {"status": "adaptive", "category": "adaptive", "params": {}}


def print_stock_info(symbol: str):
    print(f"ℹ️ {symbol}: 全自适应模式（无手工参数）")


def list_all_stocks(tier=None):
    _ = tier
    return {}
