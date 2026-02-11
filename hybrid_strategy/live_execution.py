# -*- coding: utf-8 -*-
"""实盘执行引擎（示例：Alpaca API）。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import alpaca_trade_api as tradeapi


class LiveExecutionEngine:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.pending_orders = {}
        self.position_tracker = {}
        self.slippage_history = []

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> dict:
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force="day",
                limit_price=limit_price,
                stop_price=stop_price,
            )
            self.pending_orders[order.id] = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "submit_time": datetime.now(),
                "submit_price": self._get_current_price(symbol),
                "status": order.status,
            }
            return {"order_id": order.id, "status": "submitted"}
        except Exception as exc:
            return {"order_id": None, "status": "failed", "error": str(exc)}

    def check_order_status(self, order_id: str) -> dict:
        try:
            order = self.api.get_order(order_id)
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = order.status

            if order.status == "filled" and order_id in self.pending_orders:
                submit_price = self.pending_orders[order_id]["submit_price"]
                filled_price = float(order.filled_avg_price)
                slippage_bps = (filled_price / submit_price - 1) * 10000
                self.slippage_history.append(
                    {
                        "order_id": order_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "submit_price": submit_price,
                        "filled_price": filled_price,
                        "slippage_bps": slippage_bps,
                        "timestamp": order.filled_at,
                    }
                )
                del self.pending_orders[order_id]

            return {
                "order_id": order_id,
                "status": order.status,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                "filled_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def get_position(self, symbol: str) -> Optional[dict]:
        try:
            position = self.api.get_position(symbol)
            return {
                "symbol": symbol,
                "quantity": int(position.qty),
                "avg_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
            }
        except Exception:
            return None

    def _get_current_price(self, symbol: str) -> float:
        try:
            quote = self.api.get_latest_quote(symbol)
            return (float(quote.ask_price) + float(quote.bid_price)) / 2
        except Exception:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)

    def calculate_realized_slippage(self) -> dict:
        if not self.slippage_history:
            return {}
        slippages = [s["slippage_bps"] for s in self.slippage_history]
        return {
            "mean_slippage_bps": float(np.mean(slippages)),
            "median_slippage_bps": float(np.median(slippages)),
            "std_slippage_bps": float(np.std(slippages)),
            "max_slippage_bps": float(max(slippages)),
            "min_slippage_bps": float(min(slippages)),
            "n_orders": len(slippages),
        }


class LiveRiskMonitor:
    def __init__(self, max_position_size: float = 0.30, max_daily_loss: float = 0.05, max_drawdown: float = 0.15):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.daily_start_equity = None
        self.equity_peak = 0
        self.risk_events = []

    def check_pre_trade_risk(self, current_equity: float, proposed_position_value: float, current_positions: Dict[str, dict]) -> dict:
        issues = []
        position_ratio = proposed_position_value / current_equity
        if position_ratio > self.max_position_size:
            issues.append(f"单票仓位过大: {position_ratio:.2%} > {self.max_position_size:.2%}")

        if self.daily_start_equity is None:
            self.daily_start_equity = current_equity
        daily_return = current_equity / self.daily_start_equity - 1
        if daily_return < -self.max_daily_loss:
            issues.append(f"当日亏损超限: {daily_return:.2%}")

        self.equity_peak = max(self.equity_peak, current_equity)
        drawdown = 1 - current_equity / self.equity_peak
        if drawdown > self.max_drawdown:
            issues.append(f"总回撤超限: {drawdown:.2%}")

        total_position_value = sum(p["quantity"] * p["current_price"] for p in current_positions.values())
        concentration = (total_position_value + proposed_position_value) / current_equity
        if concentration > 0.95:
            issues.append(f"持仓集中度过高: {concentration:.2%}")

        if issues:
            self.risk_events.append({"timestamp": datetime.now(), "equity": current_equity, "issues": issues})

        return {
            "approved": len(issues) == 0,
            "issues": issues,
            "position_ratio": position_ratio,
            "daily_return": daily_return,
            "drawdown": drawdown,
        }

    def reset_daily_tracker(self):
        self.daily_start_equity = None
