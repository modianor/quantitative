# -*- coding: utf-8 -*-
"""策略主体模块。"""

import backtrader as bt

from .managers import RegimeDetector, HMMRegimeDetector, PositionManager, ExitManager
from .meta_labeling import MetaLabelingFilter, TradeMetaRecorder

class OptimizedHybrid4ModeV2(bt.Strategy):
    params = dict(
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
        breakout_n=20,
        vol_ratio_min=1.2,
        ema_pullback=20,
        pullback_atr_band=1.0,
        rebound_confirm=True,
        add_breakout_n=10,
        add_vol_ratio_min=1.0,
        drawdown_tolerance=0.08,
        chand_period=22,
        chand_atr_mult=2.8,
        atr_period=14,
        stop_loss_pct=8.0,
        profit_take_pct=30.0,
        min_bars_required=210,
        stage_lookback=60,
        slope_win=10,
        high_zone_dd_th=-0.10,
        cross_top_min=12,
        atr_shrink_ratio=0.7,
        dd_drawdown_th=-0.18,
        atrp_drawdown_th=0.09,
        base_zone_dd_th=-0.35,
        base_atrp_th=0.09,
        base_hl_win=20,
        base_hl_shift=10,
        base_hl_consecutive=3,
        base_probe_cooldown=10,
        base_pyramid_profit_th=5.0,
        cooldown_bars=3,
        require_main_uptrend=True,
        allow_entry_in_top_chop=False,
        use_hmm_regime=True,
        hmm_warmup_bars=240,
        hmm_min_confidence=0.45,
        hmm_mode_buffer_days=2,
        use_meta_labeling=True,
        meta_prob_threshold=0.53,
        meta_min_samples=40,
        meta_retrain_interval=10,
        print_log=False,
        trade_start_date=None,
    )

    def log(self, txt, show_position=False):
        if self.p.print_log:
            dt = self.datas[0].datetime.date(0)

            # 基本信息
            print(f"{dt} {txt}")

            # 持仓详情（可选）
            if show_position:
                pos_size = int(self.position.size)
                cash = self.broker.cash
                value = self.broker.getvalue()

                if pos_size > 0:
                    avg_price = float(self.position.price)
                    current_price = float(self.data.close[0])
                    position_value = pos_size * current_price
                    pnl = (current_price / avg_price - 1) * 100

                    print(f"   📊 持仓: {pos_size}股 @ 均价${avg_price:.2f} | "
                          f"市值=${position_value:,.0f} | 盈亏{pnl:+.2f}%")
                    print(f"   💰 现金: ${cash:,.0f} | 总资产: ${value:,.0f}")
                else:
                    print(f"   📊 空仓")
                    print(f"   💰 现金: ${cash:,.0f} | 总资产: ${value:,.0f}")

    def __init__(self):
        super(OptimizedHybrid4ModeV2, self).__init__()  # 👈 添加这一行

        if not self.datas or self.datas[0] is None:
            raise ValueError("策略初始化失败: 未检测到有效数据源")

        d = self.datas[0]

        # 显式绑定_owner，避免在部分backtrader环境中owner推断失败(NoneType.addindicator)
        # 移除 _owner 参数
        self.atr = bt.ind.ATR(d, period=self.p.atr_period)
        self.ema20 = bt.ind.EMA(d.close, period=20)
        self.ema50 = bt.ind.EMA(d.close, period=50)
        self.ema200 = bt.ind.EMA(d.close, period=200)

        self.hh_chand = bt.ind.Highest(d.high, period=self.p.chand_period)
        self.hhv_entry = bt.ind.Highest(d.close, period=self.p.breakout_n)
        self.hhv_add = bt.ind.Highest(d.close, period=self.p.add_breakout_n)
        self.hh_stage = bt.ind.Highest(d.close, period=self.p.stage_lookback)
        self.ll_base = bt.ind.Lowest(d.low, period=self.p.base_hl_win)

        self.order = None
        self.cooldown = 0
        self.tranche = 0
        self.pb_touched = False
        self.profit_taken = False
        self.base_probe_counter = 0
        self.base_pyramid_count = 0

        self.rule_regime = RegimeDetector(self)
        self.regime = HMMRegimeDetector(self, fallback_detector=self.rule_regime) if self.p.use_hmm_regime else self.rule_regime
        self.pos_mgr = PositionManager(self)
        self.exit_mgr = ExitManager(self)

        self.last_exit_tag = None
        self.last_exit_price = None
        self.meta_filter = MetaLabelingFilter(
            prob_threshold=float(self.p.meta_prob_threshold),
            min_samples=int(self.p.meta_min_samples),
            retrain_interval=int(self.p.meta_retrain_interval),
        )
        self.meta_recorder = TradeMetaRecorder()

        self.rec_dates = []
        self.rec_close = []
        self.rec_equity = []
        self.rec_regime = []
        self.rec_mode_name = []
        self.trade_marks = []


    def _build_meta_features(self, mode_id: int):
        d = self.datas[0]
        close = float(d.close[0])
        atrp = float(self.atr[0]) / max(close, 1e-9)
        vol_ratio = float(getattr(d, "vol_ratio")[0])
        trend_score = float(getattr(d, "trend_score")[0])
        slope = (float(self.ema20[0]) / max(float(self.ema20[-1]), 1e-9) - 1.0) if len(self) > 1 else 0.0
        return [
            float(mode_id),
            atrp,
            vol_ratio,
            trend_score,
            slope,
            float(self.tranche),
        ]

    def _allow_by_meta_filter(self, mode_id: int, signal_tag: str) -> bool:
        if not bool(self.p.use_meta_labeling):
            return True

        features = self._build_meta_features(mode_id)
        allowed, proba = self.meta_filter.allow_signal(features)
        if not allowed:
            self.log(f"[META] 过滤信号 {signal_tag} | 通过概率={proba:.3f}")
            return False

        self.meta_recorder.mark_entry(features, float(self.datas[0].close[0]), signal_tag)
        return True

    def _consume_exit_for_meta(self):
        if not bool(self.p.use_meta_labeling):
            self.last_exit_tag = None
            self.last_exit_price = None
            return

        if self.last_exit_tag is None or self.last_exit_price is None:
            return

        sample = self.meta_recorder.close_trade(self.last_exit_tag, float(self.last_exit_price))
        if sample is not None:
            feature, label = sample
            self.meta_filter.register_sample(feature, label)

        self.last_exit_tag = None
        self.last_exit_price = None

    def next(self):
        d = self.datas[0]
        dt = d.datetime.date(0)

        mode_id, mode_name = self.regime.get_mode()

        self.rec_dates.append(dt)
        self.rec_close.append(float(d.close[0]))
        self.rec_equity.append(float(self.broker.getvalue()))
        self.rec_regime.append(int(mode_id))
        self.rec_mode_name.append(mode_name)

        if self.order:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.base_probe_counter > 0:
            self.base_probe_counter -= 1

        if self.p.trade_start_date is not None and dt < self.p.trade_start_date:
            return

        # 持仓：出场责任链
        if self.position:
            # 1) 止损（高波动票会跳过）
            if self.exit_mgr.check_stop_loss(mode_name):
                self.cooldown = self.p.cooldown_bars
                self._reset_state()
                return

            # 2) Regime减仓
            if self.exit_mgr.check_regime_sell(mode_id, mode_name):
                return

            # 3) 分批止盈
            if self.exit_mgr.check_profit_taking(mode_name):
                return

            # 4) Chandelier
            if self.exit_mgr.check_chandelier(mode_name):
                self.cooldown = self.p.cooldown_bars
                self._reset_state()
                return

            # 持仓：加仓
            if mode_name == "DRAWDOWN":
                return

            # BASE金字塔加仓
            if mode_name == "BASE_BUILD":
                if self.base_pyramid_count >= 2:
                    return

                close = float(d.close[0])
                cost = float(self.position.price)
                profit_pct = (close / cost - 1.0) * 100

                if profit_pct >= float(self.p.base_pyramid_profit_th):
                    if self.base_probe_counter == 0:
                        new_ratio = float(self.p.probe_ratio) * (1 + self.base_pyramid_count + 1)
                        if self._allow_by_meta_filter(mode_id, "PYRAMID"):
                            self.pos_mgr.scale_to(new_ratio, f"BASE金字塔加仓{self.base_pyramid_count + 1}", mode_name,
                                                  "PYRAMID")
                            self.base_pyramid_count += 1
                            self.base_probe_counter = self.p.base_probe_cooldown
                return

            if mode_name != "TREND_RUN":
                return

            if self.p.require_main_uptrend and getattr(d, "is_main_uptrend")[0] < 1:
                return

            if self.tranche >= len(self.p.tranche_targets):
                return

            close = float(d.close[0])
            cost = float(self.position.price)
            if close <= cost * (1.0 - float(self.p.drawdown_tolerance)):
                return

            # 第2档
            if self.tranche == 1:
                ema20 = float(self.ema20[0])
                atrv = float(self.atr[0])
                band = float(self.p.pullback_atr_band) * atrv
                lower, upper = ema20 - band, ema20 + band

                if lower <= float(d.low[0]) <= upper:
                    self.pb_touched = True

                if self.pb_touched:
                    if (not self.p.rebound_confirm) or (close > ema20):
                        if getattr(d, "vol_ratio")[0] >= 1.0:
                            if self._allow_by_meta_filter(mode_id, "TRANCHE2"):
                                self.pos_mgr.scale_to(self.p.tranche_targets[1], "第2档回踩确认", mode_name, "TRANCHE2")
                                self.tranche = 2
                            self.pb_touched = False
                return

            # 第3档
            if self.tranche == 2:
                if getattr(d, "trend_score")[0] < 4:
                    return

                if getattr(d, "vol_ratio")[0] >= float(self.p.add_vol_ratio_min):
                    if close > float(self.hhv_add[-1]):
                        if self._allow_by_meta_filter(mode_id, "TRANCHE3"):
                            self.pos_mgr.scale_to(self.p.tranche_targets[2], "第3档再突破", mode_name, "TRANCHE3")
                            self.tranche = 3
                return

            return

        # 空仓：开仓
        if self.cooldown > 0:
            return

        if mode_name == "DRAWDOWN":
            return

        if mode_name == "TOP_CHOP" and (not self.p.allow_entry_in_top_chop):
            return

        # BASE_BUILD试探仓
        if mode_name == "BASE_BUILD":
            if self.base_probe_counter > 0:
                return

            if float(d.close[0]) <= float(self.ema20[0]):
                return
            if getattr(d, "vol_ratio")[0] < 1.0:
                return

            self.tranche = 0
            self.pb_touched = False
            self.profit_taken = False
            self.base_pyramid_count = 0
            self.base_probe_counter = self.p.base_probe_cooldown
            if not self._allow_by_meta_filter(mode_id, "PROBE"):
                return
            self.pos_mgr.scale_to(float(self.p.probe_ratio), "BASE试探仓", mode_name, "PROBE")
            return

        # TREND_RUN首仓
        if mode_name == "TREND_RUN":
            if self.p.require_main_uptrend and getattr(d, "is_main_uptrend")[0] < 1:
                return

            if getattr(d, "vol_ratio")[0] < float(self.p.vol_ratio_min):
                return

            if float(d.close[0]) <= float(self.hhv_entry[-1]):
                return

            self.tranche = 1
            self.pb_touched = False
            self.profit_taken = False
            self.base_pyramid_count = 0
            if not self._allow_by_meta_filter(mode_id, "TRANCHE1"):
                return
            self.pos_mgr.scale_to(self.p.tranche_targets[0], "第1档突破首仓", mode_name, "TRANCHE1")
            return

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Completed and order.issell():
                self._consume_exit_for_meta()
                if self.position.size == 0:
                    self._reset_state()
            self.order = None

    def _reset_state(self):
        self.tranche = 0
        self.pb_touched = False
        self.profit_taken = False
        self.base_pyramid_count = 0
