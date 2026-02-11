# -*- coding: utf-8 -*-
"""策略主体模块。"""

import backtrader as bt
from collections import deque

from .managers import RegimeDetector, HMMRegimeDetector, PositionManager, ExitManager
from .meta_labeling import MetaLabelingFilter, TradeMetaRecorder
from .adaptive_profile import StockProfileLearner

class OptimizedHybrid4ModeV2(bt.Strategy):
    """四阶段自适应策略。

    参数说明（均可通过 ``run_backtest(custom_params=...)`` 覆盖）：
    - 仓位控制：``max_exposure``、``tranche_targets``、``probe_ratio``。
    - 波动率目标：``use_vol_targeting`` 与 ``target_vol_annual`` 等。
    - 入场信号：``breakout_n``、``vol_ratio_min``、``ema_pullback`` 等。
    - 出场信号：``stop_loss_pct``、``profit_take_pct``、``chand_atr_mult``。
    - 市场状态机：``use_hmm_regime`` 及 ``hmm_*`` 参数。
    - 元标签过滤：``use_meta_labeling`` 及 ``meta_*`` 参数。
    """
    params = dict(
        # ===== 1) 总体仓位/风险预算 =====
        # 账户最大持仓比例（例如 0.60 表示最多 60% 资金在场内）
        max_exposure=0.90,
        # 是否开启波动率目标仓位缩放（波动高时自动降仓）
        use_vol_targeting=True,
        # 年化目标波动率（仅在 use_vol_targeting=True 时生效）
        target_vol_annual=0.34,
        # 估算近期波动率所用窗口（bar 数）
        vol_lookback=20,
        # 年化波动下限，避免“低波动导致过度放大仓位”
        vol_floor_annual=0.10,
        # 年化波动上限，避免极端行情下缩放异常
        vol_cap_annual=0.80,
        # 波动率缩放因子最小值（最低保留仓位系数）
        min_vol_scalar=0.30,
        # 波动率缩放因子最大值（最高仓位系数）
        max_vol_scalar=1.30,
        # Realized volatility估计方法："close" 或 "parkinson"
        realized_vol_method="close",
        # TREND_RUN 模式下三段加仓目标（占 max_exposure 的比例）
        tranche_targets=(0.35, 0.72, 1.00),
        # BASE_BUILD 探针仓位比例（用于试错小仓位）
        probe_ratio=0.20,

        # ===== 2) 入场相关参数 =====
        # 突破入场窗口（收盘价创新高 N 日）
        breakout_n=20,
        # 主升浪入场最低量比要求（VOL_RATIO >= 该值）
        vol_ratio_min=0.78,
        # 回踩 EMA 周期（用于“突破后回踩确认”）
        ema_pullback=20,
        # 回踩允许偏离 ATR 带宽
        pullback_atr_band=1.0,
        # 回踩后是否需要“反弹确认”再入场
        rebound_confirm=True,
        # 加仓突破窗口（通常短于首次突破）
        add_breakout_n=10,
        # 加仓量比要求（可低于首仓）
        add_vol_ratio_min=0.62,
        # 波段回踩入场最低量比（低于突破入场）
        swing_vol_ratio_min=0.70,
        # 波段回踩入场要求的最低趋势分
        swing_trend_score_min=3,
        # 成交量z-score门槛：过滤无量试探单
        entry_vol_zscore_min=0.2,
        # 趋势质量阈值：用于加仓过滤
        trend_quality_min_tranche2=0.45,
        trend_quality_min_tranche3=0.56,
        # TREND_RUN 首仓趋势质量门槛（抑制震荡期假突破）
        trend_quality_min_entry=0.52,
        # 第2档回踩确认所需连续收盘天数
        tranche2_confirm_bars=2,
        # 波段回踩识别窗口（bar）
        swing_pullback_lookback=8,
        # 入场后可容忍回撤（超过可能减仓/退出）
        drawdown_tolerance=0.10,

        # ===== 3) 出场相关参数 =====
        # Chandelier Exit 最高价回看窗口
        chand_period=22,
        # Chandelier ATR 倍数（越大越“宽松”）
        chand_atr_mult=2.8,
        # 回撤恶化后启用“快速Chandelier”的阈值（相对入场后峰值，百分比）
        fast_exit_drawdown_pct=5.0,
        # 快速Chandelier ATR倍数（通常小于 chand_atr_mult）
        fast_chand_atr_mult=1.9,
        # 是否允许使用当日最低价触发 Chandelier（模拟日内风控）
        chand_use_intraday_low=True,
        # 按ATR%动态调节Chandelier倍数
        dynamic_chand_enabled=True,
        dynamic_chand_atrp_low=0.025,
        dynamic_chand_atrp_high=0.070,
        dynamic_chand_mult_low_vol=1.5,
        dynamic_chand_mult_high_vol=3.0,
        # ATR 指标周期
        atr_period=14,
        # 硬止损阈值（百分比），例如 8.0 表示 -8% 止损
        stop_loss_pct=9.0,
        # 分批止盈阈值（百分比）
        profit_take_pct=42.0,
        # 浮盈达到阈值后将止损抬升至保本（百分比）
        break_even_trigger_pct=4.0,
        # 保本线缓冲（百分比，防止过早扫损）
        break_even_buffer_pct=0.2,
        # 放量急拉后的“趋势保护期”（bar），保护期内放宽出场避免过早卖飞
        burst_guard_bars=4,
        # 触发趋势保护期的最低量比
        burst_vol_ratio_min=1.8,
        # 触发趋势保护期的最低实体强度（|close-open| / ATR）
        burst_body_atr_min=1.0,
        # 趋势保护期中对Chandelier附加放宽（ATR倍数）
        burst_chand_mult_bonus=0.6,
        # 趋势保护期内是否禁用“盘中最低价触发Chandelier”
        burst_disable_intraday_chand=True,
        # 趋势保护期内是否禁用保本止损
        burst_disable_break_even=True,
        # 高波动票风控自适应：降低盘中噪声触发误伤
        high_vol_relax_enabled=True,
        high_vol_intraday_stop_enabled=False,
        high_vol_stop_confirm_bars=2,
        high_vol_stop_loss_widen_pct=1.2,
        high_vol_intraday_stop_buffer_pct=0.6,
        high_vol_disable_intraday_chand=True,
        high_vol_chand_mult_bonus=0.30,
        high_vol_fast_exit_dd_bonus=1.5,
        high_vol_fast_chand_floor_bonus=0.25,
        # 震荡/波段单独参数：见好就收 + 更紧止损
        swing_stop_loss_pct=6.0,
        swing_profit_take_pct=10.0,
        swing_chand_atr_mult=2.0,
        # Regime依赖止损（None=自动回退到默认/波段）
        stop_loss_trend_pct=None,
        stop_loss_chop_pct=5.5,
        stop_loss_drawdown_pct=4.5,
        stop_loss_base_pct=6.5,
        # Regime依赖Chandelier ATR倍数（None=回退到默认/波段）
        chand_atr_mult_trend=None,
        chand_atr_mult_chop=1.8,
        chand_atr_mult_drawdown=1.6,
        chand_atr_mult_base=2.1,
        # 主升浪加仓放大（仅在高质量趋势环境下）
        trend_aggressive_scale=1.15,
        trend_confidence_atrp_max=0.07,

        # ===== 4) 模式识别/切换参数 =====
        # 最低可交易 K 线数量（确保 EMA200 等长周期指标稳定）
        min_bars_required=210,
        # 票型/阶段判断回看窗口
        stage_lookback=60,
        # 趋势斜率计算窗口
        slope_win=10,
        # 高位震荡区阈值（相对高点回撤）
        high_zone_dd_th=-0.10,
        # 高位震荡所需最少“横盘天数/交叉次数”
        cross_top_min=8,
        # ATR 收缩阈值（识别波动收敛）
        atr_shrink_ratio=0.7,
        # DRAWDOWN 区判定：回撤阈值
        dd_drawdown_th=-0.18,
        # DRAWDOWN 区判定：波动率阈值
        atrp_drawdown_th=0.09,
        # BASE_BUILD 区判定：深回撤阈值
        base_zone_dd_th=-0.35,
        # BASE_BUILD 区判定：ATR 百分比阈值
        base_atrp_th=0.09,
        # BASE 结构识别窗口（高低点结构）
        base_hl_win=20,
        # BASE 结构识别位移
        base_hl_shift=10,
        # BASE 结构连续成立次数
        base_hl_consecutive=2,
        # 是否启用“人眼K线形态”辅助判断（震荡 / 回撤中继）
        use_kline_pattern_inference=True,
        # K线形态识别回看窗口
        kline_pattern_lookback=18,
        # 震荡判定：窗口净涨跌幅绝对值上限（百分比）
        kline_chop_net_move_max=0.03,
        # 震荡判定：收盘涨跌方向翻转比例下限
        kline_chop_flip_ratio_min=0.55,
        # 震荡判定：窗口振幅上限（百分比）
        kline_chop_range_max=0.12,
        # 回撤中继判定：相对窗口高点最小回撤（负数）
        kline_pullback_min_dd=-0.12,
        # 回撤中继判定：相对窗口高点最大回撤（负数）
        kline_pullback_max_dd=-0.03,
        # 回撤中继判定：窗口整体仍需保持最小涨幅
        kline_pullback_net_up_min=0.05,
        # BASE 探针加仓冷却（bar）
        base_probe_cooldown=6,
        # BASE 模式金字塔加仓最低盈利门槛（百分比）
        base_pyramid_profit_th=3.0,
        # 是否按R倍数（浮盈/初始风险）动态放大加仓目标
        use_r_multiple_pyramiding=True,
        # R倍数对加仓目标的线性放大系数
        r_multiple_scale=0.20,
        # R倍数放大上限，避免过激进
        r_multiple_cap=2.0,
        # 平仓后冷却 bar 数，避免频繁反复交易
        cooldown_bars=1,
        # 连续试错保护：最近窗口内亏损占比过高时，延长空仓冷却
        reentry_loss_lookback=6,
        reentry_loss_ratio_threshold=0.67,
        reentry_cooldown_bars=4,

        # ===== 4.5) Time-Series Momentum 过滤 =====
        # 是否启用Moskowitz(2012)风格的TSMOM过滤
        use_tsmom_filter=True,
        # 6M/12M收益率中用于regime判定的回看（日）
        tsmom_regime_lookback_short=126,
        tsmom_regime_lookback_long=252,
        # 3M收益率触发（日）
        tsmom_trigger_lookback=63,
        # regime最低收益门槛（short+long均值）
        tsmom_regime_min_return=0.0,
        # trigger最低收益门槛
        tsmom_trigger_min_return=0.0,

        # ===== 5) 交易开关 =====
        # 是否仅在“主升浪信号”为真时允许入场
        require_main_uptrend=False,
        # 是否允许在 TOP_CHOP 模式尝试入场
        allow_entry_in_top_chop=True,

        # ===== 6) HMM Regime 参数 =====
        # 是否启用 HMM 市场状态识别（False 则使用规则引擎）
        use_hmm_regime=True,
        # HMM 热身样本数（不足时自动回退规则引擎）
        hmm_warmup_bars=240,
        # HMM 切换所需最低置信度
        hmm_min_confidence=0.38,
        # 若启用HMM，TREND_RUN开仓/加仓要求的最小趋势后验概率
        hmm_trend_prob_threshold=0.56,
        # HMM 状态切换缓冲天数（防抖）
        hmm_mode_buffer_days=1,
        # 是否按市场后验动态更新HMM转移概率
        hmm_dynamic_transition=True,
        # 动态转移矩阵更新速度
        hmm_transition_lr=0.03,

        # ===== 7) Meta Labeling 参数 =====
        # 是否启用元标签过滤器（过滤低质量入场信号）
        use_meta_labeling=True,
        # 通过信号的最低胜率概率阈值
        meta_prob_threshold=0.36,
        # Meta 2.0 分层决策阈值
        meta_reject_threshold=0.30,
        meta_probe_threshold=0.48,
        meta_half_threshold=0.62,
        # 信号被拒绝后等待bar数
        meta_wait_bars=2,
        # 训练前最少样本数
        meta_min_samples=25,
        # 模型重训练间隔（每 N 笔样本）
        meta_retrain_interval=8,
        # 启用跨资产相对强弱特征（若数据中有benchmark_close）
        use_cross_asset_meta=True,
        # 动态阈值：按市场状态自动放松/收紧过滤（负值=更容易放行）
        meta_dynamic_shift_enabled=True,
        # 全局基础偏移：默认略微降低过滤强度
        meta_base_shift=-0.08,
        # 主升浪环境放松幅度（提高上涨期弹性）
        meta_shift_uptrend_bonus=-0.09,
        # 回撤放大惩罚（继续控回撤）
        meta_shift_drawdown_penalty=0.08,
        # 波动过高惩罚（避免噪声期过度交易）
        meta_shift_vol_penalty=0.05,
        # 动态偏移夹断边界
        meta_shift_min=-0.10,
        meta_shift_max=0.12,
        # 回撤惩罚启动阈值（账户峰值回撤）
        meta_drawdown_penalty_start=0.06,
        # 回撤惩罚饱和阈值（超过该值按满额惩罚）
        meta_drawdown_penalty_full=0.18,

        # ===== 8) 市场环境因子（仅用于放行与阈值调节） =====
        env_min_breadth=0.52,
        env_max_volatility=0.06,
        env_min_liquidity=0.8,
        env_threshold_shift_weak=0.05,

        # ===== 9) 退出分型冷却 & 影子仓 =====
        cooldown_noise_bars=2,
        cooldown_trend_fail_bars=1,
        cooldown_regime_fail_bars=5,
        shadow_horizons=(5, 10, 20),

        # ===== 10) 其他 =====
        adaptive_profile_enabled=True,
        adaptive_profile_lookback=80,
        adaptive_high_vol_threshold=0.45,
        adaptive_confidence_min=0.30,
        adaptive_state_persist_enabled=True,
        adaptive_state_dir=".adaptive_state",
        adaptive_state_version="1",
        adaptive_global_weight=0.60,
        adaptive_stock_weight=0.40,
        adaptive_learning_decay=0.995,
        adaptive_learning_count_decay=0.99,
        adaptive_save_on_trade=False,
        # 是否打印详细日志
        print_log=False,
        # 交易起始日期（早于该日期仅观察不下单）
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
        self.adx = bt.ind.ADX(d, period=14)

        self.hh_chand = bt.ind.Highest(d.high, period=self.p.chand_period)
        self.hhv_entry = bt.ind.Highest(d.close, period=self.p.breakout_n)
        self.hhv_add = bt.ind.Highest(d.close, period=self.p.add_breakout_n)
        self.hh_stage = bt.ind.Highest(d.close, period=self.p.stage_lookback)
        self.ll_base = bt.ind.Lowest(d.low, period=self.p.base_hl_win)

        self.order = None
        self.cooldown = 0
        self.tranche = 0
        self.pb_touched = False
        self.tranche2_confirm_count = 0
        self.profit_taken = False
        self.base_probe_counter = 0
        self.base_pyramid_count = 0
        self.entry_peak_price = 0.0
        self.entry_profile = "NEUTRAL"
        self.current_market_bias = "NEUTRAL"
        self.breakout_guard_remaining = 0
        self.entry_context = None

        self.rule_regime = RegimeDetector(self)
        self.regime = HMMRegimeDetector(self, fallback_detector=self.rule_regime) if self.p.use_hmm_regime else self.rule_regime
        self.pos_mgr = PositionManager(self)
        self.exit_mgr = ExitManager(self)
        self.profile_learner = StockProfileLearner(self)

        self.last_exit_tag = None
        self.last_exit_price = None
        self.last_exit_reason = None
        self.stop_intraday_breach_count = 0
        self.stop_close_breach_count = 0
        self.meta_wait_count = 0
        self.engine_by_mode = {
            "TREND_RUN": "TREND_ENGINE",
            "TOP_CHOP": "RANGE_ENGINE",
            "DRAWDOWN": "RECOVERY_ENGINE",
            "BASE_BUILD": "RECOVERY_ENGINE",
        }
        self.shadow_trades = []
        self.shadow_completed = []
        self.recent_trade_results = deque(maxlen=max(int(getattr(self.p, "reentry_loss_lookback", 6)), 1))
        # 仅在“新增平仓”后评估一次连续试错保护，避免在每个bar重复刷新冷却导致长期冻结
        self.closed_trade_count = 0
        self.reentry_guard_last_applied_trade_count = -1

        self.meta_filter = MetaLabelingFilter(
            prob_threshold=float(self.p.meta_prob_threshold),
            reject_threshold=float(self.p.meta_reject_threshold),
            probe_threshold=float(self.p.meta_probe_threshold),
            half_threshold=float(self.p.meta_half_threshold),
            wait_bars_on_reject=int(self.p.meta_wait_bars),
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
        self.equity_peak = float(self.broker.getvalue())

    def get_adaptive_param(self, name: str, base_value: float):
        learner = getattr(self, "profile_learner", None)
        if learner is None:
            return float(base_value)
        return float(learner.get_adjustment(name, float(base_value)))

    def _tsmom_snapshot(self) -> dict:
        if not bool(getattr(self.p, "use_tsmom_filter", True)):
            return {"pass_regime": True, "pass_trigger": True, "regime_return": 0.0, "trigger_return": 0.0}

        d = self.datas[0]
        lb_s = int(getattr(self.p, "tsmom_regime_lookback_short", 126))
        lb_l = int(getattr(self.p, "tsmom_regime_lookback_long", 252))
        lb_t = int(getattr(self.p, "tsmom_trigger_lookback", 63))
        if len(self) <= max(lb_s, lb_l, lb_t):
            return {"pass_regime": False, "pass_trigger": False, "regime_return": 0.0, "trigger_return": 0.0}

        close = float(d.close[0])
        r6 = close / max(float(d.close[-lb_s]), 1e-9) - 1.0
        r12 = close / max(float(d.close[-lb_l]), 1e-9) - 1.0
        r3 = close / max(float(d.close[-lb_t]), 1e-9) - 1.0

        regime_ret = 0.5 * (r6 + r12)
        regime_min = float(getattr(self.p, "tsmom_regime_min_return", 0.0))
        trigger_min = float(getattr(self.p, "tsmom_trigger_min_return", 0.0))
        return {
            "pass_regime": regime_ret >= regime_min,
            "pass_trigger": r3 >= trigger_min,
            "regime_return": regime_ret,
            "trigger_return": r3,
        }

    def _allow_by_hmm_trend_prob(self, mode_name: str) -> bool:
        if mode_name != "TREND_RUN" or not bool(getattr(self.p, "use_hmm_regime", True)):
            return True

        base_threshold = float(getattr(self.p, "hmm_trend_prob_threshold", 0.70))
        threshold = self.get_adaptive_param("hmm_trend_prob_threshold", base_threshold)
        if threshold <= 0:
            return True

        trend_prob = getattr(self.regime, "get_trend_probability", lambda: 1.0)()
        return float(trend_prob) >= threshold

    def is_high_vol_stock(self) -> bool:
        """根据票型学习器识别高波动股票。"""
        if not bool(getattr(self.p, "high_vol_relax_enabled", True)):
            return False

        learner = getattr(self, "profile_learner", None)
        if learner is None:
            return False

        annual_vol = float(getattr(learner, "metrics", {}).get("annual_vol", 0.0))
        vol_th = float(getattr(self.p, "adaptive_high_vol_threshold", 0.45))
        archetype = str(getattr(learner, "archetype", "UNKNOWN"))
        conf = float(getattr(learner, "confidence", 0.0))
        conf_min = float(getattr(self.p, "adaptive_confidence_min", 0.30))

        if annual_vol >= vol_th:
            return True
        if conf >= conf_min and archetype in {"HIGH_BETA_GROWTH", "CYCLICAL"}:
            return True
        return False

    def _r_multiple_scaled_ratio(self, base_ratio: float, mode_name: str) -> float:
        if not bool(getattr(self.p, "use_r_multiple_pyramiding", True)) or not self.position:
            return float(base_ratio)

        close = float(self.datas[0].close[0])
        cost = float(self.position.price)
        if cost <= 0:
            return float(base_ratio)

        if mode_name == "TOP_CHOP":
            stop_pct = float(getattr(self.p, "swing_stop_loss_pct", 6.0))
        else:
            stop_pct = float(getattr(self.p, "stop_loss_pct", 8.0))
        stop_pct = max(stop_pct, 1e-6)

        r_multiple = ((close / cost - 1.0) * 100.0) / stop_pct
        cap = float(getattr(self.p, "r_multiple_cap", 2.0))
        scale = float(getattr(self.p, "r_multiple_scale", 0.20))
        booster = 1.0 + max(0.0, min(r_multiple, cap)) * scale
        return float(base_ratio) * booster


    def _trend_quality_score(self) -> float:
        d = self.datas[0]
        close = float(d.close[0])
        if close <= 0:
            return 0.0

        adx_raw = float(self.adx[0]) if len(self.adx) > 0 else 0.0
        adx_norm = min(max((adx_raw - 15.0) / 25.0, 0.0), 1.0)

        slope = 0.0
        if len(self) > int(self.p.slope_win):
            slope = float(self.ema20[0]) / max(float(self.ema20[-int(self.p.slope_win)]), 1e-9) - 1.0
        slope_norm = min(max((slope * 100.0) / 3.0, 0.0), 1.0)

        trend_score_norm = min(max(float(getattr(d, "trend_score")[0]) / 6.0, 0.0), 1.0)

        atr_ma = sum(float(self.atr[-i]) for i in range(min(len(self.atr), 20))) / max(min(len(self.atr), 20), 1)
        atr_ratio = float(self.atr[0]) / max(atr_ma, 1e-9)
        atr_penalty = min(max((atr_ratio - 1.0) / 0.6, 0.0), 1.0)

        score = 0.35 * adx_norm + 0.25 * slope_norm + 0.30 * trend_score_norm + 0.10 * (1.0 - atr_penalty)
        return min(max(score, 0.0), 1.0)

    def _vol_zscore(self) -> float:
        d = self.datas[0]
        if hasattr(d, "vol_zscore"):
            return float(getattr(d, "vol_zscore")[0])
        return 0.0

    def _build_meta_features(self, mode_id: int):
        d = self.datas[0]
        close = float(d.close[0])
        atrp = float(self.atr[0]) / max(close, 1e-9)
        vol_ratio = float(getattr(d, "vol_ratio")[0])
        trend_score = float(getattr(d, "trend_score")[0])
        slope = (float(self.ema20[0]) / max(float(self.ema20[-1]), 1e-9) - 1.0) if len(self) > 1 else 0.0
        relative_strength = 0.0
        if bool(getattr(self.p, "use_cross_asset_meta", True)) and hasattr(d, "benchmark_close") and len(self) > 1:
            b0 = float(getattr(d, "benchmark_close")[0])
            b1 = float(getattr(d, "benchmark_close")[-1])
            if b0 > 0 and b1 > 0:
                asset_ret = close / max(float(d.close[-1]), 1e-9) - 1.0
                bench_ret = b0 / b1 - 1.0
                relative_strength = asset_ret - bench_ret
        return [
            float(mode_id),
            atrp,
            vol_ratio,
            trend_score,
            slope,
            float(self.tranche),
            relative_strength,
        ]

    def _market_environment_snapshot(self) -> dict:
        d = self.datas[0]
        close = float(d.close[0])
        prev_close = float(d.close[-1]) if len(self) > 1 else close
        ret = close / max(prev_close, 1e-9) - 1.0

        breadth = 0.5
        if hasattr(d, "mom_score"):
            breadth = float(getattr(d, "mom_score")[0]) / 4.0
        elif hasattr(d, "trend_score"):
            breadth = float(getattr(d, "trend_score")[0]) / 6.0

        volatility = abs(ret)
        if close > 0:
            volatility = max(volatility, float(self.atr[0]) / close)

        liquidity = float(getattr(d, "vol_ratio")[0]) if hasattr(d, "vol_ratio") else 1.0
        return {
            "breadth": breadth,
            "volatility": volatility,
            "liquidity": liquidity,
            "is_weak": (
                breadth < float(self.p.env_min_breadth)
                or volatility > float(self.p.env_max_volatility)
                or liquidity < float(self.p.env_min_liquidity)
            ),
        }

    def _record_shadow_trade(self, signal_tag: str, mode_name: str, proba: float):
        d = self.datas[0]
        self.shadow_trades.append(
            {
                "entry_index": len(self),
                "entry_price": float(d.close[0]),
                "signal_tag": signal_tag,
                "mode_name": mode_name,
                "meta_proba": float(proba),
                "results": {},
            }
        )

    def _update_shadow_trades(self):
        if not self.shadow_trades:
            return

        close = float(self.datas[0].close[0])
        keep = []
        max_h = max(tuple(self.p.shadow_horizons))
        for trade in self.shadow_trades:
            age = len(self) - int(trade["entry_index"])
            for h in tuple(self.p.shadow_horizons):
                if age >= int(h) and h not in trade["results"]:
                    entry = float(trade["entry_price"])
                    trade["results"][h] = close / max(entry, 1e-9) - 1.0
            if age >= max_h:
                self.shadow_completed.append(trade)
            else:
                keep.append(trade)
        self.shadow_trades = keep

    def _active_engine(self, mode_name: str) -> str:
        return self.engine_by_mode.get(mode_name, "TREND_ENGINE")

    def _market_bias_profile(self, mode_name: str) -> str:
        """基于日线判断当前更像主升浪还是震荡波段。"""
        d = self.datas[0]
        close = float(d.close[0])
        atrp = float(self.atr[0]) / max(close, 1e-9)
        trend_score = float(getattr(d, "trend_score")[0])
        mom_score = float(getattr(d, "mom_score")[0]) if hasattr(d, "mom_score") else 0.0
        ema20 = float(self.ema20[0])
        ema50 = float(self.ema50[0])
        ema200 = float(self.ema200[0])

        archetype = getattr(getattr(self, "profile_learner", None), "archetype", "UNKNOWN")
        stage = getattr(getattr(self, "profile_learner", None), "stage", "UNKNOWN")

        strong_trend = (
            mode_name == "TREND_RUN"
            and trend_score >= 4
            and mom_score >= 2
            and close > ema20 > ema50 > ema200
            and atrp <= float(self.p.trend_confidence_atrp_max)
        ) or (archetype in {"TREND_LEADER", "HIGH_BETA_GROWTH"} and stage == "MARKUP")
        if strong_trend:
            return "MAIN_UPTREND"

        range_like = (
            mode_name == "TOP_CHOP"
            or close <= ema20
            or ema20 <= ema50
            or trend_score <= float(self.p.swing_trend_score_min)
            or archetype in {"RANGE_BOUND", "DISTRIBUTION", "CHOPPY", "CYCLICAL"}
            or stage in {"SIDEWAYS", "RANGE", "DISTRIBUTION", "MARKDOWN", "ACCUMULATION"}
        )
        if range_like:
            return "SWING_CHOP"

        return "NEUTRAL"

    def _meta_advice(self, mode_id: int, signal_tag: str, mode_name: str) -> dict:
        if not bool(self.p.use_meta_labeling):
            return {"allow": True, "size_multiplier": 1.0, "wait_bars": 0, "proba": 0.5, "tier": "OFF"}

        features = self._build_meta_features(mode_id)
        env = self._market_environment_snapshot()
        threshold_shift = self._adaptive_meta_threshold_shift(env, mode_name)
        advice = self.meta_filter.advise_signal(features, threshold_shift=threshold_shift)

        if not advice["allow"]:
            self.log(f"[META] 过滤信号 {signal_tag} | 概率={advice['proba']:.3f} | 分层={advice['tier']}")
            self.meta_wait_count = max(int(self.meta_wait_count), int(advice.get("wait_bars", 0)))
            self._record_shadow_trade(signal_tag, mode_name, advice["proba"])
            return advice

        self.meta_recorder.mark_entry(features, float(self.datas[0].close[0]), signal_tag)
        return advice

    def _adaptive_meta_threshold_shift(self, env: dict, mode_name: str) -> float:
        """动态调整 Meta 过滤阈值。

        目标：
        - 上涨期适度放松（拿回弹性）
        - 回撤/高波动期自动收紧（优先控回撤）
        """
        if not bool(getattr(self.p, "meta_dynamic_shift_enabled", True)):
            return float(self.p.env_threshold_shift_weak) if env.get("is_weak", False) else 0.0

        shift = float(getattr(self.p, "meta_base_shift", -0.03))

        # 1) 主升浪奖励：在高景气趋势里适度降低过滤阈值
        if self.current_market_bias == "MAIN_UPTREND" and mode_name == "TREND_RUN":
            shift += float(getattr(self.p, "meta_shift_uptrend_bonus", -0.04))

        # 2) 回撤惩罚：账户回撤越深，阈值越严格
        eq = float(self.broker.getvalue())
        peak = max(float(getattr(self, "equity_peak", eq)), 1e-9)
        drawdown = max(0.0, 1.0 - eq / peak)
        dd_start = float(getattr(self.p, "meta_drawdown_penalty_start", 0.06))
        dd_full = float(getattr(self.p, "meta_drawdown_penalty_full", 0.18))
        if drawdown > dd_start:
            denom = max(dd_full - dd_start, 1e-9)
            dd_score = min(1.0, (drawdown - dd_start) / denom)
            shift += dd_score * float(getattr(self.p, "meta_shift_drawdown_penalty", 0.08))

        # 3) 波动惩罚：短期波动越高，阈值越严格
        max_vol = max(float(getattr(self.p, "env_max_volatility", 0.06)), 1e-9)
        vol_score = max(0.0, float(env.get("volatility", 0.0)) / max_vol - 1.0)
        shift += min(1.0, vol_score) * float(getattr(self.p, "meta_shift_vol_penalty", 0.05))

        # 4) 弱环境保守补偿
        if env.get("is_weak", False):
            shift += float(getattr(self.p, "env_threshold_shift_weak", 0.05))

        shift_min = float(getattr(self.p, "meta_shift_min", -0.10))
        shift_max = float(getattr(self.p, "meta_shift_max", 0.12))
        return min(max(shift, shift_min), shift_max)

    def _apply_exit_cooldown(self):
        reason = self.last_exit_reason
        if reason == "REGIME_FAIL":
            self.cooldown = int(self.p.cooldown_regime_fail_bars)
        elif reason == "TREND_FAIL":
            self.cooldown = int(self.p.cooldown_trend_fail_bars)
        elif reason == "NOISE":
            self.cooldown = int(self.p.cooldown_noise_bars)
        else:
            self.cooldown = int(self.p.cooldown_bars)

    def _consume_exit_for_meta(self):
        if not bool(self.p.use_meta_labeling):
            self.last_exit_tag = None
            self.last_exit_price = None
            self.last_exit_reason = None
            return

        if self.last_exit_tag is None or self.last_exit_price is None:
            return

        sample = self.meta_recorder.close_trade(self.last_exit_tag, float(self.last_exit_price))
        if sample is not None:
            feature, label = sample
            self.meta_filter.register_sample(feature, label)

        self.last_exit_tag = None
        self.last_exit_price = None
        self.last_exit_reason = None

    def _reentry_guard_active(self, mode_name: str) -> bool:
        """在震荡试错密集期自动延长冷却，降低无效交易密度。"""
        if mode_name != "TREND_RUN":
            return False

        # 未产生新的平仓结果时，不重复触发同一轮保护，避免cooldown被每根K线重置
        if self.reentry_guard_last_applied_trade_count == self.closed_trade_count:
            return False

        window = max(int(getattr(self.p, "reentry_loss_lookback", 6)), 1)
        threshold = float(getattr(self.p, "reentry_loss_ratio_threshold", 0.67))
        if len(self.recent_trade_results) < window:
            return False

        losses = sum(1 for is_win in self.recent_trade_results if not is_win)
        loss_ratio = losses / max(len(self.recent_trade_results), 1)
        if loss_ratio < threshold:
            return False

        self.cooldown = max(self.cooldown, int(getattr(self.p, "reentry_cooldown_bars", 4)))
        self.reentry_guard_last_applied_trade_count = self.closed_trade_count
        self.log(f"[RISK] 连续试错保护触发 | 近{len(self.recent_trade_results)}笔亏损占比={loss_ratio:.2%}，延长冷却")
        return True

    def next(self):
        d = self.datas[0]
        dt = d.datetime.date(0)

        if hasattr(self, "profile_learner"):
            self.profile_learner.update()

        mode_id, mode_name = self.regime.get_mode()
        active_engine = self._active_engine(mode_name)
        self.current_market_bias = self._market_bias_profile(mode_name)
        self._update_shadow_trades()

        self.rec_dates.append(dt)
        self.rec_close.append(float(d.close[0]))
        self.rec_equity.append(float(self.broker.getvalue()))
        self.rec_regime.append(int(mode_id))
        self.rec_mode_name.append(mode_name)
        self.equity_peak = max(float(self.equity_peak), float(self.broker.getvalue()))

        if self.order:
            return

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.position and self.breakout_guard_remaining > 0:
            self.breakout_guard_remaining -= 1

        if self.base_probe_counter > 0:
            self.base_probe_counter -= 1

        if self.meta_wait_count > 0:
            self.meta_wait_count -= 1

        if self.p.trade_start_date is not None and dt < self.p.trade_start_date:
            return

        # 持仓：出场责任链
        if self.position:
            self.entry_peak_price = max(self.entry_peak_price, float(d.close[0]))

            # 1) 止损（高波动票会跳过）
            if self.exit_mgr.check_stop_loss(mode_name):
                self._apply_exit_cooldown()
                self._reset_state()
                return

            # 1.5) 浮盈后保本止损
            if self.exit_mgr.check_break_even(mode_name):
                self._apply_exit_cooldown()
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
                self._apply_exit_cooldown()
                self._reset_state()
                return

            # 持仓：加仓（由当前引擎独占发言权）
            if active_engine == "RECOVERY_ENGINE" and mode_name == "DRAWDOWN":
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
                        new_ratio = self._r_multiple_scaled_ratio(new_ratio, mode_name)
                        advice = self._meta_advice(mode_id, "PYRAMID", mode_name)
                        if advice["allow"]:
                            adj_ratio = new_ratio * float(advice["size_multiplier"])
                            self.pos_mgr.scale_to(adj_ratio, f"BASE金字塔加仓{self.base_pyramid_count + 1}", mode_name,
                                                  "PYRAMID")
                            self.base_pyramid_count += 1
                            self.base_probe_counter = self.p.base_probe_cooldown
                return

            if mode_name != "TREND_RUN":
                return

            tsmom = self._tsmom_snapshot()
            if not (tsmom["pass_regime"] and tsmom["pass_trigger"]):
                return

            if not self._allow_by_hmm_trend_prob(mode_name):
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

                tq = self._trend_quality_score()
                if tq < float(getattr(self.p, "trend_quality_min_tranche2", 0.60)):
                    self.tranche2_confirm_count = 0
                    return

                if self.pb_touched:
                    rebound_ok = (not self.p.rebound_confirm) or (close > ema20)
                    volume_ok = getattr(d, "vol_ratio")[0] >= 1.0 and self._vol_zscore() >= float(getattr(self.p, "entry_vol_zscore_min", 0.5))
                    if rebound_ok and volume_ok:
                        self.tranche2_confirm_count += 1
                    else:
                        self.tranche2_confirm_count = 0

                    if self.tranche2_confirm_count >= int(getattr(self.p, "tranche2_confirm_bars", 2)):
                        advice = self._meta_advice(mode_id, "TRANCHE2", mode_name)
                        if advice["allow"]:
                            target_ratio = self._r_multiple_scaled_ratio(float(self.p.tranche_targets[1]), mode_name)
                            target_ratio = target_ratio * float(advice["size_multiplier"])
                            self.pos_mgr.scale_to(target_ratio, "第2档回踩确认", mode_name, "TRANCHE2")
                            self.tranche = 2
                        self.pb_touched = False
                        self.tranche2_confirm_count = 0
                return

            # 第3档
            if self.tranche == 2:
                if getattr(d, "trend_score")[0] < 4:
                    return

                if self._trend_quality_score() < float(getattr(self.p, "trend_quality_min_tranche3", 0.72)):
                    return

                add_vol_ratio_min = self.get_adaptive_param("add_vol_ratio_min", float(self.p.add_vol_ratio_min))
                volume_ok = getattr(d, "vol_ratio")[0] >= add_vol_ratio_min and self._vol_zscore() >= float(getattr(self.p, "entry_vol_zscore_min", 0.5))
                if volume_ok and close > float(self.hhv_add[-1]):
                    advice = self._meta_advice(mode_id, "TRANCHE3", mode_name)
                    if advice["allow"]:
                        target_ratio = self._r_multiple_scaled_ratio(float(self.p.tranche_targets[2]), mode_name)
                        target_ratio = target_ratio * float(advice["size_multiplier"])
                        self.pos_mgr.scale_to(target_ratio, "第3档再突破", mode_name, "TRANCHE3")
                        self.tranche = 3
                return

            return

        # 空仓：开仓
        if self.cooldown > 0:
            return

        if self.meta_wait_count > 0:
            return

        if mode_name == "DRAWDOWN":
            return

        if mode_name == "TOP_CHOP" and (not self.p.allow_entry_in_top_chop):
            return

        # TOP_CHOP区间引擎：仅允许轻仓试错
        if mode_name == "TOP_CHOP":
            if active_engine != "RANGE_ENGINE" or (not self.p.allow_entry_in_top_chop):
                return
            if float(d.close[0]) <= float(self.ema20[0]):
                return
            if getattr(d, "vol_ratio")[0] < max(0.7, float(self.p.swing_vol_ratio_min)):
                return

            env = self._market_environment_snapshot()
            if env["is_weak"]:
                return

            self.tranche = 0
            self.tranche2_confirm_count = 0
            advice = self._meta_advice(mode_id, "RANGE_PROBE", mode_name)
            if not advice["allow"]:
                return
            range_ratio = min(float(self.p.probe_ratio), float(self.p.tranche_targets[0]))
            self.entry_profile = "SWING_CHOP"
            self.pos_mgr.scale_to(range_ratio * float(advice["size_multiplier"]), "区间引擎试探仓", mode_name, "RANGE_PROBE")
            return

        # BASE_BUILD试探仓
        if mode_name == "BASE_BUILD":
            if active_engine != "RECOVERY_ENGINE":
                return

            if self.base_probe_counter > 0:
                return

            if float(d.close[0]) <= float(self.ema20[0]):
                return
            if getattr(d, "vol_ratio")[0] < 1.0:
                return

            env = self._market_environment_snapshot()
            if env["is_weak"]:
                return

            self.tranche = 0
            self.pb_touched = False
            self.tranche2_confirm_count = 0
            self.profit_taken = False
            self.base_pyramid_count = 0
            self.base_probe_counter = self.p.base_probe_cooldown
            advice = self._meta_advice(mode_id, "PROBE", mode_name)
            if not advice["allow"]:
                return
            self.entry_profile = "NEUTRAL"
            self.pos_mgr.scale_to(float(self.p.probe_ratio) * float(advice["size_multiplier"]), "BASE试探仓", mode_name, "PROBE")
            return

        # TREND_RUN首仓
        if mode_name == "TREND_RUN":
            if active_engine != "TREND_ENGINE":
                return

            if self._reentry_guard_active(mode_name):
                return

            tsmom = self._tsmom_snapshot()
            if not (tsmom["pass_regime"] and tsmom["pass_trigger"]):
                return

            if not self._allow_by_hmm_trend_prob(mode_name):
                return

            if self.p.require_main_uptrend and getattr(d, "is_main_uptrend")[0] < 1:
                return

            is_breakout_entry = (
                float(getattr(d, "vol_ratio")[0]) >= self.get_adaptive_param("vol_ratio_min", float(self.p.vol_ratio_min))
                and float(d.close[0]) > float(self.hhv_entry[-1])
            )
            is_swing_entry = self._should_open_swing_entry(d)
            if not (is_breakout_entry or is_swing_entry):
                return

            if self._trend_quality_score() < float(getattr(self.p, "trend_quality_min_entry", 0.52)):
                return

            if self._vol_zscore() < float(getattr(self.p, "entry_vol_zscore_min", 0.5)):
                return

            env = self._market_environment_snapshot()
            if env["is_weak"]:
                return

            self.tranche = 1
            self.pb_touched = False
            self.tranche2_confirm_count = 0
            self.profit_taken = False
            self.base_pyramid_count = 0
            self.entry_peak_price = float(d.close[0])
            self.breakout_guard_remaining = 0
            entry_tag = "TRANCHE1" if is_breakout_entry else "SWING1"
            entry_reason = "第1档突破首仓" if is_breakout_entry else "波段回踩反弹首仓"

            advice = self._meta_advice(mode_id, entry_tag, mode_name)
            if not advice["allow"]:
                return
            base_ratio = float(self.p.tranche_targets[0])
            if self.current_market_bias == "MAIN_UPTREND" and is_breakout_entry:
                base_ratio = min(base_ratio * float(self.p.trend_aggressive_scale), float(self.p.tranche_targets[1]))
            self.entry_profile = self.current_market_bias if is_breakout_entry else "SWING_CHOP"

            if is_breakout_entry:
                body_strength = abs(float(d.close[0]) - float(d.open[0])) / max(float(self.atr[0]), 1e-9)
                vol_ratio = float(getattr(d, "vol_ratio")[0])
                if (
                    vol_ratio >= float(getattr(self.p, "burst_vol_ratio_min", 1.8))
                    and body_strength >= float(getattr(self.p, "burst_body_atr_min", 1.0))
                ):
                    self.breakout_guard_remaining = int(max(getattr(self.p, "burst_guard_bars", 0), 0))

            self.pos_mgr.scale_to(base_ratio * float(advice["size_multiplier"]), entry_reason, mode_name, entry_tag)
            return

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Completed:
                if order.isbuy() and self.position.size > 0:
                    learner = getattr(self, "profile_learner", None)
                    if learner is not None:
                        self.entry_context = learner.context_key()
                if order.issell():
                    self._consume_exit_for_meta()
                    if self.position.size == 0:
                        self._apply_exit_cooldown()
                        self._reset_state()
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        learner = getattr(self, "profile_learner", None)
        if learner is None:
            return
        pnl_pct = 0.0
        if float(trade.price) > 0:
            pnl_pct = float(trade.pnlcomm) / float(trade.price) * 100.0
        self.closed_trade_count += 1
        self.recent_trade_results.append(float(trade.pnlcomm) > 0.0)
        learner.observe_trade(pnl_pct=pnl_pct, context=self.entry_context)
        if bool(getattr(self.p, "adaptive_save_on_trade", False)):
            learner.save_state()
        self.entry_context = None

    def stop(self):
        learner = getattr(self, "profile_learner", None)
        if learner is not None:
            learner.save_state()

    def _reset_state(self):
        self.tranche = 0
        self.pb_touched = False
        self.tranche2_confirm_count = 0
        self.profit_taken = False
        self.base_pyramid_count = 0
        self.entry_peak_price = 0.0
        self.entry_profile = "NEUTRAL"
        self.breakout_guard_remaining = 0
        self.entry_context = None

    def _should_open_swing_entry(self, d) -> bool:
        if len(self) < int(self.p.swing_pullback_lookback) + 3:
            return False

        close = float(d.close[0])
        ema20 = float(self.ema20[0])
        ema50 = float(self.ema50[0])
        atrv = float(self.atr[0])

        if close <= ema20 or ema20 <= ema50:
            return False

        if float(getattr(d, "trend_score")[0]) < float(self.p.swing_trend_score_min):
            return False

        if float(getattr(d, "vol_ratio")[0]) < float(self.p.swing_vol_ratio_min):
            return False

        lookback = int(self.p.swing_pullback_lookback)
        recent_low = min(float(d.low[-i]) for i in range(1, lookback + 1))
        pullback_touched = recent_low <= ema20 + 0.5 * atrv

        rebound = close > float(d.high[-1]) and close > float(d.close[-1])
        return pullback_touched and rebound
