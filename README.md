# Quantitative：四阶段自适应趋势策略（v2.2）

本项目是一个基于 **Backtrader** 的股票趋势策略框架，核心特性包括：

- 四状态市场环境识别（TREND_RUN / TOP_CHOP / DRAWDOWN / BASE_BUILD）
- 分档建仓 + 差异化出场（止损 / 分批止盈 / Chandelier）
- 按股票独立参数配置（`stock_configs.py`）
- 可选 HMM 状态识别与 Meta-labeling 信号过滤
- 支持单标的回测、批量回测、Walk-forward 验证

---

## 1. 代码结构与参数位置总览

```text
/workspace/quantitative
├── optimized_hybrid_strategy.py      # 入口示例：单票/批量回测调用
├── stock_configs.py                  # 各股票参数配置、黑名单、默认回退配置
└── hybrid_strategy/
    ├── strategy.py                   # 策略参数全集（最终执行参数）
    ├── managers.py                   # Regime识别、仓位管理、出场规则细节
    ├── backtest.py                   # 回测参数组装、数据校验、结果输出
    ├── data_utils.py                 # 数据加载与主升浪信号打分参数
    ├── walk_forward.py               # Walk-forward参数搜索与验证区间
    └── meta_labeling.py              # Triple Barrier / 元标签模型参数
```

---

## 2. 快速开始

### 2.1 安装依赖（建议）

```bash
pip install backtrader pandas numpy matplotlib yfinance
```

### 2.2 运行示例

```bash
python optimized_hybrid_strategy.py
```

脚本中默认会执行：

- `run_backtest("NVDA")`
- `batch_backtest(symbols=["NVDA", "GOOGL", "AAPL"])`

你也可以在脚本里打开注释示例，测试黑名单股票或传入 `custom_params`。

---

## 3. 参数加载优先级（非常重要）

实际运行时，参数并不是只来自一个地方，而是按以下优先级覆盖：

1. **策略底层默认值**：`hybrid_strategy/strategy.py` 的 `OptimizedHybrid4ModeV2.params`
2. **回测入口基础参数**：`hybrid_strategy/backtest.py` 的 `strategy_params`
3. **股票级配置**：`stock_configs.py` 中 `get_stock_config(symbol)["params"]`
4. **手动覆盖参数**：`run_backtest(..., custom_params={...})`（最高优先级）

> 建议：调试时先确定你改的是哪一层，避免“参数改了但结果没变”的错觉。

---

## 4. 核心交易参数说明（位置 / 含义 / 调试范围）

> 下表给的是**策略调参时优先关注**的一组参数。
> “调试范围”分为两类：
>
> - **工程硬边界**：代码里有 clamp 或特殊逻辑时，超出会被截断/失效
> - **实盘调试建议区间**：便于从保守到激进逐步测试

| 参数 | 主要位置 | 含义 | 工程硬边界 | 建议调试范围 |
|---|---|---|---|---|
| `stop_loss_pct` | `strategy.py` / `stock_configs.py` | 止损阈值（%）；`>=999` 表示禁用止损（高波动票） | `<1` 在WF候选中会被抬到 `1`; `>=999` 视为禁用 | `5~12`（常规票）；高波动票可用 `999` |
| `profit_take_pct` | `strategy.py` / `stock_configs.py` | 三档后触发分批止盈阈值（%） | WF最低 `5` | `15~40` |
| `chand_atr_mult` | `strategy.py` / `stock_configs.py` | Chandelier 退出ATR倍数 | WF最低 `1.5` | `2.2~3.2` |
| `vol_ratio_min` | `strategy.py` / `stock_configs.py` | 首次突破入场最低量能比（`Volume/VOL_MA20`） | WF clamp 到 `[0.6,1.8]` | `0.9~1.4` |
| `add_vol_ratio_min` | `strategy.py` | 第三档加仓量能门槛 | 无显式clamp | `0.9~1.3` |
| `dd_drawdown_th` | `strategy.py` / `stock_configs.py` | 判定 DRAWDOWN 的回撤阈值（负数） | WF clamp 到 `[-0.35,-0.10]` | `-0.24~-0.14` |
| `atrp_drawdown_th` | `strategy.py` | ATR%过高触发 DRAWDOWN | 无显式clamp | `0.07~0.12` |
| `high_zone_dd_th` | `strategy.py` | 高位区域阈值，用于TOP_CHOP识别 | 无显式clamp | `-0.12~-0.06` |
| `cross_top_min` | `strategy.py` | 高位震荡判断所需 EMA20 穿越次数 | 无显式clamp | `8~16` |
| `atr_shrink_ratio` | `strategy.py` | ATR收缩判定比率 | 无显式clamp | `0.6~0.85` |
| `base_zone_dd_th` | `strategy.py` | BASE_BUILD区域阈值（负数） | 无显式clamp | `-0.45~-0.25` |
| `base_atrp_th` | `strategy.py` | BASE阶段允许的ATR%上限 | 无显式clamp | `0.06~0.12` |
| `base_hl_consecutive` | `strategy.py` | BASE阶段“抬高低点”连续次数 | 无显式clamp | `2~4` |
| `base_probe_cooldown` | `strategy.py` | BASE试探仓冷却天数 | 无显式clamp | `5~15` |
| `base_pyramid_profit_th` | `strategy.py` | BASE金字塔加仓所需浮盈（%） | 无显式clamp | `3~8` |
| `probe_ratio` | `strategy.py` | BASE试探仓比例 | 无显式clamp | `0.1~0.25` |
| `tranche_targets` | `strategy.py` | 三档目标仓位比例 | 需递增，且通常不超过 `1.0` | 例：`(0.3,0.6,1.0)` |
| `max_exposure` | `strategy.py` | 最大总暴露上限（未乘波动缩放前） | 仓位管理中最终会 clamp 到 `[0,1]` | `0.4~0.8` |
| `drawdown_tolerance` | `strategy.py` | 加仓前容忍回撤（相对成本） | 无显式clamp | `0.05~0.12` |
| `cooldown_bars` | `strategy.py` | 止损/退出后冷却bar数 | 无显式clamp | `1~5` |

---

## 5. 风险预算与仓位参数（Vol Targeting）

这组参数由 `PositionManager` 使用，决定“同样的信号下买多少仓位”。

| 参数 | 位置 | 含义 | 工程边界 | 建议调试范围 |
|---|---|---|---|---|
| `use_vol_targeting` | `strategy.py` / `backtest.py` | 是否启用波动目标仓位 | 布尔 | `True` 优先 |
| `target_vol_annual` | `strategy.py` / `backtest.py` | 目标年化波动率 | 内部最小 `1e-6` | `0.15~0.25` |
| `vol_lookback` | `strategy.py` / `backtest.py` | 波动估计回看天数 | 最小按2处理 | `20~60` |
| `vol_floor_annual` | `strategy.py` / `backtest.py` | 波动率下限 | 与 cap 一起夹断 | `0.08~0.15` |
| `vol_cap_annual` | `strategy.py` / `backtest.py` | 波动率上限 | 与 floor 一起夹断 | `0.6~1.0` |
| `min_vol_scalar` | `strategy.py` / `backtest.py` | 仓位缩放最小倍数 | clamp | `0.2~0.5` |
| `max_vol_scalar` | `strategy.py` / `backtest.py` | 仓位缩放最大倍数 | clamp | `0.8~1.2` |

---

## 6. HMM 与 Meta-labeling 参数

### 6.1 HMM状态识别参数

| 参数 | 位置 | 含义 | 建议调试范围 |
|---|---|---|---|
| `use_hmm_regime` | `strategy.py` / `backtest.py` | 启用HMM状态识别（否则用规则状态机） | `True/False` 对照实验 |
| `hmm_warmup_bars` | `strategy.py` / `backtest.py` | HMM最小热身长度 | `210~320` |
| `hmm_min_confidence` | `strategy.py` / `backtest.py` | HMM最小置信度，低于则回退规则状态机 | `0.35~0.60` |
| `hmm_mode_buffer_days` | `strategy.py` / `backtest.py` | 模式切换缓冲天数 | `1~4` |

### 6.2 Meta-labeling 参数

| 参数 | 位置 | 含义 | 建议调试范围 |
|---|---|---|---|
| `use_meta_labeling` | `strategy.py` | 是否启用在线元标签过滤 | `False`/`True` A-B测试 |
| `meta_prob_threshold` | `strategy.py` | 信号放行概率阈值 | `0.50~0.60` |
| `meta_min_samples` | `strategy.py` | 开始使用模型前最小样本数 | `20~80` |
| `meta_retrain_interval` | `strategy.py` | 每积累多少新样本重训一次 | `5~30` |

`meta_labeling.py` 还定义了独立组件参数：

- `TripleBarrierConfig.take_profit_pct`
- `TripleBarrierConfig.stop_loss_pct`
- `TripleBarrierConfig.max_holding_bars`
- `LogisticMetaModel.lr`, `LogisticMetaModel.n_iter`

如果你要单独做元标签研究，可直接在这些组件层面调参。

---

## 7. 数据与信号参数（主升浪打分）

主升浪信号由 `detect_main_uptrend` 生成，位于 `hybrid_strategy/data_utils.py`。

| 参数 | 含义 | 建议调试范围 |
|---|---|---|
| `ma_fast`, `ma_mid`, `ma_slow` | EMA快中慢周期 | `(20,50,200)` 为基线 |
| `vol_ma` | 成交量均线窗口 | `20~40` |
| `breakout_20`, `breakout_55` | 突破窗口 | `20/55` 常用 |
| `slope_win` | EMA20斜率回看 | `8~20` |
| `pullback_lookback` | 回撤统计窗口 | `40~80` |
| `pullback_dd_th` | 回撤阈值 | `-0.12~-0.05` |
| `pullback_days_th` | 回踩天数阈值 | `3~8` |
| `vol_ratio_th` | 量能阈值 | `1.0~1.5` |
| `score_threshold` | `(Trend, Mom, Pb)` 三评分门槛 | 例如 `(4,2,2)` |

---

## 8. Walk-forward 参数搜索范围（代码真实实现）

`hybrid_strategy/walk_forward.py::_candidate_params` 中，实际候选生成逻辑如下：

- `stop_loss_pct`
  - 若基准 `>=900`：固定 `999`
  - 否则：`[0.9x, 1.0x, 1.1x]`，且最低 `1.0`
- `profit_take_pct`：`[0.9x,1.0x,1.1x]`，最低 `5.0`
- `chand_atr_mult`：`base + {-0.2,0,+0.2}`，最低 `1.5`
- `vol_ratio_min`：`[0.9x,1.0x,1.1x]`，clamp 到 `[0.6,1.8]`
- `dd_drawdown_th`：`base + {-0.03,0,+0.03}`，clamp 到 `[-0.35,-0.10]`

这部分可以视作“系统内置调参网格”的**真实边界说明**。

---

## 9. 股票配置文件使用说明（`stock_configs.py`）

### 9.1 配置结构

每个股票配置格式：

```python
XXX_CONFIG = {
    "name": "...",
    "tier": "S/A/B/C",
    "category": "high_vol/low_vol/medium_vol",
    "params": {
        "stop_loss_pct": ...,
        "profit_take_pct": ...,
        "vol_ratio_min": ...,
        "chand_atr_mult": ...,
        "dd_drawdown_th": ...,
    },
    "performance": {...},
    "notes": "..."
}
```

### 9.2 黑名单机制

- `BLACKLIST_CONFIGS` 中的股票会被 `run_backtest` 直接拦截，不执行回测。
- 当状态为 `blacklisted` 时，回测函数返回 `(None, None)`。

### 9.3 未知股票默认参数

`get_stock_config` 对未配置股票返回默认参数：

- `stop_loss_pct=10.0`
- `profit_take_pct=25.0`
- `vol_ratio_min=1.2`
- `chand_atr_mult=2.8`
- `dd_drawdown_th=-0.18`

---

## 10. 调试建议（推荐流程）

1. **先固定股票池**：优先选 `Tier S/A`，避免把“标的质量问题”误判成参数问题。
2. **先调出场再调入场**：
   - 先确定 `stop_loss_pct / chand_atr_mult / profit_take_pct`
   - 再微调 `vol_ratio_min / add_vol_ratio_min`
3. **一次只改一组参数**：每轮只改 1~2 个变量，记录收益、回撤、交易次数。
4. **做A-B对照**：
   - `use_hmm_regime=True/False`
   - `use_meta_labeling=True/False`
5. **用 Walk-forward 做最终筛选**：防止参数只在单区间过拟合。

---

## 11. 常用调用方式

### 单票回测

```python
from hybrid_strategy import run_backtest

strat, df = run_backtest("NVDA")
```

### 单票 + 自定义参数覆盖

```python
strat, df = run_backtest(
    "AAPL",
    custom_params={
        "stop_loss_pct": 8.0,
        "profit_take_pct": 22.0,
        "vol_ratio_min": 1.1,
    }
)
```

### 批量回测

```python
from hybrid_strategy import batch_backtest

batch_backtest(symbols=["NVDA", "GOOGL", "AAPL"])
batch_backtest(tier="S")
```

### Walk-forward

```python
from hybrid_strategy import walk_forward_validation

walk_forward_validation(
    symbols=["NVDA", "GOOGL"],
    start="2018-01-01",
    train_years=3,
    test_years=1,
    activity_profile="balanced",
)
```

---

## 12. 注意事项

- 本项目默认从 yfinance 拉取日线数据，网络波动会影响下载稳定性。
- 回测结果依赖滑点、手续费、数据完整度，建议统一实验设置。
- 任何参数都应先在样本外（Walk-forward）验证，再考虑实盘模拟。

> 免责声明：本项目仅用于量化研究与教学演示，不构成任何投资建议。
