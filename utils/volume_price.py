# -*- coding: utf-8 -*-
"""
量价分析模块：判断资金态度

核心逻辑：
  - 放量上涨 → 健康 (+分)
  - 缩量回调 → 健康 (筹码惜售，可低吸)
  - 缩量上涨 → 买盘不足 (谨慎)
  - 放量滞涨 → 出货嫌疑 (-分)
  - 放量下跌 → 恐慌/出逃 (-分)
  - 无量涨停 → 筹码锁定好 (+分)

配置阈值：
  - VOL_RATIO_STRONG = 2.0  # 明显放量
  - VOL_RATIO_WEAK = 0.5    # 极度缩量
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# 量价分析阈值配置
VOL_RATIO_STRONG = 2.0  # 明显放量阈值
VOL_RATIO_WEAK = 0.5    # 极度缩量阈值


@dataclass
class VolumePriceAnalysis:
    """量价分析结果"""
    vol_ratio: float = 1.0        # 量比
    pattern: str = "中性"          # 形态描述
    is_healthy: bool = True       # 是否健康（资金流入）
    score_delta: int = 0          # 对总分的调整值


def analyze(df_row: dict) -> VolumePriceAnalysis:
    """
    根据行情数据判断量价形态，输出分析结果与评分调整
    
    Args:
        df_row: 包含以下字段的字典：
            - volume: 当日成交量
            - volume_ma5: 5日均量
            - change_pct: 当日涨幅%
            - close: 收盘价（可选，用于涨停判断）
    
    Returns:
        VolumePriceAnalysis: 量价分析结果
    """
    # 计算量比
    volume = df_row.get('volume', 0)
    volume_ma5 = df_row.get('volume_ma5', 0)
    change_pct = df_row.get('change_pct', 0.0)
    
    if volume_ma5 > 0:
        vol_ratio = volume / volume_ma5
    else:
        vol_ratio = 1.0
    
    result = VolumePriceAnalysis(vol_ratio=vol_ratio)
    
    # 判断涨跌方向
    if change_pct > 1.5:
        direction = "上涨"
    elif change_pct < -1.5:
        direction = "下跌"
    else:
        direction = "横盘"
    
    # === 核心规则 ===
    if vol_ratio >= VOL_RATIO_STRONG:
        # 明显放量
        if direction == "上涨":
            result.pattern = "放量上涨"
            result.is_healthy = True
            result.score_delta = +5
        elif direction == "下跌":
            result.pattern = "放量下跌"
            result.is_healthy = False
            result.score_delta = -10
        else:
            # 放量但价格不动 → 滞涨/出货
            result.pattern = "放量滞涨"
            result.is_healthy = False
            result.score_delta = -8
    elif vol_ratio <= VOL_RATIO_WEAK:
        # 缩量
        if direction == "下跌":
            result.pattern = "缩量回调"
            result.is_healthy = True  # 抛压轻，是低吸机会
            result.score_delta = +3
        elif direction == "上涨":
            result.pattern = "缩量上涨"
            result.is_healthy = False  # 惯性推升，后劲不足
            result.score_delta = -3
        else:
            result.pattern = "极度缩量横盘"
            result.is_healthy = True
            result.score_delta = 0
    else:
        # 量能温和 (0.5 ~ 2.0)
        result.pattern = f"温和放量{direction}"
        result.is_healthy = (direction != "下跌")
        result.score_delta = 1 if direction == "上涨" else (-2 if direction == "下跌" else 0)
    
    # 涨停特殊判断（涨幅>=9.5% 视为涨停）
    if change_pct >= 9.5:
        if vol_ratio <= 1.0:
            result.pattern = "无量涨停"
            result.is_healthy = True
            result.score_delta = +8
        else:
            result.pattern = "放量涨停"
            result.is_healthy = True
            result.score_delta = +5
    
    logger.debug(f"量价分析: {result.pattern}, delta={result.score_delta}")
    return result
