"""
策略模块

自动注册所有策略类
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入策略
from strategy.w_bottom_strategy import WBottomStrategy
from strategy.limit_up_pullback_strategy import LimitUpPullbackStrategy
from strategy.strong_top import StrongTopStrategy, StrongTopStrategyAdapter  # ← 导入适配器

# 策略类映射
STRATEGIES = {
    'WBottomStrategy': WBottomStrategy,
    'LimitUpPullbackStrategy': LimitUpPullbackStrategy,
    'StrongTopStrategy': StrongTopStrategyAdapter,  # ← 注册适配器
}

__all__ = [
    'WBottomStrategy',
    'LimitUpPullbackStrategy',
    'StrongTopStrategy',
    'StrongTopStrategyAdapter',
    'STRATEGIES'
]
