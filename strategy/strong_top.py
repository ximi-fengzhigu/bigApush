"""
强势股顶部识别策略模块
Strong Stock Top Detection Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class StrongTopStrategy:
    """
    强势股顶部识别策略
    
    核心逻辑：
    1. 前置过滤：大盘/行业/个股必须在MA20之上
    2. 强势评分：基于趋势、量价、板块三维度打分 (0-5分)
    3. 顶部信号检测：RSI/KDJ超买 + K线形态 + 放量滞涨
    4. 综合决策：根据强势评分动态调整信号触发门槛
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化策略参数
        
        Args:
            config: 可选参数字典，可调整所有阈值
        """
        self.config = {
            # ---- 前置过滤参数 ----
            'benchmark_ma_period': 20,
            'industry_ma_period': 20,
            'stock_ma_period': 20,
            
            # ---- 强势评分参数 ----
            'ma_periods': [5, 10, 20, 60],
            'max_bias_ratio': 0.15,      # 乖离率上限 15%
            'volume_ratio': 1.2,          # 上涨放量阈值
            'volume_shrink_ratio': 1.0,   # 回调缩量阈值
            
            # ---- 顶部信号参数 ----
            'rsi_period': 14,
            'rsi_threshold_high': 75,
            'rsi_threshold_very_high': 85,
            'kdj_period': 9,
            'k_threshold': 80,
            'j_threshold': 100,
            'shadow_body_ratio': 2.0,     # 上影线/实体 ≥ 2
            'shadow_total_ratio': 0.6,    # 上影线/总长 ≥ 60%
            'volume_surge_ratio': 1.8,    # 放量倍数
            'stall_pct_low': -0.005,      # 滞涨下限 -0.5%
            'stall_pct_high': 0.015,      # 滞涨上限 1.5%
            
            # ---- 决策阈值 ----
            'strong_score_high': 4,       # 非常强势分数线
            'strong_score_mid': 2,        # 中等强势分数线
            'strong_signal_count': 3,     # 强势股需触发的信号数
            'mid_signal_count': 2,        # 中等股需触发的信号数
            'weak_signal_count': 1,       # 弱势股需触发的信号数
        }
        
        # 用传入的config覆盖默认值
        if config:
            self.config.update(config)
    
    # ============================================================
    # 第一部分：技术指标计算
    # ============================================================
    
    @staticmethod
    def calc_ma(df: pd.DataFrame, period: int, col: str = 'close') -> pd.Series:
        """计算移动平均线"""
        return df[col].rolling(period).mean()
    
    @staticmethod
    def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calc_kdj(df: pd.DataFrame, period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算KDJ指标 (返回K, D, J)"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR（平均真实波幅），用于趋势强度辅助判断"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    # ============================================================
    # 第二部分：前置过滤（大盘/行业/个股环境）
    # ============================================================
    
    def check_preconditions(
        self,
        stock_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        benchmark_df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        检查前置条件：大盘、行业、个股是否都在MA20之上
        
        Returns:
            (是否通过, 原因说明)
        """
        ma_period = self.config['stock_ma_period']
        
        # 条件1: 大盘在MA20之上
        benchmark_ma = self.calc_ma(benchmark_df, ma_period)
        if benchmark_df['close'].iloc[-1] <= benchmark_ma.iloc[-1]:
            return False, f"大盘指数低于MA{ma_period}"
        
        # 条件2: 行业指数在MA20之上
        industry_ma = self.calc_ma(industry_df, ma_period)
        if industry_df['close'].iloc[-1] <= industry_ma.iloc[-1]:
            return False, f"行业指数低于MA{ma_period}"
        
        # 条件3: 个股在MA20之上
        stock_ma = self.calc_ma(stock_df, ma_period)
        if stock_df['close'].iloc[-1] <= stock_ma.iloc[-1]:
            return False, f"个股股价低于MA{ma_period}"
        
        return True, "前置条件全部满足"
    
    # ============================================================
    # 第三部分：强势评分（三层框架）
    # ============================================================
    
    def calc_strength_score(self, df: pd.DataFrame) -> int:
        """
        计算强势评分 (0-5分)
        分数越高代表趋势越强
        """
        score = 0
        close = df['close']
        volume = df['volume']
        
        # ---- 维度1：趋势强度 (最多3分) ----
        # 1a. 均线多头排列
        ma5 = self.calc_ma(df, 5)
        ma10 = self.calc_ma(df, 10)
        ma20 = self.calc_ma(df, 20)
        ma60 = self.calc_ma(df, 60)
        
        if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            score += 1
        
        # 1b. MA20持续上升
        if ma20.iloc[-1] > ma20.iloc[-2]:
            score += 1
        
        # 1c. 乖离率健康 (< 15%)
        bias_ratio = (close.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1]
        if bias_ratio < self.config['max_bias_ratio']:
            score += 1
        
        # ---- 维度2：量价配合 (最多2分) ----
        avg_vol_5 = volume.rolling(5).mean()
        
        # 2a. 今日上涨且放量
        if close.iloc[-1] > close.iloc[-2]:
            if volume.iloc[-1] > avg_vol_5.iloc[-1] * self.config['volume_ratio']:
                score += 1
        
        # 2b. 近3日若有回调，量能萎缩
        # 检查最近3天是否有下跌日
        recent_3 = df.tail(3)
        for i in range(len(recent_3) - 1):
            if recent_3['close'].iloc[i] > recent_3['close'].iloc[i+1]:
                # 有下跌，检查量能是否萎缩
                if recent_3['volume'].iloc[i+1] < avg_vol_5.iloc[-(3-i)]:
                    score += 1
                    break
        
        # ---- 维度3：板块效应 (依赖外部数据，这里简化) ----
        # 在外部调用时，如果有板块数据可以额外加分
        # 这里预留一个接口，由外部传入
        if hasattr(self, '_industry_extra_score'):
            score += self._industry_extra_score
        
        return min(score, 5)  # 最高5分
    
    # ============================================================
    # 第四部分：顶部信号检测
    # ============================================================
    
    def detect_top_signals(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        检测所有顶部信号
        
        Returns:
            各信号是否触发的字典
        """
        signals = {
            'rsi_overbought': False,
            'kdj_overbought': False,
            'shooting_star': False,
            'long_upper_shadow': False,
            'volume_surge_stall': False,
            'macd_divergence': False,   # 新增：MACD顶背离
        }
        
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # ----- 1. RSI超买 -----
        rsi = self.calc_rsi(df, self.config['rsi_period'])
        if rsi.iloc[-1] > self.config['rsi_threshold_high']:
            signals['rsi_overbought'] = True
        
        # ----- 2. KDJ超买 -----
        k, d, j = self.calc_kdj(df, self.config['kdj_period'])
        if k.iloc[-1] > self.config['k_threshold'] and j.iloc[-1] > self.config['j_threshold']:
            signals['kdj_overbought'] = True
        
        # ----- 3. 射击之星 -----
        body = abs(close.iloc[-1] - open_price.iloc[-1])
        upper_shadow = high.iloc[-1] - max(close.iloc[-1], open_price.iloc[-1])
        lower_shadow = min(close.iloc[-1], open_price.iloc[-1]) - low.iloc[-1]
        
        if upper_shadow >= self.config['shadow_body_ratio'] * body and lower_shadow <= 0.3 * body:
            signals['shooting_star'] = True
        
        # ----- 4. 长上影线 -----
        total_length = high.iloc[-1] - low.iloc[-1]
        if total_length > 0 and upper_shadow >= self.config['shadow_total_ratio'] * total_length:
            signals['long_upper_shadow'] = True
        
        # ----- 5. 放量滞涨 -----
        avg_vol_5 = volume.rolling(5).mean()
        pct_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        
        if volume.iloc[-1] >= self.config['volume_surge_ratio'] * avg_vol_5.iloc[-1]:
            if self.config['stall_pct_low'] <= pct_change <= self.config['stall_pct_high']:
                signals['volume_surge_stall'] = True
        
        # ----- 6. MACD顶背离 (新增) -----
        if self._check_macd_divergence(df):
            signals['macd_divergence'] = True
        
        return signals
    
    def _check_macd_divergence(self, df: pd.DataFrame, lookback: int = 20) -> bool:
        """
        检测MACD顶背离：
        股价创新高，但MACD红柱或DIFF线未创新高
        """
        if len(df) < lookback + 26:
            return False
        
        # 计算MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        diff = exp12 - exp26
        dea = diff.ewm(span=9, adjust=False).mean()
        macd_hist = diff - dea
        
        # 取最近 lookback 天的数据
        recent_high = df['high'].tail(lookback)
        recent_macd = macd_hist.tail(lookback)
        
        # 找股价的最高点和次高点
        high_max_idx = recent_high.idxmax()
        # 去掉最高点，找次高点
        high_second_max = recent_high.drop(high_max_idx).max()
        
        # 如果最高点对应的MACD不是最高的，说明背离
        macd_at_high = recent_macd.loc[high_max_idx]
        macd_max = recent_macd.max()
        
        # 股价创新高但MACD未创新高 → 顶背离
        if high_second_max > 0 and macd_at_high < macd_max * 0.95:
            return True
        
        return False
    
    # ============================================================
    # 第五部分：综合决策
    # ============================================================
    
    def make_decision(
        self,
        stock_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        benchmark_df: pd.DataFrame
    ) -> Dict:
        """
        综合决策主函数
        
        Returns:
            包含决策结果、信号详情、评分的字典
        """
        result = {
            'action': '持有',
            'action_code': 0,  # 0:持有, 1:减仓观察, 2:卖出/清仓
            'strength_score': 0,
            'top_signals': {},
            'triggered_count': 0,
            'reason': '',
            'precondition_passed': False,
        }
        
        # ----- 第一步：前置过滤 -----
        pre_passed, pre_reason = self.check_preconditions(
            stock_df, industry_df, benchmark_df
        )
        if not pre_passed:
            result['reason'] = pre_reason
            result['action'] = '持有（环境不佳）'
            return result
        
        result['precondition_passed'] = True
        
        # ----- 第二步：计算强势评分 -----
        strength_score = self.calc_strength_score(stock_df)
        result['strength_score'] = strength_score
        
        # ----- 第三步：检测顶部信号 -----
        top_signals = self.detect_top_signals(stock_df)
        result['top_signals'] = top_signals
        
        triggered = sum(top_signals.values())
        result['triggered_count'] = triggered
        
        # ----- 第四步：根据强势评分动态决策 -----
        strong_high = self.config['strong_score_high']
        strong_mid = self.config['strong_score_mid']
        
        if strength_score >= strong_high:
            # 非常强势：需要更多信号才行动
            if triggered >= self.config['strong_signal_count']:
                result['action'] = '🔴 卖出/大幅减仓'
                result['action_code'] = 2
                result['reason'] = f'强势股({strength_score}分)，触发{triggered}个顶部信号'
            elif triggered >= self.config['strong_signal_count'] - 1:
                result['action'] = '🟡 减仓1/3观察'
                result['action_code'] = 1
                result['reason'] = f'强势股({strength_score}分)，触发{triggered}个顶部信号，建议减仓观察'
            else:
                result['action'] = '🟢 继续持有'
                result['reason'] = f'强势股({strength_score}分)，仅触发{triggered}个信号，继续持有'
        
        elif strength_score >= strong_mid:
            # 中等偏强：标准门槛
            if triggered >= self.config['mid_signal_count'] + 1:
                result['action'] = '🔴 卖出/大幅减仓'
                result['action_code'] = 2
                result['reason'] = f'中等强势({strength_score}分)，触发{triggered}个顶部信号'
            elif triggered >= self.config['mid_signal_count']:
                result['action'] = '🟡 减仓1/3观察'
                result['action_code'] = 1
                result['reason'] = f'中等强势({strength_score}分)，触发{triggered}个顶部信号，建议减仓'
            else:
                result['action'] = '🟢 继续持有'
                result['reason'] = f'中等强势({strength_score}分)，仅触发{triggered}个信号，继续持有'
        
        else:
            # 弱势/赶顶：低门槛，信号敏感
            if triggered >= self.config['weak_signal_count'] + 1:
                result['action'] = '🔴 卖出/清仓'
                result['action_code'] = 2
                result['reason'] = f'弱势/赶顶({strength_score}分)，触发{triggered}个顶部信号，建议清仓'
            elif triggered >= self.config['weak_signal_count']:
                result['action'] = '🟡 减仓1/2观察'
                result['action_code'] = 1
                result['reason'] = f'弱势/赶顶({strength_score}分)，触发{triggered}个顶部信号，建议减半仓'
            else:
                result['action'] = '🟢 继续持有（收紧止损）'
                result['reason'] = f'弱势股({strength_score}分)，仅触发{triggered}个信号，继续持有但需收紧止损'
        
        # 额外：如果触发了MACD顶背离，直接升级决策
        if top_signals.get('macd_divergence', False) and result['action_code'] < 2:
            result['action'] = result['action'].replace('🟢', '🟡').replace('🟡', '🔴')
            result['action_code'] = min(result['action_code'] + 1, 2)
            result['reason'] += '；且出现MACD顶背离'
        
        return result


# ============================================================
# 第六部分：便捷函数 - 用于每日扫描
# ============================================================

def daily_scan(
    stock_data: pd.DataFrame,
    industry_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
    strategy: Optional[StrongTopStrategy] = None
) -> Dict:
    """
    每日盘后扫描单只股票
    
    Args:
        stock_data: 个股日线数据 (需包含 open, high, low, close, volume)
        industry_data: 行业指数日线数据
        benchmark_data: 大盘指数日线数据
        strategy: 策略实例，不传则使用默认参数
        
    Returns:
        决策结果字典
    """
    if strategy is None:
        strategy = StrongTopStrategy()
    
    return strategy.make_decision(stock_data, industry_data, benchmark_data)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import akshare as ak
    
    # 1. 获取数据
    # 个股：以宁德时代为例
    stock = ak.stock_zh_a_hist(
        symbol="300750", 
        period="daily", 
        start_date="20250101", 
        adjust="qfq"
    )
    # 重命名列以匹配代码
    stock.columns = ['日期', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    
    # 行业指数：以创业板指为例 (简化)
    industry = ak.stock_zh_a_hist(
        symbol="399006",
        period="daily",
        start_date="20250101",
        adjust="qfq"
    )
    industry.columns = ['日期', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    
    # 大盘：以上证指数为例
    benchmark = ak.stock_zh_a_hist(
        symbol="000001",
        period="daily",
        start_date="20250101",
        adjust="qfq"
    )
    benchmark.columns = ['日期', 'open', 'close', 'high', 'low', 'volume', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    
    # 2. 运行策略
    strategy = StrongTopStrategy()
    result = daily_scan(stock, industry, benchmark, strategy)
    
    # 3. 打印结果
    print("=" * 50)
    print("📊 强势股顶部识别结果")
    print("=" * 50)
    print(f"操作建议: {result['action']}")
    print(f"强势评分: {result['strength_score']}/5")
    print(f"触发信号数: {result['triggered_count']}")
    print(f"信号详情: {result['top_signals']}")
    print(f"原因: {result['reason']}")
    print("=" * 50)
    # ============================================================
# 第七部分：bigApush 适配器
# ============================================================

class StrongTopStrategyAdapter:
    """
    StrongTopStrategy 的 bigApush 适配器
    
    将单只股票分析策略适配为全市场选股策略：
    - 遍历全市场股票
    - 对每只股票执行顶部识别
    - 返回没有顶部信号或顶部风险较低的股票列表
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.strategy = StrongTopStrategy(config)
    
    def filter(self, df_dict: Dict[str, pd.DataFrame], **kwargs) -> List[str]:
        """
        适配 bigApush 的 filter 接口
        
        Args:
            df_dict: 股票数据字典，key为股票代码，value为日线DataFrame
            **kwargs: 额外参数，可能包含行业指数和大盘指数
            
        Returns:
            通过筛选的股票代码列表（没有顶部信号的股票）
        """
        # 获取行业和大盘数据（从kwargs或全局配置获取）
        industry_data = kwargs.get('industry_data')
        benchmark_data = kwargs.get('benchmark_data')
        
        passed_stocks = []
        
        for code, df in df_dict.items():
            try:
                # 如果数据不足，跳过
                if len(df) < 60:
                    continue
                
                # 简化版：只用个股数据快速判断
                # 如果传入了行业和大盘数据，使用完整版
                if industry_data is not None and benchmark_data is not None:
                    # 获取该股票对应的行业数据（需要映射关系）
                    # 这里简化：使用全局行业数据
                    result = self.strategy.make_decision(
                        df, 
                        industry_data, 
                        benchmark_data
                    )
                else:
                    # 简化版：仅检测顶部信号
                    result = self._quick_check(df)
                
                # 如果操作建议不是"卖出"或"大幅减仓"，认为该股票安全
                if result['action_code'] < 2:  # 0:持有, 1:减仓观察, 2:卖出
                    passed_stocks.append(code)
                    
            except Exception as e:
                # 单只股票出错不影响整体扫描
                print(f"分析 {code} 时出错: {e}")
                continue
        
        return passed_stocks
    
    def _quick_check(self, df: pd.DataFrame) -> Dict:
        """简化版快速检查，仅用个股数据"""
        # 计算关键指标
        close = df['close']
        volume = df['volume']
        
        # RSI
        rsi = self.strategy.calc_rsi(df)
        rsi_overbought = rsi.iloc[-1] > 75
        
        # KDJ
        k, d, j = self.strategy.calc_kdj(df)
        kdj_overbought = k.iloc[-1] > 80 and j.iloc[-1] > 100
        
        # 放量滞涨
        avg_vol_5 = volume.rolling(5).mean()
        pct_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        volume_surge = (
            volume.iloc[-1] >= 1.8 * avg_vol_5.iloc[-1] and 
            -0.005 <= pct_change <= 0.015
        )
        
        triggered = sum([rsi_overbought, kdj_overbought, volume_surge])
        
        return {
            'action_code': 2 if triggered >= 2 else 1 if triggered == 1 else 0,
            'action': '卖出' if triggered >= 2 else '减仓观察' if triggered == 1 else '持有',
            'triggered_count': triggered
        }
