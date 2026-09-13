# -*- coding: utf-8 -*-
"""
板块强度评分器模块（zzshare 版本）

基于 zzshare 板块行情数据计算板块强度得分。
数据来源：zzshare plates_rank 接口（同花顺板块排名）

评分维度：
  1. 板块涨幅排名得分 - 前20名 +50分
  2. 涨停家数得分 - 根据板块涨停股数量评分

综合公式：
  板块强度得分 = BASE_SCORE(50) + 涨幅排名得分 + 涨停得分
  得分范围：0 到 150

一票否决条件：
  - 板块得分 = -100 时，个股直接淘汰
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

# 导入板块强度详情模型
from trading.stock_score_models import SectorDetail
from utils.zzshare_fetcher import create_fetcher as create_sector_fetcher

# 配置日志记录器
logger = logging.getLogger(__name__)

# ============================================================
# 板块强度评分常量配置
# ============================================================

# 一票否决得分
VETO_SCORE = -100
# 基准分
BASE_SCORE = 50
# 板块涨幅排名阈值（前20名加分）
RANK_TOP_N = 20
# 涨幅排名加分
RANK_SCORE = 50
# 涨停加分（每增加1家涨停股，加5分，最高25分）
LIMIT_UP_SCORE_PER_STOCK = 5
LIMIT_UP_MAX_SCORE = 25
# 评分天数（仅当天）
SCORE_DAYS = 1

# 内存缓存 TTL（秒）
CACHE_TTL = 300  # 5分钟


class MemoryCache:
    """
    内存缓存管理器

    仅用于同一请求周期内的数据缓存，避免重复调用 API。
    缓存有效期为 5 分钟。
    """

    def __init__(self, ttl: int = CACHE_TTL):
        self._cache: Dict[str, Tuple] = {}
        self._ttl = ttl

    def get(self, key: str):
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return data
            del self._cache[key]
        return None

    def set(self, key: str, value):
        self._cache[key] = (value, time.time())


class SectorScorer:
    """
    板块强度评分器（zzshare 版本）

    根据 zzshare 板块行情数据计算板块强度得分。
    支持一票否决机制（板块得分 -100 时个股直接淘汰）。
    """

    def __init__(self, **kwargs):
        """
        初始化板块强度评分器

        参数:
            **kwargs: 兼容额外参数（如 db_manager）
        """
        # 初始化 zzshare 数据源
        self._fetcher = create_sector_fetcher()
        # 初始化内存缓存
        self._cache = MemoryCache()
        # 记录初始化日志
        logger.info("板块强度评分器（zzshare）初始化完成")

    def _format_date(self, date_str: str) -> str:
        """
        将日期字符串统一转换为 YYYYMMDD 格式

        参数:
            date_str: 日期字符串，支持 YYYYMMDD 或 YYYY-MM-DD
        返回:
            str: YYYYMMDD 格式的日期字符串
        """
        date_str = date_str.strip()
        if "-" in date_str:
            return date_str.replace("-", "")
        return date_str

    def _get_stock_sectors(self, stock_code: str) -> List[str]:
        """
        获取个股所属板块列表（通过 zzshare 板块成分股映射）

        参数:
            stock_code: 股票代码（6位数字）
        返回:
            List[str]: 板块名称列表
        """
        cache_key = f"stock_sectors_{stock_code}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            # 获取当天板块成分股映射
            stock_map = self._fetcher._build_stock_map(
                datetime.now().strftime("%Y%m%d"), 14, 100
            )
            if stock_code in stock_map:
                sectors = [stock_map[stock_code]]
                logger.debug(f"个股 {stock_code} 所属板块: {sectors}")
                self._cache.set(cache_key, sectors)
                return sectors
            else:
                logger.debug(f"个股 {stock_code} 未找到所属板块")
                self._cache.set(cache_key, [])
                return []
        except Exception as e:
            logger.warning(f"获取个股板块映射失败: {stock_code}, {e}")
            return []

    def _get_sector_daily_data(self, date: str) -> Optional[pd.DataFrame]:
        """
        获取指定日期所有板块的行情数据

        参数:
            date: 交易日期（YYYYMMDD 格式）
        返回:
            DataFrame: 板块行情数据，失败返回 None
        """
        cache_key = f"sector_daily_{date}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            data = self._fetcher.fetch(
                plate_type=14,  # 行业板块
                date=date,
                top_n=100,
            )
            if data and "sectors" in data:
                df = pd.DataFrame(data["sectors"])
                self._cache.set(cache_key, df)
                return df
            return None
        except Exception as e:
            logger.error(f"获取板块行情失败: {date}, {e}")
            return None

    def _calculate_sector_score(
        self, sector_name: str, date: str
    ) -> Tuple[float, SectorDetail]:
        """
        计算单个板块的综合得分

        综合公式：
          板块强度得分 = 基准分(50) + 涨幅排名得分 + 涨停得分

        参数:
            sector_name: 板块名称
            date: 交易日期（YYYYMMDD 格式）
        返回:
            Tuple[float, SectorDetail]: (板块得分, 板块详情对象)
        """
        detail = SectorDetail()
        detail.sector_name = sector_name

        # 获取当日板块行情
        df = self._get_sector_daily_data(date)
        if df is None or df.empty:
            logger.debug(f"无板块行情数据: {date}")
            return BASE_SCORE, detail

        # 查找目标板块
        sector_rows = df[df["plate_name"] == sector_name]
        if sector_rows.empty:
            logger.debug(f"板块 {sector_name} 当日无行情数据")
            return BASE_SCORE, detail

        # 计算涨幅排名得分
        rank = int(sector_rows.iloc[0].get("rank", 999))
        if rank <= RANK_TOP_N:
            rank_score = RANK_SCORE
            detail.rank_score = rank_score
            logger.debug(f"板块 {sector_name} 排名第{rank}，+{rank_score}分")
        else:
            detail.rank_score = 0

        # 计算涨停得分（从同花顺涨停热榜获取）
        limit_up_score = self._calculate_limit_up_score(sector_name, date)
        detail.limit_up_score = limit_up_score

        # 综合得分
        total_score = BASE_SCORE + rank_score + limit_up_score
        return total_score, detail

    def _calculate_limit_up_score(self, sector_name: str, date: str) -> float:
        """
        计算板块涨停得分

        参数:
            sector_name: 板块名称
            date: 交易日期（YYYYMMDD 格式）
        返回:
            float: 涨停得分
        """
        try:
            # 获取涨停热榜
            hot_stocks = self._fetcher.get_uplimit_hot(date)
            if not hot_stocks:
                return 0

            # 统计该板块涨停股数量（简化：通过成分股映射）
            # 实际项目中需要从更精确的数据源获取
            # 这里暂时返回 0，后续可根据需要优化
            return 0
        except Exception as e:
            logger.debug(f"计算涨停得分失败: {e}")
            return 0

    # ============================================================
    # 公开接口方法
    # ============================================================

    def calculate_score(
        self, stock_code: str, score_date: str
    ) -> Tuple[float, SectorDetail]:
        """
        计算指定股票在指定日期的板块强度得分

        流程：
          1. 获取个股所属板块列表
          2. 计算每个板块的综合得分
          3. 取最高分板块作为个股板块得分
          4. 检查一票否决条件

        参数:
            stock_code: 股票代码（6位数字）
            score_date: 评分日期，格式 YYYY-MM-DD 或 YYYYMMDD
        返回:
            Tuple[float, SectorDetail]: (板块强度得分, 板块详情对象)
        """
        logger.debug(f"开始计算板块强度得分: {stock_code}, 日期: {score_date}")

        # 检查是否为交易日
        from utils.trade_date_utils import is_trading_day
        if not is_trading_day(score_date):
            logger.debug(f"日期 {score_date} 不是交易日，跳过板块强度评分")
            detail = SectorDetail()
            return 0, detail

        # 统一日期格式
        formatted_date = self._format_date(score_date)

        # 获取个股所属板块
        sectors = self._get_stock_sectors(stock_code)
        if not sectors:
            logger.debug(f"个股无板块映射: {stock_code}，返回基准分")
            return BASE_SCORE, SectorDetail()

        # 计算每个板块的得分，取最高分
        best_score = BASE_SCORE
        best_detail = SectorDetail(sector_name=sectors[0])

        for sector_name in sectors:
            sector_score, sector_detail = self._calculate_sector_score(
                sector_name, formatted_date
            )
            logger.debug(f"板块 {sector_name} 得分: {sector_score}")
            if sector_score > best_score:
                best_score = sector_score
                best_detail = sector_detail

        # 检查一票否决条件
        if best_score <= VETO_SCORE:
            best_detail.veto = True
            best_detail.veto_reason = (
                f"板块 {best_detail.sector_name} 得分 {best_score}，"
                f"触发一票否决（得分 ≤ {VETO_SCORE}）"
            )
            logger.warning(f"股票 {stock_code} 板块强度一票否决: {best_detail.veto_reason}")
            return VETO_SCORE, best_detail

        logger.debug(f"股票 {stock_code} 板块强度得分: {best_score}")
        return best_score, best_detail

    def check_veto(
        self,
        stock_code: str,
        score_date: str,
        sector_score: float = None,
    ) -> Tuple[bool, str]:
        """
        检查板块强度一票否决条件

        一票否决条件：
          - 板块得分 = -100 时，个股直接淘汰

        参数:
            stock_code: 股票代码（6位数字）
            score_date: 评分日期
            sector_score: 已计算的板块得分（可选）
        返回:
            Tuple[bool, str]: (是否触发一票否决, 否决原因)
        """
        if sector_score is None:
            sector_score, detail = self.calculate_score(stock_code, score_date)
        else:
            detail = SectorDetail()

        if sector_score <= VETO_SCORE:
            reason = f"板块得分 {sector_score}，触发一票否决（得分 ≤ {VETO_SCORE}）"
            logger.warning(f"股票 {stock_code} 板块一票否决: {reason}")
            return True, reason

        return False, ""
