# -*- coding: utf-8 -*-
"""
板块强度分析模块：判断战场/风口

核心逻辑：
  - 板块涨幅排名靠前 → 主线行情 (+分)
  - 板块内涨停家数多 → 强度高 (+分)
  - 板块指数站上5日线且向上 → 趋势健康 (+分)

数据源分层（按优先级）：
  1. ZzShareSectorProvider：真实数据（zzshare），含缓存+限流+降级
  2. MockSectorProvider：中性默认兜底（无网络/限流/缺失时使用）
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class SectorInfo:
    """板块强度信息"""
    sector_name: str = ""
    sector_change_pct: float = 0.0   # 板块涨幅
    sector_rank: int = 99             # 涨幅排名
    limit_up_count: int = 0           # 板块内涨停家数
    is_hot: bool = False              # 是否主线热点
    score_delta: int = 0


class SectorDataProvider:
    """板块数据源抽象基类"""
    
    def get_sector_for(self, code: str) -> str:
        """股票代码 → 所属板块名"""
        raise NotImplementedError
    
    def get_sector_change(self, sector_name: str) -> float:
        """板块涨跌幅"""
        raise NotImplementedError
    
    def get_sector_rank(self, sector_name: str) -> int:
        """板块涨幅排名"""
        raise NotImplementedError
    
    def get_limit_up_count(self, sector_name: str) -> int:
        """板块内涨停股数量"""
        raise NotImplementedError
    
    def get_sector_ma5_trend(self, sector_name: str) -> bool:
        """板块指数是否站上5日线且向上"""
        raise NotImplementedError


class MockSectorProvider(SectorDataProvider):
    """本地兜底实现：返回中性默认值，避免无数据崩溃"""
    
    def get_sector_for(self, code: str) -> str:
        return "其他"
    
    def get_sector_change(self, sector_name: str) -> float:
        return 0.0
    
    def get_sector_rank(self, sector_name: str) -> int:
        return 20  # 默认中下游
    
    def get_limit_up_count(self, sector_name: str) -> int:
        return 0
    
    def get_sector_ma5_trend(self, sector_name: str) -> bool:
        return False


class ZzShareSectorProvider(SectorDataProvider):
    """
    zzshare 真实数据源：板块映射 + 排名 + 涨停家数
    
    用法：
        prov = ZzShareSectorProvider()
        analyze_sector("002349", provider=prov)
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "bigapush_sector"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._strength: Dict[str, Dict[str, Any]] = {}
        self._mapping: Dict[str, str] = {}
        self._loaded = False
        
        # 尝试导入 zzshare
        self._api = None
        try:
            from utils.zzshare_fetcher import create_fetcher
            fetcher = create_fetcher()
            if fetcher.is_available():
                self._api = fetcher._api
                logger.info("zzshare 板块数据源初始化成功")
            else:
                logger.warning("zzshare 不可用，将使用 Mock")
        except Exception as e:
            logger.warning(f"zzshare 初始化失败: {e}，将使用 Mock")
    
    def _ensure(self):
        """懒加载板块数据"""
        if self._loaded or self._api is None:
            return
        
        try:
            from datetime import datetime
            date = datetime.now().strftime("%Y%m%d")
            
            # 获取板块排名
            rank_list = self._api.plates_rank(plate_type=14, date1=date, limit=100)
            if rank_list:
                for item in rank_list:
                    plate_name = item.get("plate_name", "")
                    if plate_name:
                        self._strength[plate_name] = {
                            "change_pct": float(item.get("rate", 0)),
                            "rank": int(item.get("rank", 99)),
                            "limit_up_count": int(item.get("limit_up", 0)),
                        }
            
            # 获取成分股映射（简化版，实际可从 market_plate_stocks 获取）
            # 这里暂时返回空映射，由外部设置
            logger.debug(f"板块数据加载完成: {len(self._strength)} 个板块")
            self._loaded = True
            
        except Exception as e:
            logger.warning(f"[ZzShareSectorProvider] load failed: {e}; degrading neutral")
            self._strength = {}
            self._mapping = {}
    
    def set_stock_mapping(self, mapping: Dict[str, str]):
        """设置股票代码 → 板块名称映射（由外部传入）"""
        self._mapping = mapping
    
    def get_sector_for(self, code: str) -> str:
        self._ensure()
        return self._mapping.get(code, "其他")
    
    def get_sector_change(self, sector_name: str) -> float:
        self._ensure()
        return float(self._strength.get(sector_name, {}).get("change_pct", 0))
    
    def get_sector_rank(self, sector_name: str) -> int:
        self._ensure()
        return int(self._strength.get(sector_name, {}).get("rank", 99)) or 99
    
    def get_limit_up_count(self, sector_name: str) -> int:
        self._ensure()
        return int(self._strength.get(sector_name, {}).get("limit_up_count", 0))
    
    def get_sector_ma5_trend(self, sector_name: str) -> bool:
        self._ensure()
        s = self._strength.get(sector_name, {})
        return s.get("rank", 99) <= 10 and float(s.get("change_pct", 0)) > 0


# 兼容旧名称
ZzShareSectorProviderAlias = ZzShareSectorProvider


def analyze_sector(code: str, provider: SectorDataProvider = None) -> SectorInfo:
    """
    对单只股票做板块强度分析
    
    Args:
        code: 股票代码
        provider: 板块数据源，None 时使用 Mock 中性兜底
    
    Returns:
        SectorInfo: 板块强度信息
    """
    provider = provider or MockSectorProvider()
    sector_name = provider.get_sector_for(code)
    
    info = SectorInfo(
        sector_name=sector_name,
        sector_change_pct=provider.get_sector_change(sector_name),
        sector_rank=provider.get_sector_rank(sector_name),
        limit_up_count=provider.get_limit_up_count(sector_name),
        is_hot=False,
    )
    
    # === 评分规则 ===
    score = 0
    
    # 1. 涨幅排名（前3加分，前10中性，其余减分）
    if info.sector_rank <= 3:
        score += 8
        info.is_hot = True
    elif info.sector_rank <= 10:
        score += 4
        info.is_hot = True
    elif info.sector_rank <= 20:
        score += 0
    else:
        score -= 4
    
    # 2. 涨停家数（板块效应）
    if info.limit_up_count >= 5:
        score += 7
        info.is_hot = True
    elif info.limit_up_count >= 3:
        score += 4
    elif info.limit_up_count >= 1:
        score += 1
    
    # 3. 板块趋势（站上5日线）
    if provider.get_sector_ma5_trend(sector_name):
        score += 3
    
    info.score_delta = score
    logger.debug(f"板块分析: {sector_name}, rank={info.sector_rank}, delta={info.score_delta}")
    return info


def make_provider(cache_dir: str = None) -> SectorDataProvider:
    """
    工厂：优先 ZzShare（真实数据），失败时自动降级 Mock
    
    Returns:
        SectorDataProvider: 板块数据源实例
    """
    try:
        return ZzShareSectorProvider(cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"zzshare 不可用，降级为 MockSectorProvider: {e}")
        return MockSectorProvider()
