"""
zzshare 板块数据源 - 适配 bigApush DataSource 架构
参考: https://pypi.org/project/zzshare/
接口: plates_rank / market_plate_stocks / uplimit_hot / rt_k / plates_list

实际API字段（2026-09验证）:
  - plates_rank: date1, plate_code, score, plate_name, rate(涨跌幅), speed, volume_ration
  - market_plate_stocks: stock_name, stock_code, rank, rank_diff, attention, last_pct
  - uplimit_hot: dict格式，含涨停股列表
  - rt_k: close, high_limit, low_limit, volume

plates_type:
  - 14 = 行业板块
  - 15 = 概念板块
  - 17 = 题材板块
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
from utils.base_fetcher import DataSource

logger = logging.getLogger(__name__)

# 兼容 LoongArch64 的 zzshare
try:
    from zzshare.client import DataApi
    ZZSHARE_AVAILABLE = True
except ImportError:
    ZZSHARE_AVAILABLE = False
    DataApi = None
    logger.warning("zzshare 未安装，板块数据将降级为 Mock")


class TokenBucket:
    """令牌桶限流器"""

    def __init__(self, capacity: int = 30, refill_period: int = 60):
        self.capacity = capacity
        self.refill_period = refill_period
        self._tokens = float(capacity)
        self._last = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self._last
        self._tokens = min(self.capacity, self._tokens + elapsed * self.capacity / self.refill_period)
        self._last = now

    def acquire(self) -> bool:
        self._refill()
        if self._tokens >= 1:
            self._tokens -= 1
            return True
        wait = (1 - self._tokens) * self.refill_period / self.capacity
        time.sleep(wait)
        return True


def _ensure_df(data):
    """确保返回 DataFrame（zzshare 返回 list 或 dict，统一转换）"""
    if data is None:
        return None
    import pandas as pd
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        # dict 转 DataFrame
        df = pd.DataFrame(list(data.items()), columns=['key', 'value'])
        return df
    return data


class ZzShareSectorFetcher(DataSource):
    """
    zzshare 板块数据源实现
    继承 DataSource 接口，适配 bigApush 数据源架构
    """

    def __init__(self, cache_dir: str = None, token: str = None):
        super().__init__("zzshare", priority=2)  # 优先级低于 Tushare/EastMoney
        self.cache_dir = Path(cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "bigapush_sector"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._api = self._create_api(token)
        self._rate_limiter = TokenBucket(capacity=28)  # 留2次缓冲
        self._sector_map_cache: Dict[str, Dict] = {}
        self._daily_cache: Dict[str, Dict] = {}

        logger.info(f"ZzShareSectorFetcher 初始化完成 (api={'ok' if self._api else 'fail'})")

    @staticmethod
    def _create_api(token: str = None) -> Optional[DataApi]:
        try:
            return DataApi(token=token) if token else DataApi()
        except Exception as e:
            logger.warning(f"zzshare 连接失败: {e}")
            return None

    def is_available(self) -> bool:
        return ZZSHARE_AVAILABLE and self._api is not None

    def fetch(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        统一入口：获取板块数据
        kwargs:
            plate_type: 14=行业, 15=概念, 17=题材
            date: YYYYMMDD，默认今天
            top_n: 返回前N个板块
        """
        plate_type = kwargs.get("plate_type", 14)
        date = kwargs.get("date", datetime.now().strftime("%Y%m%d"))
        top_n = kwargs.get("top_n", 50)

        if not self.is_available():
            logger.warning("zzshare 不可用，降级 Mock")
            return self._mock_result(plate_type, date, top_n)

        try:
            self._rate_limiter.acquire()

            # 获取板块排名
            rank_list = self._api.plates_rank(plate_type=plate_type, date1=date, limit=top_n)
            rank_df = _ensure_df(rank_list)

            if rank_df is None or len(rank_df) == 0:
                logger.warning(f"无板块数据返回: date={date}")
                return None

            # 构建结果（适配实际API字段）
            sectors = []
            for _, row in rank_df.iterrows():
                sectors.append({
                    "plate_name": str(row.get("plate_name", "")),
                    "plate_code": str(row.get("plate_code", "")),
                    "change_pct": float(row.get("rate", 0)),  # rate=涨跌幅
                    "volume_ration": float(row.get("volume_ration", 1.0)),  # 量比
                    "score": float(row.get("score", 0)),  # 综合评分
                })

            # 按涨跌幅排序（取前N个）
            sectors.sort(key=lambda x: x["change_pct"], reverse=True)
            for i, s in enumerate(sectors):
                s["rank"] = i + 1

            # 获取成分股映射（可选）
            include_stocks = kwargs.get("include_stocks", False)
            stock_map = {}
            if include_stocks:
                stock_map = self._build_stock_map(date, plate_type, top_n)

            return {
                "plate_type": plate_type,
                "date": date,
                "sectors": sectors[:top_n],
                "stock_map": stock_map,
                "source": "zzshare",
            }

        except Exception as e:
            logger.error(f"zzshare 获取板块数据失败: {e}")
            return None

    def _build_stock_map(self, date: str, plate_type: int, top_n: int) -> Dict[str, str]:
        """构建股票→板块映射"""
        cache_key = f"stock_map_{date}_{plate_type}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        # 尝试缓存（7天TTL）
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                cached_at = datetime.fromisoformat(data["cached_at"])
                ttl = timedelta(seconds=604800)  # 7天
                if datetime.now() - cached_at < ttl:
                    return data["mapping"]
            except Exception:
                pass

        # 重新拉取
        mapping = {}
        try:
            rank_list = self._api.plates_rank(plate_type=plate_type, date1=date, limit=top_n)
            rank_df = _ensure_df(rank_list)

            for _, row in rank_df.iterrows():
                plate_name = str(row.get("plate_name", ""))
                plate_code = str(row.get("plate_code", ""))
                if not plate_name or not plate_code:
                    continue

                stocks_list = self._api.market_plate_stocks(
                    plate_type=plate_type, plate_code=plate_code, date1=date, limit=200
                )
                stocks_df = _ensure_df(stocks_list)

                if stocks_df is not None and len(stocks_df) > 0:
                    for _, s in stocks_df.iterrows():
                        stock_code = str(s.get("stock_code", ""))
                        if stock_code:
                            mapping[stock_code] = plate_name

            # 写入缓存
            cache_file.write_text(json.dumps({
                "mapping": mapping,
                "cached_at": datetime.now().isoformat(),
            }, ensure_ascii=False))

            logger.info(f"板块映射已缓存: {len(mapping)}只股票, {len(rank_df)}个板块")

        except Exception as e:
            logger.error(f"构建股票映射失败: {e}")

        return mapping

    def get_uplimit_hot(self, date: str = None) -> List[str]:
        """获取历史当日涨停池（防未来函数）"""
        if not self.is_available():
            return []

        date = date or datetime.now().strftime("%Y%m%d")
        try:
            self._rate_limiter.acquire()
            hot_data = self._api.uplimit_hot(date1=date)

            stocks = []
            if isinstance(hot_data, dict):
                # dict 格式，尝试提取 stock_code
                for k, v in hot_data.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                stock_code = item.get("stock_code") or item.get("ts_code")
                                if stock_code:
                                    stocks.append(str(stock_code))
                    elif isinstance(v, str):
                        # 处理逗号分隔的字符串
                        for code in v.split(","):
                            code = code.strip()
                            if len(code) == 6 and code.isdigit():
                                stocks.append(code)
                    elif isinstance(v, (int, float)) and len(str(v)) == 6:
                        stocks.append(str(int(v)))
            elif isinstance(hot_data, list):
                for item in hot_data:
                    if isinstance(item, dict):
                        stock_code = item.get("stock_code") or item.get("ts_code")
                        if stock_code:
                            stocks.append(str(stock_code))
                    elif isinstance(item, str) and len(item) >= 6:
                        # 处理逗号分隔
                        for code in item.split(","):
                            code = code.strip()
                            if len(code) == 6 and code.isdigit():
                                stocks.append(code)

            return stocks

        except Exception as e:
            logger.error(f"获取涨停池失败: {e}")

        return []

    def get_rt_k(self, ts_code: str) -> Optional[Dict]:
        """获取实时K线数据（含涨跌停价字段）"""
        if not self.is_available():
            return None

        try:
            self._rate_limiter.acquire()
            # 补全交易所后缀
            if "." not in ts_code:
                ts_code = ts_code + (".SH" if ts_code.startswith("6") else ".SZ")

            k_list = self._api.rt_k(ts_code=ts_code)
            k_df = _ensure_df(k_list)
            if k_df is not None and len(k_df) > 0:
                row = k_df.iloc[-1]
                close = float(row.get("close", 0))
                high_limit = float(row.get("high_limit", 0))
                return {
                    "ts_code": ts_code,
                    "close": close,
                    "high_limit": high_limit,
                    "low_limit": float(row.get("low_limit", 0)),
                    "is_zt": close >= high_limit - 1e-6,
                }
        except Exception as e:
            logger.error(f"获取K线失败 {ts_code}: {e}")

        return None

    def _mock_result(self, plate_type: int, date: str, top_n: int) -> Dict:
        """Mock 数据兜底"""
        return {
            "plate_type": plate_type,
            "date": date,
            "sectors": [],
            "stock_map": {},
            "source": "mock",
        }


class MockSectorFetcher(DataSource):
    """Mock 板块数据源：zzshare 不可用时的降级兜底"""

    def __init__(self):
        super().__init__("mock_sector", priority=99)

    def is_available(self) -> bool:
        return True

    def fetch(self, **kwargs) -> Dict:
        return {
            "plate_type": kwargs.get("plate_type", 14),
            "date": kwargs.get("date", datetime.now().strftime("%Y%m%d")),
            "sectors": [],
            "stock_map": {},
            "source": "mock",
        }


def create_fetcher(**kwargs) -> DataSource:
    """工厂函数：创建板块数据源 Fetcher"""
    if ZZSHARE_AVAILABLE:
        try:
            fetcher = ZzShareSectorFetcher(**kwargs)
            if fetcher.is_available():
                logger.info("板块数据源：ZzShare")
                return fetcher
        except Exception as e:
            logger.warning(f"ZzShare 初始化失败，降级 Mock: {e}")

    logger.info("板块数据源：Mock（离线兜底）")
    return MockSectorFetcher()
