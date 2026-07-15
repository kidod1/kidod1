"""바이낸스 공개 API에서 캔들(K-line) 데이터를 가져오는 모듈.

API 키 없이 사용 가능한 공개 엔드포인트(/api/v3/klines)를 사용한다.
한 번의 요청으로 최대 1000개의 캔들만 받을 수 있으므로,
그 이상이 필요하면 endTime을 거슬러 올라가며 여러 번 요청한다.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_PER_REQUEST = 1000

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """바이낸스에서 캔들 데이터를 받아 DataFrame으로 반환한다.

    Args:
        symbol: 거래쌍 (예: BTCUSDT, ETHUSDT)
        interval: 캔들 간격 (1m, 5m, 15m, 1h, 4h, 1d 등)
        limit: 가져올 캔들 개수 (1000 초과 시 자동으로 나눠서 요청)
        session: 재사용할 requests 세션 (옵션)

    Returns:
        open_time 오름차순으로 정렬된 OHLCV DataFrame
    """
    sess = session or requests.Session()
    frames: list[pd.DataFrame] = []
    remaining = limit
    end_time: int | None = None

    while remaining > 0:
        batch = min(remaining, MAX_PER_REQUEST)
        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": batch,
        }
        if end_time is not None:
            params["endTime"] = end_time

        resp = sess.get(BASE_URL + KLINES_ENDPOINT, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break

        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        frames.append(df)
        remaining -= len(rows)
        # 다음 요청은 이번 배치의 가장 오래된 캔들 직전까지
        end_time = int(rows[0][0]) - 1
        if len(rows) < batch:
            break  # 더 이상 과거 데이터가 없음
        if remaining > 0:
            time.sleep(0.25)  # rate limit 여유

    if not frames:
        raise RuntimeError(f"{symbol} {interval} 캔들 데이터를 받지 못했습니다.")

    out = pd.concat(frames, ignore_index=True)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base", "taker_buy_quote"]
    out[numeric_cols] = out[numeric_cols].astype(float)
    out["trades"] = out["trades"].astype(int)
    out["open_time"] = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], unit="ms", utc=True)
    out = out.drop(columns=["ignore"])
    out = out.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return out
