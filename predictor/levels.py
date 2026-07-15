"""지지선/저항선 자동 탐지.

방식:
  1. 스윙 고점/저점(피벗) 탐지 — 좌우 window개 캔들보다 높은 고가/낮은 저가
  2. 가까운 피벗 가격끼리 클러스터링 (허용 오차 = 가격의 tolerance 비율)
  3. 터치 횟수(해당 레벨 근처에 온 캔들 수)로 강도 산정
현재가 아래의 레벨은 지지선, 위의 레벨은 저항선으로 분류한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Level:
    """지지 또는 저항 레벨 하나."""
    price: float
    kind: str          # "지지" 또는 "저항"
    touches: int       # 이 레벨 근처에 닿은 캔들 수 (강도)
    distance_pct: float  # 현재가 대비 거리 (지지는 음수, 저항은 양수)


def _find_pivot_prices(df: pd.DataFrame, window: int = 5) -> np.ndarray:
    """좌우 window개 캔들 기준의 스윙 고점/저점 가격 배열을 반환한다."""
    size = 2 * window + 1
    high = df["high"]
    low = df["low"]
    pivot_highs = high[high == high.rolling(size, center=True).max()]
    pivot_lows = low[low == low.rolling(size, center=True).min()]
    return np.concatenate([pivot_highs.to_numpy(), pivot_lows.to_numpy()])


def _cluster(prices: np.ndarray, tolerance: float) -> list[list[float]]:
    """정렬 후 인접 가격이 tolerance 비율 이내면 같은 클러스터로 묶는다."""
    clusters: list[list[float]] = []
    for p in np.sort(prices):
        if clusters and (p - clusters[-1][-1]) / clusters[-1][-1] <= tolerance:
            clusters[-1].append(float(p))
        else:
            clusters.append([float(p)])
    return clusters


def find_levels(
    df: pd.DataFrame,
    window: int = 5,
    tolerance: float = 0.005,
    min_touches: int = 2,
    max_levels: int = 4,
) -> tuple[list[Level], list[Level]]:
    """OHLCV 데이터에서 (지지선 목록, 저항선 목록)을 찾는다.

    Args:
        df: fetch_klines()가 반환한 OHLCV DataFrame
        window: 피벗 판정에 사용할 좌우 캔들 수
        tolerance: 같은 레벨로 묶을 가격 허용 오차 비율 (0.005 = 0.5%)
        min_touches: 레벨로 인정할 최소 터치 횟수
        max_levels: 지지/저항 각각 반환할 최대 개수 (현재가에 가까운 순)

    Returns:
        (지지선 목록, 저항선 목록) — 각각 현재가에 가까운 순으로 정렬
    """
    if len(df) < 2 * window + 1:
        return [], []

    current = float(df["close"].iloc[-1])
    pivots = _find_pivot_prices(df, window)
    if len(pivots) == 0:
        return [], []

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    supports: list[Level] = []
    resistances: list[Level] = []
    for cluster in _cluster(pivots, tolerance):
        price = float(np.mean(cluster))
        # 터치 = 캔들의 고가~저가 범위가 레벨의 tolerance 밴드와 겹침
        band = price * tolerance
        touches = int(np.sum((low <= price + band) & (high >= price - band)))
        if touches < min_touches:
            continue
        distance = (price - current) / current
        if price < current:
            supports.append(Level(price, "지지", touches, distance))
        elif price > current:
            resistances.append(Level(price, "저항", touches, distance))

    supports.sort(key=lambda lv: -lv.price)     # 현재가 바로 아래부터
    resistances.sort(key=lambda lv: lv.price)   # 현재가 바로 위부터
    return supports[:max_levels], resistances[:max_levels]


def format_levels(supports: list[Level], resistances: list[Level],
                  current: float) -> str:
    """지지/저항 레벨을 사람이 읽기 좋은 문자열로 만든다."""
    lines = [f"현재가 {current:,.4f}"]
    if resistances:
        lines.append("저항선 (위):")
        for lv in reversed(resistances):
            lines.append(f"  🔺 {lv.price:,.4f} ({lv.distance_pct:+.1%}, 터치 {lv.touches}회)")
    else:
        lines.append("저항선: 근처에 뚜렷한 레벨 없음")
    if supports:
        lines.append("지지선 (아래):")
        for lv in supports:
            lines.append(f"  🔻 {lv.price:,.4f} ({lv.distance_pct:+.1%}, 터치 {lv.touches}회)")
    else:
        lines.append("지지선: 근처에 뚜렷한 레벨 없음")
    return "\n".join(lines)
