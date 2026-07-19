"""상위 시간대 추세 전환 감지.

SMA7과 SMA25의 교차(골든/데드 크로스)를 추세 기준으로 삼는다:
  SMA7 > SMA25 → 상승 추세, SMA7 < SMA25 → 하락 추세

닫힌 캔들만 사용하고, 직전 실행에서 저장한 추세와 비교해 달라졌을 때만
"추세 전환" 이벤트를 만든다 (돌파 알림과 같은 상태 파일 방식 → 중복 알림 없음).
전환을 예측하는 것이 아니라 확정된 전환을 즉시 알리는 방식이라 오보가 적다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FAST_SMA = 7
SLOW_SMA = 25


@dataclass
class TrendShift:
    """추세 전환 이벤트 하나."""
    symbol: str
    interval: str
    old_trend: str   # "상승" 또는 "하락"
    new_trend: str
    close: float


def current_trend(ohlcv: pd.DataFrame) -> str | None:
    """닫힌 캔들 기준 현재 추세를 반환한다 (데이터 부족 시 None)."""
    closed = ohlcv.iloc[:-1]
    if len(closed) < SLOW_SMA + 1:
        return None
    close = closed["close"]
    fast = close.rolling(FAST_SMA).mean().iloc[-1]
    slow = close.rolling(SLOW_SMA).mean().iloc[-1]
    return "상승" if fast > slow else "하락"


def check_trend_shift(
    ohlcv: pd.DataFrame,
    symbol: str,
    interval: str,
    stored_trend: str | None,
) -> tuple[TrendShift | None, str | None]:
    """저장된 추세와 현재 추세를 비교해 (전환 이벤트, 새로 저장할 추세)를 반환한다.

    첫 실행(stored_trend가 None)은 기록만 하고 알리지 않는다.
    """
    trend = current_trend(ohlcv)
    if trend is None:
        return None, stored_trend
    if stored_trend is not None and stored_trend != trend:
        shift = TrendShift(
            symbol=symbol.upper(), interval=interval,
            old_trend=stored_trend, new_trend=trend,
            close=float(ohlcv["close"].iloc[-2]),
        )
        return shift, trend
    return None, trend


def format_trend_shift(shift: TrendShift) -> str:
    """추세 전환 이벤트를 텔레그램 메시지 문자열로 만든다."""
    arrow = "📈" if shift.new_trend == "상승" else "📉"
    cross = "상향" if shift.new_trend == "상승" else "하향"
    lines = [
        f"📢 {shift.symbol} {shift.interval} 추세 전환",
        "",
        f"{arrow} {shift.old_trend} → {shift.new_trend}",
        f"기준: {FAST_SMA}캔들 평균이 {SLOW_SMA}캔들 평균을 {cross} 교차 (종가 기준)",
        f"현재가: {shift.close:,.4f}",
        "",
        "※ 참고용이며 재무적 조언이 아닙니다.",
    ]
    return "\n".join(lines)
