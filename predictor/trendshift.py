"""상위 시간대 추세 전환 감지 — 휩쏘(가짜 전환)를 걸러내는 보수적 판정.

SMA 단순 교차(fast > slow)만 쓰면 두 평균선이 붙어 있을 때 미세하게 스치기만
해도 전환으로 잡혀 알림이 시시각각 뒤집힌다. 그래서 네 겹의 조건을 모두
통과해야만 "추세 전환"으로 인정한다:

  1. 이격 마진 — 두 평균선이 가격의 MARGIN 비율 이상 벌어져야 방향으로 인정.
     그 사이는 '중립'이며, 중립을 오가는 것만으로는 전환 알림이 나가지 않는다.
  2. 확인 캔들 — 새 방향이 CONFIRM_CANDLES개 연속 닫힌 캔들에서 유지돼야 한다.
  3. 추세 강도 — ADX가 MIN_ADX 미만인 횡보장에서는 전환을 선언하지 않는다.
  4. 쿨다운 — 직전 전환 후 COOLDOWN_CANDLES개가 지나기 전에는 다시 알리지 않는다.

전환을 예측하는 것이 아니라, 확정된 전환만 늦더라도 정확하게 알리는 쪽을 택한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import adx

FAST_SMA = 7
SLOW_SMA = 25
MARGIN = 0.004          # 두 평균선의 최소 이격 (가격의 0.4%)
CONFIRM_CANDLES = 3     # 새 방향이 유지돼야 하는 연속 캔들 수
MIN_ADX = 20.0          # 이 미만이면 횡보장으로 보고 전환 판정 보류
COOLDOWN_CANDLES = 6    # 직전 전환 후 이만큼은 새 전환을 알리지 않음


@dataclass
class TrendShift:
    """추세 전환 이벤트 하나."""
    symbol: str
    interval: str
    old_trend: str   # "상승" 또는 "하락"
    new_trend: str
    close: float
    adx: float       # 전환 시점의 추세 강도
    held: int        # 새 방향이 유지된 캔들 수


def _direction_series(closed: pd.DataFrame) -> pd.Series:
    """캔들별 방향 라벨 시리즈 — 이격이 마진 미만이면 '중립'."""
    close = closed["close"]
    fast = close.rolling(FAST_SMA).mean()
    slow = close.rolling(SLOW_SMA).mean()
    gap = (fast - slow) / close
    return pd.Series(
        ["상승" if g >= MARGIN else "하락" if g <= -MARGIN else "중립" for g in gap],
        index=closed.index,
    )


def current_trend(ohlcv: pd.DataFrame,
                  confirm: int = CONFIRM_CANDLES) -> str | None:
    """확인 캔들까지 통과한 현재 추세를 반환한다.

    마지막 confirm개 닫힌 캔들이 모두 같은 방향이어야 그 방향으로 인정하고,
    하나라도 다르거나 '중립'이 섞이면 None(판정 보류)을 반환한다.
    """
    closed = ohlcv.iloc[:-1]
    if len(closed) < SLOW_SMA + confirm:
        return None
    recent = _direction_series(closed).iloc[-confirm:]
    first = recent.iloc[0]
    if first == "중립" or not (recent == first).all():
        return None
    return first


def _held_candles(ohlcv: pd.DataFrame, trend: str) -> int:
    """현재 방향이 몇 개 캔들 연속 유지됐는지 센다."""
    labels = _direction_series(ohlcv.iloc[:-1])
    held = 0
    for label in reversed(labels.tolist()):
        if label != trend:
            break
        held += 1
    return held


def check_trend_shift(
    ohlcv: pd.DataFrame,
    symbol: str,
    interval: str,
    stored: dict | str | None,
) -> tuple[TrendShift | None, dict | None]:
    """저장된 상태와 비교해 (전환 이벤트, 새로 저장할 상태)를 반환한다.

    첫 실행(stored 없음)은 기록만 하고 알리지 않는다. 판정이 보류(중립·확인
    미달)면 저장된 추세를 그대로 유지해 '중립을 스치는' 왕복 알림을 막는다.

    Args:
        stored: 이전 상태. 구버전 호환을 위해 문자열도 받는다.
    """
    # 구버전 상태(문자열) 호환
    if isinstance(stored, str):
        stored = {"trend": stored, "candles_since_shift": COOLDOWN_CANDLES}
    prev_trend = (stored or {}).get("trend")
    since = (stored or {}).get("candles_since_shift", COOLDOWN_CANDLES)

    trend = current_trend(ohlcv)
    if trend is None:
        # 판정 보류 — 이전 추세를 유지한 채 쿨다운만 진행
        if stored is None:
            return None, None
        return None, {"trend": prev_trend, "candles_since_shift": since + 1}

    if prev_trend is None:
        return None, {"trend": trend, "candles_since_shift": COOLDOWN_CANDLES}

    if prev_trend == trend:
        return None, {"trend": trend, "candles_since_shift": since + 1}

    # 방향이 바뀌었다 — 추세 강도와 쿨다운을 확인
    closed = ohlcv.iloc[:-1]
    adx_val = float(adx(closed, 14)[0].iloc[-1])
    if adx_val < MIN_ADX or since < COOLDOWN_CANDLES:
        # 횡보장이거나 쿨다운 중 — 알리지 않고 이전 추세를 유지
        return None, {"trend": prev_trend, "candles_since_shift": since + 1}

    shift = TrendShift(
        symbol=symbol.upper(), interval=interval,
        old_trend=prev_trend, new_trend=trend,
        close=float(closed["close"].iloc[-1]),
        adx=adx_val, held=_held_candles(ohlcv, trend),
    )
    return shift, {"trend": trend, "candles_since_shift": 0}


def format_trend_shift(shift: TrendShift) -> str:
    """추세 전환 이벤트를 텔레그램 메시지 문자열로 만든다."""
    arrow = "📈" if shift.new_trend == "상승" else "📉"
    cross = "상향" if shift.new_trend == "상승" else "하향"
    lines = [
        f"📢 {shift.symbol} {shift.interval} 추세 전환 (확정)",
        "",
        f"{arrow} {shift.old_trend} → {shift.new_trend}",
        f"현재가: {shift.close:,.4f}",
        "",
        "확정 조건 (모두 충족):",
        f"  • {FAST_SMA}/{SLOW_SMA}캔들 평균이 {MARGIN:.1%} 이상 벌어진 채 {cross} 교차",
        f"  • {shift.held}캔들 연속 같은 방향 유지 (최소 {CONFIRM_CANDLES})",
        f"  • ADX {shift.adx:.0f} — 추세 국면 확인 (최소 {MIN_ADX:.0f})",
        f"  • 직전 전환 이후 {COOLDOWN_CANDLES}캔들 이상 경과",
        "",
        "※ 참고용이며 재무적 조언이 아닙니다.",
    ]
    return "\n".join(lines)
