"""지지/저항 돌파 감지.

캔들 '종가' 기준으로 레벨을 넘어서면 돌파로 판정한다 (꼬리만 스친 가짜 돌파 제외).
확정을 위해 레벨 가격에서 buffer(기본 0.2%) 이상 벗어나야 한다.

- 저항 상향 돌파 → 방향 '상승', 목표가 = 다음 저항 (없으면 +2×ATR)
- 지지 하향 돌파 → 방향 '하락', 목표가 = 다음 지지 (없으면 -2×ATR)
돌파된 레벨은 '재이탈 주의선'이 된다 (다시 반대로 넘어오면 돌파 무효).

진행 중인(아직 안 닫힌) 캔들은 종가가 계속 변하므로 판정에서 제외한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import atr
from .levels import find_raw_levels

BREAK_BUFFER = 0.002    # 돌파 확정에 필요한 레벨 이탈 비율 (0.2%)
ATR_TARGET_MULT = 2.0   # 다음 레벨이 없을 때 목표 거리 = ATR × 2
MAX_LOOKBACK = 12       # 상태 파일이 없을 때 되돌아볼 최대 캔들 수


@dataclass
class Breakout:
    """돌파 이벤트 하나."""
    symbol: str
    interval: str
    direction: str        # "상승" 또는 "하락"
    level_price: float    # 돌파된 레벨
    touches: int          # 그 레벨의 터치 횟수 (강도)
    close: float          # 돌파 확정 캔들의 종가
    candle_time: pd.Timestamp
    target: float         # 목표가
    target_kind: str      # "다음 저항" / "다음 지지" / "ATR 기준"


def detect_breakouts(
    ohlcv: pd.DataFrame,
    symbol: str,
    interval: str,
    since: pd.Timestamp | None = None,
    tolerance: float = 0.005,
    min_touches: int = 3,
) -> tuple[list[Breakout], pd.Timestamp | None]:
    """since 이후에 닫힌 캔들들에서 돌파를 찾는다.

    Returns:
        (돌파 목록, 마지막으로 검사한 캔들의 open_time)
        — 두 번째 값을 상태로 저장했다가 다음 실행의 since로 넘기면 중복 알림이 없다.
    """
    # 진행 중인 마지막 캔들 제외 (종가 미확정)
    closed = ohlcv.iloc[:-1]
    if len(closed) < 30:
        return [], None
    last_time = closed["open_time"].iloc[-1]

    # 검사 대상: since 이후 캔들 (상태가 없으면 최근 2개만 — 과거 소급 알림 방지)
    if since is not None:
        scan_idx = closed.index[closed["open_time"] > since]
        scan_idx = scan_idx[-MAX_LOOKBACK:]
    else:
        scan_idx = closed.index[-2:]
    if len(scan_idx) == 0:
        return [], last_time

    # 레벨은 검사 구간 이전 데이터로 계산 (돌파할 대상은 그 전에 형성된 레벨)
    base = closed.loc[:scan_idx[0] - 1]
    levels = find_raw_levels(base, tolerance=tolerance, min_touches=min_touches)
    if not levels:
        return [], last_time

    atr_last = float(atr(closed, 14).iloc[-1])
    closes = closed["close"]
    events: list[Breakout] = []

    for i in scan_idx:
        if i == 0:
            continue
        prev_c = float(closes.loc[i - 1])
        cur_c = float(closes.loc[i])
        for price, touches in levels:
            broke_up = prev_c <= price and cur_c > price * (1 + BREAK_BUFFER)
            broke_down = prev_c >= price and cur_c < price * (1 - BREAK_BUFFER)
            if not (broke_up or broke_down):
                continue

            if broke_up:
                above = [p for p, _ in levels if p > price * (1 + BREAK_BUFFER)]
                target = min(above) if above else cur_c + ATR_TARGET_MULT * atr_last
                kind = "다음 저항" if above else "ATR 기준"
                direction = "상승"
            else:
                below = [p for p, _ in levels if p < price * (1 - BREAK_BUFFER)]
                target = max(below) if below else cur_c - ATR_TARGET_MULT * atr_last
                kind = "다음 지지" if below else "ATR 기준"
                direction = "하락"

            events.append(Breakout(
                symbol=symbol.upper(), interval=interval, direction=direction,
                level_price=price, touches=touches, close=cur_c,
                candle_time=closed["open_time"].loc[i],
                target=target, target_kind=kind,
            ))
    return events, last_time


def format_breakout(b: Breakout) -> str:
    """돌파 이벤트를 텔레그램 메시지 문자열로 만든다."""
    arrow = "📈" if b.direction == "상승" else "📉"
    kind = "저항" if b.direction == "상승" else "지지"
    kst = b.candle_time.tz_convert("Asia/Seoul") if b.candle_time.tzinfo else b.candle_time
    lines = [
        f"🚨 {b.symbol} {kind} 돌파! ({b.interval} 종가 기준)",
        "",
        f"{arrow} 방향: {b.direction}",
        f"돌파선: {b.level_price:,.4f} (터치 {b.touches}회)",
        f"종가: {b.close:,.4f} ({kst:%m/%d %H:%M} KST 캔들)",
        f"목표가: {b.target:,.4f} ({b.target_kind})",
        f"재이탈 주의선: {b.level_price:,.4f}",
        f"  (종가가 이 선을 다시 {'하향' if b.direction == '상승' else '상향'} 이탈하면 돌파 무효)",
        "",
        "※ 참고용이며 재무적 조언이 아닙니다.",
    ]
    return "\n".join(lines)


def load_state(path: str | Path) -> dict[str, str]:
    """심볼별 마지막 검사 캔들 시각을 담은 상태를 읽는다 (없으면 빈 dict)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: str | Path, state: dict[str, str]) -> None:
    Path(path).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
