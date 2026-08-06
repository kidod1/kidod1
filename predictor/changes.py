"""의미 있는 변화 감지 — 주기적으로 다 보내는 대신 달라졌을 때만 알린다.

직전에 알린 상태를 심볼별로 기억해두고, 지금 상태와 비교해 아래 중 하나라도
해당하면 알림 대상으로 본다:

  1. 타이밍 판단 변화 (관망 → 매수 등)
  2. 방향 판단 변화 (중립 → 상승, 상승 → 하락 등)
  3. 가격이 마지막 알림 시점 대비 임계값(기본 1.5%) 이상 움직임
  4. 반전 주의 경고가 새로 켜짐
  5. 지지/저항선에 새로 근접 (1% 이내)

돌파·추세 전환은 이미 이벤트 기반이라 여기서 다루지 않는다.
아무 변화가 없으면 아무것도 보내지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_STATE_FILE = ".changes_state.json"
PRICE_MOVE_PCT = 0.015   # 가격 변화 알림 임계값 (1.5%)
NEAR_LEVEL_PCT = 0.01    # 지지/저항 근접 판정 (1%)

# 판단의 강도 순서 — 인접 단계 이동인지 큰 도약인지 구분할 때 사용
ACTION_RANK = {"강한 매도": -2, "매도": -1, "관망": 0, "매수": 1, "강한 매수": 2}


def load_state(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(path: str | Path, state: dict) -> None:
    Path(path).write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")


def snapshot_state(snap) -> dict:
    """스냅샷에서 '알림 여부 판단에 쓰는' 상태만 뽑는다."""
    near_support = bool(snap.supports
                        and abs(snap.supports[0].distance_pct) <= NEAR_LEVEL_PCT)
    near_resistance = bool(snap.resistances
                           and abs(snap.resistances[0].distance_pct) <= NEAR_LEVEL_PCT)
    return {
        "action": snap.advice.action,
        "verdict": snap.verdict,
        "close": snap.close,
        "warning": bool(snap.advice.warning),
        "near_support": near_support,
        "near_resistance": near_resistance,
    }


def detect_changes(prev: dict | None, cur: dict,
                   price_move_pct: float = PRICE_MOVE_PCT) -> list[str]:
    """직전 알림 상태와 비교해 변화 사유 목록을 만든다 (빈 목록 = 알릴 것 없음).

    첫 관측(prev 없음)은 기준선을 잡는 용도이므로 알리지 않는다.
    """
    if prev is None:
        return []

    reasons: list[str] = []

    if prev.get("action") != cur["action"]:
        old_rank = ACTION_RANK.get(prev.get("action", "관망"), 0)
        new_rank = ACTION_RANK.get(cur["action"], 0)
        jump = "  (2단계 이상 급변)" if abs(new_rank - old_rank) >= 2 else ""
        reasons.append(f"판단 변경: {prev.get('action')} → {cur['action']}{jump}")

    if prev.get("verdict") != cur["verdict"]:
        reasons.append(f"방향 전환: {prev.get('verdict')} → {cur['verdict']}")

    old_close = prev.get("close")
    if old_close:
        move = cur["close"] / old_close - 1
        if abs(move) >= price_move_pct:
            reasons.append(f"가격 {move:+.1%} 변동 (마지막 알림 대비)")

    if cur["warning"] and not prev.get("warning"):
        reasons.append("반전 주의 신호 발생")

    if cur["near_resistance"] and not prev.get("near_resistance"):
        reasons.append("저항선 근접")
    if cur["near_support"] and not prev.get("near_support"):
        reasons.append("지지선 근접")

    return reasons


def format_change_alert(snap, reasons: list[str], body: str) -> str:
    """변화 알림 메시지를 만든다 — 무엇이 바뀌었는지를 맨 위에 둔다."""
    lines = [f"🔔 {snap.symbol} 변화 감지"]
    lines += [f"  • {r}" for r in reasons]
    lines += ["", body, "", "※ 참고용이며 재무적 조언이 아닙니다."]
    return "\n".join(lines)
