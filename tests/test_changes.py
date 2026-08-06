"""변화 감지 로직을 검증한다.

실행: python tests/test_changes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.changes import (
    detect_changes,
    load_state,
    save_state,
)

BASE = {
    "action": "관망", "verdict": "중립", "close": 100.0,
    "warning": False, "near_support": False, "near_resistance": False,
}


def _with(**kw) -> dict:
    return {**BASE, **kw}


def test_no_change_is_silent():
    assert detect_changes(BASE, dict(BASE)) == []
    # 임계값 미만의 가격 변동은 무시
    assert detect_changes(BASE, _with(close=100.9)) == []
    print("test_no_change_is_silent 통과")


def test_first_observation_is_silent():
    """첫 관측은 기준선을 잡는 용도이므로 알리지 않는다."""
    assert detect_changes(None, BASE) == []
    print("test_first_observation_is_silent 통과")


def test_action_change():
    r = detect_changes(BASE, _with(action="매수"))
    assert len(r) == 1 and "판단 변경: 관망 → 매수" in r[0]
    assert "급변" not in r[0]
    # 2단계 이상 도약이면 표시
    r2 = detect_changes(BASE, _with(action="강한 매수"))
    assert "급변" in r2[0]
    print("test_action_change 통과")


def test_verdict_change():
    r = detect_changes(BASE, _with(verdict="상승"))
    assert any("방향 전환: 중립 → 상승" in x for x in r)
    print("test_verdict_change 통과")


def test_price_move():
    r = detect_changes(BASE, _with(close=102.0))  # +2% > 1.5% 임계값
    assert any("가격 +2.0% 변동" in x for x in r)
    r2 = detect_changes(BASE, _with(close=98.0))
    assert any("가격 -2.0% 변동" in x for x in r2)
    # 임계값을 높이면 같은 변동도 무시
    assert detect_changes(BASE, _with(close=102.0), price_move_pct=0.05) == []
    print("test_price_move 통과")


def test_warning_and_levels():
    assert any("반전 주의" in x for x in detect_changes(BASE, _with(warning=True)))
    # 이미 켜져 있던 경고는 다시 알리지 않음
    assert detect_changes(_with(warning=True), _with(warning=True)) == []
    assert any("저항선 근접" in x
               for x in detect_changes(BASE, _with(near_resistance=True)))
    assert any("지지선 근접" in x
               for x in detect_changes(BASE, _with(near_support=True)))
    # 근접이 해제되는 것은 알리지 않음 (소음 방지)
    assert detect_changes(_with(near_support=True), BASE) == []
    print("test_warning_and_levels 통과")


def test_multiple_changes_listed():
    r = detect_changes(BASE, _with(action="매수", verdict="상승", close=103.0))
    assert len(r) == 3
    print(f"test_multiple_changes_listed 통과 ({len(r)}건 동시 감지)")


def test_state_roundtrip():
    path = Path("/tmp/claude-0/-home-user-kidod1/4949cad9-14cd-5230-9187-bc26f65b6a13/scratchpad/chg_state.json")
    save_state(path, {"BTCUSDT": BASE})
    assert load_state(path)["BTCUSDT"]["action"] == "관망"
    assert load_state(path.with_name("nope.json")) == {}
    path.unlink()
    print("test_state_roundtrip 통과")


if __name__ == "__main__":
    test_no_change_is_silent()
    test_first_observation_is_silent()
    test_action_change()
    test_verdict_change()
    test_price_move()
    test_warning_and_levels()
    test_multiple_changes_listed()
    test_state_roundtrip()
    print("\n모든 변화 감지 테스트 통과!")
