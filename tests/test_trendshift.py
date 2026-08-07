"""추세 전환 판정의 보수성(휩쏘 억제)을 검증한다.

실행: python tests/test_trendshift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.trendshift import (
    COOLDOWN_CANDLES,
    check_trend_shift,
    current_trend,
    format_trend_shift,
)
from tests.test_reversal import _ohlcv_from_close


def _run(closes: np.ndarray, stored=None, symbol="BTCUSDT"):
    return check_trend_shift(_ohlcv_from_close(closes), symbol, "4h", stored)


UP = np.linspace(100, 160, 120)                                   # 뚜렷한 상승
DOWN = np.concatenate([np.linspace(100, 160, 80),
                       np.linspace(160, 100, 60)])                # 뚜렷한 하락 전환


def test_clear_trend_is_detected():
    assert current_trend(_ohlcv_from_close(UP)) == "상승"
    assert current_trend(_ohlcv_from_close(DOWN)) == "하락"
    print("test_clear_trend_is_detected 통과")


def test_flat_market_is_neutral():
    """평평한 시장에서는 두 평균선이 붙어 있으므로 방향을 선언하지 않는다."""
    rng = np.random.default_rng(3)
    flat = 100 + rng.normal(0, 0.05, 200)  # 거의 움직이지 않음
    assert current_trend(_ohlcv_from_close(flat)) is None
    print("test_flat_market_is_neutral 통과")


def test_whipsaw_produces_no_alert():
    """평균선이 아슬아슬하게 오가는 구간에서 전환 알림이 나가면 안 된다.

    미세한 사인파로 SMA7과 SMA25가 반복 교차하는 상황을 만든다 (기존 로직이라면
    매 캔들 전환이 발생했을 패턴).
    """
    t = np.arange(300)
    wobble = 100 + 0.15 * np.sin(2 * np.pi * t / 18)  # 진폭 0.15% 수준
    df = _ohlcv_from_close(wobble)

    alerts = 0
    stored = {"trend": "상승", "candles_since_shift": COOLDOWN_CANDLES}
    # 캔들을 하나씩 늘려가며 실제 운영처럼 반복 검사
    for end in range(60, len(df)):
        shift, stored = check_trend_shift(df.iloc[:end], "BTCUSDT", "4h", stored)
        if shift:
            alerts += 1
    assert alerts == 0, f"휩쏘 구간에서 {alerts}건 오탐 발생"
    print("test_whipsaw_produces_no_alert 통과 (오탐 0건)")


def test_first_run_records_only():
    shift, state = _run(UP, stored=None)
    assert shift is None and state["trend"] == "상승"
    print("test_first_run_records_only 통과")


def test_same_trend_is_silent():
    shift, state = _run(UP, stored={"trend": "상승", "candles_since_shift": 10})
    assert shift is None and state["candles_since_shift"] == 11
    print("test_same_trend_is_silent 통과")


def test_real_shift_alerts_with_evidence():
    shift, state = _run(DOWN, stored={"trend": "상승",
                                      "candles_since_shift": COOLDOWN_CANDLES})
    assert shift is not None, "뚜렷한 전환은 알려야 함"
    assert shift.old_trend == "상승" and shift.new_trend == "하락"
    assert shift.held >= 3 and shift.adx >= 20
    assert state["candles_since_shift"] == 0
    msg = format_trend_shift(shift)
    assert "확정" in msg and "ADX" in msg and "연속" in msg
    print("test_real_shift_alerts_with_evidence 통과")
    print(msg)


def test_cooldown_blocks_rapid_reversal():
    """직전 전환 직후의 재전환은 쿨다운으로 막힌다."""
    shift, state = _run(DOWN, stored={"trend": "상승", "candles_since_shift": 1})
    assert shift is None, "쿨다운 중에는 알리지 않아야 함"
    assert state["trend"] == "상승", "쿨다운 중에는 이전 추세를 유지해야 함"
    print("test_cooldown_blocks_rapid_reversal 통과")


def test_neutral_keeps_previous_trend():
    """판정 보류(중립) 구간이 이전 추세를 지워 왕복 알림을 만들지 않는다."""
    rng = np.random.default_rng(5)
    flat = 100 + rng.normal(0, 0.05, 200)
    shift, state = _run(flat, stored={"trend": "상승", "candles_since_shift": 9})
    assert shift is None and state["trend"] == "상승"
    print("test_neutral_keeps_previous_trend 통과")


def test_legacy_string_state_compatible():
    """구버전 상태 파일(문자열)이 있어도 동작해야 한다."""
    shift, state = _run(UP, stored="상승")
    assert shift is None and isinstance(state, dict) and state["trend"] == "상승"
    print("test_legacy_string_state_compatible 통과")


if __name__ == "__main__":
    test_clear_trend_is_detected()
    test_flat_market_is_neutral()
    test_whipsaw_produces_no_alert()
    test_first_run_records_only()
    test_same_trend_is_silent()
    test_real_shift_alerts_with_evidence()
    test_cooldown_blocks_rapid_reversal()
    test_neutral_keeps_previous_trend()
    test_legacy_string_state_compatible()
    print("\n모든 추세 전환 테스트 통과!")
