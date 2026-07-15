"""지지/저항 돌파 감지를 합성 데이터로 검증한다.

실행: python tests/test_breakout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.breakout import detect_breakouts, format_breakout, load_state, save_state
from tests.test_levels_signals import make_ranging_ohlcv


def _set_candle(df: pd.DataFrame, i: int, close: float) -> None:
    """i번째 캔들의 가격을 지정 종가 기준으로 일관되게 덮어쓴다."""
    df.loc[i, "open"] = df.loc[i - 1, "close"] if i > 0 else close
    df.loc[i, "close"] = close
    df.loc[i, "high"] = max(df.loc[i, "open"], close) + 0.1
    df.loc[i, "low"] = min(df.loc[i, "open"], close) - 0.1


def make_breakout_df() -> pd.DataFrame:
    """98~102 횡보 후 마지막에 저항(~102)을 상향 돌파하는 데이터."""
    df = make_ranging_ohlcv(n=400)
    n = len(df)
    # n-4: 저항 아래, n-3: 돌파 확정, n-2: 상승 지속, n-1: 진행 중(제외됨)
    _set_candle(df, n - 4, 101.0)
    _set_candle(df, n - 3, 103.5)
    _set_candle(df, n - 2, 103.8)
    _set_candle(df, n - 1, 104.0)
    return df


def test_upward_breakout():
    df = make_breakout_df()
    events, last_time = detect_breakouts(df, "BTCUSDT", "1h")
    assert last_time is not None
    assert len(events) >= 1, "저항 돌파를 감지해야 함"
    ev = events[0]
    assert ev.direction == "상승"
    assert 101.5 < ev.level_price < 103.0, f"돌파선이 ~102여야 함: {ev.level_price}"
    assert ev.close > ev.level_price
    assert ev.target > ev.close, "상승 돌파 목표가는 종가보다 높아야 함"
    msg = format_breakout(ev)
    assert "돌파" in msg and "목표가" in msg and "재이탈" in msg
    print(f"test_upward_breakout 통과 (돌파선 {ev.level_price:.2f}, "
          f"목표 {ev.target:.2f}, {ev.target_kind})")
    print(msg)


def test_no_breakout_on_range():
    df = make_ranging_ohlcv(n=400)
    events, _ = detect_breakouts(df, "BTCUSDT", "1h")
    assert events == [], f"횡보장에서 돌파가 없어야 함: {events}"
    print("test_no_breakout_on_range 통과")


def test_dedup_with_state():
    df = make_breakout_df()
    events1, last_time = detect_breakouts(df, "BTCUSDT", "1h")
    assert len(events1) >= 1
    # 같은 데이터를 상태(since)와 함께 다시 검사하면 중복 알림이 없어야 함
    events2, _ = detect_breakouts(df, "BTCUSDT", "1h", since=last_time)
    assert events2 == []
    print("test_dedup_with_state 통과")


def test_state_roundtrip(tmp_path: Path | None = None):
    path = (tmp_path or Path("/tmp")) / "breakout_state_test.json"
    save_state(path, {"BTCUSDT:1h": "2026-07-15 09:00:00+00:00"})
    assert load_state(path) == {"BTCUSDT:1h": "2026-07-15 09:00:00+00:00"}
    assert load_state(path.with_name("no_such_file.json")) == {}
    path.unlink()
    print("test_state_roundtrip 통과")


if __name__ == "__main__":
    test_upward_breakout()
    test_no_breakout_on_range()
    test_dedup_with_state()
    test_state_roundtrip()
    print("\n모든 돌파 테스트 통과!")
