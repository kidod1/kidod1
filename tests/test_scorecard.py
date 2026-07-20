"""예측 성적표를 합성 데이터로 검증한다.

실행: python tests/test_scorecard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.scorecard import (
    format_scorecard,
    load_log,
    record_prediction,
    resolve_pending,
    save_log,
)
from tests.test_pipeline import make_synthetic_ohlcv


def test_record_and_dedupe():
    log = {"pending": [], "resolved": []}
    t = pd.Timestamp("2026-07-16 09:00:00+00:00")
    assert record_prediction(log, "BTCUSDT", "1h", t, 100.0, 0.62, 4) is True
    assert record_prediction(log, "BTCUSDT", "1h", t, 100.0, 0.62, 4) is False  # 중복
    assert record_prediction(log, "ETHUSDT", "1h", t, 50.0, 0.45, 4) is True  # 다른 심볼
    # 호라이즌(4캔들) 구간과 겹치는 예측은 건너뜀 — 독립 베팅만 기록
    assert record_prediction(log, "BTCUSDT", "1h",
                             t + pd.Timedelta(hours=1), 101.0, 0.60, 4) is False
    assert record_prediction(log, "BTCUSDT", "1h",
                             t + pd.Timedelta(hours=3), 101.0, 0.60, 4) is False
    assert record_prediction(log, "BTCUSDT", "1h",
                             t + pd.Timedelta(hours=4), 101.0, 0.60, 4) is True
    assert len(log["pending"]) == 3
    print("test_record_and_dedupe 통과")


def test_resolve_correctness():
    df = make_synthetic_ohlcv(n=200)
    log = {"pending": [], "resolved": []}
    # 과거 시점(-10번째 캔들)의 예측 두 개: 하나는 상승 예측, 하나는 하락 예측
    t = df["open_time"].iloc[-10]
    entry_close = float(df["close"].iloc[-10])
    actual = float(df.loc[df["open_time"] == t + pd.Timedelta(hours=4), "close"].iloc[0])
    went_up = actual > entry_close

    record_prediction(log, "BTCUSDT", "1h", t, entry_close, 0.70, 4)  # 상승 예측
    record_prediction(log, "ETHUSDT", "1h", t, entry_close, 0.30, 4)  # 하락 예측
    n = resolve_pending(log, {"BTCUSDT": df, "ETHUSDT": df})
    assert n == 2 and not log["pending"]

    by_symbol = {r["symbol"]: r for r in log["resolved"]}
    assert by_symbol["BTCUSDT"]["correct"] == went_up          # 상승 예측 → 실제 상승이면 적중
    assert by_symbol["ETHUSDT"]["correct"] == (not went_up)    # 하락 예측 → 반대
    print(f"test_resolve_correctness 통과 (실제 {'상승' if went_up else '하락'})")


def test_pending_stays_until_horizon():
    df = make_synthetic_ohlcv(n=200)
    log = {"pending": [], "resolved": []}
    # 마지막 캔들 시점의 예측 → 아직 4캔들이 안 지나서 보류돼야 함
    t = df["open_time"].iloc[-1]
    record_prediction(log, "BTCUSDT", "1h", t, float(df["close"].iloc[-1]), 0.6, 4)
    n = resolve_pending(log, {"BTCUSDT": df})
    assert n == 0 and len(log["pending"]) == 1
    print("test_pending_stays_until_horizon 통과")


def test_format_and_roundtrip():
    log = {"pending": [], "resolved": [
        {"symbol": "BTCUSDT", "interval": "1h", "candle_time": "t1",
         "close": 1, "prob_up": 0.6, "horizon": 4, "actual_close": 2,
         "correct": True, "with_trend": True},
        {"symbol": "BTCUSDT", "interval": "1h", "candle_time": "t2",
         "close": 1, "prob_up": 0.6, "horizon": 4, "actual_close": 0.5,
         "correct": False, "with_trend": False},
        {"symbol": "ETHUSDT", "interval": "1h", "candle_time": "t3",
         "close": 1, "prob_up": 0.45, "horizon": 4, "actual_close": 0.9,
         "correct": True, "with_trend": None},
    ]}
    text = format_scorecard(log)
    assert "추세 순응 예측: 1/1건 (100%)" in text
    assert "역추세 예측: 0/1건 (0%)" in text
    assert "1건씩 독립 예측만 기록" in text
    assert "2/3" in text and "67%" in text
    assert "BTCUSDT 50%" in text and "ETHUSDT 100%" in text
    # 산정 기준 설명이 포함돼야 함
    assert "산정 기준" in text
    assert "향후 4캔들(1h) 방향" in text
    assert "50% 이상인 쪽" in text
    # 강한 신호(0.6 이상 확신) 세부: prob 0.6 두 건(적중 1) → 1/2
    assert "확신 60% 이상 신호만: 1/2건 (50%)" in text
    assert format_scorecard({"pending": [], "resolved": []}) is None

    path = Path("/tmp/claude-0/-home-user-kidod1/4949cad9-14cd-5230-9187-bc26f65b6a13/scratchpad/scorecard_rt.json")
    save_log(path, log)
    assert len(load_log(path)["resolved"]) == 3
    path.unlink()
    print("test_format_and_roundtrip 통과")
    print(text)


if __name__ == "__main__":
    test_record_and_dedupe()
    test_resolve_correctness()
    test_pending_stays_until_horizon()
    test_format_and_roundtrip()
    print("\n모든 성적표 테스트 통과!")
