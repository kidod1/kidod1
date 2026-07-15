"""지지/저항선 탐지와 타이밍 신호를 합성 데이터로 검증한다.

실행: python tests/test_levels_signals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.features import build_features
from predictor.levels import find_levels, format_levels
from predictor.model import Prediction
from predictor.signals import advise, format_advice
from tests.test_pipeline import make_synthetic_ohlcv


def make_ranging_ohlcv(n: int = 400) -> pd.DataFrame:
    """98~102 사이를 오가는 횡보 데이터 — 100 근처가 아닌 98(지지), 102(저항)에
    여러 번 닿도록 사인파 + 소량의 노이즈로 만든다."""
    rng = np.random.default_rng(3)
    t = np.arange(n)
    close = 100 + 2 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.05, n)
    high = close + 0.15
    low = close - 0.15
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    times = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    volume = rng.uniform(100, 200, n)
    return pd.DataFrame({
        "open_time": times, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
        "close_time": times + pd.Timedelta(hours=1),
        "quote_volume": volume * close, "trades": rng.integers(100, 500, n),
        "taker_buy_base": volume * 0.5, "taker_buy_quote": volume * close * 0.5,
    })


def test_levels_on_range():
    df = make_ranging_ohlcv()
    supports, resistances = find_levels(df, tolerance=0.005)
    current = float(df["close"].iloc[-1])
    # 횡보장의 천장(~102)과 바닥(~98)이 잡혀야 한다
    assert supports, "지지선을 하나 이상 찾아야 함"
    assert resistances, "저항선을 하나 이상 찾아야 함"
    assert all(lv.price < current for lv in supports)
    assert all(lv.price > current for lv in resistances)
    assert any(lv.price < 99.5 for lv in supports + resistances) or \
           any(lv.price > 100.5 for lv in supports + resistances)
    report = format_levels(supports, resistances, current)
    assert "지지" in report or "저항" in report
    print("test_levels_on_range 통과")
    print(report)


def test_levels_random_walk_no_crash():
    df = make_synthetic_ohlcv(n=800)
    supports, resistances = find_levels(df)
    for lv in supports:
        assert lv.touches >= 2 and lv.distance_pct <= 0
    for lv in resistances:
        assert lv.touches >= 2 and lv.distance_pct >= 0
    print(f"test_levels_random_walk_no_crash 통과 "
          f"(지지 {len(supports)}개, 저항 {len(resistances)}개)")


def _make_pred(prob_up: float) -> Prediction:
    return Prediction(prob_up=prob_up, prob_down=1 - prob_up,
                      last_close=100.0, last_open_time=pd.Timestamp("2025-01-01", tz="UTC"))


def test_advise_directions():
    df = make_synthetic_ohlcv(n=600)
    featured = build_features(df)
    supports, resistances = find_levels(df)

    bullish = advise(featured, _make_pred(0.72), supports, resistances)
    bearish = advise(featured, _make_pred(0.28), supports, resistances)
    # 다른 조건이 같을 때 상승 확률이 높으면 점수가 더 높아야 한다
    assert bullish.score > bearish.score
    assert bullish.action in ("강한 매수", "매수", "관망")
    assert bearish.action in ("강한 매도", "매도", "관망")
    assert any("모델" in r for r in bullish.reasons)
    text = format_advice(bullish)
    assert "타이밍 판단" in text
    print(f"test_advise_directions 통과 "
          f"(상승 시나리오 {bullish.score:+d} / 하락 시나리오 {bearish.score:+d})")


def test_advise_neutral():
    df = make_synthetic_ohlcv(n=600)
    featured = build_features(df)
    neutral = advise(featured, _make_pred(0.50), [], [])
    assert -1 <= neutral.score <= 1
    print(f"test_advise_neutral 통과 (점수 {neutral.score:+d}, {neutral.action})")


if __name__ == "__main__":
    test_levels_on_range()
    test_levels_random_walk_no_crash()
    test_advise_directions()
    test_advise_neutral()
    print("\n모든 레벨/신호 테스트 통과!")
