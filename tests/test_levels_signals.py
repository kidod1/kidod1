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
    # 다른 조건이 같을 때 상승 확률이 높으면 점수가 4점(±2씩) 더 높아야 한다
    assert bullish.score == bearish.score + 4
    assert any("모델" in r for r in bullish.reasons)
    text = format_advice(bullish)
    assert "타이밍 판단" in text
    print(f"test_advise_directions 통과 "
          f"(상승 시나리오 {bullish.score:+d} / 하락 시나리오 {bearish.score:+d})")


def test_advise_htf_trends():
    """상위 시간대 추세가 점수에 반영되는지 확인."""
    df = make_synthetic_ohlcv(n=600)
    featured = build_features(df)
    base = advise(featured, _make_pred(0.50), [], [])
    up = advise(featured, _make_pred(0.50), [], [],
                htf_trends={"4h": "상승", "1d": "상승"})
    down = advise(featured, _make_pred(0.50), [], [],
                  htf_trends={"4h": "하락", "1d": "중립"})
    assert up.score == base.score + 2
    assert down.score == base.score - 1
    assert any("4h 추세 상승" in r for r in up.reasons)
    print("test_advise_htf_trends 통과")


def test_advise_stop_target():
    """매수/매도 판단일 때만 ATR 손절/목표가가 제안되는지 확인."""
    df = make_synthetic_ohlcv(n=600)
    featured = build_features(df)
    # 강한 상승 조건을 인위적으로 만들어 매수 판단 유도
    strong = advise(featured, _make_pred(0.75), [], [],
                    htf_trends={"4h": "상승", "1d": "상승"})
    assert strong.score >= 2
    assert strong.stop_loss is not None and strong.take_profit is not None
    price = float(featured["close"].iloc[-1])
    assert strong.stop_loss < price < strong.take_profit
    assert "손절" in format_advice(strong)
    print(f"test_advise_stop_target 통과 "
          f"(손절 {strong.stop_loss:,.2f} < 현재 {price:,.2f} < 목표 {strong.take_profit:,.2f})")


def test_advise_neutral():
    from predictor.signals import _score_to_action
    df = make_synthetic_ohlcv(n=600)
    featured = build_features(df)
    neutral = advise(featured, _make_pred(0.50), [], [])
    # 점수와 판단이 일관돼야 하고, 관망이면 손절/목표가는 없어야 한다
    assert neutral.action == _score_to_action(neutral.score)
    if neutral.action == "관망":
        assert neutral.stop_loss is None
    print(f"test_advise_neutral 통과 (점수 {neutral.score:+d}, {neutral.action})")


if __name__ == "__main__":
    test_levels_on_range()
    test_levels_random_walk_no_crash()
    test_advise_directions()
    test_advise_htf_trends()
    test_advise_stop_target()
    test_advise_neutral()
    print("\n모든 레벨/신호 테스트 통과!")
