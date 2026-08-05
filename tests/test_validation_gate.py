"""모델 검증 게이트(neutral_reason, oos_confidence_stats)를 검증한다.

실행: python tests/test_validation_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from predictor.model import (
    DirectionStats,
    Prediction,
    drop_open_candle,
    oos_confidence_stats,
    prepare_dataset,
)
from tests.test_pipeline import make_synthetic_ohlcv

T = pd.Timestamp("2026-07-20", tz="UTC")


def _pred(prob_up: float) -> Prediction:
    return Prediction(prob_up=prob_up, prob_down=1 - prob_up,
                      last_close=100.0, last_open_time=T)


def _stats(direction: str, hit: float, baseline: float, n: int = 50) -> dict:
    return {direction: DirectionStats(hit_rate=hit, samples=n, baseline=baseline)}


def test_neutral_reason_branches():
    conf, weak = _pred(0.60), _pred(0.52)
    good = _stats("상승", 0.60, 0.52)  # 우위 +8%p
    # 확신 낮음
    assert bot.neutral_reason(weak, {}, good) == "확신 낮음"
    # 역추세: 4h/1d 모두 반대
    r = bot.neutral_reason(conf, {"4h": "하락", "1d": "하락"}, good)
    assert r and "역추세" in r
    # 검증 미달: 적중률은 높지만 기준선(상승 편향 시장)을 못 넘음
    r = bot.neutral_reason(conf, {"4h": "상승"}, _stats("상승", 0.56, 0.58))
    assert r and "검증 미달" in r, r
    # 표본 부족(20건 미만)이면 검증 게이트 미적용
    assert bot.neutral_reason(conf, {"4h": "상승"},
                              _stats("상승", 0.30, 0.58, n=5)) is None
    # 해당 방향 통계가 없으면 게이트 미적용
    assert bot.neutral_reason(conf, {"4h": "상승"}, _stats("하락", 0.30, 0.58)) is None
    # 채택: 확신 + 추세 순응 + 기준선 대비 우위
    assert bot.neutral_reason(conf, {"4h": "상승", "1d": "중립"}, good) is None
    print("test_neutral_reason_branches 통과")


def test_gate_uses_baseline_not_absolute():
    """상승 편향 시장에서 '적중률 58%'는 기준선 60%보다 낮으므로 차단돼야 한다.
    (절대 기준만 봤다면 통과했을 값)"""
    conf = _pred(0.62)
    r = bot.neutral_reason(conf, {}, _stats("상승", 0.58, 0.60))
    assert r is not None and "검증 미달" in r
    print("test_gate_uses_baseline_not_absolute 통과")


def test_drop_open_candle():
    """진행 중(close_time이 미래) 캔들이 제거되는지."""
    df = make_synthetic_ohlcv(n=50)
    now = pd.Timestamp.now(tz="UTC")
    # 마지막 캔들을 '진행 중'으로 만든다
    df.loc[df.index[-1], "close_time"] = now + pd.Timedelta(hours=1)
    df.loc[df.index[-1], "open_time"] = now - pd.Timedelta(minutes=5)
    trimmed = drop_open_candle(df)
    assert len(trimmed) == len(df) - 1, "진행 중 캔들이 제거돼야 함"
    # 모두 닫힌 캔들이면 그대로
    assert len(drop_open_candle(make_synthetic_ohlcv(n=50))) == 50
    print("test_drop_open_candle 통과")


def test_oos_stats_on_random_walk():
    """예측 불가능한 랜덤워크에서는 어느 방향도 기준선 대비 우위가 없어야 한다."""
    train, _ = prepare_dataset(make_synthetic_ohlcv(n=1200))
    dir_stats, up_rate = oos_confidence_stats(train)
    assert 0.0 <= up_rate <= 1.0
    for name, st in dir_stats.items():
        assert 0.0 <= st.hit_rate <= 1.0
        if st.samples >= bot.MIN_OOS_SAMPLE:
            # 랜덤워크에서 기준선을 크게 이기면 데이터 누수를 의심해야 함
            assert st.edge < 0.15, f"{name} 우위가 비정상적으로 큼: {st.edge:+.0%}"
    print(f"test_oos_stats_on_random_walk 통과 "
          f"({ {k: f'{v.edge:+.0%}' for k, v in dir_stats.items()} })")


if __name__ == "__main__":
    test_neutral_reason_branches()
    test_gate_uses_baseline_not_absolute()
    test_drop_open_candle()
    test_oos_stats_on_random_walk()
    print("\n모든 검증 게이트 테스트 통과!")
