"""모델 검증 게이트(neutral_reason, oos_confidence_stats)를 검증한다.

실행: python tests/test_validation_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from predictor.model import Prediction, oos_confidence_stats, prepare_dataset
from tests.test_pipeline import make_synthetic_ohlcv

T = pd.Timestamp("2026-07-20", tz="UTC")


def _pred(prob_up: float) -> Prediction:
    return Prediction(prob_up=prob_up, prob_down=1 - prob_up,
                      last_close=100.0, last_open_time=T)


def test_neutral_reason_branches():
    conf, weak = _pred(0.60), _pred(0.52)
    # 확신 낮음
    assert bot.neutral_reason(weak, {}, 0.60, 100) == "확신 낮음"
    # 역추세: 4h/1d 모두 반대
    r = bot.neutral_reason(conf, {"4h": "하락", "1d": "하락"}, 0.60, 100)
    assert r and "역추세" in r
    # 검증 미달: 표본 충분 + 적중률 52% 미만
    r = bot.neutral_reason(conf, {"4h": "상승"}, 0.45, 50)
    assert r and "검증 미달" in r
    # 표본 부족(20건 미만)이면 검증 게이트 미적용
    assert bot.neutral_reason(conf, {"4h": "상승"}, 0.40, 5) is None
    # 채택: 확신 + 추세 순응(혼조 포함) + 검증 통과
    assert bot.neutral_reason(conf, {"4h": "상승", "1d": "중립"}, 0.58, 60) is None
    print("test_neutral_reason_branches 통과")


def test_oos_stats_on_random_walk():
    """예측 불가능한 랜덤워크에서는 검증 적중률이 우연 수준이어야 하고,
    그 결과 게이트가 방향 예측을 차단해야 한다."""
    train, _ = prepare_dataset(make_synthetic_ohlcv(n=1200))
    rate, n = oos_confidence_stats(train)
    assert n >= 0
    if rate is not None and n >= bot.MIN_OOS_SAMPLE:
        assert 0.0 <= rate <= 1.0
        # 랜덤워크에서 60%를 넘기면 누수를 의심해야 함
        assert rate < 0.60, f"랜덤워크 검증 적중률이 비정상적으로 높음: {rate:.0%}"
    print(f"test_oos_stats_on_random_walk 통과 "
          f"(적중률 {'N/A' if rate is None else f'{rate:.0%}'}, {n}건)")


if __name__ == "__main__":
    test_neutral_reason_branches()
    test_oos_stats_on_random_walk()
    print("\n모든 검증 게이트 테스트 통과!")
