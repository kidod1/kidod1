"""매매 전략 백테스트를 합성 데이터로 검증한다.

실행: python tests/test_strategy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.model import prepare_dataset, walk_forward_probabilities
from predictor.strategy import format_report, simulate, sweep_thresholds
from tests.test_pipeline import make_synthetic_ohlcv


def test_simulate_mechanics():
    """수수료·포지션 로직이 수식대로 동작하는지 손으로 계산한 값과 비교."""
    probs = pd.DataFrame({
        "prob_up": [0.7, 0.7, 0.4, 0.3, 0.8],
        "forward_return": [0.01, -0.02, 0.05, 0.01, 0.03],
    })
    r = simulate(probs, threshold=0.6, fee=0.001, allow_short=False)
    # 포지션: [1, 1, 0, 0, 1] → 변경 3회 (진입, 청산, 재진입)
    assert r.n_trades == 3
    assert r.time_in_market == 0.6
    # 수익: (1+0.01-0.001)*(1-0.02)*(1-0.001)*(1)*(1+0.03-0.001) - 1
    expected = (1.009) * (0.98) * (0.999) * 1 * (1.029) - 1
    assert abs(r.total_return - expected) < 1e-9
    print("test_simulate_mechanics 통과")


def test_simulate_short():
    probs = pd.DataFrame({
        "prob_up": [0.2, 0.9],
        "forward_return": [-0.03, 0.02],
    })
    r = simulate(probs, threshold=0.6, fee=0.0, allow_short=True)
    # 숏으로 -3% 하락 → +3%, 롱으로 +2% → 누적 (1.03)(1.02)-1
    assert abs(r.total_return - ((1.03) * (1.02) - 1)) < 1e-9
    assert r.win_rate == 1.0
    print("test_simulate_short 통과")


def test_walk_forward_and_report():
    df = make_synthetic_ohlcv(n=1200)
    train, _ = prepare_dataset(df)
    probs = walk_forward_probabilities(train, n_splits=3)
    assert {"prob_up", "forward_return"} <= set(probs.columns)
    assert len(probs) > 500
    assert probs["prob_up"].between(0, 1).all()

    results = sweep_thresholds(probs, (0.5, 0.6), fee=0.001)
    report = format_report(results, "TESTUSDT", "1h")
    assert "TESTUSDT" in report and "0.50" in report and "0.60" in report
    print("test_walk_forward_and_report 통과")
    print("--- 리포트 미리보기 ---")
    print(report)


if __name__ == "__main__":
    test_simulate_mechanics()
    test_simulate_short()
    test_walk_forward_and_report()
    print("\n모든 전략 테스트 통과!")
