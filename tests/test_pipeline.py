"""네트워크 없이 합성 데이터로 전체 파이프라인을 검증하는 테스트.

실행: python -m pytest tests/  또는  python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.features import FEATURE_COLUMNS, build_features
from predictor.model import backtest, predict_next, prepare_dataset


def make_synthetic_ohlcv(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """랜덤워크 기반 합성 OHLCV 데이터를 생성한다."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n)
    close = 30000 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.uniform(100, 1000, n)
    times = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open_time": times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": times + pd.Timedelta(hours=1),
        "quote_volume": volume * close,
        "trades": rng.integers(100, 5000, n),
        "taker_buy_base": volume * rng.uniform(0.3, 0.7, n),
        "taker_buy_quote": volume * close * 0.5,
    })


def test_features():
    df = make_synthetic_ohlcv()
    featured = build_features(df)
    assert all(col in featured.columns for col in FEATURE_COLUMNS)
    # 마지막 행의 target은 NaN (아직 다음 캔들이 없음)
    assert np.isnan(featured["target"].iloc[-1])
    # 지표 계산 초기 구간(최장 sma200 + 변동성 72) 이후에는 NaN이 없어야 함
    assert not featured[FEATURE_COLUMNS].iloc[280:].isna().any(axis=None)
    print("test_features 통과")


def test_prediction():
    df = make_synthetic_ohlcv()
    train, latest = prepare_dataset(df)
    assert len(train) > 1000
    pred = predict_next(train, latest)
    assert 0.0 <= pred.prob_up <= 1.0
    assert abs(pred.prob_up + pred.prob_down - 1.0) < 1e-9
    print(f"test_prediction 통과 (상승 확률 예시: {pred.prob_up:.1%})")


def test_backtest():
    df = make_synthetic_ohlcv()
    train, _ = prepare_dataset(df)
    result = backtest(train, n_splits=3)
    assert len(result.fold_accuracies) == 3
    assert all(0.0 <= a <= 1.0 for a in result.fold_accuracies)
    # 랜덤워크 데이터이므로 정확도는 50% 근처여야 정상
    assert 0.3 <= result.mean_accuracy <= 0.7
    print(f"test_backtest 통과 (랜덤워크 정확도: {result.mean_accuracy:.1%})")


if __name__ == "__main__":
    test_features()
    test_prediction()
    test_backtest()
    print("\n모든 테스트 통과!")
