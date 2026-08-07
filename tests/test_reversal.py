"""추세 전환 대응 기능(다이버전스·반전 캔들·추세 전환 감지)을 검증한다.

실행: python tests/test_reversal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predictor.features import divergence_flags, rsi
from predictor.signals import _candle_patterns
from predictor.trendshift import check_trend_shift, current_trend, format_trend_shift
from tests.test_pipeline import make_synthetic_ohlcv


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    times = pd.date_range("2025-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({
        "open_time": times, "open": open_,
        "high": np.maximum(open_, close) + 0.05,
        "low": np.minimum(open_, close) - 0.05,
        "close": close, "volume": np.full(len(close), 100.0),
    })


def test_bearish_divergence():
    """느려지는 상승(가격 신고점 + RSI 하락)에서 약세 다이버전스가 잡히는지."""
    # 급등 후 완만한 추가 상승: 두 번째 고점은 더 높지만 상승 탄력은 약함
    up_fast = np.linspace(100, 130, 30)          # 가파른 상승 → 첫 고점
    pull = np.linspace(130, 124, 10)             # 조정
    up_slow = np.linspace(124, 131, 30)          # 완만한 상승 → 더 높은 고점
    tail = np.linspace(131, 129, 10)             # 확정용 하락
    close = np.concatenate([up_fast, pull, up_slow, tail])
    df = _ohlcv_from_close(close)
    bear, bull = divergence_flags(df, rsi(df["close"]))
    assert bear.sum() > 0, "약세 다이버전스를 감지해야 함"
    print(f"test_bearish_divergence 통과 (신호 캔들 {int(bear.sum())}개)")


def test_no_divergence_on_clean_trend():
    """일정한 기울기의 깨끗한 추세에서는 다이버전스가 드물어야 함."""
    rng = np.random.default_rng(1)
    close = np.linspace(100, 150, 300) + rng.normal(0, 0.05, 300)
    df = _ohlcv_from_close(close)
    bear, bull = divergence_flags(df, rsi(df["close"]))
    assert bear.sum() <= len(df) * 0.1
    print("test_no_divergence_on_clean_trend 통과")


def test_candle_patterns():
    # 하락 장악형: 직전 양봉을 완전히 덮는 음봉
    prev = pd.Series({"open": 100.0, "close": 102.0, "high": 102.5, "low": 99.5})
    row = pd.Series({"open": 102.5, "close": 99.0, "high": 103.0, "low": 98.5})
    bear, bull = _candle_patterns(row, prev)
    assert bear and not bull

    # 망치형: 긴 아랫꼬리 + 작은 몸통
    prev2 = pd.Series({"open": 100.0, "close": 99.0, "high": 100.5, "low": 98.5})
    row2 = pd.Series({"open": 99.0, "close": 99.2, "high": 99.3, "low": 96.0})
    bear2, bull2 = _candle_patterns(row2, prev2)
    assert bull2 and not bear2
    print("test_candle_patterns 통과")


def test_trend_shift_detection():
    """뚜렷한 추세와 전환이 감지되는지 (휩쏘 억제 등 상세는 test_trendshift.py)."""
    from predictor.trendshift import COOLDOWN_CANDLES

    n = 120
    up = _ohlcv_from_close(np.linspace(100, 150, n))
    assert current_trend(up) == "상승"
    down = _ohlcv_from_close(np.concatenate([
        np.linspace(100, 150, n - 40), np.linspace(150, 110, 40)]))
    assert current_trend(down) == "하락"

    # 첫 실행: 기록만 (알림 없음)
    shift, stored = check_trend_shift(up, "BTCUSDT", "4h", None)
    assert shift is None and stored["trend"] == "상승"
    # 추세 유지: 알림 없음
    shift, stored = check_trend_shift(up, "BTCUSDT", "4h", stored)
    assert shift is None and stored["trend"] == "상승"
    # 전환: 쿨다운이 지난 뒤에만 알림
    shift, stored = check_trend_shift(
        down, "BTCUSDT", "4h",
        {"trend": "상승", "candles_since_shift": COOLDOWN_CANDLES})
    assert shift is not None and stored["trend"] == "하락"
    assert shift.old_trend == "상승" and shift.new_trend == "하락"
    msg = format_trend_shift(shift)
    assert "추세 전환" in msg and "상승 → 하락" in msg
    print("test_trend_shift_detection 통과")
    print(msg)


def test_features_include_divergence():
    from predictor.features import FEATURE_COLUMNS, build_features
    df = make_synthetic_ohlcv(n=600)
    featured = build_features(df)
    assert "bearish_divergence" in FEATURE_COLUMNS
    assert set(featured["bearish_divergence"].unique()) <= {0.0, 1.0}
    print("test_features_include_divergence 통과")


if __name__ == "__main__":
    test_bearish_divergence()
    test_no_divergence_on_clean_trend()
    test_candle_patterns()
    test_trend_shift_detection()
    test_features_include_divergence()
    print("\n모든 반전 대응 테스트 통과!")
