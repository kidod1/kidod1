"""OHLCV 데이터에서 기술적 지표 피처를 만드는 모듈.

외부 TA 라이브러리 없이 pandas만으로 계산한다.
타깃(label)은 "다음 캔들의 종가가 현재 종가보다 높은가" (1=상승, 0=하락)이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Wilder 방식의 지수이동평균 사용)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 라인, 시그널 라인, 히스토그램을 반환한다."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에 피처와 타깃 컬럼을 추가해 반환한다.

    반환된 DataFrame의 마지막 행은 target이 NaN이다(다음 캔들이 아직 없으므로).
    이 마지막 행이 "지금 예측할" 시점이다.
    """
    out = df.copy()
    close = out["close"]

    # 수익률 계열 (단기 + 장기 호라이즌)
    out["return_1"] = close.pct_change()
    out["return_3"] = close.pct_change(3)
    out["return_6"] = close.pct_change(6)
    out["return_12"] = close.pct_change(12)
    out["return_24"] = close.pct_change(24)
    out["return_72"] = close.pct_change(72)

    # 이동평균 대비 위치 (추세; 200은 장기 추세 대용)
    for period in (7, 25, 99, 200):
        sma = close.rolling(period).mean()
        out[f"close_vs_sma{period}"] = close / sma - 1

    # 변동성 (단기/중기 + 변동성 국면 전환)
    out["volatility_12"] = out["return_1"].rolling(12).std()
    out["volatility_24"] = out["return_1"].rolling(24).std()
    out["volatility_72"] = out["return_1"].rolling(72).std()
    out["vol_regime"] = out["volatility_12"] / out["volatility_72"].replace(0, np.nan)

    # 캔들 모양
    body = (out["close"] - out["open"]).abs()
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_ratio"] = body / candle_range
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / candle_range
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / candle_range

    # RSI (단기 + 장기 호라이즌)
    out["rsi_14"] = rsi(close, 14)
    out["rsi_56"] = rsi(close, 56)

    # 시간대 주기성 (암호화폐는 요일/시간대 패턴이 존재)
    hour = out["open_time"].dt.hour
    dow = out["open_time"].dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # MACD (가격 스케일에 의존하지 않도록 종가로 정규화)
    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line / close
    out["macd_signal"] = signal_line / close
    out["macd_hist"] = hist / close

    # 볼린저 밴드 내 위치 (0=하단, 1=상단)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_position"] = (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    out["bb_width"] = (4 * std20) / sma20

    # 거래량
    vol_sma = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / vol_sma.replace(0, np.nan)
    out["taker_buy_ratio"] = out["taker_buy_base"] / out["volume"].replace(0, np.nan)

    # 타깃: 다음 캔들 종가가 오르면 1, 내리면 0
    out["target"] = (close.shift(-1) > close).astype(float)
    out.loc[out.index[-1], "target"] = np.nan
    # 다음 캔들 수익률 (매매 전략 백테스트용; 학습 피처로는 사용하지 않음)
    out["forward_return"] = close.shift(-1) / close - 1

    return out


FEATURE_COLUMNS = [
    "return_1", "return_3", "return_6", "return_12", "return_24", "return_72",
    "close_vs_sma7", "close_vs_sma25", "close_vs_sma99", "close_vs_sma200",
    "volatility_12", "volatility_24", "volatility_72", "vol_regime",
    "body_ratio", "upper_wick", "lower_wick",
    "rsi_14", "rsi_56", "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width",
    "volume_ratio", "taker_buy_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
