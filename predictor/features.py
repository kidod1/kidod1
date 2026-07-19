"""OHLCV 데이터에서 기술적 지표 피처를 만드는 모듈.

외부 TA 라이브러리 없이 pandas만으로 계산한다.
타깃(label)은 "HORIZON개 캔들 뒤 종가가 현재 종가보다 높은가" (1=상승, 0=하락)이다.
단일 캔들보다 노이즈가 적어 예측 대상으로 더 적합하다.

알트코인 예측 시 btc_df를 넘기면 비트코인 동향 피처(btc_*)가 추가된다
(알트코인은 BTC를 따라 움직이는 경향이 강하다).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 예측 호라이즌: 앞으로 몇 개 캔들 뒤의 방향을 맞출 것인가
HORIZON = 4


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


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(고가-저가, |고가-전종가|, |저가-전종가|)."""
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Wilder 방식) — 평균 변동폭."""
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series]:
    """(ADX, DI차이)를 반환한다.

    ADX는 추세의 '강도'(방향 무관, 25 이상이면 추세장),
    DI차이(+DI - -DI)는 추세의 '방향'(양수=상승 추세)이다.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1 / period
    tr_smooth = true_range(df).ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False).mean(), plus_di - minus_di


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """MFI (Money Flow Index) — 거래량 가중 RSI. 20 이하 과매도, 80 이상 과매수."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    flow = typical * df["volume"]
    direction = typical.diff()
    pos = flow.where(direction > 0, 0.0).rolling(period).sum()
    neg = flow.where(direction < 0, 0.0).rolling(period).sum().replace(0, np.nan)
    return 100 - 100 / (1 + pos / neg)


def obv(df: pd.DataFrame) -> pd.Series:
    """OBV (On-Balance Volume) — 상승 캔들 거래량은 더하고 하락은 빼는 누적 흐름."""
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def stoch_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """스토캐스틱 RSI — RSI가 최근 period 범위에서 어디쯤인지 (0~1)."""
    r = rsi(close, period)
    lo = r.rolling(period).min()
    hi = r.rolling(period).max()
    return (r - lo) / (hi - lo).replace(0, np.nan)


def divergence_flags(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    window: int = 3,
    hold: int = 6,
) -> tuple[pd.Series, pd.Series]:
    """(약세 다이버전스, 강세 다이버전스) 플래그 시리즈를 반환한다.

    약세: 가격은 직전 스윙 고점보다 높은 고점인데 RSI는 더 낮음 → 상승 동력 약화
    강세: 가격은 직전 스윙 저점보다 낮은 저점인데 RSI는 더 높음 → 하락 동력 약화

    스윙 피벗은 좌우 window개 캔들이 지나야 확정되므로, 신호는 피벗 확정
    시점(피벗 + window 캔들)부터 hold개 캔들 동안 켜진다 (미래 정보 누수 없음).
    """
    size = 2 * window + 1
    high, low = df["high"], df["low"]
    pivot_high = (high == high.rolling(size, center=True).max()).fillna(False)
    pivot_low = (low == low.rolling(size, center=True).min()).fillna(False)

    bear = np.zeros(len(df))
    bull = np.zeros(len(df))

    ph_idx = np.flatnonzero(pivot_high.to_numpy())
    for prev, cur in zip(ph_idx, ph_idx[1:]):
        if high.iloc[cur] > high.iloc[prev] and \
                rsi_series.iloc[cur] < rsi_series.iloc[prev]:
            start = cur + window  # 피벗 확정 시점
            bear[start:start + hold] = 1

    pl_idx = np.flatnonzero(pivot_low.to_numpy())
    for prev, cur in zip(pl_idx, pl_idx[1:]):
        if low.iloc[cur] < low.iloc[prev] and \
                rsi_series.iloc[cur] > rsi_series.iloc[prev]:
            start = cur + window
            bull[start:start + hold] = 1

    return pd.Series(bear, index=df.index), pd.Series(bull, index=df.index)


def ichimoku(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """일목균형표 — (전환선, 기준선, 선행스팬A, 선행스팬B)를 반환한다.

    선행스팬은 26캔들 앞에 그려지므로, '현재 캔들 위치의 구름'은
    26캔들 전에 계산된 값이다 (shift(26) 적용 완료 상태로 반환).
    """
    def midline(period: int) -> pd.Series:
        return (df["high"].rolling(period).max() + df["low"].rolling(period).min()) / 2

    tenkan = midline(9)     # 전환선
    kijun = midline(26)     # 기준선
    senkou_a = ((tenkan + kijun) / 2).shift(26)  # 선행스팬A (현재 위치의 구름)
    senkou_b = midline(52).shift(26)             # 선행스팬B
    return tenkan, kijun, senkou_a, senkou_b


def build_features(
    df: pd.DataFrame,
    btc_df: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """OHLCV DataFrame에 피처와 타깃 컬럼을 추가해 반환한다.

    마지막 horizon개 행은 target이 NaN이다(미래 캔들이 아직 없으므로).
    마지막 행이 "지금 예측할" 시점이다.

    Args:
        df: 대상 심볼의 OHLCV
        btc_df: BTCUSDT의 같은 간격 OHLCV (없으면 df 자신을 사용 —
                BTCUSDT 자체를 분석할 때와 동일한 결과)
        horizon: 타깃 호라이즌 (캔들 수)
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

    # ATR — 평균 변동폭 (가격 대비 비율로 정규화)
    out["atr_ratio"] = atr(out, 14) / close

    # ADX — 추세 강도와 방향
    adx_val, di_diff = adx(out, 14)
    out["adx_14"] = adx_val
    out["di_diff"] = di_diff

    # 자금 흐름 — MFI, OBV 기울기 (최근 12캔들 변화를 거래량 합으로 정규화)
    out["mfi_14"] = mfi(out, 14)
    obv_series = obv(out)
    out["obv_slope"] = (obv_series - obv_series.shift(12)) / \
        out["volume"].rolling(24).sum().replace(0, np.nan)

    # 스토캐스틱 RSI
    out["stoch_rsi"] = stoch_rsi(close, 14)

    # 일목균형표 — 구름 대비 가격 위치, 전환선/기준선 관계
    tenkan, kijun, senkou_a, senkou_b = ichimoku(out)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    out["price_vs_cloud_top"] = close / cloud_top - 1
    out["price_vs_cloud_bottom"] = close / cloud_bottom - 1
    out["tenkan_vs_kijun"] = (tenkan - kijun) / close

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

    # 다이버전스 — 추세 전환의 선행 신호
    bear_div, bull_div = divergence_flags(out, out["rsi_14"])
    out["bearish_divergence"] = bear_div
    out["bullish_divergence"] = bull_div

    # 비트코인 동향 피처 (알트코인은 BTC를 따라가는 경향)
    ref = btc_df if btc_df is not None else df
    btc = ref[["open_time", "close"]].rename(columns={"close": "btc_close"})
    out = out.merge(btc, on="open_time", how="left")
    btc_close = out["btc_close"].ffill()
    out["btc_return_1"] = btc_close.pct_change()
    out["btc_return_6"] = btc_close.pct_change(6)
    out["btc_return_24"] = btc_close.pct_change(24)
    out["btc_vs_sma25"] = btc_close / btc_close.rolling(25).mean() - 1
    out = out.drop(columns=["btc_close"])

    # 타깃: horizon개 캔들 뒤 종가가 오르면 1, 내리면 0
    out["target"] = (close.shift(-horizon) > close).astype(float)
    out.iloc[-horizon:, out.columns.get_loc("target")] = np.nan
    # 다음 1캔들 수익률 (매매 전략 백테스트용; 학습 피처로는 사용하지 않음)
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
    "atr_ratio", "adx_14", "di_diff",
    "mfi_14", "obv_slope", "stoch_rsi",
    "price_vs_cloud_top", "price_vs_cloud_bottom", "tenkan_vs_kijun",
    "btc_return_1", "btc_return_6", "btc_return_24", "btc_vs_sma25",
    "bearish_divergence", "bullish_divergence",
]
