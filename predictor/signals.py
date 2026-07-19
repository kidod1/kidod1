"""매수/매도 타이밍 신호.

여러 근거를 점수로 합산해 5단계 판단(강한 매수~강한 매도)과 근거 목록을 만든다.

근거와 배점:
  - 모델 상승/하락 확률 (60% 이상 ±2, 55% 이상 ±1)
  - RSI 과매도(<30)/과매수(>70) ±1
  - 스토캐스틱 RSI 극단(<0.1 / >0.9) ±1
  - MFI 자금 유출입 극단(<20 / >80) ±1
  - MACD 히스토그램 부호 전환 (골든/데드 크로스) ±1
  - 볼린저 밴드 하단/상단 이탈 ±1
  - ADX 25 이상 추세장에서 DI 방향 ±1 (25 미만이면 횡보장 참고만)
  - 일목균형표 구름 위/아래 ±1
  - 상위 시간대(4h/1d) 추세 일치 각 ±1
  - 지지선/저항선 1% 이내 근접 ±1
점수 합계: +5 이상 강한 매수, +2 이상 매수, -2 이하 매도, -5 이하 강한 매도, 그 외 관망.

매수/매도 판단일 때는 ATR(평균 변동폭) 기반 손절가·목표가를 함께 제안한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .levels import Level
from .model import Prediction

NEAR_LEVEL_PCT = 0.01   # 지지/저항 근접 판정 거리 (1%)
ATR_STOP_MULT = 1.5     # 손절 거리 = ATR × 1.5
ATR_TARGET_MULT = 2.0   # 목표 거리 = ATR × 2.0

# 상위 시간대 추세 판정에 쓰는 간격의 분 단위 크기
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080,
}


@dataclass
class TimingAdvice:
    """타이밍 판단 결과."""
    action: str                       # 강한 매수 / 매수 / 관망 / 매도 / 강한 매도
    score: int
    reasons: list[str] = field(default_factory=list)
    stop_loss: float | None = None    # ATR 기반 손절 참고가 (매수/매도 판단일 때만)
    take_profit: float | None = None  # ATR 기반 목표 참고가
    warning: str | None = None        # 반전 신호가 겹칠 때의 경고 문구


def _candle_patterns(row: pd.Series, prev: pd.Series) -> tuple[bool, bool]:
    """(약세 반전 캔들, 강세 반전 캔들) 여부를 반환한다.

    약세: 하락 장악형(직전 양봉 몸통을 덮는 음봉) 또는 유성형(긴 윗꼬리)
    강세: 상승 장악형 또는 망치형(긴 아랫꼬리)
    """
    body = abs(row["close"] - row["open"])
    prev_body = abs(prev["close"] - prev["open"])
    candle_range = max(row["high"] - row["low"], 1e-12)
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]

    bearish_engulf = (row["close"] < row["open"] and prev["close"] > prev["open"]
                      and body > prev_body
                      and row["close"] < prev["open"] and row["open"] > prev["close"])
    shooting_star = upper_wick > 2 * body and upper_wick / candle_range > 0.6

    bullish_engulf = (row["close"] > row["open"] and prev["close"] < prev["open"]
                      and body > prev_body
                      and row["close"] > prev["open"] and row["open"] < prev["close"])
    hammer = lower_wick > 2 * body and lower_wick / candle_range > 0.6

    return (bearish_engulf or shooting_star), (bullish_engulf or hammer)


def _score_to_action(score: int) -> str:
    if score >= 5:
        return "강한 매수"
    if score >= 2:
        return "매수"
    if score <= -5:
        return "강한 매도"
    if score <= -2:
        return "매도"
    return "관망"


def trend_direction(ohlcv: pd.DataFrame) -> str:
    """단순 추세 판정 — 종가가 SMA25 위(+0.5%)면 상승, 아래(-0.5%)면 하락."""
    close = ohlcv["close"]
    if len(close) < 26:
        return "중립"
    sma25 = close.rolling(25).mean().iloc[-1]
    last = float(close.iloc[-1])
    if last > sma25 * 1.005:
        return "상승"
    if last < sma25 * 0.995:
        return "하락"
    return "중립"


def higher_timeframes(interval: str) -> list[str]:
    """기준 간격보다 큰 확인용 상위 시간대 목록을 반환한다."""
    base = INTERVAL_MINUTES.get(interval, 60)
    return [itv for itv in ("4h", "1d") if INTERVAL_MINUTES[itv] > base]


def advise(
    featured: pd.DataFrame,
    pred: Prediction,
    supports: list[Level],
    resistances: list[Level],
    htf_trends: dict[str, str] | None = None,
) -> TimingAdvice:
    """지표·예측·레벨·상위 시간대 추세를 종합해 타이밍 판단을 만든다.

    Args:
        featured: build_features() 결과 (마지막 행 = 현재 캔들)
        pred: predict_next() 결과
        supports / resistances: find_levels() 결과
        htf_trends: 상위 시간대 추세 (예: {"4h": "상승", "1d": "하락"}), 없으면 생략
    """
    row = featured.iloc[-1]
    prev = featured.iloc[-2] if len(featured) >= 2 else row
    score = 0
    reasons: list[str] = []

    # 1) 모델 확률
    if pred.prob_up >= 0.60:
        score += 2
        reasons.append(f"모델 상승 확률 높음 ({pred.prob_up:.0%}) +2")
    elif pred.prob_up >= 0.55:
        score += 1
        reasons.append(f"모델 상승 우세 ({pred.prob_up:.0%}) +1")
    elif pred.prob_down >= 0.60:
        score -= 2
        reasons.append(f"모델 하락 확률 높음 ({pred.prob_down:.0%}) -2")
    elif pred.prob_down >= 0.55:
        score -= 1
        reasons.append(f"모델 하락 우세 ({pred.prob_down:.0%}) -1")

    # 2) RSI
    rsi = row["rsi_14"]
    if rsi < 30:
        score += 1
        reasons.append(f"RSI 과매도 ({rsi:.0f}) +1")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI 과매수 ({rsi:.0f}) -1")

    # 2-1) 스토캐스틱 RSI 극단
    srsi = row["stoch_rsi"]
    if not pd.isna(srsi):
        if srsi < 0.1:
            score += 1
            reasons.append(f"스토캐스틱RSI 바닥권 ({srsi:.2f}) +1")
        elif srsi > 0.9:
            score -= 1
            reasons.append(f"스토캐스틱RSI 천장권 ({srsi:.2f}) -1")

    # 2-2) MFI 자금 흐름 극단
    mfi_val = row["mfi_14"]
    if not pd.isna(mfi_val):
        if mfi_val < 20:
            score += 1
            reasons.append(f"MFI 자금 유출 과다 ({mfi_val:.0f}) +1")
        elif mfi_val > 80:
            score -= 1
            reasons.append(f"MFI 자금 유입 과열 ({mfi_val:.0f}) -1")

    # 3) MACD 히스토그램 부호 전환
    if prev["macd_hist"] <= 0 < row["macd_hist"]:
        score += 1
        reasons.append("MACD 골든크로스 +1")
    elif prev["macd_hist"] >= 0 > row["macd_hist"]:
        score -= 1
        reasons.append("MACD 데드크로스 -1")

    # 4) 볼린저 밴드 이탈
    bb = row["bb_position"]
    if bb < 0.05:
        score += 1
        reasons.append("볼린저 밴드 하단 이탈 +1")
    elif bb > 0.95:
        score -= 1
        reasons.append("볼린저 밴드 상단 이탈 -1")

    # 5) ADX 추세 국면 — 추세장(25 이상)에서만 방향 점수, 아니면 참고 표기
    adx_val, di_diff = row["adx_14"], row["di_diff"]
    if not pd.isna(adx_val) and not pd.isna(di_diff):
        if adx_val >= 25:
            if di_diff > 0:
                score += 1
                reasons.append(f"ADX {adx_val:.0f} 상승 추세장 +1")
            elif di_diff < 0:
                score -= 1
                reasons.append(f"ADX {adx_val:.0f} 하락 추세장 -1")
        elif adx_val < 20:
            reasons.append(f"ADX {adx_val:.0f} 횡보장 (추세 신호 신뢰도 낮음)")

    # 6) 일목균형표 구름 위치
    above_cloud = row["price_vs_cloud_top"]
    below_cloud = row["price_vs_cloud_bottom"]
    if not pd.isna(above_cloud) and not pd.isna(below_cloud):
        if above_cloud > 0:
            score += 1
            reasons.append("일목 구름 위 (상승 배열) +1")
        elif below_cloud < 0:
            score -= 1
            reasons.append("일목 구름 아래 (하락 배열) -1")

    # 7) 상위 시간대 추세 일치
    for itv, trend in (htf_trends or {}).items():
        if trend == "상승":
            score += 1
            reasons.append(f"{itv} 추세 상승 +1")
        elif trend == "하락":
            score -= 1
            reasons.append(f"{itv} 추세 하락 -1")

    # 8) 지지/저항 근접
    near_support = bool(supports and abs(supports[0].distance_pct) <= NEAR_LEVEL_PCT)
    near_resistance = bool(resistances
                           and abs(resistances[0].distance_pct) <= NEAR_LEVEL_PCT)
    if near_support:
        score += 1
        reasons.append(
            f"지지선 {supports[0].price:,.4f} 근접 (터치 {supports[0].touches}회) +1")
    if near_resistance:
        score -= 1
        reasons.append(
            f"저항선 {resistances[0].price:,.4f} 근접 (터치 {resistances[0].touches}회) -1")

    # 9) 추세 전환(반전) 신호 — 다이버전스 + 반전 캔들 패턴
    rev_bear: list[str] = []
    rev_bull: list[str] = []
    if row.get("bearish_divergence", 0) == 1:
        score -= 1
        reasons.append("약세 다이버전스 (가격 신고점인데 RSI 하락) -1")
        rev_bear.append("약세 다이버전스")
    if row.get("bullish_divergence", 0) == 1:
        score += 1
        reasons.append("강세 다이버전스 (가격 신저점인데 RSI 상승) +1")
        rev_bull.append("강세 다이버전스")

    bear_candle, bull_candle = _candle_patterns(row, prev)
    if bear_candle and near_resistance:
        score -= 1
        reasons.append("저항선 부근 약세 반전 캔들 -1")
        rev_bear.append("저항선 반전 캔들")
    elif bear_candle:
        rev_bear.append("약세 반전 캔들")
    if bull_candle and near_support:
        score += 1
        reasons.append("지지선 부근 강세 반전 캔들 +1")
        rev_bull.append("지지선 반전 캔들")
    elif bull_candle:
        rev_bull.append("강세 반전 캔들")

    if not reasons:
        reasons.append("뚜렷한 신호 없음")

    # 반전 증거가 2개 이상 겹치면 경고 (추세 지표는 후행이라 전환점에서 늦음)
    warning = None
    if len(rev_bear) >= 2:
        warning = "⚠️ 하락 반전 주의: " + " + ".join(rev_bear)
    elif len(rev_bull) >= 2:
        warning = "⚠️ 상승 반전 주의: " + " + ".join(rev_bull)

    # 매수/매도 판단이면 ATR 기반 손절가·목표가 제안
    action = _score_to_action(score)
    stop_loss = take_profit = None
    atr_ratio = row["atr_ratio"]
    if not pd.isna(atr_ratio) and atr_ratio > 0:
        price = float(row["close"])
        if action in ("매수", "강한 매수"):
            stop_loss = price * (1 - ATR_STOP_MULT * atr_ratio)
            take_profit = price * (1 + ATR_TARGET_MULT * atr_ratio)
        elif action in ("매도", "강한 매도"):
            stop_loss = price * (1 + ATR_STOP_MULT * atr_ratio)
            take_profit = price * (1 - ATR_TARGET_MULT * atr_ratio)

    return TimingAdvice(action=action, score=score, reasons=reasons,
                        stop_loss=stop_loss, take_profit=take_profit,
                        warning=warning)


def format_advice(advice: TimingAdvice) -> str:
    """타이밍 판단을 사람이 읽기 좋은 문자열로 만든다."""
    icon = {"강한 매수": "🟢🟢", "매수": "🟢", "관망": "⚪",
            "매도": "🔴", "강한 매도": "🔴🔴"}[advice.action]
    lines = [f"{icon} 타이밍 판단: {advice.action} (점수 {advice.score:+d})"]
    if advice.warning:
        lines.append(advice.warning)
    lines.append("근거:")
    lines += [f"  • {r}" for r in advice.reasons]
    if advice.stop_loss is not None and advice.take_profit is not None:
        lines += [
            "참고 가격 (ATR 변동폭 기준):",
            f"  손절: {advice.stop_loss:,.4f} / 목표: {advice.take_profit:,.4f}",
        ]
    return "\n".join(lines)
