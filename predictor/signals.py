"""매수/매도 타이밍 신호.

여러 근거를 점수로 합산해 5단계 판단(강한 매수~강한 매도)과 근거 목록을 만든다.

근거와 배점:
  - 모델 상승/하락 확률 (60% 이상 ±2, 55% 이상 ±1)
  - RSI 과매도(<30)/과매수(>70) ±1
  - MACD 히스토그램 부호 전환 (골든/데드 크로스) ±1
  - 볼린저 밴드 하단/상단 이탈 ±1
  - 지지선/저항선 1% 이내 근접 ±1
점수 합계: +4 이상 강한 매수, +2 이상 매수, -2 이하 매도, -4 이하 강한 매도, 그 외 관망.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .levels import Level
from .model import Prediction

NEAR_LEVEL_PCT = 0.01  # 지지/저항 근접 판정 거리 (1%)


@dataclass
class TimingAdvice:
    """타이밍 판단 결과."""
    action: str                       # 강한 매수 / 매수 / 관망 / 매도 / 강한 매도
    score: int
    reasons: list[str] = field(default_factory=list)


def _score_to_action(score: int) -> str:
    if score >= 4:
        return "강한 매수"
    if score >= 2:
        return "매수"
    if score <= -4:
        return "강한 매도"
    if score <= -2:
        return "매도"
    return "관망"


def advise(
    featured: pd.DataFrame,
    pred: Prediction,
    supports: list[Level],
    resistances: list[Level],
) -> TimingAdvice:
    """지표·예측·레벨을 종합해 타이밍 판단을 만든다.

    Args:
        featured: build_features() 결과 (마지막 행 = 현재 캔들)
        pred: predict_next() 결과
        supports / resistances: find_levels() 결과
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

    # 5) 지지/저항 근접
    if supports and abs(supports[0].distance_pct) <= NEAR_LEVEL_PCT:
        score += 1
        reasons.append(
            f"지지선 {supports[0].price:,.4f} 근접 (터치 {supports[0].touches}회) +1")
    if resistances and abs(resistances[0].distance_pct) <= NEAR_LEVEL_PCT:
        score -= 1
        reasons.append(
            f"저항선 {resistances[0].price:,.4f} 근접 (터치 {resistances[0].touches}회) -1")

    if not reasons:
        reasons.append("뚜렷한 신호 없음")

    return TimingAdvice(action=_score_to_action(score), score=score, reasons=reasons)


def format_advice(advice: TimingAdvice) -> str:
    """타이밍 판단을 사람이 읽기 좋은 문자열로 만든다."""
    icon = {"강한 매수": "🟢🟢", "매수": "🟢", "관망": "⚪",
            "매도": "🔴", "강한 매도": "🔴🔴"}[advice.action]
    lines = [f"{icon} 타이밍 판단: {advice.action} (점수 {advice.score:+d})", "근거:"]
    lines += [f"  • {r}" for r in advice.reasons]
    return "\n".join(lines)
