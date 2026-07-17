"""예측 성적표 — 보낸 예측을 기록해뒀다가 실제 결과와 비교한다.

리포트가 나갈 때마다 예측(방향·확률·기준가)을 로그에 기록하고,
호라이즌이 지난 예측은 실제 종가와 비교해 적중 여부를 확정한다.
누적 적중률이 리포트 하단에 표시되므로 "이 모델이 실전에서 얼마나
맞는지"를 백테스트가 아닌 실제 기록으로 확인할 수 있다.

로그 파일 구조 (JSON):
    {"pending": [...], "resolved": [...]}
    pending 항목: symbol, interval, candle_time, close, prob_up, horizon
    resolved 항목: 위 + actual_close, correct
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .signals import INTERVAL_MINUTES

DEFAULT_LOG_FILE = ".predictions_log.json"
MAX_RESOLVED = 500  # 로그 파일이 무한히 커지지 않도록 유지할 확정 기록 수


def load_log(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"pending": [], "resolved": []}
    try:
        log = json.loads(p.read_text())
        return {"pending": log.get("pending", []), "resolved": log.get("resolved", [])}
    except (json.JSONDecodeError, OSError):
        return {"pending": [], "resolved": []}


def save_log(path: str | Path, log: dict) -> None:
    log["resolved"] = log["resolved"][-MAX_RESOLVED:]
    Path(path).write_text(json.dumps(log, indent=1, sort_keys=True) + "\n")


def record_prediction(
    log: dict,
    symbol: str,
    interval: str,
    candle_time: pd.Timestamp,
    close: float,
    prob_up: float,
    horizon: int,
) -> bool:
    """예측 하나를 기록한다. 같은 캔들에 대한 중복 기록은 건너뛴다(False 반환)."""
    key = (symbol.upper(), interval, str(candle_time))
    for entry in log["pending"] + log["resolved"]:
        if (entry["symbol"], entry["interval"], entry["candle_time"]) == key:
            return False
    log["pending"].append({
        "symbol": symbol.upper(),
        "interval": interval,
        "candle_time": str(candle_time),
        "close": close,
        "prob_up": round(prob_up, 4),
        "horizon": horizon,
    })
    return True


def resolve_pending(log: dict, ohlcv_by_symbol: dict[str, pd.DataFrame]) -> int:
    """호라이즌이 지난 예측을 실제 종가와 비교해 확정한다.

    Args:
        ohlcv_by_symbol: 심볼(대문자) → 최신 OHLCV. 리포트 생성 시 이미
            받아둔 데이터를 재사용하므로 추가 API 호출이 없다.

    Returns:
        이번에 확정된 예측 수
    """
    still_pending: list[dict] = []
    resolved_now = 0

    for entry in log["pending"]:
        df = ohlcv_by_symbol.get(entry["symbol"])
        if df is None:
            still_pending.append(entry)
            continue
        minutes = INTERVAL_MINUTES.get(entry["interval"], 60)
        target_time = pd.Timestamp(entry["candle_time"]) + \
            pd.Timedelta(minutes=minutes * entry["horizon"])
        # 대상 캔들이 닫힌 뒤에만 확정 (마지막 행은 진행 중이므로 제외)
        closed = df.iloc[:-1]
        match = closed[closed["open_time"] == target_time]
        if match.empty:
            if closed["open_time"].iloc[-1] > target_time + pd.Timedelta(minutes=minutes * 24):
                continue  # 너무 오래돼 대조 불가능한 기록은 폐기
            still_pending.append(entry)
            continue
        actual = float(match["close"].iloc[0])
        went_up = actual > entry["close"]
        predicted_up = entry["prob_up"] >= 0.5
        log["resolved"].append({**entry,
                                "actual_close": actual,
                                "correct": went_up == predicted_up})
        resolved_now += 1

    log["pending"] = still_pending
    return resolved_now


def format_scorecard(log: dict, last_n: int = 100) -> str | None:
    """최근 확정 기록의 적중률 요약 문자열을 만든다 (기록이 없으면 None)."""
    resolved = log["resolved"][-last_n:]
    if not resolved:
        return None
    correct = sum(1 for r in resolved if r["correct"])
    lines = [f"🎯 실전 적중률: {correct}/{len(resolved)}건 ({correct / len(resolved):.0%})"]

    by_symbol: dict[str, list[bool]] = {}
    for r in resolved:
        by_symbol.setdefault(r["symbol"], []).append(r["correct"])
    if len(by_symbol) > 1:
        parts = [f"{sym} {sum(v) / len(v):.0%}" for sym, v in sorted(by_symbol.items())]
        lines.append("  " + " · ".join(parts))
    return "\n".join(lines)
