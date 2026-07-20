"""텔레그램 봇 — 바이낸스 상승/하락 확률 예측 결과를 휴대폰으로 받아본다.

봇 토큰 발급: 텔레그램에서 @BotFather 에게 /newbot 을 보내면 토큰을 준다.

사용 예:
    # 대화형 봇만 실행 (텔레그램에서 /predict BTCUSDT 로 조회)
    TELEGRAM_BOT_TOKEN=123:abc python bot.py

    # 1시간마다 BTCUSDT, ETHUSDT 예측을 자동으로 보내주는 모드 포함
    TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_CHAT_ID=987654321 \
        python bot.py --watch BTCUSDT ETHUSDT --every 3600

chat id는 봇에게 아무 메시지나 보낸 뒤 봇이 알려준다(/start 응답에 포함).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import requests

from predictor.data import fetch_klines
from predictor.features import HORIZON, build_features
from predictor.levels import find_levels, format_levels
from predictor.model import backtest, predict_next, prepare_dataset
from predictor.scorecard import (
    DEFAULT_LOG_FILE,
    format_scorecard,
    load_log,
    record_prediction,
    resolve_pending,
    save_log,
)
from predictor.signals import (
    INTERVAL_MINUTES,
    advise,
    format_advice,
    higher_timeframes,
    trend_direction,
)

DEFAULT_INTERVAL = "1h"
DEFAULT_LIMIT = 1500
DEFAULT_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

HELP_TEXT = (
    "사용할 수 있는 명령어:\n"
    "/predict SYMBOL [간격] — 예측 실행\n"
    "    예: /predict BTCUSDT\n"
    "    예: /predict ETHUSDT 15m\n"
    "    간격: 1m 5m 15m 1h 4h 1d (기본 1h)\n"
    "/scan [SYMBOL...] — 여러 코인 한 번에 스캔\n"
    "    예: /scan  (기본 5개 코인)\n"
    "    예: /scan BTCUSDT SOLUSDT\n"
    "/signal SYMBOL [간격] — 종합 타이밍 판단\n"
    "    (예측 + 지지/저항 + 매수·매도 신호)\n"
    "/levels SYMBOL [간격] — 지지선/저항선만 빠르게\n"
    "/report [SYMBOL...] — 전체 코인 종합 리포트 한 장\n"
    "    (예측 + 지지/저항 + 판단을 코인별로 요약)\n"
    "/help — 이 도움말\n\n"
    "※ 참고용 통계 모델이며 재무적 조언이 아닙니다."
)


class TelegramClient:
    """텔레그램 Bot API 최소 래퍼 (long polling)."""

    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()

    def get_updates(self, offset: int | None, timeout: int = 30) -> list[dict]:
        params: dict[str, int] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        resp = self.session.get(
            f"{self.base}/getUpdates", params=params, timeout=timeout + 10
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", []) if data.get("ok") else []

    def send_message(self, chat_id: int | str, text: str) -> None:
        resp = self.session.post(
            f"{self.base}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        resp.raise_for_status()


def horizon_label(interval: str, horizon: int = HORIZON) -> str:
    """호라이즌을 사람이 읽는 시간 단위로 바꾼다 (예: 1h×4 → '4시간')."""
    minutes = INTERVAL_MINUTES.get(interval, 60) * horizon
    if minutes % 1440 == 0:
        return f"{minutes // 1440}일"
    if minutes % 60 == 0:
        return f"{minutes // 60}시간"
    return f"{minutes}분"


def format_prediction_message(
    symbol: str,
    interval: str,
    pred,
    accuracy: float | None = None,
    baseline: float | None = None,
) -> str:
    """예측 결과를 텔레그램 메시지 문자열로 만든다."""
    bar_len = 10
    filled = round(pred.prob_up * bar_len)
    bar = "🟩" * filled + "🟥" * (bar_len - filled)
    lines = [
        f"📊 {symbol} — 향후 {horizon_label(interval, pred.horizon)} 방향 ({interval} 캔들)",
        f"기준 시각: {pred.last_open_time:%Y-%m-%d %H:%M} UTC",
        f"현재 종가: {pred.last_close:,.4f}",
        "",
        bar,
        f"상승 {pred.prob_up:.1%}  /  하락 {pred.prob_down:.1%}",
        f"판단: {pred.direction} 우세" if pred.is_confident
        else "판단: 중립 (확신 55% 미만 — 방향 판단 유보)",
    ]
    if accuracy is not None and baseline is not None:
        lines += [
            "",
            f"백테스트 정확도 {accuracy:.1%} (기준선 {baseline:.1%})",
        ]
    lines += ["", "※ 참고용이며 재무적 조언이 아닙니다."]
    return "\n".join(lines)


def analyze_symbol(symbol: str, interval: str, limit: int = DEFAULT_LIMIT,
                   btc_ohlcv=None):
    """데이터 수집→학습→예측을 수행하고 (예측, 백테스트 결과)를 반환한다.

    알트코인이면 BTC 동향 피처를 위해 BTCUSDT 캔들도 함께 사용한다
    (btc_ohlcv를 넘기면 재사용, 없으면 새로 조회).
    """
    ohlcv = fetch_klines(symbol, interval, limit)
    if symbol.upper() != "BTCUSDT" and btc_ohlcv is None:
        btc_ohlcv = fetch_klines("BTCUSDT", interval, limit)
    train, latest = prepare_dataset(ohlcv, btc_ohlcv=btc_ohlcv)
    result = backtest(train, n_splits=3)
    pred = predict_next(train, latest)
    return pred, result


def run_prediction(symbol: str, interval: str, limit: int = DEFAULT_LIMIT) -> str:
    """데이터 수집→학습→예측을 수행하고 결과 메시지를 반환한다."""
    pred, result = analyze_symbol(symbol, interval, limit)
    return format_prediction_message(
        symbol.upper(), interval, pred,
        accuracy=result.mean_accuracy, baseline=result.baseline_accuracy,
    )


def run_levels(symbol: str, interval: str, limit: int = DEFAULT_LIMIT) -> str:
    """지지선/저항선만 빠르게 조회한다 (모델 학습 없음)."""
    ohlcv = fetch_klines(symbol, interval, limit)
    supports, resistances = find_levels(ohlcv)
    current = float(ohlcv["close"].iloc[-1])
    return (f"📐 {symbol.upper()} {interval} 지지/저항선\n\n"
            + format_levels(supports, resistances, current))


def fetch_htf_trends(symbol: str, interval: str) -> dict[str, str]:
    """기준 간격보다 큰 상위 시간대(4h/1d)의 추세 방향을 조회한다."""
    trends: dict[str, str] = {}
    for itv in higher_timeframes(interval):
        try:
            trends[itv] = trend_direction(fetch_klines(symbol, itv, 200))
        except Exception:  # noqa: BLE001 — 상위 시간대 조회 실패는 치명적이지 않음
            continue
    return trends


def run_signal(symbol: str, interval: str, limit: int = DEFAULT_LIMIT) -> str:
    """예측 + 지지/저항 + 타이밍 판단을 종합한 메시지를 만든다."""
    ohlcv = fetch_klines(symbol, interval, limit)
    btc_ohlcv = None
    if symbol.upper() != "BTCUSDT":
        btc_ohlcv = fetch_klines("BTCUSDT", interval, limit)
    train, latest = prepare_dataset(ohlcv, btc_ohlcv=btc_ohlcv)
    result = backtest(train, n_splits=3)
    pred = predict_next(train, latest)
    featured = build_features(ohlcv)
    supports, resistances = find_levels(ohlcv)
    advice = advise(featured, pred, supports, resistances,
                    htf_trends=fetch_htf_trends(symbol, interval))

    prediction_part = format_prediction_message(
        symbol.upper(), interval, pred,
        accuracy=result.mean_accuracy, baseline=result.baseline_accuracy,
    )
    # 예측 메시지의 마지막 면책 문구는 종합 메시지 끝에 한 번만 두기 위해 제거
    prediction_part = prediction_part.rsplit("\n\n※", 1)[0]
    return "\n\n".join([
        prediction_part,
        format_levels(supports, resistances, pred.last_close),
        format_advice(advice),
        "※ 참고용이며 재무적 조언이 아닙니다.",
    ])


def full_report(
    symbols: list[str],
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    log_path: str = DEFAULT_LOG_FILE,
) -> str:
    """모든 코인의 예측·지지/저항·타이밍 판단을 한 장으로 정리한 종합 리포트.

    예측을 성적표 로그에 기록하고, 호라이즌이 지난 과거 예측은 실제 결과와
    비교해 하단에 누적 적중률을 표시한다.
    """
    from datetime import datetime, timedelta, timezone

    kst = datetime.now(timezone(timedelta(hours=9)))
    blocks = [f"📋 종합 리포트 — {interval} 캔들 · 향후 {horizon_label(interval)} 방향\n"
              f"({kst:%m/%d %H:%M} KST)"]

    # BTC 캔들은 한 번만 받아 모든 심볼의 BTC 동향 피처에 재사용
    try:
        btc_ohlcv = fetch_klines("BTCUSDT", interval, limit)
    except Exception:  # noqa: BLE001
        btc_ohlcv = None

    log = load_log(log_path)
    ohlcv_cache: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            if symbol.upper() == "BTCUSDT" and btc_ohlcv is not None:
                ohlcv = btc_ohlcv
            else:
                ohlcv = fetch_klines(symbol, interval, limit)
            ohlcv_cache[symbol.upper()] = ohlcv
            train, latest = prepare_dataset(ohlcv, btc_ohlcv=btc_ohlcv)
            result = backtest(train, n_splits=3)
            pred = predict_next(train, latest)
            featured = build_features(ohlcv, btc_df=btc_ohlcv)
            supports, resistances = find_levels(ohlcv)
            htf = fetch_htf_trends(symbol, interval)
            advice = advise(featured, pred, supports, resistances, htf_trends=htf)
            # 예측 방향이 상위 시간대 추세와 일치하는지 (순응/역추세/혼조)
            directional = [v for v in htf.values() if v in ("상승", "하락")]
            if directional and all(t == pred.direction for t in directional):
                with_trend = True
            elif directional and all(t != pred.direction for t in directional):
                with_trend = False
            else:
                with_trend = None
            # 확신 55% 미만(중립)은 방향 예측이 아니므로 성적표에 기록하지 않음
            # (같은 호라이즌 구간과 겹치는 예측은 record_prediction이 걸러냄)
            if pred.is_confident:
                record_prediction(log, symbol, interval, pred.last_open_time,
                                  pred.last_close, pred.prob_up, pred.horizon,
                                  with_trend=with_trend)
        except Exception as exc:  # noqa: BLE001
            blocks.append(f"❌ {symbol.upper()}: 조회 실패 ({exc})")
            continue

        if pred.is_confident:
            arrow = "📈" if pred.prob_up >= 0.5 else "📉"
            pred_line = (f"  예측: {pred.direction} {pred.confidence:.1%}"
                         f" (정확도 {result.mean_accuracy:.0%}"
                         f"/기준 {result.baseline_accuracy:.0%})")
        else:
            arrow = "⚪"
            pred_line = (f"  예측: 중립 — 확신 낮음 (상승 {pred.prob_up:.1%}, "
                         f"성적표 제외)")
        lines = [
            f"{arrow} {symbol.upper()}  {pred.last_close:,.4f}",
            pred_line,
        ]
        if htf:
            lines.append("  추세: " + " / ".join(f"{k} {v}" for k, v in htf.items()))
            # 확신 있는 예측이 상위 시간대 추세와 정반대면 역추세 표시
            trends = [v for v in htf.values() if v in ("상승", "하락")]
            if (pred.is_confident and trends
                    and all(t != pred.direction for t in trends)):
                lines.append("  ⚠ 역추세 예측 (4h/1d 추세와 반대 — 신뢰도 낮음)")
        if supports:
            lines.append(f"  지지: {supports[0].price:,.4f} ({supports[0].distance_pct:+.1%},"
                         f" 터치 {supports[0].touches}회)")
        if resistances:
            lines.append(f"  저항: {resistances[0].price:,.4f} ({resistances[0].distance_pct:+.1%},"
                         f" 터치 {resistances[0].touches}회)")
        lines.append(f"  판단: {advice.action} (점수 {advice.score:+d})")
        if advice.warning:
            lines.append(f"  {advice.warning}")
        if advice.stop_loss is not None and advice.take_profit is not None:
            lines.append(f"  손절 {advice.stop_loss:,.4f} / 목표 {advice.take_profit:,.4f}")
        blocks.append("\n".join(lines))

    # 성적표: 호라이즌이 지난 예측을 실제 결과와 대조하고 적중률 표시
    resolve_pending(log, ohlcv_cache)
    save_log(log_path, log)
    scorecard = format_scorecard(log)
    if scorecard:
        blocks.append(scorecard)

    blocks.append("※ 참고용이며 재무적 조언이 아닙니다.")
    return "\n\n".join(blocks)


def scan_symbols(
    symbols: list[str],
    interval: str = DEFAULT_INTERVAL,
    min_confidence: float = 0.0,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """여러 심볼을 스캔해 한 장의 요약 메시지를 만든다.

    min_confidence가 0보다 크면 그 이상 확신하는 신호에 🔥 표시를 붙인다.
    """
    lines = [f"🔍 스캔 결과 ({interval} 캔들 기준)", ""]
    for symbol in symbols:
        try:
            pred, result = analyze_symbol(symbol, interval, limit)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  {symbol.upper()}: 조회 실패 ({exc})")
            continue
        confidence = max(pred.prob_up, pred.prob_down)
        arrow = "📈" if pred.prob_up >= 0.5 else "📉"
        strong = " 🔥" if min_confidence > 0 and confidence >= min_confidence else ""
        lines.append(
            f"  {arrow} {symbol.upper()}: {pred.direction} {confidence:.1%}"
            f" (정확도 {result.mean_accuracy:.0%}/기준 {result.baseline_accuracy:.0%}){strong}"
        )
    lines += ["", "※ 참고용이며 재무적 조언이 아닙니다."]
    return "\n".join(lines)


def handle_command(text: str) -> str:
    """봇 명령어를 해석해 응답 메시지를 만든다."""
    parts = text.strip().split()
    if not parts:
        return HELP_TEXT
    cmd = parts[0].lower().split("@")[0]  # 그룹 채팅의 /predict@botname 형태 지원

    if cmd in ("/start", "/help"):
        return HELP_TEXT

    if cmd == "/predict":
        if len(parts) < 2:
            return "심볼을 입력해주세요. 예: /predict BTCUSDT 1h"
        symbol = parts[1].upper()
        interval = parts[2] if len(parts) > 2 else DEFAULT_INTERVAL
        try:
            return run_prediction(symbol, interval)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                return f"'{symbol}' 심볼 또는 '{interval}' 간격이 올바르지 않습니다."
            return f"바이낸스 API 오류: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"예측 실패: {exc}"

    if cmd == "/scan":
        symbols = [p.upper() for p in parts[1:]] or DEFAULT_WATCHLIST
        return scan_symbols(symbols)

    if cmd == "/report":
        symbols = [p.upper() for p in parts[1:]] or DEFAULT_WATCHLIST
        return full_report(symbols)

    if cmd in ("/signal", "/levels"):
        if len(parts) < 2:
            return f"심볼을 입력해주세요. 예: {cmd} BTCUSDT 1h"
        symbol = parts[1].upper()
        interval = parts[2] if len(parts) > 2 else DEFAULT_INTERVAL
        runner = run_signal if cmd == "/signal" else run_levels
        try:
            return runner(symbol, interval)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                return f"'{symbol}' 심볼 또는 '{interval}' 간격이 올바르지 않습니다."
            return f"바이낸스 API 오류: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"조회 실패: {exc}"

    return HELP_TEXT


def main() -> int:
    parser = argparse.ArgumentParser(description="바이낸스 예측 텔레그램 봇")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"),
                        help="봇 토큰 (기본: TELEGRAM_BOT_TOKEN 환경변수)")
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"),
                        help="자동 예측을 보낼 chat id (기본: TELEGRAM_CHAT_ID 환경변수)")
    parser.add_argument("--watch", nargs="*", default=[],
                        help="주기적으로 자동 예측할 심볼 목록 (예: --watch BTCUSDT ETHUSDT)")
    parser.add_argument("--watch-interval", default=DEFAULT_INTERVAL,
                        help="자동 예측에 사용할 캔들 간격 (기본: 1h)")
    parser.add_argument("--every", type=int, default=3600,
                        help="자동 예측 주기(초, 기본 3600)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="자동 예측에서 이 확률 이상 확신할 때만 알림 전송 "
                             "(예: 0.6 — 기본 0.0은 항상 전송)")
    parser.add_argument("--report", action="store_true",
                        help="자동 전송을 개별 신호 대신 종합 리포트 한 장으로 보냄 "
                             "(예측+지지/저항+판단 요약)")
    args = parser.parse_args()

    if not args.token:
        print("봇 토큰이 없습니다. TELEGRAM_BOT_TOKEN 환경변수 또는 --token 을 지정하세요.",
              file=sys.stderr)
        return 1
    if args.watch and not args.chat_id:
        print("--watch 를 쓰려면 TELEGRAM_CHAT_ID 환경변수 또는 --chat-id 가 필요합니다.",
              file=sys.stderr)
        return 1

    tg = TelegramClient(args.token)
    offset: int | None = None
    next_push = time.time() if args.watch else None

    print("봇이 시작되었습니다. 텔레그램에서 /predict BTCUSDT 를 보내보세요. (Ctrl+C 종료)")
    while True:
        try:
            # 1) 자동(watch) 예측 푸시
            if next_push is not None and time.time() >= next_push:
                if args.report:
                    # 종합 리포트 한 장으로 전송
                    tg.send_message(args.chat_id,
                                    full_report(args.watch, args.watch_interval))
                else:
                    # 코인별 개별 신호 (min-confidence 필터 적용)
                    for symbol in args.watch:
                        try:
                            pred, result = analyze_symbol(symbol, args.watch_interval)
                        except Exception as exc:  # noqa: BLE001
                            tg.send_message(args.chat_id, f"{symbol} 자동 예측 실패: {exc}")
                            continue
                        confidence = max(pred.prob_up, pred.prob_down)
                        if confidence < args.min_confidence:
                            continue  # 확신이 낮은 신호는 조용히 넘어감
                        tg.send_message(args.chat_id, format_prediction_message(
                            symbol.upper(), args.watch_interval, pred,
                            accuracy=result.mean_accuracy,
                            baseline=result.baseline_accuracy,
                        ))
                next_push = time.time() + args.every

            # 2) 사용자 명령 처리 (long polling)
            for update in tg.get_updates(offset, timeout=25):
                offset = update["update_id"] + 1
                message = update.get("message") or {}
                text = message.get("text")
                chat = message.get("chat") or {}
                chat_id = chat.get("id")
                if not text or chat_id is None:
                    continue
                if text.strip().lower().startswith("/start"):
                    tg.send_message(
                        chat_id,
                        f"안녕하세요! 이 채팅의 chat id는 {chat_id} 입니다.\n"
                        f"(자동 예측을 받으려면 TELEGRAM_CHAT_ID={chat_id} 로 설정)\n\n"
                        + HELP_TEXT,
                    )
                    continue
                if text.strip().lower().startswith(("/predict", "/scan", "/signal", "/report")):
                    tg.send_message(chat_id, "분석 중입니다... (10~30초 소요) ⏳")
                tg.send_message(chat_id, handle_command(text))

        except KeyboardInterrupt:
            print("\n봇을 종료합니다.")
            return 0
        except Exception:  # noqa: BLE001 — 네트워크 오류 등은 잠시 후 재시도
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
