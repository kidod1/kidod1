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
from predictor.model import backtest, predict_next, prepare_dataset

DEFAULT_INTERVAL = "1h"
DEFAULT_LIMIT = 1500

HELP_TEXT = (
    "사용할 수 있는 명령어:\n"
    "/predict SYMBOL [간격] — 예측 실행\n"
    "    예: /predict BTCUSDT\n"
    "    예: /predict ETHUSDT 15m\n"
    "    간격: 1m 5m 15m 1h 4h 1d (기본 1h)\n"
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
        f"📊 {symbol} — 다음 {interval} 캔들",
        f"기준 시각: {pred.last_open_time:%Y-%m-%d %H:%M} UTC",
        f"현재 종가: {pred.last_close:,.4f}",
        "",
        bar,
        f"상승 {pred.prob_up:.1%}  /  하락 {pred.prob_down:.1%}",
        f"판단: {pred.direction} 우세",
    ]
    if accuracy is not None and baseline is not None:
        lines += [
            "",
            f"백테스트 정확도 {accuracy:.1%} (기준선 {baseline:.1%})",
        ]
    lines += ["", "※ 참고용이며 재무적 조언이 아닙니다."]
    return "\n".join(lines)


def run_prediction(symbol: str, interval: str, limit: int = DEFAULT_LIMIT) -> str:
    """데이터 수집→학습→예측을 수행하고 결과 메시지를 반환한다."""
    ohlcv = fetch_klines(symbol, interval, limit)
    train, latest = prepare_dataset(ohlcv)
    result = backtest(train, n_splits=3)
    pred = predict_next(train, latest)
    return format_prediction_message(
        symbol.upper(), interval, pred,
        accuracy=result.mean_accuracy, baseline=result.baseline_accuracy,
    )


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
                for symbol in args.watch:
                    try:
                        msg = run_prediction(symbol, args.watch_interval)
                    except Exception as exc:  # noqa: BLE001
                        msg = f"{symbol} 자동 예측 실패: {exc}"
                    tg.send_message(args.chat_id, msg)
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
                if text.strip().lower().startswith("/predict"):
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
