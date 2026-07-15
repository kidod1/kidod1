"""일회성 예측 알림 — 지정한 심볼들을 예측해 텔레그램으로 보내고 종료한다.

GitHub Actions 같은 스케줄 실행 환경에서 사용한다 (상주 프로세스 불필요).

사용 예:
    TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_CHAT_ID=987654321 \
        python notify.py --symbols BTCUSDT ETHUSDT --interval 1h
"""

from __future__ import annotations

import argparse
import os
import sys

from bot import TelegramClient, run_prediction


def main() -> int:
    parser = argparse.ArgumentParser(description="예측 결과를 텔레그램으로 1회 전송")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"],
                        help="예측할 심볼 목록 (기본: BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="캔들 간격 (기본: 1h)")
    parser.add_argument("--limit", type=int, default=1500,
                        help="학습에 사용할 캔들 개수 (기본: 1500)")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 가 필요합니다.", file=sys.stderr)
        return 1

    tg = TelegramClient(args.token)
    failures = 0
    for symbol in args.symbols:
        try:
            message = run_prediction(symbol, args.interval, args.limit)
        except Exception as exc:  # noqa: BLE001
            message = f"{symbol} 예측 실패: {exc}"
            failures += 1
            print(message, file=sys.stderr)
        tg.send_message(args.chat_id, message)
        print(f"{symbol}: 전송 완료")

    return 1 if failures == len(args.symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
