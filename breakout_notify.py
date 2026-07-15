"""지지/저항 돌파 감지 후 텔레그램 알림 — 1회 실행하고 종료한다.

GitHub Actions 스케줄 실행용. 마지막으로 검사한 캔들 시각을 상태 파일에
기록해두므로, 실행 간격이 불규칙해도 같은 돌파를 두 번 알리지 않는다.

사용 예:
    TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... \
        python breakout_notify.py --symbols BTCUSDT ETHUSDT --interval 1h
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from bot import TelegramClient
from predictor.breakout import (
    detect_breakouts,
    format_breakout,
    load_state,
    save_state,
)
from predictor.data import fetch_klines

DEFAULT_STATE_FILE = ".breakout_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="지지/저항 돌파 감지 및 알림")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=600,
                        help="레벨 계산에 사용할 캔들 수 (기본: 600)")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                        help=f"중복 알림 방지용 상태 파일 (기본: {DEFAULT_STATE_FILE})")
    parser.add_argument("--min-touches", type=int, default=3,
                        help="돌파 알림 대상 레벨의 최소 터치 횟수 (기본: 3)")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 가 필요합니다.", file=sys.stderr)
        return 1

    tg = TelegramClient(args.token)
    state = load_state(args.state_file)
    alerts = 0
    failures = 0

    for symbol in args.symbols:
        key = f"{symbol.upper()}:{args.interval}"
        since_str = state.get(key)
        since = pd.Timestamp(since_str) if since_str else None
        try:
            ohlcv = fetch_klines(symbol, args.interval, args.limit)
            events, last_time = detect_breakouts(
                ohlcv, symbol, args.interval,
                since=since, min_touches=args.min_touches,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{symbol}: 검사 실패 ({exc})", file=sys.stderr)
            continue

        for event in events:
            tg.send_message(args.chat_id, format_breakout(event))
            alerts += 1
            print(f"{symbol}: {event.direction} 돌파 알림 전송 "
                  f"(레벨 {event.level_price:,.4f})")
        if last_time is not None:
            state[key] = str(last_time)
        if not events:
            print(f"{symbol}: 돌파 없음")

    save_state(args.state_file, state)
    print(f"완료: 알림 {alerts}건, 실패 {failures}건")
    return 1 if failures == len(args.symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
