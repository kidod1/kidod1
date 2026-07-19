"""지지/저항 돌파 + 상위 시간대 추세 전환 감지 후 텔레그램 알림 — 1회 실행 후 종료.

GitHub Actions 스케줄 실행용. 마지막으로 검사한 캔들 시각과 추세 상태를
상태 파일에 기록해두므로, 실행 간격이 불규칙해도 같은 이벤트를 두 번 알리지 않는다.

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
from predictor.trendshift import check_trend_shift, format_trend_shift

DEFAULT_STATE_FILE = ".breakout_state.json"
TREND_INTERVALS = ("4h", "1d")  # 추세 전환을 감시할 상위 시간대


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

        # 상위 시간대(4h/1d) 추세 전환 감시
        for itv in TREND_INTERVALS:
            trend_key = f"trend:{symbol.upper()}:{itv}"
            try:
                htf = fetch_klines(symbol, itv, 200)
            except Exception as exc:  # noqa: BLE001
                print(f"{symbol} {itv}: 추세 확인 실패 ({exc})", file=sys.stderr)
                continue
            shift, new_trend = check_trend_shift(
                htf, symbol, itv, state.get(trend_key))
            if shift is not None:
                tg.send_message(args.chat_id, format_trend_shift(shift))
                alerts += 1
                print(f"{symbol} {itv}: 추세 전환 알림 전송 "
                      f"({shift.old_trend}→{shift.new_trend})")
            if new_trend is not None:
                state[trend_key] = new_trend

    save_state(args.state_file, state)
    print(f"완료: 알림 {alerts}건, 실패 {failures}건")
    return 1 if failures == len(args.symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
