"""통합 서버 — 내 컴퓨터에서 이 하나만 실행하면 전부 동작한다.

한 프로세스에서 세 가지를 동시에 처리한다:
  1. 실시간 명령 응답 (/predict, /signal, /scan, /report, /levels)
  2. 주기적 종합 리포트 전송 (예측 + 지지/저항 + 판단 + 성적표)
  3. 주기적 돌파/추세 전환 감시 (이벤트 발생 시에만 알림)

GitHub Actions와 달리 정확한 간격으로 실행되고, 컴퓨터가 켜져 있는 한 24시간 돈다.

사용 예 (Windows):
    set TELEGRAM_BOT_TOKEN=123:abc
    set TELEGRAM_CHAT_ID=987654321
    python serve.py

옵션:
    --report-every 300   종합 리포트 주기(초, 기본 300 = 5분)
    --alert-every 300    돌파/추세 감시 주기(초, 기본 300 = 5분)
    --symbols BTCUSDT ETHUSDT ...   감시할 코인 (기본 5종)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from bot import (
    DEFAULT_WATCHLIST,
    HELP_TEXT,
    TelegramClient,
    full_report,
    handle_command,
)
from breakout_notify import scan_breakouts_and_trends


def main() -> int:
    parser = argparse.ArgumentParser(description="바이낸스 알림 통합 서버")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_WATCHLIST,
                        help="감시할 코인 목록 (기본: BTC/ETH/SOL/XRP/BNB)")
    parser.add_argument("--interval", default="1h", help="캔들 간격 (기본: 1h)")
    parser.add_argument("--report-every", type=int, default=300,
                        help="종합 리포트 주기(초, 기본 300=5분). 0이면 리포트 끔")
    parser.add_argument("--alert-every", type=int, default=300,
                        help="돌파/추세 감시 주기(초, 기본 300=5분). 0이면 감시 끔")
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 가 필요합니다.\n"
              "Windows: set TELEGRAM_BOT_TOKEN=봇토큰  /  set TELEGRAM_CHAT_ID=챗아이디",
              file=sys.stderr)
        return 1

    tg = TelegramClient(args.token)
    offset: int | None = None
    now = time.time()
    next_report = now if args.report_every > 0 else None
    next_alert = now if args.alert_every > 0 else None

    print("통합 서버가 시작되었습니다. (Ctrl+C 로 종료)")
    print(f"  감시 코인 : {' '.join(args.symbols)}")
    print(f"  리포트    : {'끔' if next_report is None else f'{args.report_every}초마다'}")
    print(f"  돌파/추세 : {'끔' if next_alert is None else f'{args.alert_every}초마다'}")
    tg.send_message(args.chat_id,
                    "🟢 통합 서버 시작됨\n"
                    f"감시: {', '.join(args.symbols)}\n"
                    f"리포트 {args.report_every}초 · 돌파/추세 {args.alert_every}초 간격\n\n"
                    + HELP_TEXT)

    while True:
        try:
            # 1) 주기적 종합 리포트
            if next_report is not None and time.time() >= next_report:
                print("[리포트] 생성 중...")
                try:
                    tg.send_message(args.chat_id,
                                    full_report(args.symbols, args.interval))
                    print("[리포트] 전송 완료")
                except Exception as exc:  # noqa: BLE001
                    print(f"[리포트] 실패: {exc}", file=sys.stderr)
                next_report = time.time() + args.report_every

            # 2) 주기적 돌파/추세 전환 감시
            if next_alert is not None and time.time() >= next_alert:
                print("[감시] 돌파/추세 검사 중...")
                try:
                    n, f = scan_breakouts_and_trends(
                        tg, args.chat_id, args.symbols, args.interval)
                    print(f"[감시] 완료 (알림 {n}건, 실패 {f}건)")
                except Exception as exc:  # noqa: BLE001
                    print(f"[감시] 실패: {exc}", file=sys.stderr)
                next_alert = time.time() + args.alert_every

            # 3) 실시간 명령 처리 (long polling)
            for update in tg.get_updates(offset, timeout=15):
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
                        f"안녕하세요! 이 채팅의 chat id는 {chat_id} 입니다.\n\n"
                        + HELP_TEXT,
                    )
                    continue
                if text.strip().lower().startswith(
                        ("/predict", "/scan", "/signal", "/report")):
                    tg.send_message(chat_id, "분석 중입니다... (10~30초 소요) ⏳")
                tg.send_message(chat_id, handle_command(text))

        except KeyboardInterrupt:
            print("\n서버를 종료합니다.")
            return 0
        except Exception:  # noqa: BLE001 — 네트워크 오류 등은 잠시 후 재시도
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
