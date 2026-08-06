"""통합 서버 — 내 컴퓨터에서 이 하나만 실행하면 전부 동작한다.

주기적으로 다 보내지 않고, **의미 있는 변화가 있을 때만** 알린다:
  1. 실시간 명령 응답 (/predict, /signal, /scan, /report, /levels)
  2. 변화 감지 알림 — 판단/방향 변경, 큰 가격 변동, 반전 경고, 레벨 근접
  3. 돌파/추세 전환 알림 (이벤트 발생 시에만)
  4. 안부 리포트 (기본 6시간마다 — 서버가 살아있는지 확인용, 끌 수 있음)

조용하다는 건 시장에 변화가 없다는 뜻이다. 전체 현황이 궁금하면 언제든
텔레그램에서 /report 를 보내면 즉시 받아볼 수 있다.

사용 예 (Windows):
    set TELEGRAM_BOT_TOKEN=123:abc
    set TELEGRAM_CHAT_ID=987654321
    python serve.py

옵션:
    --check-every 300     변화 감지 주기(초, 기본 300 = 5분)
    --price-move 0.015    가격 변동 알림 임계값 (기본 1.5%)
    --heartbeat-hours 6   안부 리포트 간격(시간, 0이면 끔)
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
    analyze_full,
    format_symbol_block,
    full_report,
    handle_command,
    record_snapshot,
)
from breakout_notify import scan_breakouts_and_trends
from predictor.changes import (
    DEFAULT_STATE_FILE,
    detect_changes,
    format_change_alert,
    load_state,
    save_state,
    snapshot_state,
)
from predictor.data import fetch_klines
from predictor.scorecard import DEFAULT_LOG_FILE, load_log, resolve_pending, save_log


def check_changes(tg, chat_id: str, symbols: list[str], interval: str,
                  price_move: float, state_file: str,
                  log_path: str = DEFAULT_LOG_FILE) -> int:
    """모든 심볼을 분석해 변화가 있는 것만 알린다. 보낸 알림 수를 반환한다."""
    state = load_state(state_file)
    log = load_log(log_path)
    ohlcv_cache = {}
    sent = 0

    try:
        btc_ohlcv = fetch_klines("BTCUSDT", interval, 1500)
    except Exception:  # noqa: BLE001
        btc_ohlcv = None

    for symbol in symbols:
        try:
            snap = analyze_full(symbol, interval, 1500, btc_ohlcv)
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: 분석 실패 ({exc})", file=sys.stderr)
            continue

        ohlcv_cache[snap.symbol] = snap.ohlcv
        record_snapshot(log, snap)

        cur = snapshot_state(snap)
        reasons = detect_changes(state.get(snap.symbol), cur, price_move)
        if reasons:
            tg.send_message(chat_id,
                            format_change_alert(snap, reasons,
                                                format_symbol_block(snap)))
            sent += 1
            print(f"  {snap.symbol}: 변화 알림 전송 ({len(reasons)}건) — {reasons[0]}")
            state[snap.symbol] = cur  # 알린 시점의 상태만 기준으로 갱신
        elif snap.symbol not in state:
            state[snap.symbol] = cur  # 첫 관측: 기준선만 기록
            print(f"  {snap.symbol}: 기준 상태 기록 (첫 관측)")
        else:
            print(f"  {snap.symbol}: 변화 없음")

    resolve_pending(log, ohlcv_cache)
    save_log(log_path, log)
    save_state(state_file, state)
    return sent


def main() -> int:
    parser = argparse.ArgumentParser(description="바이낸스 알림 통합 서버")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_WATCHLIST,
                        help="감시할 코인 목록 (기본: BTC/ETH/SOL/XRP/BNB)")
    parser.add_argument("--interval", default="1h", help="캔들 간격 (기본: 1h)")
    parser.add_argument("--check-every", type=int, default=300,
                        help="변화 감지 주기(초, 기본 300=5분). 0이면 끔")
    parser.add_argument("--price-move", type=float, default=0.015,
                        help="가격 변동 알림 임계값 (기본 0.015 = 1.5%%)")
    parser.add_argument("--heartbeat-hours", type=float, default=6.0,
                        help="안부 리포트 간격(시간, 기본 6). 0이면 끔")
    parser.add_argument("--alert-every", type=int, default=300,
                        help="돌파/추세 감시 주기(초, 기본 300=5분). 0이면 감시 끔")
    parser.add_argument("--state-file", default=DEFAULT_STATE_FILE,
                        help="변화 감지 상태 파일")
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 가 필요합니다.\n"
              "Windows: set TELEGRAM_BOT_TOKEN=봇토큰  /  set TELEGRAM_CHAT_ID=챗아이디",
              file=sys.stderr)
        return 1

    tg = TelegramClient(args.token)
    offset: int | None = None
    now = time.time()
    next_check = now if args.check_every > 0 else None
    next_alert = now if args.alert_every > 0 else None
    heartbeat_secs = int(args.heartbeat_hours * 3600)
    # 첫 안부 리포트는 시작 직후가 아니라 한 주기 뒤에 (시작 메시지와 중복 방지)
    next_heartbeat = now + heartbeat_secs if heartbeat_secs > 0 else None

    print("통합 서버가 시작되었습니다. (Ctrl+C 로 종료)")
    print(f"  감시 코인   : {' '.join(args.symbols)}")
    print(f"  변화 감지   : {'끔' if next_check is None else f'{args.check_every}초마다'}")
    print(f"  돌파/추세   : {'끔' if next_alert is None else f'{args.alert_every}초마다'}")
    print(f"  안부 리포트 : {'끔' if next_heartbeat is None else f'{args.heartbeat_hours}시간마다'}")
    tg.send_message(
        args.chat_id,
        "🟢 통합 서버 시작됨 (변화 감지 모드)\n"
        f"감시: {', '.join(args.symbols)}\n\n"
        "이제 주기적으로 다 보내지 않고, 아래 변화가 있을 때만 알립니다:\n"
        "  • 매수/매도 판단 변경\n"
        "  • 방향 예측 전환 (중립↔상승/하락)\n"
        f"  • 가격 {args.price_move:.1%} 이상 변동\n"
        "  • 반전 주의 신호 발생\n"
        "  • 지지/저항선 근접\n"
        "  • 돌파 / 추세 전환\n\n"
        "조용하면 변화가 없다는 뜻입니다. 전체 현황은 /report 로 언제든 확인하세요.\n\n"
        + HELP_TEXT)

    while True:
        try:
            # 1) 변화 감지 — 달라진 심볼만 알림
            if next_check is not None and time.time() >= next_check:
                print("[변화감지] 검사 중...")
                try:
                    n = check_changes(tg, args.chat_id, args.symbols, args.interval,
                                      args.price_move, args.state_file)
                    print(f"[변화감지] 완료 (알림 {n}건)")
                except Exception as exc:  # noqa: BLE001
                    print(f"[변화감지] 실패: {exc}", file=sys.stderr)
                next_check = time.time() + args.check_every

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

            # 3) 안부 리포트 — 변화가 없어도 서버가 살아있음을 알림
            if next_heartbeat is not None and time.time() >= next_heartbeat:
                print("[안부] 리포트 전송 중...")
                try:
                    tg.send_message(args.chat_id,
                                    full_report(args.symbols, args.interval))
                    print("[안부] 전송 완료")
                except Exception as exc:  # noqa: BLE001
                    print(f"[안부] 실패: {exc}", file=sys.stderr)
                next_heartbeat = time.time() + heartbeat_secs

            # 4) 실시간 명령 처리 (long polling)
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
