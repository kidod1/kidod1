"""일회성 예측 알림 — 지정한 심볼들을 예측해 텔레그램으로 보내고 종료한다.

GitHub Actions 같은 스케줄 실행 환경에서 사용한다 (상주 프로세스 불필요).

동작 모드:
  - 기본: 모든 심볼의 예측을 각각 전송
  - --min-confidence 0.6: 확률이 60% 이상인 강한 신호만 상세 전송
    (강한 신호가 하나도 없으면 아무것도 보내지 않음)
  - --digest: 개별 메시지 대신 전체 심볼을 한 장의 요약으로 전송

사용 예:
    # 매시간: 강한 신호만 알림
    python notify.py --symbols BTCUSDT ETHUSDT SOLUSDT --min-confidence 0.6

    # 하루 한 번: 전체 요약
    python notify.py --symbols BTCUSDT ETHUSDT SOLUSDT --digest
"""

from __future__ import annotations

import argparse
import os
import sys

from bot import TelegramClient, analyze_symbol, format_prediction_message, scan_symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="예측 결과를 텔레그램으로 1회 전송")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"],
                        help="예측할 심볼 목록 (기본: BTCUSDT)")
    parser.add_argument("--interval", default="1h", help="캔들 간격 (기본: 1h)")
    parser.add_argument("--limit", type=int, default=1500,
                        help="학습에 사용할 캔들 개수 (기본: 1500)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="이 확률 이상 확신하는 신호만 전송 (기본 0.0 = 전부 전송)")
    parser.add_argument("--digest", action="store_true",
                        help="개별 메시지 대신 전체 요약 한 장으로 전송")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    args = parser.parse_args()

    if not args.token or not args.chat_id:
        print("TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 가 필요합니다.", file=sys.stderr)
        return 1

    tg = TelegramClient(args.token)

    if args.digest:
        message = scan_symbols(args.symbols, args.interval,
                               min_confidence=args.min_confidence, limit=args.limit)
        tg.send_message(args.chat_id, message)
        print("요약 전송 완료")
        return 0

    failures = 0
    sent = 0
    for symbol in args.symbols:
        try:
            pred, result = analyze_symbol(symbol, args.interval, args.limit)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{symbol} 예측 실패: {exc}", file=sys.stderr)
            tg.send_message(args.chat_id, f"{symbol} 예측 실패: {exc}")
            continue

        confidence = max(pred.prob_up, pred.prob_down)
        if confidence < args.min_confidence:
            print(f"{symbol}: 확신 {confidence:.1%} < {args.min_confidence:.1%}, 전송 생략")
            continue

        tg.send_message(args.chat_id, format_prediction_message(
            symbol.upper(), args.interval, pred,
            accuracy=result.mean_accuracy, baseline=result.baseline_accuracy,
        ))
        sent += 1
        print(f"{symbol}: 전송 완료 (확신 {confidence:.1%})")

    print(f"완료: {sent}건 전송, {failures}건 실패, "
          f"{len(args.symbols) - sent - failures}건 필터링됨")
    return 1 if failures == len(args.symbols) else 0


if __name__ == "__main__":
    raise SystemExit(main())
