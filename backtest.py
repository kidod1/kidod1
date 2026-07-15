"""매매 전략 백테스트 CLI — 예측대로 매매했다면 수익이 났을지 확인한다.

사용 예:
    python backtest.py --symbol BTCUSDT --interval 1h --limit 3000
    python backtest.py --symbol ETHUSDT --interval 4h --allow-short --fee 0.00075
"""

from __future__ import annotations

import argparse
import sys

from predictor.data import fetch_klines
from predictor.model import prepare_dataset, walk_forward_probabilities
from predictor.strategy import format_report, sweep_thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="예측 확률 기반 매매 전략 백테스트")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=3000,
                        help="사용할 캔들 개수 (기본: 3000)")
    parser.add_argument("--fee", type=float, default=0.001,
                        help="포지션 변경 1회당 수수료 비율 (기본: 0.001 = 0.1%%)")
    parser.add_argument("--allow-short", action="store_true",
                        help="하락 신호에서 숏 진입 허용")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.50, 0.55, 0.60, 0.65],
                        help="비교할 진입 임계값 목록")
    args = parser.parse_args()

    print(f"[1/3] {args.symbol} {args.interval} 캔들 {args.limit}개 수집 중...")
    try:
        ohlcv = fetch_klines(args.symbol, args.interval, args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"데이터 수집 실패: {exc}", file=sys.stderr)
        return 1

    print("[2/3] 워크포워드 아웃오브샘플 확률 생성 중 (몇십 초 걸릴 수 있음)...")
    train, _ = prepare_dataset(ohlcv)
    probs = walk_forward_probabilities(train)

    print("[3/3] 전략 시뮬레이션...")
    results = sweep_thresholds(
        probs, tuple(args.thresholds), fee=args.fee, allow_short=args.allow_short
    )
    print()
    print(format_report(results, args.symbol.upper(), args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
