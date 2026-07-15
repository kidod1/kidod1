"""바이낸스 상승/하락 확률 예측 CLI.

사용 예:
    python main.py --symbol BTCUSDT --interval 1h --limit 2000
    python main.py --symbol ETHUSDT --interval 15m --no-backtest
"""

from __future__ import annotations

import argparse
import sys

from predictor.data import fetch_klines
from predictor.features import build_features
from predictor.levels import find_levels, format_levels
from predictor.model import backtest, predict_next, prepare_dataset
from predictor.signals import advise, format_advice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="바이낸스 캔들 기반 상승/하락 확률 예측")
    parser.add_argument("--symbol", default="BTCUSDT", help="거래쌍 (기본: BTCUSDT)")
    parser.add_argument("--interval", default="1h",
                        help="캔들 간격: 1m 5m 15m 1h 4h 1d 등 (기본: 1h)")
    parser.add_argument("--limit", type=int, default=2000,
                        help="학습에 사용할 캔들 개수 (기본: 2000)")
    parser.add_argument("--no-backtest", action="store_true",
                        help="백테스트를 건너뛰고 예측만 수행")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[1/3] 바이낸스에서 {args.symbol} {args.interval} 캔들 {args.limit}개 요청 중...")
    try:
        ohlcv = fetch_klines(args.symbol, args.interval, args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"데이터 수집 실패: {exc}", file=sys.stderr)
        return 1
    print(f"      {len(ohlcv)}개 수신 "
          f"({ohlcv['open_time'].iloc[0]} ~ {ohlcv['open_time'].iloc[-1]})")

    print("[2/3] 기술적 지표 계산 및 데이터셋 구성...")
    train, latest = prepare_dataset(ohlcv)
    print(f"      학습 샘플 {len(train)}개 "
          f"(상승 비율 {train['target'].mean():.1%})")

    if not args.no_backtest:
        print("[3/3] 시계열 교차검증(백테스트) 실행 중...")
        result = backtest(train)
        print(f"      평균 정확도    : {result.mean_accuracy:.1%} "
              f"(폴드별: {', '.join(f'{a:.1%}' for a in result.fold_accuracies)})")
        print(f"      다수클래스 기준: {result.baseline_accuracy:.1%} "
              f"(모델이 이 값보다 높아야 의미가 있음)")
        print(f"      Brier score    : {result.mean_brier:.4f} (낮을수록 확률이 잘 보정됨)")
    else:
        print("[3/3] 백테스트 생략")

    pred = predict_next(train, latest)

    print()
    print("=" * 52)
    print(f"  {args.symbol} — 다음 {args.interval} 캔들 예측")
    print("=" * 52)
    print(f"  기준 캔들 시각 : {pred.last_open_time} (UTC)")
    print(f"  현재 종가      : {pred.last_close:,.4f}")
    print(f"  상승 확률      : {pred.prob_up:.1%}")
    print(f"  하락 확률      : {pred.prob_down:.1%}")
    print(f"  판단           : {pred.direction} 우세")
    print("=" * 52)

    # 지지/저항선 + 타이밍 판단 (상위 시간대 추세 포함)
    from predictor.signals import higher_timeframes, trend_direction
    htf_trends = {}
    for itv in higher_timeframes(args.interval):
        try:
            htf_trends[itv] = trend_direction(fetch_klines(args.symbol, itv, 200))
        except Exception:  # noqa: BLE001
            pass
    supports, resistances = find_levels(ohlcv)
    featured = build_features(ohlcv)
    advice = advise(featured, pred, supports, resistances, htf_trends=htf_trends)
    print()
    print(format_levels(supports, resistances, pred.last_close))
    print()
    print(format_advice(advice))
    print()
    print("※ 참고용 통계 모델입니다. 투자 손실에 대한 책임은")
    print("  본인에게 있으며, 재무적 조언이 아닙니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
