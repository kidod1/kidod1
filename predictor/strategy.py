"""매매 전략 백테스트 — 예측 확률대로 매매했다면 수익이 났을지 시뮬레이션한다.

규칙:
  - 상승 확률 >= threshold      → 다음 캔들 동안 롱(매수) 보유
  - 하락 확률 >= threshold      → 숏(allow_short=True 일 때만), 아니면 현금
  - 그 외                        → 현금 (포지션 없음)
포지션이 바뀔 때마다 수수료(fee)를 차감한다. 진입/청산 각각 1회씩 부과되므로
롱→현금 전환은 fee 1회, 롱→숏 전환은 fee 2회에 해당한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyResult:
    """전략 시뮬레이션 결과 지표 모음."""
    threshold: float
    allow_short: bool
    fee: float
    n_candles: int
    n_trades: int            # 포지션 변경 횟수
    time_in_market: float    # 포지션 보유 시간 비율
    total_return: float      # 전략 누적 수익률
    buy_hold_return: float   # 같은 기간 단순 보유 수익률
    win_rate: float          # 포지션 보유 캔들 중 수익 캔들 비율
    max_drawdown: float      # 최대 낙폭 (음수)
    sharpe: float            # 캔들 단위 샤프 비율 (연환산 아님)


def simulate(
    probs: pd.DataFrame,
    threshold: float = 0.55,
    fee: float = 0.001,
    allow_short: bool = False,
) -> StrategyResult:
    """워크포워드 확률(prob_up)과 다음 캔들 수익률(forward_return)로 전략을 시뮬레이션한다.

    Args:
        probs: walk_forward_probabilities()의 반환값
               (prob_up, forward_return 컬럼 필요)
        threshold: 진입에 필요한 최소 확률 (0.5 = 항상 진입)
        fee: 포지션 변경 1회당 수수료 비율 (0.001 = 0.1%)
        allow_short: 하락 신호에서 숏 진입 허용 여부
    """
    df = probs.dropna(subset=["prob_up", "forward_return"]).reset_index(drop=True)
    prob_up = df["prob_up"].to_numpy()
    fwd = df["forward_return"].to_numpy()

    position = np.zeros(len(df))
    position[prob_up >= threshold] = 1.0
    if allow_short:
        position[(1 - prob_up) >= threshold] = -1.0

    # 포지션 변경량에 비례해 수수료 부과 (0→1 은 1회, 1→-1 은 2회분)
    change = np.abs(np.diff(position, prepend=0.0))
    strat_returns = position * fwd - change * fee

    equity = np.cumprod(1 + strat_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1

    in_market = position != 0
    held_returns = strat_returns[in_market]
    std = strat_returns.std()

    return StrategyResult(
        threshold=threshold,
        allow_short=allow_short,
        fee=fee,
        n_candles=len(df),
        n_trades=int((change > 0).sum()),
        time_in_market=float(in_market.mean()),
        total_return=float(equity[-1] - 1) if len(equity) else 0.0,
        buy_hold_return=float(np.prod(1 + fwd) - 1),
        win_rate=float((held_returns > 0).mean()) if len(held_returns) else 0.0,
        max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
        sharpe=float(strat_returns.mean() / std) if std > 0 else 0.0,
    )


def sweep_thresholds(
    probs: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65),
    fee: float = 0.001,
    allow_short: bool = False,
) -> list[StrategyResult]:
    """여러 임계값에 대해 전략을 시뮬레이션해 비교 목록을 반환한다."""
    return [simulate(probs, t, fee, allow_short) for t in thresholds]


def format_report(results: list[StrategyResult], symbol: str, interval: str) -> str:
    """전략 결과 목록을 사람이 읽기 좋은 표 문자열로 만든다."""
    if not results:
        return "결과 없음"
    header = results[0]
    lines = [
        f"{symbol} {interval} 매매 전략 백테스트 "
        f"(캔들 {header.n_candles}개, 수수료 {header.fee:.2%}/회, "
        f"{'롱+숏' if header.allow_short else '롱 전용'})",
        "",
        f"{'임계값':>6} {'누적수익':>9} {'단순보유':>9} {'승률':>7} "
        f"{'거래수':>6} {'보유비율':>8} {'최대낙폭':>9} {'샤프':>7}",
    ]
    for r in results:
        lines.append(
            f"{r.threshold:>6.2f} {r.total_return:>9.1%} {r.buy_hold_return:>9.1%} "
            f"{r.win_rate:>7.1%} {r.n_trades:>6d} {r.time_in_market:>8.1%} "
            f"{r.max_drawdown:>9.1%} {r.sharpe:>7.3f}"
        )
    lines += [
        "",
        "※ 워크포워드 아웃오브샘플 확률 기준. 과거 성과가 미래를 보장하지 않으며,",
        "  슬리피지·호가 스프레드는 반영되지 않았습니다.",
    ]
    return "\n".join(lines)
