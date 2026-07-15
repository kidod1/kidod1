"""상승/하락 확률 예측 모델.

Gradient Boosting 분류기를 사용하며, 시계열 특성을 지키기 위해
학습/검증 분할은 항상 시간 순서를 유지한다(과거로 학습, 미래로 검증).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS, build_features


@dataclass
class BacktestResult:
    """시계열 교차검증 결과."""
    fold_accuracies: list[float] = field(default_factory=list)
    fold_brier_scores: list[float] = field(default_factory=list)
    baseline_accuracy: float = 0.0  # 항상 다수 클래스로만 찍었을 때의 정확도

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_accuracies)) if self.fold_accuracies else 0.0

    @property
    def mean_brier(self) -> float:
        return float(np.mean(self.fold_brier_scores)) if self.fold_brier_scores else 0.0


@dataclass
class Prediction:
    """최신 캔들 기준 예측 결과."""
    prob_up: float
    prob_down: float
    last_close: float
    last_open_time: pd.Timestamp

    @property
    def direction(self) -> str:
        return "상승" if self.prob_up >= 0.5 else "하락"


def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_samples_leaf=30,
        random_state=42,
    )


def prepare_dataset(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """피처를 만들고 (학습용 데이터, 예측용 마지막 행)으로 나눈다."""
    featured = build_features(ohlcv)
    # 지표 계산 초기 구간(NaN)이 있는 행 제거하되, 마지막 행(예측 대상)은 유지
    train = featured.iloc[:-1].dropna(subset=FEATURE_COLUMNS + ["target"])
    latest = featured.iloc[[-1]]
    if latest[FEATURE_COLUMNS].isna().any(axis=None):
        raise ValueError("데이터가 부족해 최신 캔들의 지표를 계산할 수 없습니다. limit을 늘려주세요.")
    return train, latest


def backtest(train: pd.DataFrame, n_splits: int = 5) -> BacktestResult:
    """TimeSeriesSplit으로 과거→미래 방향 교차검증을 수행한다."""
    X = train[FEATURE_COLUMNS].to_numpy()
    y = train["target"].to_numpy()

    result = BacktestResult()
    majority = float(max(y.mean(), 1 - y.mean()))
    result.baseline_accuracy = majority

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X):
        model = _make_model()
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(float)
        result.fold_accuracies.append(accuracy_score(y[test_idx], pred))
        result.fold_brier_scores.append(brier_score_loss(y[test_idx], proba))
    return result


def predict_next(train: pd.DataFrame, latest: pd.DataFrame) -> Prediction:
    """전체 데이터로 학습한 뒤 최신 캔들의 다음 방향 확률을 예측한다."""
    model = _make_model()
    model.fit(train[FEATURE_COLUMNS].to_numpy(), train["target"].to_numpy())
    proba_up = float(model.predict_proba(latest[FEATURE_COLUMNS].to_numpy())[0, 1])
    return Prediction(
        prob_up=proba_up,
        prob_down=1 - proba_up,
        last_close=float(latest["close"].iloc[0]),
        last_open_time=latest["open_time"].iloc[0],
    )
