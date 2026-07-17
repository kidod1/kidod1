"""상승/하락 확률 예측 모델.

Gradient Boosting 분류기를 사용하며, 시계열 특성을 지키기 위해
학습/검증 분할은 항상 시간 순서를 유지한다(과거로 학습, 미래로 검증).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, HORIZON, build_features


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
    horizon: int = HORIZON  # 향후 몇 개 캔들의 방향인지

    @property
    def direction(self) -> str:
        return "상승" if self.prob_up >= 0.5 else "하락"


def _make_model() -> VotingClassifier:
    """트리 부스팅 + 선형 모델의 소프트 보팅 앙상블.

    두 모델은 실수 패턴이 달라(비선형 상호작용 vs 선형 추세) 확률을 평균하면
    단일 모델보다 분산이 줄어든다.
    """
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        l2_regularization=1.0,
        min_samples_leaf=30,
        random_state=42,
    )
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=1000),
    )
    return VotingClassifier([("hgb", hgb), ("logit", logit)], voting="soft")


def _fit_calibrator(X: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    """뒤쪽 15% 검증 구간으로 Platt scaling 보정기를 학습한다.

    검증 구간에 두 클래스가 모두 없으면 보정을 건너뛴다(None 반환).
    """
    split = int(len(X) * 0.85)
    if split < 100 or len(np.unique(y[split:])) < 2:
        return None
    base = _make_model()
    base.fit(X[:split], y[:split])
    p_val = base.predict_proba(X[split:])[:, 1]
    calibrator = LogisticRegression()
    calibrator.fit(p_val.reshape(-1, 1), y[split:])
    return calibrator


def prepare_dataset(
    ohlcv: pd.DataFrame,
    btc_ohlcv: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """피처를 만들고 (학습용 데이터, 예측용 마지막 행)으로 나눈다."""
    featured = build_features(ohlcv, btc_df=btc_ohlcv, horizon=horizon)
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


def walk_forward_probabilities(train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """워크포워드 방식으로 아웃오브샘플 예측 확률을 만든다.

    각 폴드마다 과거 데이터로만 학습하고 미래 구간의 확률을 예측하므로,
    반환된 확률은 실전에서 그 시점에 실제로 얻을 수 있었던 값과 같은 조건이다.
    매매 전략 백테스트의 입력으로 사용한다.
    """
    X = train[FEATURE_COLUMNS].to_numpy()
    y = train["target"].to_numpy()

    parts: list[pd.DataFrame] = []
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X):
        model = _make_model()
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        fold = train.iloc[test_idx][["open_time", "close", "forward_return", "target"]].copy()
        fold["prob_up"] = proba
        parts.append(fold)
    return pd.concat(parts, ignore_index=True)


def predict_next(train: pd.DataFrame, latest: pd.DataFrame,
                 horizon: int = HORIZON) -> Prediction:
    """전체 데이터로 학습한 뒤 향후 horizon캔들의 방향 확률을 예측한다.

    검증 구간으로 학습한 Platt scaling 보정기가 있으면 확률을 보정해
    과신(overconfidence)을 줄인다.
    """
    X = train[FEATURE_COLUMNS].to_numpy()
    y = train["target"].to_numpy()
    calibrator = _fit_calibrator(X, y)
    model = _make_model()
    model.fit(X, y)
    proba_up = float(model.predict_proba(latest[FEATURE_COLUMNS].to_numpy())[0, 1])
    if calibrator is not None:
        proba_up = float(calibrator.predict_proba([[proba_up]])[0, 1])
    return Prediction(
        prob_up=proba_up,
        prob_down=1 - proba_up,
        last_close=float(latest["close"].iloc[0]),
        last_open_time=latest["open_time"].iloc[0],
        horizon=horizon,
    )
