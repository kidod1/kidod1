"""상승/하락 확률 예측 모델.

Gradient Boosting 분류기를 사용하며, 시계열 특성을 지키기 위해
학습/검증 분할은 항상 시간 순서를 유지한다(과거로 학습, 미래로 검증).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, HORIZON, MIN_MOVE, build_features

# 최근 데이터 가중 학습: 이 개수만큼 과거로 갈 때마다 가중치가 절반이 됨
RECENCY_HALF_LIFE = 500

# 이 확신 미만의 예측은 "중립"으로 취급 (방향 판단 유보, 성적표 제외)
CONFIDENT_THRESHOLD = 0.55


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

    @property
    def confidence(self) -> float:
        return max(self.prob_up, self.prob_down)

    @property
    def is_confident(self) -> bool:
        """확신이 중립 기준 이상인지 — 미만이면 방향 판단을 유보해야 한다."""
        return self.confidence >= CONFIDENT_THRESHOLD


class SoftVoteModel:
    """트리 부스팅 + 선형 모델의 소프트 보팅 앙상블 (샘플 가중치 지원).

    두 모델은 실수 패턴이 달라(비선형 상호작용 vs 선형 추세) 확률을 평균하면
    단일 모델보다 분산이 줄어든다. sklearn VotingClassifier는 Pipeline에
    sample_weight를 전달하지 못해 직접 구현한다.
    """

    def __init__(self):
        self.hgb = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=30,
            random_state=42,
        )
        self.logit = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=1000),
        )

    def fit(self, X, y, sample_weight=None):
        self.hgb.fit(X, y, sample_weight=sample_weight)
        self.logit.fit(X, y, logisticregression__sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return (self.hgb.predict_proba(X) + self.logit.predict_proba(X)) / 2


def _make_model() -> SoftVoteModel:
    return SoftVoteModel()


def _recency_weights(n: int, half_life: int = RECENCY_HALF_LIFE) -> np.ndarray:
    """최근 표본일수록 큰 지수 가중치 — 현재 시장 국면에 빨리 적응하게 한다."""
    age = np.arange(n - 1, -1, -1)  # 마지막 표본의 나이 = 0
    return 0.5 ** (age / half_life)


def _fit_calibrator(X: np.ndarray, y: np.ndarray) -> LogisticRegression | None:
    """뒤쪽 15% 검증 구간으로 Platt scaling 보정기를 학습한다.

    검증 구간에 두 클래스가 모두 없으면 보정을 건너뛴다(None 반환).
    """
    split = int(len(X) * 0.85)
    if split < 100 or len(np.unique(y[split:])) < 2:
        return None
    base = _make_model()
    base.fit(X[:split], y[:split], sample_weight=_recency_weights(split))
    p_val = base.predict_proba(X[split:])[:, 1]
    calibrator = LogisticRegression()
    calibrator.fit(p_val.reshape(-1, 1), y[split:])
    return calibrator


def drop_open_candle(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """아직 닫히지 않은 마지막 캔들을 제거한다.

    바이낸스 klines는 진행 중인 캔들을 마지막에 포함한다. 그 캔들은 거래량이
    부분적으로만 쌓였고 종가·고저가도 확정되지 않아, 완성된 캔들로만 학습한
    모델에게는 학습 분포 밖의 입력이 된다(특히 volume_ratio가 크게 왜곡됨).
    예측 입력을 마지막 '닫힌' 캔들로 맞춰 학습과 추론 조건을 일치시킨다.
    """
    if "close_time" not in ohlcv.columns or ohlcv.empty:
        return ohlcv
    now = pd.Timestamp.now(tz="UTC")
    close_time = ohlcv["close_time"]
    if close_time.dt.tz is None:
        close_time = close_time.dt.tz_localize("UTC")
    return ohlcv[close_time <= now]


def prepare_dataset(
    ohlcv: pd.DataFrame,
    btc_ohlcv: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """피처를 만들고 (학습용 데이터, 예측용 마지막 행)으로 나눈다.

    진행 중인 캔들은 제외하므로, 예측은 항상 마지막으로 '닫힌' 캔들 기준이다.
    """
    ohlcv = drop_open_candle(ohlcv)
    if btc_ohlcv is not None:
        btc_ohlcv = drop_open_candle(btc_ohlcv)
    featured = build_features(ohlcv, btc_df=btc_ohlcv, horizon=horizon)
    # 지표 계산 초기 구간(NaN)이 있는 행 제거하되, 마지막 행(예측 대상)은 유지
    train = featured.iloc[:-1].dropna(subset=FEATURE_COLUMNS + ["target"])
    # 사실상 무방향인 미세 변동 라벨은 노이즈이므로 학습에서 제외
    train = train[train["forward_return_h"].abs() >= MIN_MOVE]
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
        model.fit(X[train_idx], y[train_idx],
                  sample_weight=_recency_weights(len(train_idx)))
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
        model.fit(X[train_idx], y[train_idx],
                  sample_weight=_recency_weights(len(train_idx)))
        proba = model.predict_proba(X[test_idx])[:, 1]
        fold = train.iloc[test_idx][["open_time", "close", "forward_return", "target"]].copy()
        fold["prob_up"] = proba
        parts.append(fold)
    return pd.concat(parts, ignore_index=True)


@dataclass
class DirectionStats:
    """한 방향(상승/하락) 예측의 아웃오브샘플 검증 성적."""
    hit_rate: float
    samples: int
    baseline: float  # 같은 구간에서 그 방향을 항상 찍었을 때의 적중률

    @property
    def edge(self) -> float:
        """기준선 대비 우위 (양수여야 예측할 가치가 있음)."""
        return self.hit_rate - self.baseline


def oos_confidence_stats(
    train: pd.DataFrame,
    min_conf: float = CONFIDENT_THRESHOLD,
    n_splits: int = 3,
    last_n: int = 300,
) -> tuple[dict[str, DirectionStats], float]:
    """워크포워드 아웃오브샘플에서 방향별 검증 성적과 시장 상승 비율을 낸다.

    절대 적중률만 보면 함정에 빠진다: 검증 구간의 시장이 상승 편향이면
    "항상 상승"만 해도 높은 적중률이 나오므로, 그 기준선을 넘지 못하는 예측은
    정보가 없는 것이다. 그래서 상승/하락 각각에 대해 '그 방향을 항상 찍었을
    때의 적중률(기준선)'과 비교한 우위(edge)를 함께 계산한다.

    Returns:
        ({"상승": DirectionStats, "하락": ...}, 검증 구간의 시장 상승 비율)
        — 표본이 없는 방향은 딕셔너리에서 빠진다.
    """
    probs = walk_forward_probabilities(train, n_splits=n_splits).tail(last_n)
    probs = probs.dropna(subset=["target"])
    if probs.empty:
        return {}, 0.5

    up_rate = float((probs["target"] == 1.0).mean())  # 시장 상승 비율
    out: dict[str, DirectionStats] = {}
    for name, mask, baseline in (
        ("상승", probs["prob_up"] >= min_conf, up_rate),
        ("하락", probs["prob_up"] <= 1 - min_conf, 1 - up_rate),
    ):
        subset = probs[mask]
        if subset.empty:
            continue
        hit = float(((subset["prob_up"] >= 0.5) == (subset["target"] == 1.0)).mean())
        out[name] = DirectionStats(hit, len(subset), baseline)
    return out, up_rate


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
    model.fit(X, y, sample_weight=_recency_weights(len(X)))
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
