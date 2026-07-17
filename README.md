# Binance 상승/하락 확률 예측기

바이낸스 공개 API에서 캔들(K-line) 차트 데이터를 가져와, 기술적 지표를 기반으로
**향후 4캔들(1시간봉 기준 4시간) 방향**을 확률로 예측하는 프로그램입니다.
API 키가 필요 없습니다. 단일 캔들 대신 여러 캔들 누적 방향을 예측해 노이즈를 줄였고,
알트코인 예측에는 비트코인 동향 피처가 함께 사용됩니다.

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

```bash
# 기본: BTCUSDT 1시간봉 2000개로 학습 + 백테스트 + 예측
python main.py

# 다른 코인 / 캔들 간격
python main.py --symbol ETHUSDT --interval 15m --limit 3000

# 백테스트 생략하고 빠르게 예측만
python main.py --no-backtest
```

### 출력에 포함되는 것

- 다음 캔들 상승/하락 확률과 백테스트 정확도
- **지지선/저항선**: 스윙 고점·저점을 클러스터링해 터치 횟수(강도)와 현재가 대비 거리 표시
- **매수/매도 타이밍 판단**: 모델 확률 + RSI/스토캐스틱RSI/MFI + MACD 교차 + 볼린저밴드
  + ADX 추세 국면 + 일목균형표 구름 + 상위 시간대(4h/1d) 추세 일치 + 지지/저항 근접을
  점수로 합산해 5단계(강한 매수~강한 매도)와 근거 목록 제시
- **ATR 손절/목표가 제안**: 매수/매도 판단일 때 평균 변동폭 기준 참고 가격 표시
- **예측 성적표**: 리포트마다 예측을 기록(`.predictions_log.json`)해뒀다가 호라이즌이
  지나면 실제 결과와 대조 — 리포트 하단에 실전 누적 적중률이 표시됩니다

## 매매 전략 백테스트

"예측대로 매매했다면 수익이 났을까?"를 워크포워드(항상 과거로만 학습) 방식으로
시뮬레이션합니다. 수수료를 반영하고, 진입 임계값별로 누적 수익률·승률·최대낙폭·샤프를
단순 보유(buy & hold)와 비교합니다.

```bash
python backtest.py --symbol BTCUSDT --interval 1h --limit 3000

# 숏 포지션 허용, 수수료 조정, 임계값 직접 지정
python backtest.py --symbol ETHUSDT --allow-short --fee 0.00075 --thresholds 0.55 0.6 0.65
```

### 출력 예시

```
====================================================
  BTCUSDT — 다음 1h 캔들 예측
====================================================
  기준 캔들 시각 : 2026-07-15 00:00:00+00:00 (UTC)
  현재 종가      : 112,345.6700
  상승 확률      : 56.3%
  하락 확률      : 43.7%
  판단           : 상승 우세
====================================================
```

## 텔레그램 봇 (휴대폰에서 받아보기)

이 프로그램은 PC/서버에서 실행되지만, 텔레그램 봇을 통해 휴대폰으로 결과를 받아볼 수 있습니다.

1. 텔레그램에서 **@BotFather** 에게 `/newbot` 을 보내 봇을 만들고 토큰을 받습니다.
2. 봇을 실행합니다:

```bash
# 대화형 모드: 텔레그램에서 명령을 보내 조회
TELEGRAM_BOT_TOKEN=123456:ABC... python bot.py
```

3. 텔레그램에서 자기 봇에게 `/start` 를 보내면 chat id를 알려줍니다.
4. 자동 예측 알림을 받으려면:

```bash
# 1시간마다 BTCUSDT, ETHUSDT 예측을 자동 전송
TELEGRAM_BOT_TOKEN=123456:ABC... TELEGRAM_CHAT_ID=987654321 \
    python bot.py --watch BTCUSDT ETHUSDT --every 3600
```

**봇 명령어:**
- `/predict BTCUSDT` — 1시간봉 기준 예측
- `/predict ETHUSDT 15m` — 캔들 간격 지정
- `/scan` — 기본 5개 코인(BTC/ETH/SOL/XRP/BNB) 일괄 스캔 요약
- `/scan BTCUSDT SOLUSDT` — 원하는 코인만 스캔
- `/signal BTCUSDT` — 종합 타이밍 판단 (예측 + 지지/저항 + 매수·매도 신호)
- `/levels BTCUSDT` — 지지선/저항선만 빠르게 조회
- `/report` — 전체 코인 종합 리포트 한 장 (예측+지지/저항+판단 요약)
- `/help` — 도움말

자동 알림 모드 두 가지:

```bash
# 5분마다 종합 리포트 한 장 (예측+지지/저항+판단)
python bot.py --watch BTCUSDT ETHUSDT --every 300 --report

# 확률 60% 이상인 강한 신호만 개별 알림 (그 외에는 조용함)
python bot.py --watch BTCUSDT ETHUSDT --every 3600 --min-confidence 0.6
```

GitHub Actions 워크플로는 **지지/저항 돌파 알림** 방식으로 동작합니다:
5개 코인을 감시하다가 캔들 종가가 레벨을 돌파하면 방향·목표가·재이탈 주의선을
알려줍니다. 마지막 검사 캔들을 상태 파일(`.breakout_state.json`)에 기록하므로
실행 간격이 불규칙해도 돌파를 놓치거나 중복 알림하지 않습니다.

돌파 없이 조용할 때 정기 리포트도 받고 싶으면 서버에서 `--report` 모드를
사용하세요 (GitHub 스케줄은 5분 간격을 보장하지 못합니다 — deploy/DEPLOY.md 참고).

## 24시간 무료로 돌리기 (클라우드)

- **GitHub Actions (추천 시작점)** — 저장소에 포함된 `.github/workflows/predict.yml` 이
  매시간 예측을 텔레그램으로 보내줍니다. 저장소 Secrets에 `TELEGRAM_BOT_TOKEN` 과
  `TELEGRAM_CHAT_ID` 만 등록하면 끝. 서버가 필요 없습니다.
- **Oracle Cloud 무료 VM** — 실시간 `/predict` 조회까지 하려면 무료 서버에
  `deploy/setup.sh` 한 번 실행으로 설치됩니다.

자세한 단계는 [deploy/DEPLOY.md](deploy/DEPLOY.md) 를 참고하세요.

## 동작 원리

1. **데이터 수집** (`predictor/data.py`) — `/api/v3/klines` 공개 엔드포인트에서
   OHLCV 캔들을 가져옵니다. 1000개 초과 시 자동으로 나눠서 요청합니다.
2. **피처 엔지니어링** (`predictor/features.py`) — 수익률, 이동평균 이격도,
   RSI, MACD, 볼린저밴드, ATR, ADX, MFI, OBV, 스토캐스틱RSI, 일목균형표,
   변동성 국면, 캔들 모양, 거래량 비율, 시간대 주기성, BTC 동향(알트코인용)
   등 43개 지표를 계산합니다. 타깃은 "향후 4캔들 뒤 종가 방향"입니다.
3. **모델** (`predictor/model.py`) — scikit-learn의 `HistGradientBoostingClassifier`로
   "다음 캔들 종가가 현재보다 높은가"를 이진 분류하고, `predict_proba`로 확률을 출력합니다.
4. **백테스트** — `TimeSeriesSplit`으로 항상 과거 데이터로 학습하고 미래 데이터로
   검증합니다(look-ahead bias 방지). 다수 클래스 기준선(baseline)과 비교해
   모델이 실제로 정보를 담고 있는지 확인할 수 있습니다.

## 테스트

네트워크 없이 합성 데이터로 전체 파이프라인을 검증할 수 있습니다:

```bash
python tests/test_pipeline.py
```

## 주의사항

- 암호화폐 단기 가격은 예측이 매우 어렵습니다. 백테스트 정확도가 다수 클래스
  기준선과 비슷하다면(대략 50% 초반) 그 구간에서는 모델에 예측력이 거의 없다는 뜻입니다.
- 이 프로그램은 **학습·참고용**이며 재무적 조언이 아닙니다. 실제 투자 판단과
  손실에 대한 책임은 전적으로 사용자에게 있습니다.
