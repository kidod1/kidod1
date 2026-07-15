# Binance 상승/하락 확률 예측기

바이낸스 공개 API에서 캔들(K-line) 차트 데이터를 가져와, 기술적 지표를 기반으로
**다음 캔들이 오를지 내릴지 확률**을 예측하는 프로그램입니다. API 키가 필요 없습니다.

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
- `/help` — 도움말

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
   RSI, MACD, 볼린저밴드 위치, 변동성, 캔들 모양, 거래량 비율 등 20개 지표를 계산합니다.
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
