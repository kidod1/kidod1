# 무료 클라우드 서버 구축 가이드

봇을 24시간 돌리는 두 가지 무료 방법을 정리했습니다.

## 방법 1: GitHub Actions (서버 불필요, 가장 간단) ✅

이 저장소에 이미 워크플로(`.github/workflows/predict.yml`)가 포함되어 있습니다.
매시간 BTCUSDT, ETHUSDT 예측을 텔레그램으로 보내줍니다.

**설정 (5분 소요):**

1. 텔레그램 @BotFather 에게 `/newbot` → 봇 토큰 발급
2. 만든 봇에게 아무 메시지나 보낸 뒤, PC에서 잠깐 `python bot.py` 를 실행하고
   봇에게 `/start` 를 보내면 chat id를 알려줍니다.
3. GitHub 저장소 → **Settings → Secrets and variables → Actions → New repository secret**
   - `TELEGRAM_BOT_TOKEN` = 봇 토큰
   - `TELEGRAM_CHAT_ID` = chat id
4. 끝. 매시 5분에 자동 실행됩니다. **Actions 탭 → Hourly prediction → Run workflow**
   로 즉시 테스트할 수 있습니다.

- 장점: 완전 무료, 서버 관리 불필요, 컴퓨터 꺼놔도 동작
- 한계: 정해진 시간에 알림만 옴. 텔레그램에서 `/predict` 로 실시간 조회는 불가
  (그건 상주 프로세스가 필요 → 방법 2)
- 참고: 스케줄은 몇 분 정도 지연될 수 있고, 저장소가 60일간 커밋이 없으면
  GitHub가 스케줄을 자동 비활성화합니다(메일로 알려줌, 버튼 한 번으로 재활성화).

## 방법 2: Oracle Cloud 무료 서버 (실시간 대화형 봇까지)

Oracle Cloud "Always Free" 티어는 기간 제한 없이 무료인 VM을 제공합니다
(ARM 4 OCPU / 24GB RAM까지 — 이 봇에는 차고 넘칩니다).

**계정/서버 생성 (본인 인증이 필요해 직접 하셔야 합니다):**

1. https://www.oracle.com/kr/cloud/free/ 에서 가입 (카드 인증 필요, 과금은 안 됨.
   홈 리전은 서울/춘천 선택 가능)
2. 콘솔에서 **Compute → Instances → Create instance**
   - Image: **Ubuntu 22.04** (또는 24.04)
   - Shape: **Ampere A1.Flex** (Always Free 표시 확인, 1 OCPU / 6GB면 충분)
   - SSH 공개키 등록 후 생성
3. 생성된 인스턴스의 공인 IP로 접속: `ssh ubuntu@<공인IP>`

**봇 설치 (서버 접속 후 3줄):**

```bash
git clone https://github.com/kidod1/kidod1.git ~/binance-predictor
cd ~/binance-predictor/deploy
TELEGRAM_BOT_TOKEN=봇토큰 TELEGRAM_CHAT_ID=챗아이디 bash setup.sh
```

`setup.sh` 가 파이썬 설치, 의존성 설치, systemd 서비스 등록까지 자동으로 처리합니다.
서버가 재부팅돼도 봇이 자동으로 다시 시작됩니다.

**운영 명령어:**

```bash
sudo systemctl status binance-bot      # 상태 확인
journalctl -u binance-bot -f           # 실시간 로그
sudo systemctl restart binance-bot     # 재시작
git -C ~/binance-predictor pull && sudo systemctl restart binance-bot  # 업데이트
```

자동 알림 심볼/주기를 바꾸려면:

```bash
WATCH_SYMBOLS="BTCUSDT SOLUSDT" WATCH_EVERY=1800 \
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... bash setup.sh   # 재실행하면 갱신됨
```

**참고:** 서버 리전이 미국이면 `api.binance.com` 이 지역 차단(HTTP 451)될 수 있습니다.
그 경우 `.env` 파일에 `BINANCE_BASE_URL=https://data-api.binance.vision` 한 줄을
추가하고 재시작하세요. 서울 리전이면 문제없습니다.

## 어떤 방법을 고를까?

| | 방법 1: GitHub Actions | 방법 2: Oracle Cloud VM |
|---|---|---|
| 비용 | 무료 | 무료 (Always Free) |
| 준비 시간 | ~5분 | ~30분 (가입 포함) |
| 정기 알림 | ✅ | ✅ |
| `/predict` 실시간 조회 | ❌ | ✅ |
| 관리 부담 | 없음 | 거의 없음 (systemd 자동 재시작) |

먼저 방법 1로 시작하고, 실시간 조회가 필요해지면 방법 2로 넘어가는 것을 추천합니다.
