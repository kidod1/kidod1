# 서버 구축 가이드

봇을 24시간 돌리는 방법들을 정리했습니다. 변화 감지 알림 + 실시간 명령까지
전부 쓰려면 **방법 0(내 컴퓨터에서 통합 서버)** 이 가장 확실합니다.

## 방법 0: 내 컴퓨터에서 통합 서버 실행 (Windows) ⭐

`serve.py` 하나로 전부 동작합니다. 주기적으로 다 보내지 않고
**의미 있는 변화가 있을 때만** 알립니다:

- 매수/매도 판단 변경 (관망 → 매수 등)
- 방향 예측 전환 (중립 ↔ 상승/하락)
- 가격이 마지막 알림 대비 1.5% 이상 변동
- 반전 주의 신호 발생, 지지/저항선 근접
- 돌파 / 추세 전환

조용하다는 건 시장에 변화가 없다는 뜻입니다. 전체 현황은 텔레그램에서
`/report` 를 보내면 언제든 즉시 받아볼 수 있고, 기본 6시간마다 안부
리포트가 와서 서버가 살아있는지도 확인됩니다.

**준비 (처음 한 번):**

1. https://www.python.org/downloads/ 에서 파이썬 설치
   (설치 화면에서 **"Add Python to PATH"** 체크 필수)
2. 이 저장소를 내려받습니다. Git이 있으면:
   ```
   git clone https://github.com/kidod1/kidod1.git
   ```
   Git이 없으면 GitHub 저장소 페이지 → **Code → Download ZIP** → 압축 해제
3. 명령 프롬프트(cmd)를 열고 폴더로 이동한 뒤 라이브러리 설치:
   ```
   cd 내려받은_폴더_경로
   pip install -r requirements.txt
   ```

**실행 (간편 방법):**

`deploy\run_windows.bat` 파일을 메모장으로 열어 봇 토큰과 chat id 두 줄을
본인 값으로 바꾸고 저장한 뒤, 그 파일을 더블클릭하면 됩니다.

**실행 (직접 명령):**

```
set TELEGRAM_BOT_TOKEN=여기에_봇토큰
set TELEGRAM_CHAT_ID=여기에_챗아이디
python serve.py
```

`🟢 통합 서버 시작됨` 메시지가 텔레그램으로 오면 성공입니다. 이 창을 켜둔 동안
5분마다 변화를 검사해 달라진 것이 있을 때만 알리고, 명령에는 실시간으로 답합니다.

**옵션:**

```
python serve.py --price-move 0.03        가격 알림을 3% 이상 변동일 때만
python serve.py --check-every 900        변화 감지를 15분마다 (기본 300=5분)
python serve.py --heartbeat-hours 0      안부 리포트 끄기 (완전히 조용하게)
python serve.py --heartbeat-hours 24     안부 리포트를 하루 한 번만
python serve.py --symbols BTCUSDT SOLUSDT   원하는 코인만 감시
```

알림이 너무 잦으면 `--price-move` 를 올리고(예: 0.03), 너무 뜸하면
내리세요(예: 0.01).

**24시간 자동 실행 (컴퓨터 켤 때마다 자동 시작):**

Windows **작업 스케줄러**로 등록하면 창을 안 띄우고도 백그라운드로 돕니다.
1. 시작 메뉴 → "작업 스케줄러" 실행 → **작업 만들기**
2. 일반 탭: 이름 입력, **"사용자가 로그온할 때만 실행"**, "가장 높은 권한으로 실행" 체크
3. 트리거 탭 → 새로 만들기 → **"로그온할 때"**
4. 동작 탭 → 새로 만들기 → 프로그램: `run_windows.bat` 의 전체 경로 지정
5. 확인 → 이제 PC를 켜면 서버가 자동으로 시작됩니다

> 참고: 이 방법을 쓰면 GitHub Actions 알림과 **중복**됩니다. 저장소 Actions 탭 →
> Crypto alerts → `···` → **Disable workflow** 로 GitHub 쪽을 꺼두세요.
> (돌파 상태 파일은 각자 로컬에서 관리되므로 서로 간섭하지 않습니다.)

---

## 클라우드에서 돌리는 무료 방법 (컴퓨터를 끄고 싶을 때)

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
