#!/usr/bin/env bash
# 우분투/데비안 계열 서버에 봇을 설치하고 systemd 서비스로 등록하는 스크립트.
#
# 서버에서 실행:
#   git clone https://github.com/kidod1/kidod1.git ~/binance-predictor
#   cd ~/binance-predictor/deploy
#   TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_CHAT_ID=987654321 bash setup.sh
#
# 옵션 (환경변수):
#   WATCH_SYMBOLS  자동 알림 심볼 목록 (기본: "BTCUSDT ETHUSDT")
#   WATCH_EVERY    자동 알림 주기 초 (기본: 3600)

set -euo pipefail

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "TELEGRAM_BOT_TOKEN 과 TELEGRAM_CHAT_ID 환경변수를 지정하고 실행하세요." >&2
    echo "예: TELEGRAM_BOT_TOKEN=123:abc TELEGRAM_CHAT_ID=987 bash setup.sh" >&2
    exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WATCH_SYMBOLS="${WATCH_SYMBOLS:-BTCUSDT ETHUSDT}"
WATCH_EVERY="${WATCH_EVERY:-3600}"

echo "[1/4] 시스템 패키지 설치..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip

echo "[2/4] 가상환경 및 의존성 설치..."
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "[3/4] 환경변수 파일 생성 ($APP_DIR/.env)..."
cat > "$APP_DIR/.env" <<EOF
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID
EOF
chmod 600 "$APP_DIR/.env"

echo "[4/4] systemd 서비스 등록..."
sudo tee /etc/systemd/system/binance-bot.service > /dev/null <<EOF
[Unit]
Description=Binance up/down prediction Telegram bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python bot.py --watch $WATCH_SYMBOLS --every $WATCH_EVERY
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now binance-bot

echo
echo "설치 완료! 상태 확인:"
echo "  sudo systemctl status binance-bot"
echo "  journalctl -u binance-bot -f     # 실시간 로그"
echo
echo "텔레그램에서 봇에게 /predict BTCUSDT 를 보내보세요."
