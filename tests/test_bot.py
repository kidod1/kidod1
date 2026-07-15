"""텔레그램 봇의 메시지 포맷과 명령어 파싱을 네트워크 없이 검증한다.

실행: python tests/test_bot.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from predictor.model import predict_next, prepare_dataset
from tests.test_pipeline import make_synthetic_ohlcv


def test_format_message():
    df = make_synthetic_ohlcv()
    train, latest = prepare_dataset(df)
    pred = predict_next(train, latest)
    msg = bot.format_prediction_message("BTCUSDT", "1h", pred,
                                        accuracy=0.53, baseline=0.51)
    assert "BTCUSDT" in msg
    assert "상승" in msg and "하락" in msg
    assert "53.0%" in msg
    print("test_format_message 통과")
    print("--- 메시지 미리보기 ---")
    print(msg)
    print("----------------------")


def test_handle_command():
    # 도움말 계열
    assert "명령어" in bot.handle_command("/start")
    assert "명령어" in bot.handle_command("/help")
    assert "명령어" in bot.handle_command("안녕")
    # 심볼 누락
    assert "심볼" in bot.handle_command("/predict")

    # /predict 는 run_prediction 으로 위임되는지 (네트워크 호출은 mock)
    with patch.object(bot, "run_prediction", return_value="MOCK_RESULT") as mock:
        assert bot.handle_command("/predict btcusdt 15m") == "MOCK_RESULT"
        mock.assert_called_once_with("BTCUSDT", "15m")
    # 그룹 채팅의 /predict@botname 형태
    with patch.object(bot, "run_prediction", return_value="MOCK_RESULT"):
        assert bot.handle_command("/predict@mybot ETHUSDT") == "MOCK_RESULT"
    print("test_handle_command 통과")


if __name__ == "__main__":
    test_format_message()
    test_handle_command()
    print("\n모든 봇 테스트 통과!")
