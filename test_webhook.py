import os
from typing import Any

import requests


BASE_URL = "http://localhost:8000/api/webhook"
VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN", "change-me")


def test_verify_webhook() -> None:
    params: dict[str, str] = {
        "hub.mode": "subscribe",
        "hub.verify_token": VERIFY_TOKEN,
        "hub.challenge": "TEST_XAC_THUC_OK",
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    print(f"GET verify status={response.status_code} body={response.text}")


def test_receive_webhook() -> None:
    payload: dict[str, Any] = {
        "object": "page",
        "entry": [
            {
                "id": "page_001",
                "time": 1790787600,
                "messaging": [
                    {
                        "sender": {"id": "user_123"},
                        "recipient": {"id": "page_001"},
                        "timestamp": 1790787600,
                        "message": {
                            "mid": "mid_test_user_123",
                            "text": "Tôi muốn tư vấn",
                        },
                    }
                ],
            }
        ],
    }
    response = requests.post(BASE_URL, json=payload, timeout=10)
    print(f"POST receive status={response.status_code} body={response.text}")


def main() -> None:
    try:
        test_verify_webhook()
        test_receive_webhook()
    except requests.exceptions.ConnectionError:
        print("Khong the ket noi FastAPI. Hay chay: uvicorn src.main:app --reload")
    except requests.exceptions.RequestException as exc:
        print(f"Loi khi gui request toi webhook: {exc}")


if __name__ == "__main__":
    main()
