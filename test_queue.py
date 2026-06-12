import asyncio
from typing import Any

import requests

from src.database import close_redis, get_redis_client
from src.services.session_service import get_session_history


WEBHOOK_URL = "http://localhost:8000/api/webhook"
SENDER_ID = "user_123"
SESSION_KEY = f"session:{SENDER_ID}"


def build_facebook_payload() -> dict[str, Any]:
    return {
        "object": "page",
        "entry": [
            {
                "id": "page_123",
                "time": 1710000000000,
                "messaging": [
                    {
                        "sender": {"id": SENDER_ID},
                        "recipient": {"id": "page_123"},
                        "timestamp": 1710000000000,
                        "message": {
                            "mid": "mid.test_message",
                            "text": "T\u00f4i mu\u1ed1n t\u01b0 v\u1ea5n",
                        },
                    }
                ],
            }
        ],
    }


async def poll_session_history(timeout_seconds: int = 10) -> list[dict[str, Any]]:
    for _ in range(timeout_seconds):
        history = await get_session_history(SENDER_ID)
        roles = {message.get("role") for message in history}
        if {"user", "assistant"}.issubset(roles):
            return history
        await asyncio.sleep(1)

    return []


async def reset_test_session() -> None:
    redis_client = await get_redis_client()
    await redis_client.delete(SESSION_KEY)


async def main() -> None:
    await reset_test_session()

    payload = build_facebook_payload()
    response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
    response.raise_for_status()

    print("Webhook response:", response.json())

    try:
        history = await poll_session_history()
        redis_client = await get_redis_client()
        ttl = await redis_client.ttl(SESSION_KEY)

        if history:
            print("Session history:", history)
            print(f"TTL for {SESSION_KEY}:", ttl)
        else:
            print("User and assistant messages were not written within 10 seconds.")
            print(f"TTL for {SESSION_KEY}:", ttl)
    finally:
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
