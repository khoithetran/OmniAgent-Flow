from typing import Any, Protocol

from loguru import logger

from src.services.session_service import get_session_history


class AIProvider(Protocol):
    async def generate_response(
        self,
        sender_id: str,
        user_message: str,
        session_history: list[dict[str, Any]],
    ) -> str:
        """Generate an assistant response from a provider implementation."""


class MockAIProvider:
    async def generate_response(
        self,
        sender_id: str,
        user_message: str,
        session_history: list[dict[str, Any]],
    ) -> str:
        normalized_message = user_message.lower()

        if "tu van" in normalized_message or "tư vấn" in normalized_message:
            return (
                "Cảm ơn bạn đã quan tâm. Tôi có thể tư vấn giải pháp phù hợp "
                "dựa trên nhu cầu, quy mô đội ngũ và kênh chăm sóc khách hàng hiện tại."
            )

        if "gia" in normalized_message or "giá" in normalized_message:
            return (
                "Chi phí sẽ phụ thuộc vào số lượng kênh tích hợp, khối lượng hội thoại "
                "và mức độ tự động hóa. Tôi có thể ghi nhận nhu cầu để đội ngũ báo giá chi tiết."
            )

        return (
            "Tôi đã nhận được thông tin của bạn. Đội ngũ hỗ trợ sẽ tiếp tục trao đổi "
            "để nắm rõ nhu cầu và hướng xử lý phù hợp."
        )


ai_provider: AIProvider = MockAIProvider()


async def generate_agent_response(sender_id: str, user_message: str) -> str:
    session_history = await get_session_history(sender_id)
    response = await ai_provider.generate_response(
        sender_id=sender_id,
        user_message=user_message,
        session_history=session_history,
    )
    logger.info(
        "Generated mock assistant response",
        sender_id=sender_id,
        history_size=len(session_history),
    )
    return response
