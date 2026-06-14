import json
from typing import Any

from loguru import logger

from src.database import connect_postgres


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id BIGSERIAL PRIMARY KEY,
        channel TEXT NOT NULL,
        external_sender_id TEXT NOT NULL,
        last_intent TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE (channel, external_sender_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        intent TEXT,
        action TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_created
    ON conversation_messages (conversation_id, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_messages_intent
    ON conversation_messages (intent)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_sender
    ON conversations (external_sender_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS hubspot_lead_syncs (
        id BIGSERIAL PRIMARY KEY,
        conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        hubspot_contact_id TEXT,
        status TEXT NOT NULL,
        action TEXT,
        reason TEXT,
        intent TEXT,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_hubspot_lead_syncs_conversation_created
    ON hubspot_lead_syncs (conversation_id, created_at)
    """,
)


async def init_conversation_schema() -> None:
    pool = await connect_postgres()

    async with pool.acquire() as connection:
        for statement in SCHEMA_STATEMENTS:
            await connection.execute(statement)

    logger.info("PostgreSQL conversation schema is ready")


async def upsert_conversation(
    sender_id: str,
    channel: str = "facebook",
    last_intent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    pool = await connect_postgres()
    metadata_payload = json.dumps(metadata or {}, ensure_ascii=False)

    async with pool.acquire() as connection:
        conversation_id = await connection.fetchval(
            """
            INSERT INTO conversations (
                channel,
                external_sender_id,
                last_intent,
                metadata
            )
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (channel, external_sender_id)
            DO UPDATE SET
                last_intent = COALESCE(EXCLUDED.last_intent, conversations.last_intent),
                metadata = conversations.metadata || EXCLUDED.metadata,
                updated_at = NOW()
            RETURNING id
            """,
            channel,
            sender_id,
            last_intent,
            metadata_payload,
        )

    if not isinstance(conversation_id, int):
        raise RuntimeError("PostgreSQL did not return a conversation id")

    logger.info(
        "Upserted conversation",
        conversation_id=conversation_id,
        channel=channel,
        sender_id=sender_id,
        last_intent=last_intent,
    )
    return conversation_id


async def save_conversation_message(
    sender_id: str,
    role: str,
    content: str,
    channel: str = "facebook",
    intent: str | None = None,
    action: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    if role not in {"user", "assistant", "system"}:
        raise ValueError("role must be one of: user, assistant, system")

    conversation_id = await upsert_conversation(
        sender_id=sender_id,
        channel=channel,
        last_intent=intent,
        metadata=metadata,
    )
    pool = await connect_postgres()
    metadata_payload = json.dumps(metadata or {}, ensure_ascii=False)

    async with pool.acquire() as connection:
        message_id = await connection.fetchval(
            """
            INSERT INTO conversation_messages (
                conversation_id,
                role,
                content,
                intent,
                action,
                metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            RETURNING id
            """,
            conversation_id,
            role,
            content,
            intent,
            action,
            metadata_payload,
        )

    if not isinstance(message_id, int):
        raise RuntimeError("PostgreSQL did not return a message id")

    logger.info(
        "Saved conversation message to PostgreSQL",
        conversation_id=conversation_id,
        message_id=message_id,
        sender_id=sender_id,
        role=role,
        intent=intent,
    )
    return message_id


async def save_hubspot_sync_event(
    sender_id: str,
    status: str,
    channel: str = "facebook",
    hubspot_contact_id: str | None = None,
    action: str | None = None,
    reason: str | None = None,
    intent: str | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    conversation_id = await upsert_conversation(
        sender_id=sender_id,
        channel=channel,
        last_intent=intent,
    )
    pool = await connect_postgres()
    payload_json = json.dumps(payload or {}, ensure_ascii=False)

    async with pool.acquire() as connection:
        sync_id = await connection.fetchval(
            """
            INSERT INTO hubspot_lead_syncs (
                conversation_id,
                hubspot_contact_id,
                status,
                action,
                reason,
                intent,
                payload
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            RETURNING id
            """,
            conversation_id,
            hubspot_contact_id,
            status,
            action,
            reason,
            intent,
            payload_json,
        )

    if not isinstance(sync_id, int):
        raise RuntimeError("PostgreSQL did not return a HubSpot sync id")

    logger.info(
        "Saved HubSpot sync event to PostgreSQL",
        sync_id=sync_id,
        conversation_id=conversation_id,
        sender_id=sender_id,
        status=status,
        hubspot_contact_id=hubspot_contact_id,
    )
    return sync_id


async def get_conversation_messages(
    sender_id: str,
    channel: str = "facebook",
    limit: int = 50,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be greater than 0")

    pool = await connect_postgres()

    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT
                cm.id,
                cm.role,
                cm.content,
                cm.intent,
                cm.action,
                cm.metadata::text AS metadata,
                cm.created_at
            FROM conversation_messages cm
            INNER JOIN conversations c ON c.id = cm.conversation_id
            WHERE c.channel = $1 AND c.external_sender_id = $2
            ORDER BY cm.created_at DESC, cm.id DESC
            LIMIT $3
            """,
            channel,
            sender_id,
            limit,
        )

    messages: list[dict[str, Any]] = []
    for row in reversed(rows):
        raw_metadata = row["metadata"]
        metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else {}
        messages.append(
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "intent": row["intent"],
                "action": row["action"],
                "metadata": metadata,
                "created_at": row["created_at"].isoformat(),
            }
        )

    return messages
