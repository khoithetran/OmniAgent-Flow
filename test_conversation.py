from src.services.conversation_service import SCHEMA_STATEMENTS


def test_conversation_schema_contains_required_tables() -> None:
    schema_sql = "\n".join(SCHEMA_STATEMENTS)

    assert "CREATE TABLE IF NOT EXISTS conversations" in schema_sql
    assert "CREATE TABLE IF NOT EXISTS conversation_messages" in schema_sql
    assert "CREATE TABLE IF NOT EXISTS hubspot_lead_syncs" in schema_sql
    assert "metadata JSONB" in schema_sql
    assert "last_intent TEXT" in schema_sql


if __name__ == "__main__":
    test_conversation_schema_contains_required_tables()
