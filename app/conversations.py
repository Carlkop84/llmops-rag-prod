"""Chat history storage: conversations and their turns."""

from __future__ import annotations

import json

from psycopg import sql

from app.search_index import connection

CHATS = "chats"
MESSAGES = "chat_messages"

ROLES = ("user", "assistant")
MAX_TITLE = 60


def _table(name: str) -> sql.Identifier:
    return sql.Identifier(name)


def _fetch_dicts(statement, params=None) -> list[dict]:
    with connection().cursor() as cursor:
        cursor.execute(statement, params)
        names = [column.name for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


def _loads(raw):
    """Tolerate a NULL, a bad row, or a jsonb column that arrives pre-parsed."""
    if raw is None:
        return []
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


# --- Schema -----------------------------------------------------------------
def create_tables() -> None:
    with connection().cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {chats} (
                    chat_id bigserial PRIMARY KEY,
                    title text NOT NULL,
                    started_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(chats=_table(CHATS))
        )
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {messages} (
                    message_id bigserial PRIMARY KEY,
                    chat_id bigint NOT NULL
                        REFERENCES {chats}(chat_id) ON DELETE CASCADE,
                    role text NOT NULL CHECK (role IN ('user', 'assistant')),
                    content text NOT NULL,
                    sources text NOT NULL DEFAULT '[]',
                    created_at timestamptz NOT NULL DEFAULT now()
                )
                """
            ).format(messages=_table(MESSAGES), chats=_table(CHATS))
        )

        # Every read of a conversation is "all the messages in this chat, in order".
        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {messages} (chat_id, message_id)"
            ).format(
                name=sql.Identifier(f"{MESSAGES}_chat"),
                messages=_table(MESSAGES),
            )
        )
        # And every sidebar render is "the chats, most recently touched first".
        cursor.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {chats} (updated_at DESC)"
            ).format(
                name=sql.Identifier(f"{CHATS}_updated"),
                chats=_table(CHATS),
            )
        )


# --- Writes -----------------------------------------------------------------
def start_chat(question: str) -> int:
    # The chat's title is just the first question, flattened and trimmed so long
    # ones don't wreck the sidebar layout.
    title = " ".join(question.split()) or "New chat"
    if len(title) > MAX_TITLE:
        title = title[: MAX_TITLE - 3].rstrip() + "..."

    with connection().cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "INSERT INTO {chats} (title) VALUES (%s) RETURNING chat_id"
            ).format(chats=_table(CHATS)),
            (title,),
        )
        return cursor.fetchone()[0]


def add_message(chat_id: int, role: str, content: str, sources=None) -> None:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, not {role!r}")

    # One statement, so a stored message can never be left with a stale
    # updated_at and sink to the bottom of the sidebar.
    statement = sql.SQL(
        """
        WITH inserted AS (
            INSERT INTO {messages} (chat_id, role, content, sources)
            VALUES (%s, %s, %s, %s)
            RETURNING chat_id
        )
        UPDATE {chats} SET updated_at = now()
        WHERE chat_id = (SELECT chat_id FROM inserted)
        """
    ).format(messages=_table(MESSAGES), chats=_table(CHATS))

    with connection().cursor() as cursor:
        cursor.execute(
            statement, (chat_id, role, content, json.dumps(sources or []))
        )


def delete_chat(chat_id: int) -> bool:
    with connection().cursor() as cursor:
        cursor.execute(
            sql.SQL("DELETE FROM {chats} WHERE chat_id = %s").format(
                chats=_table(CHATS)
            ),
            (chat_id,),
        )
        return cursor.rowcount > 0


# --- Reads ------------------------------------------------------------------
def list_chats() -> list[dict]:
    return _fetch_dicts(
        sql.SQL(
            "SELECT chat_id, title FROM {chats} ORDER BY updated_at DESC"
        ).format(chats=_table(CHATS))
    )


def get_messages(chat_id: int) -> list[dict]:
    # Returns the turns in the shape main.py keeps in st.session_state.messages,
    # so a reopened chat redraws exactly like a live one.
    rows = _fetch_dicts(
        sql.SQL(
            """
            SELECT role, content, sources
            FROM {messages}
            WHERE chat_id = %s
            ORDER BY message_id
            """
        ).format(messages=_table(MESSAGES)),
        (chat_id,),
    )
    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": _loads(row["sources"]),
        }
        for row in rows
    ]