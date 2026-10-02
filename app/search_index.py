"""Postgres + pgvector storage for policy document chunks."""

from __future__ import annotations

from functools import cache

import psycopg
from psycopg import sql

from app import config

# The fields stored with every chunk. These are what an answer cites, so a chunk
# retrieved out of context can still say where it came from and whether it is
# still in force.
FIELDS = [
    "document_id",
    "title",
    "filename",
    "section",
    "category",
    "region",
    "version",
    "status",
    "effective_date",
    "content",
]


# --- Connection -------------------------------------------------------------
@cache
def _connect() -> psycopg.Connection:
    return psycopg.connect(config.DATABASE_URL, autocommit=True)


def connection() -> psycopg.Connection:
    conn = _connect()
    if conn.closed:
        _connect.cache_clear()
        conn = _connect()
    return conn


# --- SQL helpers ------------------------------------------------------------
def _table() -> sql.Identifier:
    return sql.Identifier(config.POLICY_INDEX)


def _field_list() -> sql.Composed:
    return sql.SQL(", ").join(sql.Identifier(name) for name in FIELDS)


def _as_literal(vector) -> str:
    """pgvector wants '[1.0, 2.0]'. str() on a numpy array drops the commas."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return str([float(value) for value in vector])


def _fetch_dicts(statement, params) -> list[dict]:
    with connection().cursor() as cursor:
        cursor.execute(statement, params)
        names = [column.name for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]


# --- Schema -----------------------------------------------------------------
def create_index() -> None:
    columns = sql.SQL(",\n    ").join(
        sql.SQL("{} text").format(sql.Identifier(name)) for name in FIELDS
    )
    with connection().cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(_table()))
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE {table} (
                    chunk_id bigserial PRIMARY KEY,
                    {columns},
                    embedding vector({size})
                )
                """
            ).format(
                table=_table(),
                columns=columns,
                size=sql.Literal(config.EMBEDDING_SIZE),
            )
        )
        cursor.execute(
            sql.SQL(
                "CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)"
            ).format(_table())
        )
        cursor.execute(sql.SQL("CREATE INDEX ON {} (status)").format(_table()))


def count() -> int:
    statement = sql.SQL(
        "SELECT COUNT(*) FROM {} WHERE status = 'current'"
    ).format(_table())
    with connection().cursor() as cursor:
        try:
            cursor.execute(statement)
        except psycopg.errors.UndefinedTable:
            return 0
        return cursor.fetchone()[0]


# --- Retrieval --------------------------------------------------------------
def search(vector, limit: int | None = None) -> list[dict]:
    limit = limit or config.RESULTS_PER_SEARCH
    statement = sql.SQL(
        """
        SELECT {fields}, 1 - (embedding <=> %(vector)s::vector) AS similarity
        FROM {table}
        WHERE status = 'current'
        ORDER BY embedding <=> %(vector)s::vector
        LIMIT %(limit)s
        """
    ).format(fields=_field_list(), table=_table())
    return _fetch_dicts(statement, {"vector": _as_literal(vector), "limit": limit})


def keyword_search(text: str, limit: int | None = None) -> list[dict]:
    limit = limit or config.RESULTS_PER_SEARCH
    statement = sql.SQL(
        """
        SELECT {fields},
               ts_rank(
                   to_tsvector('english', content),
                   plainto_tsquery('english', %(text)s)
               ) AS rank
        FROM {table}
        WHERE status = 'current'
          AND to_tsvector('english', content) @@ plainto_tsquery('english', %(text)s)
        ORDER BY rank DESC
        LIMIT %(limit)s
        """
    ).format(fields=_field_list(), table=_table())
    return _fetch_dicts(statement, {"text": text, "limit": limit})


# --- Writes -----------------------------------------------------------------
def add_chunks(chunks) -> None:
    rows = [
        tuple(chunk.get(field) for field in FIELDS)
        + (_as_literal(chunk["embedding"]),)
        for chunk in chunks
    ]
    if not rows:
        return

    columns = FIELDS + ["embedding"]
    placeholders = sql.SQL(", ").join(
        [sql.Placeholder()] * len(FIELDS)
        + [sql.SQL("{}::vector").format(sql.Placeholder())]
    )
    statement = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({values})").format(
        table=_table(),
        columns=sql.SQL(", ").join(sql.Identifier(name) for name in columns),
        values=placeholders,
    )
    with connection().cursor() as cursor:
        cursor.executemany(statement, rows)