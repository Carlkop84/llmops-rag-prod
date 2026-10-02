"""Ingest: validate the manifest, chunk the documents, embed, and index."""

from __future__ import annotations

import json
from pathlib import Path

from app import chunking, config, documents, embeddings, search_index

# Metadata copied from the manifest entry onto every chunk of that document.
CARRIED_FIELDS = [
    "document_id",
    "title",
    "category",
    "region",
    "version",
    "status",
    "effective_date",
]

# Fields each chunk supplies itself rather than inheriting from the manifest.
CHUNK_FIELDS = ["filename", "section", "content"]

REQUIRED_ENTRY_KEYS = {"filename", "title", "document_id", "status"}
VALID_STATUSES = ("current", "superseded")


class IngestError(Exception):
    """Raised when the manifest or the documents are not fit to index."""


def _check_field_coverage() -> None:
    """The chunk dicts must carry exactly the columns the index stores."""
    expected = set(search_index.FIELDS)
    actual = set(CARRIED_FIELDS) | set(CHUNK_FIELDS)
    if actual != expected:
        raise IngestError(
            "chunk fields do not match search_index.FIELDS — "
            f"missing {sorted(expected - actual)}, "
            f"unexpected {sorted(actual - expected)}"
        )


# --- Manifest ---------------------------------------------------------------
def load_manifest(documents_dir) -> list[dict]:
    documents_dir = Path(documents_dir)
    manifest_path = documents_dir / "manifest.json"

    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IngestError(f"no manifest.json in {documents_dir}") from None
    except json.JSONDecodeError as error:
        raise IngestError(f"manifest.json is not valid JSON: {error}") from None

    if not isinstance(entries, list):
        raise IngestError("manifest.json should contain a list of entries")

    problems: list[str] = []
    listed: set[str] = set()

    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            problems.append(
                f"entry {position}: expected an object, found {type(entry).__name__}"
            )
            continue

        filename = entry.get("filename")
        label = filename or f"entry {position}"

        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            problems.append(f"{label}: manifest entry is missing {sorted(missing)}")

        status = entry.get("status")
        if status not in VALID_STATUSES:
            problems.append(
                f"{label}: status is {status!r}, expected current or superseded"
            )

        if filename:
            if filename in listed:
                problems.append(f"{filename}: listed in the manifest more than once")
            listed.add(filename)
            if not (documents_dir / filename).exists():
                problems.append(
                    f"{filename}: listed in the manifest but not in the folder"
                )

    on_disk = {
        item.name
        for item in documents_dir.iterdir()
        if item.is_file()
        and item.name != "manifest.json"
        and not item.name.startswith(".")
    }
    for orphan in sorted(on_disk - listed):
        problems.append(
            f"{orphan}: in the folder but not in the manifest, so never searchable"
        )

    if problems:
        raise IngestError(
            "manifest.json does not match documents:\n  " + "\n  ".join(problems)
        )
    return entries


# --- Chunking ---------------------------------------------------------------
def build_chunks(documents_dir, entries) -> list[dict]:
    documents_dir = Path(documents_dir)
    chunks: list[dict] = []
    empty: list[str] = []

    for entry in entries:
        path = documents_dir / entry["filename"]
        sections = documents.read_document(path)
        produced = 0

        for piece in chunking.chunk_sections(sections):
            chunk = {field: entry.get(field, "") for field in CARRIED_FIELDS}
            chunk["filename"] = entry["filename"]
            chunk["section"] = piece["section"]
            chunk["content"] = piece["content"]
            chunks.append(chunk)
            produced += 1

        if produced == 0:
            empty.append(entry["filename"])

    if empty:
        raise IngestError(
            "these documents produced no chunks, so nothing in them is "
            "searchable:\n  " + "\n  ".join(sorted(empty))
        )
    return chunks


def embedding_text(chunk) -> str:
    header = " - ".join(
        part for part in (chunk.get("title"), chunk.get("section")) if part
    )
    return f"{header}\n{chunk['content']}" if header else chunk["content"]


# --- Pipeline ---------------------------------------------------------------
def ingest(documents_dir) -> dict:
    _check_field_coverage()
    entries = load_manifest(documents_dir)
    chunks = build_chunks(documents_dir, entries)

    vectors = embeddings.embed([embedding_text(chunk) for chunk in chunks])
    if len(vectors) != len(chunks):
        raise IngestError(
            f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
        )
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    # Only now, with every chunk embedded, is it safe to drop the live index.
    search_index.create_index()
    search_index.add_chunks(chunks)
    return {"documents": len(entries), "chunks": len(chunks)}


if __name__ == "__main__":
    try:
        summary = ingest(config.DOCUMENTS_DIR)
    except IngestError as error:
        raise SystemExit(str(error))
    print(
        f"Indexed {summary['chunks']} chunks "
        f"from {summary['documents']} documents."
    )