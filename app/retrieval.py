"""Retrieval: find the policy excerpts that answer a question, and format them
with the provenance an answer needs in order to cite them."""

from __future__ import annotations

from app import config, embeddings, search_index


def search_policies(question: str) -> list[dict]:
    question_vector = embeddings.embed([question])[0]
    chunks = search_index.search(question_vector)

    if chunks:
        # Results come back nearest first from both stores, but sort anyway
        # rather than trusting it.
        chunks.sort(key=lambda chunk: chunk["similarity"], reverse=True)
        best = chunks[0]["similarity"]

        if best >= config.MIN_BEST_SIMILARITY:
            cutoff = max(
                config.MIN_BEST_SIMILARITY, best - config.MAX_GAP_FROM_BEST
            )
            return [chunk for chunk in chunks if chunk["similarity"] >= cutoff]

    # The vectors found nothing convincing. Before giving up, try matching the
    # words themselves: an acronym the documents never spell out ("pto",
    # "401k") is invisible to an embedding model and obvious to keyword search.
    keyword_hits = search_index.keyword_search(question)
    for hit in keyword_hits:
        hit.setdefault("similarity", None)
    return keyword_hits[: config.RESULTS_PER_SEARCH]


def cite(chunks) -> list[dict]:
    return [
        {
            "number": number,
            "title": chunk.get("title"),
            "section": chunk.get("section"),
            "filename": chunk.get("filename"),
            "version": chunk.get("version"),
            "effective_date": chunk.get("effective_date"),
            "region": chunk.get("region"),
        }
        for number, chunk in enumerate(chunks, start=1)
    ]


def _header(number: int, chunk) -> str:
    header = f"[{number}] {chunk.get('title') or chunk.get('filename') or 'Untitled'}"

    if chunk.get("section"):
        header += f" - {chunk['section']}"

    if chunk.get("version"):
        header += f" (version {chunk['version']}"
        if chunk.get("effective_date"):
            header += f", effective {chunk['effective_date']}"
        header += ")"
    elif chunk.get("effective_date"):
        header += f" (effective {chunk['effective_date']})"

    region = chunk.get("region")
    if region and region != "global":
        header += f" — APPLIES TO: {region.upper()} ONLY"

    return header


def format_excerpts(chunks) -> str:
    if not chunks:
        return "NO POLICY EXCERPTS WERE FOUND FOR THIS QUESTION."

    return "\n\n".join(
        f"{_header(number, chunk)}\n{chunk['content']}"
        for number, chunk in enumerate(chunks, start=1)
    )