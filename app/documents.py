"""Document readers: load .md, .docx, .pdf, .csv, .txt and .html files
into a common list of (heading, text) sections."""

from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from pathlib import Path

from docx import Document
from pypdf import PdfReader

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


# --- Markdown ---------------------------------------------------------------
def read_md(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    body: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if "\n".join(body).strip():
                sections.append((heading, "\n".join(body).strip()))
            heading = match.group(2).strip()
            body = []
        else:
            body.append(line)

    if "\n".join(body).strip():
        sections.append((heading, "\n".join(body).strip()))
    return sections


# --- Word -------------------------------------------------------------------
def read_docx(path: Path) -> list[tuple[str, str]]:
    document = Document(str(path))
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    body: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading"):
            if body:
                sections.append((heading, "\n".join(body).strip()))
            heading = text
            body = []
        else:
            body.append(text)

    if body:
        sections.append((heading, "\n".join(body).strip()))
    return sections


# --- PDF --------------------------------------------------------------------
def read_pdf(path: Path) -> list[tuple[str, str]]:
    reader = PdfReader(str(path))
    return [
        (f"Page {number}", page.extract_text() or "")
        for number, page in enumerate(reader.pages, start=1)
    ]


# --- CSV --------------------------------------------------------------------
def read_csv(path: Path) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    # The generator writes a provenance comment as the first row;
    # the real header follows.
    if rows and rows[0] and rows[0][0].startswith("#"):
        rows = rows[1:]
    if not rows:
        return []

    header, data = rows[0], rows[1:]
    sections: list[tuple[str, str]] = []

    for row in data:
        if not any(cell.strip() for cell in row):
            continue
        line = ", ".join(f"{column}: {value}" for column, value in zip(header, row))
        # Name the section after the first column that is not a date.
        label = next(
            (
                value
                for column, value in zip(header, row)
                if value.strip() and column.lower() not in ("date", "day")
            ),
            "Row",
        )
        sections.append((label, line))

    return sections


# --- Plain text -------------------------------------------------------------
def read_txt(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    return [("Introduction", text)] if text else []


# --- HTML -------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def read_html(path: Path) -> list[tuple[str, str]]:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8"))
    text = "\n".join(parser.parts).strip()
    return [("Introduction", text)] if text else []


# --- Dispatch ---------------------------------------------------------------
READERS = {
    ".md": read_md,
    ".markdown": read_md,
    ".docx": read_docx,
    ".pdf": read_pdf,
    ".csv": read_csv,
    ".txt": read_txt,
    ".html": read_html,
    ".htm": read_html,
}


def read_document(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported document type: {path.name}")
    return reader(path)