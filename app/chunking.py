from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import config
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)
def chunk_sections(sections):
    chunks = []
    for section, text in sections:
        for piece in splitter.split_text(text):
            chunks.append({
                "section": section,
                "content": piece,
            })
    return chunks
