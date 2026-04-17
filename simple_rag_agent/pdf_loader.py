"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Load PDFs from input_pdf, chunk text, and build embeddings for simple RAG.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

from oci_models import build_embedding_client
from simple_rag_agent.in_memory_vector_store import InMemoryVectorStore
from utils import collect_oci_runtime_config

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 16


def list_pdf_files(input_dir: Path) -> List[Path]:
    """Return sorted PDF file paths from the input directory."""
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.glob("*.pdf") if path.is_file())


def build_documents_from_texts(
    pdf_text_items: Sequence[Tuple[str, str]],
) -> List[Document]:
    """Build LangChain documents from pairs of source name and text."""
    documents: List[Document] = []
    for source_name, text in pdf_text_items:
        normalized_text = text.strip()
        if not normalized_text:
            continue
        documents.append(
            Document(page_content=normalized_text, metadata={"source": source_name})
        )
    return documents


def load_pdf_documents(input_dir: Path) -> List[Document]:
    """Read all PDFs from input_dir and return one raw document per PDF."""
    pdf_paths = list_pdf_files(input_dir)
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {input_dir}.")

    pdf_text_items: List[Tuple[str, str]] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        page_texts: List[str] = []
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")
        pdf_text_items.append((pdf_path.name, "\n".join(page_texts)))

    return build_documents_from_texts(pdf_text_items)


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into chunks with LangChain RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def _collect_pdf_runtime_config() -> Dict[str, str]:
    """Collect OCI runtime settings required by PDF embedding pipeline."""
    runtime_config = collect_oci_runtime_config()

    embed_model_id = os.getenv("OCI_EMBED_MODEL_ID", "").strip()
    if not embed_model_id:
        raise ValueError("Set OCI_EMBED_MODEL_ID environment variable.")

    top_k_raw = os.getenv("SIMPLE_RAG_TOP_K", "4").strip()
    try:
        top_k = int(top_k_raw)
    except ValueError as error:
        raise ValueError("Set SIMPLE_RAG_TOP_K as a positive integer.") from error
    if top_k < 1:
        raise ValueError("Set SIMPLE_RAG_TOP_K as a positive integer.")

    runtime_config["OCI_EMBED_MODEL_ID"] = embed_model_id
    runtime_config["SIMPLE_RAG_TOP_K"] = str(top_k)
    return runtime_config


def embed_chunks(
    chunk_documents_list: Sequence[Document],
    embedding_client: Any,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[Sequence[float]]:
    """Build embeddings for all chunks in batches with a progress bar."""
    vectors: List[Sequence[float]] = []
    total_chunks = len(chunk_documents_list)

    for start_index in tqdm(
        range(0, total_chunks, batch_size),
        desc="Embedding chunks",
        unit="chunk",
    ):
        chunk_batch = chunk_documents_list[start_index : start_index + batch_size]
        batch_vectors = embedding_client.embed_documents(
            [document.page_content for document in chunk_batch]
        )
        vectors.extend(batch_vectors)

    return vectors


def build_pdf_vector_store(
    input_dir: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> InMemoryVectorStore:
    """Load PDFs, create chunk embeddings, and return an indexed in-memory store."""
    logging.info("START PdfLoadingAndIndexing")
    runtime_config = _collect_pdf_runtime_config()
    top_k = int(runtime_config["SIMPLE_RAG_TOP_K"])

    raw_documents = load_pdf_documents(input_dir)
    chunks = chunk_documents(
        documents=raw_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedding_client = build_embedding_client(runtime_config)
    vectors = embed_chunks(chunks, embedding_client=embedding_client)

    vector_store = InMemoryVectorStore(base_documents=chunks, top_k=top_k)
    loaded_documents = vector_store.set_index(chunks, vectors)
    logging.info("Loaded documents: %s", loaded_documents)
    logging.info("END PdfLoadingAndIndexing")
    return vector_store
