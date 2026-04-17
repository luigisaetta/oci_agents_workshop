"""
Author: L. Saetta
Date last modified: 2026-04-17
License: MIT
Description: Load PDFs from input_pdf, chunk text, and build embeddings for simple RAG.
"""

from __future__ import annotations

# pylint: disable=duplicate-code

import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm import tqdm

from oci_models import build_embedding_client
from utils import collect_oci_runtime_config

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 16


def list_pdf_files(input_dir: Path) -> List[Path]:
    """Return sorted PDF file paths from the input directory."""
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.glob("*.pdf") if path.is_file())


def _resolve_pdf_title(reader: PdfReader, pdf_path: Path) -> str:
    """Resolve a readable title from PDF metadata with filename fallback."""
    metadata = reader.metadata
    if metadata and metadata.title:
        return str(metadata.title).strip() or pdf_path.stem
    return pdf_path.stem


def load_pdf_documents(input_dir: Path) -> List[Document]:
    """Read all PDFs from input_dir and return one raw document per page."""
    pdf_paths = list_pdf_files(input_dir)
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {input_dir}.")

    documents: List[Document] = []
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        title = _resolve_pdf_title(reader, pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": pdf_path.name,
                        "title": title,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No extractable text found in PDF files in {input_dir}.")
    return documents


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


def add_chunks_to_vector_store(
    chunk_documents_list: Sequence[Document],
    vector_store: InMemoryVectorStore,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> int:
    """Embed and add all chunks to vector store in batches with progress bar."""
    total_chunks = len(chunk_documents_list)
    total_added = 0

    for start_index in tqdm(
        range(0, total_chunks, batch_size),
        desc="Embedding chunks",
        unit="chunk",
    ):
        chunk_batch = chunk_documents_list[start_index : start_index + batch_size]
        added_ids = vector_store.add_documents(list(chunk_batch))
        total_added += len(added_ids)

    return total_added


def build_pdf_vector_store(
    input_dir: Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> InMemoryVectorStore:
    """Load PDFs, create chunk embeddings, and return an indexed in-memory store."""
    logging.info("START PdfLoadingAndIndexing")
    runtime_config = _collect_pdf_runtime_config()

    raw_documents = load_pdf_documents(input_dir)
    chunks = chunk_documents(
        documents=raw_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    embedding_client = build_embedding_client(runtime_config)
    vector_store = InMemoryVectorStore(embedding=embedding_client)
    loaded_documents = add_chunks_to_vector_store(chunks, vector_store=vector_store)
    logging.info("Loaded documents: %s", loaded_documents)
    logging.info("END PdfLoadingAndIndexing")
    return vector_store
