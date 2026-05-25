"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: Render scanned PDF pages to PNG images.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import fitz


def render_pdf_to_png(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
) -> List[Path]:
    """Render each PDF page to a PNG image.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory where generated PNG files are written.
        dpi: Rendering resolution in dots per inch.

    Returns:
        List of PNG image paths, ordered by page number.

    Raises:
        FileNotFoundError: If the source PDF does not exist.
        ValueError: If the source file is not a PDF or the DPI is invalid.
        RuntimeError: If PyMuPDF is not installed.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")
    if dpi <= 0:
        raise ValueError("DPI must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    rendered_paths: List[Path] = []

    document = fitz.open(pdf_path)
    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1:03d}.png"
            pixmap.save(image_path)
            rendered_paths.append(image_path)
    finally:
        document.close()

    return rendered_paths
