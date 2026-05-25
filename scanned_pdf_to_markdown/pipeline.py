"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: End-to-end scanned PDF to Markdown backend pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from scanned_pdf_to_markdown.document_writer import (
    assemble_markdown_document,
    write_markdown_document,
)
from scanned_pdf_to_markdown.pdf_images import render_pdf_to_png
from scanned_pdf_to_markdown.vision_extractor import (
    DEFAULT_EXTRACTION_PROMPT,
    extract_markdown_from_image,
)


@dataclass(frozen=True)
class ConversionOptions:
    """Options for scanned PDF conversion.

    Attributes:
        dpi: PDF rendering resolution.
        prompt: Prompt used for each page image.
        include_page_markers: Whether to add page markers in the final document.
    """

    dpi: int = 200
    prompt: str = DEFAULT_EXTRACTION_PROMPT
    include_page_markers: bool = True


def convert_scanned_pdf_to_markdown(
    pdf_path: Path,
    output_path: Path,
    image_output_dir: Path,
    llm: Any,
    options: ConversionOptions = ConversionOptions(),
) -> Path:
    """Convert a scanned PDF into a single Markdown file.

    Args:
        pdf_path: Source scanned PDF path.
        output_path: Destination Markdown file path.
        image_output_dir: Directory where page PNG files are generated.
        llm: LangChain multimodal model used for image extraction.
        options: Conversion options.

    Returns:
        Path to the generated Markdown file.
    """
    image_paths = render_pdf_to_png(
        pdf_path=pdf_path,
        output_dir=image_output_dir,
        dpi=options.dpi,
    )
    page_markdown = [
        extract_markdown_from_image(
            image_path=image_path,
            llm=llm,
            prompt=options.prompt,
        )
        for image_path in image_paths
    ]
    document_markdown = assemble_markdown_document(
        page_markdown=page_markdown,
        include_page_markers=options.include_page_markers,
    )

    return write_markdown_document(
        output_path=output_path,
        markdown_text=document_markdown,
    )


def default_output_path(pdf_path: Path, output_path: Optional[Path] = None) -> Path:
    """Resolve the default Markdown output path for a PDF.

    Args:
        pdf_path: Source scanned PDF path.
        output_path: Optional explicit output path.

    Returns:
        Resolved output path.
    """
    if output_path is not None:
        return output_path
    return Path("output") / f"{pdf_path.stem}.md"


def default_image_output_dir(
    pdf_path: Path, image_output_dir: Optional[Path] = None
) -> Path:
    """Resolve the default image output directory for a PDF.

    Args:
        pdf_path: Source scanned PDF path.
        image_output_dir: Optional explicit image output directory.

    Returns:
        Resolved image output directory.
    """
    if image_output_dir is not None:
        return image_output_dir
    return Path("output") / f"{pdf_path.stem}_pages"
