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
from scanned_pdf_to_markdown.pdf_images import (
    DEFAULT_DPI,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SIDE,
    ImageRenderOptions,
    render_pdf_to_images,
)
from scanned_pdf_to_markdown.vision_extractor import (
    DEFAULT_EXTRACTION_PROMPT,
    extract_markdown_from_image,
)


@dataclass(frozen=True)
class ConversionOptions:
    """Options for scanned PDF conversion.

    Attributes:
        dpi: PDF rendering resolution.
        image_format: Output image format for rendered pages.
        max_side: Maximum width or height after resizing.
        jpeg_quality: JPEG quality used when ``image_format`` is ``jpeg``.
        prompt: Prompt used for each page image.
        include_page_markers: Whether to add page markers in the final document.
    """

    dpi: int = DEFAULT_DPI
    image_format: str = DEFAULT_IMAGE_FORMAT
    max_side: int = DEFAULT_MAX_SIDE
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    prompt: str = DEFAULT_EXTRACTION_PROMPT
    include_page_markers: bool = True


#
# This is the pipeline
#
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
        image_output_dir: Directory where page image files are generated.
        llm: LangChain multimodal model used for image extraction.
        options: Conversion options.

    Returns:
        Path to the generated Markdown file.
    """
    # 1. extract jpg images for all pages in the PDF
    image_paths = render_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=image_output_dir,
        options=ImageRenderOptions(
            dpi=options.dpi,
            image_format=options.image_format,
            max_side=options.max_side,
            jpeg_quality=options.jpeg_quality,
        ),
    )
    # 2. extract markdown from each page image
    page_markdown = [
        extract_markdown_from_image(
            image_path=image_path,
            llm=llm,
            prompt=options.prompt,
            jpeg_quality=options.jpeg_quality,
        )
        for image_path in image_paths
    ]
    # 3. assemble page markdown into a single document and write to output
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
