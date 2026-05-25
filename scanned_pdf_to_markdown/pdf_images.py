"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: Render scanned PDF pages to model-ready image files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz
from PIL import Image

DEFAULT_DPI = 200
DEFAULT_IMAGE_FORMAT = "jpeg"
DEFAULT_JPEG_QUALITY = 85
DEFAULT_MAX_SIDE = 1600


@dataclass(frozen=True)
class ImageRenderOptions:
    """Options used to render PDF pages as model-ready images.

    Attributes:
        dpi: Rendering resolution in dots per inch.
        image_format: Output image format, either ``jpeg`` or ``png``.
        max_side: Maximum width or height after resizing.
        jpeg_quality: JPEG quality used when ``image_format`` is ``jpeg``.
    """

    dpi: int = DEFAULT_DPI
    image_format: str = DEFAULT_IMAGE_FORMAT
    max_side: int = DEFAULT_MAX_SIDE
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


def validate_image_options(options: ImageRenderOptions) -> str:
    """Validate rendering options and normalize the image format.

    Args:
        options: Image rendering options.

    Returns:
        Normalized image format.

    Raises:
        ValueError: If any option is invalid.
    """
    if options.dpi <= 0:
        raise ValueError("DPI must be greater than zero.")
    if options.max_side <= 0:
        raise ValueError("Maximum image side must be greater than zero.")
    if not 1 <= options.jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    normalized_format = options.image_format.lower().strip()
    if normalized_format not in {"jpeg", "png"}:
        raise ValueError("Image format must be either 'jpeg' or 'png'.")
    return normalized_format


def _pixmap_to_image(pixmap: fitz.Pixmap) -> Image.Image:
    """Convert a PyMuPDF pixmap to a PIL image.

    Args:
        pixmap: Rendered PyMuPDF pixmap.

    Returns:
        RGB PIL image.
    """
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
    """Resize an image when its largest side exceeds ``max_side``.

    Args:
        image: Source PIL image.
        max_side: Maximum width or height.

    Returns:
        Original or resized PIL image.
    """
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image

    target_size = (int(width * scale), int(height * scale))
    return image.resize(target_size, Image.Resampling.LANCZOS)


def _save_image(
    image: Image.Image,
    image_path: Path,
    image_format: str,
    jpeg_quality: int,
) -> None:
    """Save an image as JPEG or PNG.

    Args:
        image: PIL image to save.
        image_path: Destination path.
        image_format: Normalized image format.
        jpeg_quality: JPEG quality used for JPEG output.

    Returns:
        None.
    """
    if image_format == "jpeg":
        image.save(image_path, format="JPEG", quality=jpeg_quality, optimize=True)
        return

    image.save(image_path, format="PNG", compress_level=6, optimize=True)


def _render_page_to_image(
    page: fitz.Page,
    matrix: fitz.Matrix,
    max_side: int,
) -> Image.Image:
    """Render one PDF page and resize it for model input.

    Args:
        page: PyMuPDF page object.
        matrix: Rendering scale matrix.
        max_side: Maximum output width or height.

    Returns:
        Rendered and resized PIL image.
    """
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return _resize_image(_pixmap_to_image(pixmap), max_side=max_side)


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    options: ImageRenderOptions = ImageRenderOptions(),
) -> List[Path]:
    """Render each PDF page to a model-ready image file.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory where generated image files are written.
        options: Image rendering options.

    Returns:
        List of image paths, ordered by page number.

    Raises:
        FileNotFoundError: If the source PDF does not exist.
        ValueError: If the source file or rendering options are invalid.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")

    normalized_format = validate_image_options(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = options.dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    rendered_paths: List[Path] = []
    extension = "jpg" if normalized_format == "jpeg" else "png"

    document = fitz.open(pdf_path)
    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            image = _render_page_to_image(
                page=page,
                matrix=matrix,
                max_side=options.max_side,
            )
            image_path = (
                output_dir / f"{pdf_path.stem}_page_{page_index + 1:03d}.{extension}"
            )
            _save_image(
                image=image,
                image_path=image_path,
                image_format=normalized_format,
                jpeg_quality=options.jpeg_quality,
            )
            rendered_paths.append(image_path)
    finally:
        document.close()

    return rendered_paths


def render_pdf_to_png(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = DEFAULT_DPI,
) -> List[Path]:
    """Render each PDF page to a PNG image.

    Args:
        pdf_path: Path to the source PDF file.
        output_dir: Directory where generated PNG files are written.
        dpi: Rendering resolution in dots per inch.

    Returns:
        List of PNG image paths, ordered by page number.
    """
    return render_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir,
        options=ImageRenderOptions(
            dpi=dpi,
            image_format="png",
            max_side=DEFAULT_MAX_SIDE,
        ),
    )
