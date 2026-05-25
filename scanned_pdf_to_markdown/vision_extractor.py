"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: Extract Markdown text from page images with OCI vision models.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI
from PIL import Image

from common.utils import extract_text

DEFAULT_MODEL_ID = "cohere.command-a-vision"

DEFAULT_EXTRACTION_PROMPT = """You are performing OCR on a scanned document page.
Return ONLY the transcribed text.

Rules:
- Do not return JSON.
- Do not wrap the output in Markdown fences.
- Do not add page numbers.
- Do not summarize.
- Do not translate.
- Preserve reading order, paragraphs, line breaks, headings, lists, and numbering.
- Keep units, symbols, mathematical signs, and special characters exactly as in the source.
- Do not guess or invent missing characters.
- If text or symbols are unreadable, write [ILLEGIBLE].

Tables:
- If you detect a table, output it as a GitHub-flavored Markdown table using pipes '|'.
- Flatten multi-row or multi-level headers into a single explicit header row when possible.
- Do not use spaces to align columns. Use only Markdown pipes.
- Keep each data row on a single Markdown row.
- If a cell is empty or the source shows '-', output '-'."""


def image_file_to_data_url(image_path: Path, jpeg_quality: int = 85) -> str:
    """Convert an image file to a JPEG data URL for multimodal model input.

    Args:
        image_path: Path to the rendered page image.
        jpeg_quality: JPEG quality for the in-memory encoded payload.

    Returns:
        Base64 data URL with ``image/jpeg`` MIME type.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If JPEG quality is invalid.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_vision_model(
    runtime_config: Dict[str, str],
    model_id: str = DEFAULT_MODEL_ID,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatOCIGenAI:
    """Create a LangChain OCI multimodal chat model.

    Args:
        runtime_config: OCI runtime configuration dictionary.
        model_id: OCI model ID for the vision model.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens for each page extraction.

    Returns:
        Configured ``ChatOCIGenAI`` instance.
    """
    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=runtime_config["OCI_SERVICE_ENDPOINT"],
        compartment_id=runtime_config["OCI_COMPARTMENT_ID"],
        provider="cohere",
        auth_type=runtime_config["OCI_AUTH_TYPE"],
        auth_profile=runtime_config["OCI_AUTH_PROFILE"],
        model_kwargs={"temperature": temperature, "max_tokens": max_tokens},
    )


def clean_markdown_response(markdown_text: str) -> str:
    """Normalize model output before writing it into the final document.

    Args:
        markdown_text: Raw model output.

    Returns:
        Markdown text without outer Markdown code fences.
    """
    text = markdown_text.strip()
    if text.startswith("```markdown") and text.endswith("```"):
        return text[len("```markdown") : -len("```")].strip()
    if text.startswith("```") and text.endswith("```"):
        return text[len("```") : -len("```")].strip()
    return text


def extract_markdown_from_image(
    image_path: Path,
    llm: Any,
    prompt: str = DEFAULT_EXTRACTION_PROMPT,
    jpeg_quality: int = 85,
) -> str:
    """Extract Markdown text from one page image.

    Args:
        image_path: Image path for one rendered PDF page.
        llm: LangChain chat model with vision support.
        prompt: Extraction prompt sent with the image.
        jpeg_quality: JPEG quality for the in-memory data URL payload.

    Returns:
        Markdown text extracted from the image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    data_url = image_file_to_data_url(
        image_path=image_path,
        jpeg_quality=jpeg_quality,
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    response = llm.invoke([message])

    return clean_markdown_response(extract_text(response))
