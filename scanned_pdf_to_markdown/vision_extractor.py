"""
Author: L. Saetta
Date last modified: 2026-05-27
License: MIT
Description: Extract Markdown text from page images with OCI vision models.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_oci import ChatOCIGenAI
from PIL import Image

DEFAULT_MODEL_ID = "cohere.command-a-vision"

DEFAULT_EXTRACTION_PROMPT = "Extract all the text in the image."

LOGGER = logging.getLogger(__name__)


def _normalize_image_format(image_format: str) -> str:
    """Normalize and validate model payload image format.

    Args:
        image_format: Requested image format.

    Returns:
        Normalized image format.

    Raises:
        ValueError: If the format is not supported.
    """
    normalized_format = image_format.lower().strip()
    if normalized_format not in {"jpeg", "png"}:
        raise ValueError("Image format must be either 'jpeg' or 'png'.")
    return normalized_format


def image_to_data_url(
    image: Image.Image,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> str:
    """Convert an image object to a data URL for multimodal model input.

    Args:
        image: Rendered page image.
        image_format: Payload image format, either ``jpeg`` or ``png``.
        jpeg_quality: JPEG quality for the in-memory encoded payload when using JPEG.

    Returns:
        Base64 data URL with the selected image MIME type.

    Raises:
        ValueError: If image format or JPEG quality is invalid.
    """
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100.")

    normalized_format = _normalize_image_format(image_format)
    buffer = io.BytesIO()
    if normalized_format == "jpeg":
        rgb_image = image.convert("RGB")
        rgb_image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime_type = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime_type = "image/png"

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_file_to_data_url(
    image_path: Path,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> str:
    """Convert an image file to a data URL for multimodal model input.

    Args:
        image_path: Path to the rendered page image.
        image_format: Payload image format, either ``jpeg`` or ``png``.
        jpeg_quality: JPEG quality for the in-memory encoded payload when using JPEG.

    Returns:
        Base64 data URL with the selected image MIME type.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If image format or JPEG quality is invalid.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        return image_to_data_url(
            image=image,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )


def build_model_kwargs(
    model_id: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build model kwargs with provider-specific token parameter names.

    Args:
        model_id: OCI model ID for the vision model.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens for each page extraction.

    Returns:
        Model keyword arguments for ``ChatOCIGenAI``.
    """
    token_parameter_name = "max_tokens"
    normalized_model_id = model_id.lower()
    if normalized_model_id.startswith("gpt-5") or ".gpt-5" in normalized_model_id:
        token_parameter_name = "max_completion_tokens"

    return {
        "temperature": temperature,
        token_parameter_name: max_tokens,
    }


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
        auth_type=runtime_config["OCI_AUTH_TYPE"],
        auth_profile=runtime_config["OCI_AUTH_PROFILE"],
        model_kwargs=build_model_kwargs(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
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


def extract_text(response: Any) -> str:
    """Extract plain text from common LangChain model response shapes.

    Args:
        response: Model response object, string, or list-like structured payload.

    Returns:
        Extracted text content.
    """
    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        return "".join(text_parts)

    return str(content)


def invoke_llm_with_timing(llm: Any, messages: list[HumanMessage]) -> Any:
    """Invoke the LLM and log the model call duration.

    Args:
        llm: LangChain chat model with vision support.
        messages: Messages to send to the model.

    Returns:
        Model response returned by ``llm.invoke``.
    """
    start_time = time.perf_counter()
    response = llm.invoke(messages)
    elapsed_seconds = time.perf_counter() - start_time
    LOGGER.info("LLM call duration: %.2f secs", elapsed_seconds)
    return response


def extract_markdown_from_image(
    image_path: Path,
    llm: Any,
    prompt: str = DEFAULT_EXTRACTION_PROMPT,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> str:
    """Extract Markdown text from one page image.

    Args:
        image_path: Image path for one rendered PDF page.
        llm: LangChain chat model with vision support.
        prompt: Extraction prompt sent with the image.
        image_format: Payload image format, either ``jpeg`` or ``png``.
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
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    response = invoke_llm_with_timing(llm, [message])

    return clean_markdown_response(extract_text(response))


def extract_markdown_from_image_object(
    image: Image.Image,
    llm: Any,
    prompt: str = DEFAULT_EXTRACTION_PROMPT,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> str:
    """Extract Markdown text from one in-memory page image.

    Args:
        image: Rendered page image.
        llm: LangChain chat model with vision support.
        prompt: Extraction prompt sent with the image.
        image_format: Payload image format, either ``jpeg`` or ``png``.
        jpeg_quality: JPEG quality for the in-memory data URL payload.

    Returns:
        Markdown text extracted from the image.

    Raises:
        ValueError: If image format or JPEG quality is invalid.
    """
    data_url = image_to_data_url(
        image=image,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    response = invoke_llm_with_timing(llm, [message])

    return clean_markdown_response(extract_text(response))
