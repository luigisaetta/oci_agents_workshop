"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: Tests for the scanned PDF to Markdown backend example.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scanned_pdf_to_markdown import (
    cli,
    document_writer,
    pdf_images,
    pipeline,
    vision_extractor,
)


def _fake_invoke(messages):
    """Return a deterministic Markdown response.

    Args:
        messages: LangChain messages sent to the fake model.

    Returns:
        SimpleNamespace: Object with a content field.
    """
    assert len(messages) == 1
    content = messages[0].content
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    return SimpleNamespace(content="```markdown\n# Extracted\n\nText\n```")


def test_clean_markdown_response_removes_outer_fence() -> None:
    """It should remove Markdown fences around model output."""
    raw_text = "```markdown\n# Title\n\nBody\n```"

    assert vision_extractor.clean_markdown_response(raw_text) == "# Title\n\nBody"


def test_assemble_markdown_document_adds_page_markers() -> None:
    """It should assemble pages in order with page comments."""
    markdown = document_writer.assemble_markdown_document(["# Page one", "# Page two"])

    assert markdown == (
        "<!-- Page 1 -->\n\n# Page one\n\n" "<!-- Page 2 -->\n\n# Page two\n"
    )


def test_write_markdown_document_creates_parent_directory(tmp_path: Path) -> None:
    """It should create parent folders before writing output."""
    output_path = tmp_path / "nested" / "document.md"

    result = document_writer.write_markdown_document(output_path, "# Hello\n")

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "# Hello\n"


def test_extract_markdown_from_image_uses_multimodal_message(tmp_path: Path) -> None:
    """It should send text and image content to the model."""
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    markdown = vision_extractor.extract_markdown_from_image(
        image_path=image_path,
        llm=SimpleNamespace(invoke=_fake_invoke),
        prompt="Extract text.",
    )

    assert markdown == "# Extracted\n\nText"


def test_image_file_to_data_url_encodes_jpeg_payload(tmp_path: Path) -> None:
    """It should encode rendered images as JPEG data URLs."""
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    data_url = vision_extractor.image_file_to_data_url(image_path)

    assert data_url.startswith("data:image/jpeg;base64,")


def test_extract_markdown_from_image_requires_existing_image(tmp_path: Path) -> None:
    """It should reject missing image files before model invocation."""
    with pytest.raises(FileNotFoundError):
        vision_extractor.extract_markdown_from_image(
            image_path=tmp_path / "missing.png",
            llm=SimpleNamespace(invoke=_fake_invoke),
        )


def test_default_paths_are_based_on_pdf_name() -> None:
    """It should build default output paths from the PDF stem."""
    pdf_path = Path("input_pdf/sample.pdf")

    assert pipeline.default_output_path(pdf_path) == Path("output/sample.md")
    assert pipeline.default_image_output_dir(pdf_path) == Path("output/sample_pages")


def test_default_conversion_options_match_model_ready_image_defaults() -> None:
    """It should use the model-ready JPEG defaults for page rendering."""
    options = pipeline.ConversionOptions()

    assert options.dpi == 200
    assert options.image_format == "jpeg"
    assert options.max_side == 1600
    assert options.jpeg_quality == 85


def test_apply_model_override_updates_printed_runtime_config() -> None:
    """It should expose the CLI-selected model in the effective runtime config."""
    runtime_config = {
        "OCI_MODEL_ID": "openai.gpt-oss-120b",
        "OCI_REGION": "us-chicago-1",
    }

    effective_config = cli.apply_model_override(
        runtime_config,
        "cohere.command-a-vision",
    )

    assert effective_config["OCI_MODEL_ID"] == "cohere.command-a-vision"
    assert runtime_config["OCI_MODEL_ID"] == "openai.gpt-oss-120b"


def test_validate_image_options_rejects_invalid_format() -> None:
    """It should reject unsupported rendered image formats."""
    with pytest.raises(ValueError):
        pdf_images.validate_image_options(
            pdf_images.ImageRenderOptions(
                dpi=200,
                image_format="gif",
                jpeg_quality=85,
                max_side=1600,
            )
        )


def test_convert_scanned_pdf_to_markdown_orchestrates_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """It should render pages, extract Markdown, and write one output file."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    image_dir = tmp_path / "pages"
    output_path = tmp_path / "output.md"
    image_paths = [image_dir / "page_001.jpg", image_dir / "page_002.jpg"]

    def _fake_render_pdf_to_images(pdf_path, output_dir, options):
        assert pdf_path.name == "sample.pdf"
        assert output_dir == image_dir
        assert options.dpi == 200
        assert options.image_format == "jpeg"
        assert options.max_side == 1600
        assert options.jpeg_quality == 85
        return image_paths

    def _fake_extract_markdown_from_image(image_path, llm, prompt, jpeg_quality):
        assert llm == "fake-llm"
        assert prompt == "Extract."
        assert jpeg_quality == 85
        return f"# {image_path.stem}"

    monkeypatch.setattr(
        pipeline,
        "render_pdf_to_images",
        _fake_render_pdf_to_images,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_markdown_from_image",
        _fake_extract_markdown_from_image,
    )

    result = pipeline.convert_scanned_pdf_to_markdown(
        pdf_path=pdf_path,
        output_path=output_path,
        image_output_dir=image_dir,
        llm="fake-llm",
        options=pipeline.ConversionOptions(prompt="Extract."),
    )

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == (
        "<!-- Page 1 -->\n\n# page_001\n\n" "<!-- Page 2 -->\n\n# page_002\n"
    )
