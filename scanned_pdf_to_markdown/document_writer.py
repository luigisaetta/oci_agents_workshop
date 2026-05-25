"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: Assemble page Markdown into a single output document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def assemble_markdown_document(
    page_markdown: Sequence[str],
    include_page_markers: bool = True,
) -> str:
    """Assemble page-level Markdown into one document.

    Args:
        page_markdown: Markdown text extracted from each page.
        include_page_markers: Whether to add page headings between pages.

    Returns:
        A single Markdown document.
    """
    sections: list[str] = []
    for page_index, markdown_text in enumerate(page_markdown, start=1):
        content = markdown_text.strip()
        if include_page_markers:
            sections.append(f"<!-- Page {page_index} -->\n\n{content}")
        else:
            sections.append(content)

    return "\n\n".join(sections).strip() + "\n"


def write_markdown_document(output_path: Path, markdown_text: str) -> Path:
    """Write Markdown text to disk.

    Args:
        output_path: Destination file path.
        markdown_text: Markdown content to write.

    Returns:
        The output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")
    return output_path
