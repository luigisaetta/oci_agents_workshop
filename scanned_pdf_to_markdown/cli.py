"""
Author: L. Saetta
Date last modified: 2026-05-25
License: MIT
Description: Command line entry point for scanned PDF Markdown extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

from common.utils import collect_oci_runtime_config, print_oci_runtime_config
from scanned_pdf_to_markdown.pdf_images import (
    DEFAULT_DPI,
    DEFAULT_IMAGE_FORMAT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_SIDE,
)
from scanned_pdf_to_markdown.pipeline import (
    ConversionOptions,
    convert_scanned_pdf_to_markdown,
    default_image_output_dir,
    default_output_path,
)
from scanned_pdf_to_markdown.vision_extractor import (
    DEFAULT_MODEL_ID,
    build_vision_model,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert a scanned PDF into Markdown using OCI vision models."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the scanned PDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination Markdown file. Defaults to output/<pdf-name>.md.",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=None,
        help="Directory for generated page image files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PDF rendering resolution. Default: {DEFAULT_DPI}.",
    )
    parser.add_argument(
        "--image-format",
        choices=("jpeg", "png"),
        default=DEFAULT_IMAGE_FORMAT,
        help=f"Rendered page image format. Default: {DEFAULT_IMAGE_FORMAT}.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=DEFAULT_MAX_SIDE,
        help=f"Maximum rendered image width or height. Default: {DEFAULT_MAX_SIDE}.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality when --image-format=jpeg. Default: {DEFAULT_JPEG_QUALITY}.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"OCI vision model ID. Default: {DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--no-page-markers",
        action="store_true",
        help="Do not add page markers to the final Markdown document.",
    )
    return parser


def apply_model_override(
    runtime_config: Dict[str, str], model_id: str
) -> Dict[str, str]:
    """Return runtime configuration with the effective model ID.

    Args:
        runtime_config: Runtime configuration loaded from environment variables.
        model_id: Model ID selected by the CLI arguments.

    Returns:
        Runtime configuration copy with ``OCI_MODEL_ID`` set to the effective model.
    """
    effective_config = dict(runtime_config)
    effective_config["OCI_MODEL_ID"] = model_id
    return effective_config


def main() -> None:
    """Run the scanned PDF to Markdown pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
    runtime_config = collect_oci_runtime_config()
    effective_config = apply_model_override(runtime_config, args.model_id)
    print_oci_runtime_config(effective_config)

    llm = build_vision_model(
        runtime_config=effective_config,
        model_id=effective_config["OCI_MODEL_ID"],
    )
    output_path = default_output_path(args.pdf_path, args.output)
    image_output_dir = default_image_output_dir(args.pdf_path, args.image_output_dir)

    result_path = convert_scanned_pdf_to_markdown(
        pdf_path=args.pdf_path,
        output_path=output_path,
        image_output_dir=image_output_dir,
        llm=llm,
        options=ConversionOptions(
            dpi=args.dpi,
            image_format=args.image_format,
            max_side=args.max_side,
            jpeg_quality=args.jpeg_quality,
            include_page_markers=not args.no_page_markers,
        ),
    )
    print(f"Markdown file written to: {result_path}")


if __name__ == "__main__":
    main()
