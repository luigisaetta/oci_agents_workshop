# Scanned PDF To Markdown

This example shows a backend pipeline that converts a scanned PDF into a
single Markdown text file by sending each rendered page image to an OCI
multimodal model through `langchain-oci`.

The pipeline does four things:

- renders each PDF page as a PNG image;
- sends each image to `cohere.command-a-vision`;
- asks the model to extract all visible text as Markdown;
- assembles all page outputs into one Markdown file.

## Requirements

Install the project from the repository root:

```bash
pip install -e ".[dev]"
```

Configure OCI credentials as described in the root `README.md`, then set at least:

```text
OCI_COMPARTMENT_ID=<your-compartment-ocid>
OCI_REGION=<your-region>
OCI_AUTH_TYPE=API_KEY
OCI_AUTH_PROFILE=DEFAULT
```

The example uses `cohere.command-a-vision` by default. You can override it with
`--model-id` if your tenancy uses a different model identifier or a dedicated
endpoint.

## Run

From the repository root:

```bash
python -m scanned_pdf_to_markdown.cli input_pdf/sample_scanned.pdf
```

The default output paths are:

```text
output/sample_scanned_pages/
output/sample_scanned.md
```

You can choose explicit paths:

```bash
python -m scanned_pdf_to_markdown.cli \
  input_pdf/sample_scanned.pdf \
  --image-output-dir output/scanned_pages \
  --output output/sample_scanned.md
```

## Use From Python Code

You can also import the backend pipeline from another Python module. This is the
recommended approach when you want to integrate scanned PDF extraction into an
API, a batch job, or a larger document-processing workflow.

```python
from pathlib import Path

from dotenv import load_dotenv

from common.utils import collect_oci_runtime_config
from scanned_pdf_to_markdown.pipeline import (
    ConversionOptions,
    convert_scanned_pdf_to_markdown,
)
from scanned_pdf_to_markdown.vision_extractor import build_vision_model


def convert_document(pdf_path: Path) -> Path:
    """Convert one scanned PDF into Markdown.

    Args:
        pdf_path: Path to the scanned PDF file.

    Returns:
        Path to the generated Markdown file.
    """
    load_dotenv(".env")

    runtime_config = collect_oci_runtime_config()
    llm = build_vision_model(runtime_config)

    output_path = Path("output") / f"{pdf_path.stem}.md"
    image_output_dir = Path("output") / f"{pdf_path.stem}_pages"

    return convert_scanned_pdf_to_markdown(
        pdf_path=pdf_path,
        output_path=output_path,
        image_output_dir=image_output_dir,
        llm=llm,
        options=ConversionOptions(
            dpi=200,
            include_page_markers=True,
        ),
    )


if __name__ == "__main__":
    generated_file = convert_document(Path("input_pdf/sample_scanned.pdf"))
    print(f"Generated Markdown file: {generated_file}")
```

For custom extraction behavior, pass your own prompt through `ConversionOptions`:

```python
options = ConversionOptions(
    dpi=250,
    prompt=(
        "Extract all visible text from this scanned page as Markdown. "
        "Preserve tables and list numbering. Return only Markdown."
    ),
    include_page_markers=False,
)
```

## Prompt

The default prompt asks the model to return only Markdown, preserving visible
reading order, headings, paragraphs, lists, tables, and footnotes. It also asks
the model to mark unreadable words as `[unreadable]` instead of guessing.

## Notes

- The PDF is rendered with PyMuPDF and produces PNG files.
- The Markdown extraction is performed one page at a time.
- Page markers are added as HTML comments, for example `<!-- Page 1 -->`.
- Use `--no-page-markers` if you want the final Markdown without page markers.
