"""
Mistral OCR module for converting PDF files to Markdown.

This module handles PDF to Markdown conversion using Mistral OCR API with
skip-if-exists logic to avoid re-OCR'ing already processed files.
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from mistralai import Mistral
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MistralOCR:
    """Wrapper for Mistral OCR API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Mistral OCR client.

        Args:
            api_key: Mistral API key. If None, reads from MISTRAL_API_KEY env var.
        """
        self.api_key = api_key or self._get_api_key()
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-ocr-latest"

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = self._get_env("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables. "
                "Please set it in .env file or pass as parameter."
            )
        return api_key

    @staticmethod
    def _get_env(key: str) -> Optional[str]:
        """Get environment variable."""
        import os
        return os.getenv(key)
    @staticmethod
    def _encode_pdf_to_base64(pdf_path: Path) -> str:
        """
        Encode a PDF file to base64 string.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Base64 encoded string of the PDF content.
        """
        import base64
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def pdf_to_markdown(
        self,
        pdf_path: Path,
        output_dir: Path,
        force: bool = False
    ) -> Path:
        """
        Convert a PDF file to Markdown using Mistral OCR.

        Args:
            pdf_path: Path to the input PDF file.
            output_dir: Directory to save the output Markdown file.
            force: If True, re-OCR even if output exists. Defaults to False.

        Returns:
            Path to the generated Markdown file.

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            ValueError: If PDF file doesn't have .pdf extension.
        """
        # Validate input
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        # Generate output filename from path
        # Remove the 'data/raw_pdfs/' prefix to get the meaningful part
        try:
            relative_path = pdf_path.relative_to("data/raw_pdfs")
        except ValueError:
            # If not under data/raw_pdfs, use filename only
            relative_path = pdf_path.name
        encoded_filename = self._encode_path_to_filename(relative_path)
        output_path = output_dir / encoded_filename

        # Skip if output exists and not forcing
        if output_path.exists() and not force:
            logger.info(f"Skipping {pdf_path.name} - output already exists")
            return output_path

        logger.info(f"Processing: {pdf_path.name} -> {encoded_filename}")

        try:
            # Read PDF as base64
            pdf_data = self._encode_pdf_to_base64(pdf_path)

            # Call Mistral OCR
            logger.info(f"Calling Mistral OCR API...")
            ocr_response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_data}"
                },
                table_format="markdown",  # Changed from "html" to "markdown" for better RAG integration
                extract_header=True,
                extract_footer=True,
                include_image_base64=True
            )

            # Extract and merge markdown content from all pages
            page_count = len(ocr_response.pages)
            markdown_content = "\n\n".join([page.markdown for page in ocr_response.pages])

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. Save the full OCR response JSON (using SDK's native serialization)
            ocr_json_path = output_path.with_suffix(".ocr.json")
            with open(ocr_json_path, "w", encoding="utf-8") as f:
                json.dump(ocr_response.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved full OCR response: {ocr_json_path.name}")

            # 2. Save the markdown file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"✓ Saved Markdown ({page_count} pages): {output_path.name}")

            # 3. Write metadata file
            meta_path = output_path.with_suffix(".meta.json")
            metadata = {
                "source_path": str(relative_path),
                "company": str(relative_path.parts[0]),
                "product_type": str(relative_path.parts[1] if len(relative_path.parts) > 1 else "unknown"),
                "risk_profile": str(relative_path.parts[2] if len(relative_path.parts) > 2 else "n/a"),
                "ocr_timestamp": datetime.now(timezone.utc).isoformat(),
                "page_count": page_count,
                "model": self.model,
                "table_format": "markdown",
                "files": {
                    "markdown": str(output_path.name),
                    "ocr_json": str(ocr_json_path.name),
                    "metadata": str(meta_path.name)
                }
            }

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Saved metadata: {meta_path.name}")

            return output_path

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {type(e).__name__}: {e}")
            # Print more details if available
            if hasattr(e, 'response'):
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            if hasattr(e, 'body'):
                logger.error(f"Error body: {e.body}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _encode_path_to_filename(self, path: Path) -> str:
        """
        Encode a file path into a single filename.

        Example:
            barclays/sipp/guide.pdf -> barclays_sipp_guide.md

        Args:
            path: Relative path to the PDF file.

        Returns:
            Encoded filename with .md extension.
        """
        # Get all parts except the final filename
        parts = list(path.parts)

        # Replace spaces with underscores in each part
        parts = [part.replace(" ", "_") for part in parts]

        # Join with underscores and change extension to .md
        encoded = "_".join(parts)

        # Change extension to .md
        encoded = encoded.replace(".pdf", ".md")

        return encoded

    def process_directory(
        self,
        pdf_dir: Path,
        output_dir: Path,
        force: bool = False,
        recursive: bool = True
    ) -> list[Path]:
        """
        Process all PDF files in a directory.

        Args:
            pdf_dir: Directory containing PDF files.
            output_dir: Directory to save output Markdown files.
            force: If True, re-OCR all files.
            recursive: If True, search subdirectories.

        Returns:
            List of paths to generated Markdown files.
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)

        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        # Find all PDF files
        if recursive:
            pdf_files = list(pdf_dir.rglob("*.pdf"))
        else:
            pdf_files = list(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process each PDF
        output_files = []
        for pdf_file in pdf_files:
            try:
                output_file = self.pdf_to_markdown(pdf_file, output_dir, force=force)
                output_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                continue

        logger.info(f"✓ Processed {len(output_files)}/{len(pdf_files)} files successfully")

        return output_files


def main():
    """Example usage of MistralOCR."""
    import sys

    # Initialize OCR client
    try:
        ocr = MistralOCR()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Define paths
    pdf_dir = Path("data/raw_pdfs")
    output_dir = Path("data/ocr_markdown")

    # Process all PDFs
    try:
        output_files = ocr.process_directory(pdf_dir, output_dir, force=False, recursive=True)
        print(f"\n✓ Generated {len(output_files)} Markdown files")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
