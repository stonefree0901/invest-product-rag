"""
Chunking module for splitting markdown documents into smaller chunks.

This module handles:
1. Reading markdown files and their metadata
2. Splitting content into chunks using NLTK sentence tokenization
3. Handling tables as whole chunks (TODO)
4. Saving chunks to JSONL format
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Generator
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")


class DocumentChunker:
    """Split markdown documents into chunks."""

    def __init__(
        self,
        sentences_per_chunk: int = 15,
        overlap_sentences: int = 5,
        min_sentences: int = 3
    ):
        """
        Initialize chunker.

        Args:
            sentences_per_chunk: Number of sentences per chunk.
            overlap_sentences: Number of overlapping sentences between chunks.
            min_sentences: Minimum sentences for a valid chunk (shorter chunks merged).
        """
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.min_sentences = min_sentences

    def chunk_document(
        self,
        md_path: Path,
        meta_path: Path
    ) -> List[Dict]:
        """
        Chunk a single markdown document.

        Args:
            md_path: Path to the markdown file.
            meta_path: Path to the metadata JSON file.

        Returns:
            List of chunk dictionaries.
        """
        # 1. Read metadata
        metadata = self._read_metadata(meta_path)
        if not metadata:
            logger.error(f"Could not read metadata: {meta_path}")
            return []

        # 2. Read markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 3. Remove image placeholders (TODO: you can change this)
        content = self._remove_images(content)

        # 4. TODO: Split into chunks
        # You need to implement this part:
        # - Detect and extract tables (keep as whole chunks)
        # - For non-table content: split into sentences using NLTK
        # - Create chunks with overlap
        # - Merge chunks that are too short
        chunks = self._create_chunks(content, metadata)

        logger.info(f"✓ Created {len(chunks)} chunks from {md_path.name}")
        return chunks

    def _read_metadata(self, meta_path: Path) -> Dict:
        """
        Read metadata JSON file.

        Args:
            meta_path: Path to metadata JSON file.

        Returns:
            Metadata dictionary or empty dict if error.
        """
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error reading metadata {meta_path}: {e}")
            return {}

    def _remove_images(self, content: str) -> str:
        """
        Remove image placeholders from markdown content.

        Args:
            content: Markdown content with image placeholders.

        Returns:
            Content with images removed.
        """
        import re
        # Remove image markdown: ![alt](url)
        pattern = r'!\[([^\]]*)\]\([^)]+\)'
        content = re.sub(pattern, '', content)
        return content

    def _create_chunks(
        self,
        content: str,
        metadata: Dict
    ) -> List[Dict]:
        """
        Create chunks from content.

        TODO: You need to implement the actual chunking logic here.

        Steps:
        1. Detect tables in content (lines starting with |)
        2. Extract tables as whole chunks
        3. For remaining content, split into sentences
        4. Create chunks with overlap
        5. Merge short chunks

        Args:
            content: The markdown content.
            metadata: Document metadata.

        Returns:
            List of chunk dictionaries.
        """
        chunks = []

        # TODO: Your implementation here
        # Example placeholder:
        sentences = self._split_into_sentences(content)
        for i in range(0, len(sentences), 10):  # 简化：每10句一个chunk                                                                 
            chunk = { ... }                                                                                                             
            chunks.append(chunk)    

        # Create chunks with overlap
        # This is where you implement the 15-sentence-per-chunk logic
        # with 5-sentence overlap

        # For now, return a dummy chunk as example
        chunk = {
            "chunk_id": self._generate_chunk_id(metadata, 0),
            "content": content[:500],  # Dummy: just first 500 chars
            "company": metadata.get("company", "unknown"),
            "product_type": metadata.get("product_type", "unknown"),
            "risk_profile": metadata.get("risk_profile", "n/a"),
            "source_file": metadata.get("files", {}).get("markdown", "unknown"),
            "source_path": metadata.get("source_path", "unknown"),
            "chunk_index": 0,
            "total_chunks": 1,
            "is_table": False,
            "doc_id": self._generate_doc_id(metadata.get("source_path", ""))
        }
        chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, content: str) -> List[str]:
        """
        Split content into sentences using NLTK.

        Args:
            content: The text content.

        Returns:
            List of sentences.
        """
        sentences = nltk.sent_tokenize(content)
        return sentences

    def _generate_chunk_id(self, metadata: Dict, chunk_index: int) -> str:
        """
        Generate a unique chunk ID.

        Format: {doc_id}_{chunk_index:04d}

        Args:
            metadata: Document metadata.
            chunk_index: Index of this chunk.

        Returns:
            Unique chunk ID.
        """
        doc_id = self._generate_doc_id(metadata.get("source_path", ""))
        return f"{doc_id}_{chunk_index:04d}"

    def _generate_doc_id(self, source_path: str) -> str:
        """
        Generate a document ID from source path.

        Uses first 12 characters of SHA256 hash.

        Args:
            source_path: The source file path.

        Returns:
            12-character document ID.
        """
        hash_obj = hashlib.sha256(source_path.encode())
        return hash_obj.hexdigest()[:12]

    def chunk_directory(
        self,
        markdown_dir: Path,
        output_path: Path
    ) -> int:
        """
        Chunk all markdown documents in a directory.

        Args:
            markdown_dir: Directory containing markdown files.
            output_path: Path to output JSONL file.

        Returns:
            Number of chunks created.
        """
        markdown_dir = Path(markdown_dir)
        output_path = Path(output_path)

        if not markdown_dir.exists():
            logger.error(f"Markdown directory not found: {markdown_dir}")
            return 0

        # Find all markdown files
        md_files = list(markdown_dir.glob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        all_chunks = []

        for md_file in md_files:
            # Find corresponding metadata file
            meta_file = md_file.with_suffix(".meta.json")
            if not meta_file.exists():
                logger.warning(f"Metadata not found: {meta_file}, skipping {md_file}")
                continue

            # Chunk this document
            chunks = self.chunk_document(md_file, meta_file)
            all_chunks.extend(chunks)

        # Save to JSONL
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        logger.info(f"✓ Saved {len(all_chunks)} chunks to {output_path.name}")
        return len(all_chunks)


def main():
    """Example usage."""
    chunker = DocumentChunker(
        sentences_per_chunk=15,
        overlap_sentences=5,
        min_sentences=3
    )

    num_chunks = chunker.chunk_directory(
        markdown_dir=Path("data/ocr_markdown"),
        output_path=Path("data/chunks/chunks.jsonl")
    )

    print(f"\n✓ Total chunks created: {num_chunks}")


if __name__ == "__main__":
    main()
