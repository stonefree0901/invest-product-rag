"""
Chunking module for splitting markdown documents into smaller chunks.

This module handles:
1. Reading markdown files and their metadata
2. Splitting content into chunks using NLTK sentence tokenization
3. Handling tables as whole chunks
4. Saving chunks to JSONL format
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Union
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

        # 3. Remove image placeholders
        content = self._remove_images(content)

        # 4. Split into chunks
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

    def _is_table_header_line(self, line: str) -> bool:
        """
        Check if a line is a table header.

        A table header line:
        - Starts and ends with '|'
        - Contains multiple pipe separators (at least 2 pipes total)
        - Either has multiple spaces (like "|   |") or has content between pipes

        Args:
            line: The line to check.

        Returns:
            True if this line looks like a table header.
        """
        line = line.strip()
        if not line:
            return False

        # Must start and end with |
        if not (line.startswith('|') and line.endswith('|')):
            return False

        # Must have at least 2 pipes (at least 2 columns)
        pipe_count = line.count('|')
        if pipe_count < 2:
            return False

        # Check if it looks like a table row
        # Split by pipe and check each cell
        cells = [cell.strip() for cell in line.split('|')[1:-1]]  # Remove empty first and last

        # A valid table row has at least 2 cells
        if len(cells) < 2:
            return False

        # Check if cells are either empty (spaces) or contain content
        # This catches both "|   |" and "|  Asset Class |" patterns
        return True

    def _parse_content_blocks(self, content: str) -> List[Dict]:
        """
        Parse content into blocks of text or tables.

        This analyzes line by line to identify table boundaries.
        A table starts with a header line that looks like "|   |" or "|  Asset Class |"

        Args:
            content: The markdown content.

        Returns:
            List of blocks, each with 'type' and 'content' keys.
        """
        blocks = []
        lines = content.split('\n')

        current_block = []
        in_table = False

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this line could be a table header
            is_potential_header = self._is_table_header_line(stripped)

            if not in_table and is_potential_header:
                # Check if next few lines are also table rows
                # This helps distinguish from regular text with pipes
                is_table = False

                # Look ahead to see if there are more table-like lines
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_line = lines[j]
                    if not next_line.strip():
                        # Empty line after potential header - might be a table
                        is_table = True
                        break
                    if self._is_table_header_line(next_line.strip()):
                        # Multiple table-like lines - definitely a table
                        is_table = True
                        break

                if is_table:
                    # Save current text block if exists
                    if current_block:
                        blocks.append({
                            "type": "text",
                            "content": "\n".join(current_block)
                        })
                        current_block = []

                    # Start table block
                    in_table = True
                    current_block = [line]

            elif in_table:
                # Still in table
                current_block.append(line)
                # Check if table ends (empty line or line that doesn't look like a table row)
                if i == len(lines) - 1:
                    # End of file
                    blocks.append({
                        "type": "table",
                        "content": "\n".join(current_block)
                    })
                    current_block = []
                    in_table = False
                elif not stripped:
                    # Empty line marks end of table
                    blocks.append({
                        "type": "table",
                        "content": "\n".join(current_block)
                    })
                    current_block = []
                    in_table = False
                elif not stripped.startswith('|'):
                    # Line doesn't start with | - end of table
                    blocks.append({
                        "type": "table",
                        "content": "\n".join(current_block)
                    })
                    current_block = [line]  # Start new text block with this line
                    in_table = False
            else:
                # Text line
                current_block.append(line)

            i += 1

        # Save remaining content
        if current_block:
            content_type = "table" if in_table else "text"
            blocks.append({
                "type": content_type,
                "content": "\n".join(current_block)
            })

        return blocks

    def _is_list_paragraph(self, paragraph: str) -> bool:
        """
        Check if a paragraph is a list.

        Lists start with bullet points or numbers:
        - * - bullet
        - • - bullet
        - 1. 2. - numbered list

        Args:
            paragraph: The paragraph content.

        Returns:
            True if this paragraph is a list.
        """
        lines = paragraph.strip().split('\n')
        if not lines:
            return False

        # Check if at least 2 lines are list items
        list_item_count = 0
        list_patterns = [r'^-\s', r'^\*\s', r'^•\s', r'^\d+\.\s']

        for line in lines:
            import re
            for pattern in list_patterns:
                if re.match(pattern, line.strip()):
                    list_item_count += 1
                    break

        return list_item_count >= 2

    def _get_surrounding_blocks(self, blocks: List[Dict], table_index: int, context_blocks: int = 1) -> tuple:
        """
        Get text blocks before and after a table.

        Args:
            blocks: List of all blocks.
            table_index: Index of the table block.
            context_blocks: Number of blocks to include as context.

        Returns:
            Tuple of (before_blocks, after_blocks) lists.
        """
        before_blocks = []
        after_blocks = []

        # Get context before (text blocks only)
        for i in range(table_index - 1, max(-1, table_index - context_blocks - 1), -1):
            if i >= 0 and blocks[i]["type"] == "text":
                before_blocks.insert(0, blocks[i])
            else:
                break

        # Get context after (text blocks only)
        for i in range(table_index + 1, min(len(blocks), table_index + context_blocks + 1)):
            if i < len(blocks) and blocks[i]["type"] == "text":
                after_blocks.append(blocks[i])
            else:
                break

        return before_blocks, after_blocks

    def _add_context_to_table_block(self, table_block: Dict, context: tuple) -> str:
        """
        Add context before and after a table block.

        Args:
            table_block: The table block dict.
            context: Tuple of (before_blocks, after_blocks) lists.

        Returns:
            Table with context added.
        """
        before_blocks, after_blocks = context

        parts = []

        # Add context before (take last 2 sentences from each context block)
        if before_blocks:
            context_text = []
            for block in before_blocks:
                sentences = self._split_into_sentences(block["content"])
                # Take last 2 sentences for context
                context_text.extend(sentences[-2:] if len(sentences) > 2 else sentences)
            if context_text:
                parts.append(" ".join(context_text))

        # Add table
        parts.append(table_block["content"].strip())

        # Add context after (take first 2 sentences from each context block)
        if after_blocks:
            context_text = []
            for block in after_blocks:
                sentences = self._split_into_sentences(block["content"])
                # Take first 2 sentences for context
                context_text.extend(sentences[:2] if len(sentences) > 2 else sentences)
            if context_text:
                parts.append(" ".join(context_text))

        return "\n\n".join(parts)

    def _create_overlapping_chunks(self, text_units: list, metadata: Dict, start_index: int) -> List[Dict]:
        """
        Create chunks from text units with overlap.

        Args:
            text_units: List of text units (sentences or list paragraphs).
            metadata: Document metadata.
            start_index: Starting chunk index.

        Returns:
            List of chunk dictionaries.
        """
        chunks = []

        if not text_units:
            return chunks

        chunk_size = self.sentences_per_chunk
        overlap = self.overlap_sentences
        step = chunk_size - overlap

        chunk_index = start_index

        for i in range(0, len(text_units), step):
            # Get text for this chunk
            chunk_text_units = text_units[i:i + chunk_size]

            # Skip if too few units (will be merged later)
            if len(chunk_text_units) < self.min_sentences:
                # Merge with previous chunk if exists
                if chunks:
                    chunks[-1]["content"] += " " + " ".join(chunk_text_units)
                continue

            # Create chunk
            chunk = {
                "chunk_id": self._generate_chunk_id(metadata, chunk_index),
                "content": " ".join(chunk_text_units),
                "company": metadata.get("company", "unknown"),
                "product_type": metadata.get("product_type", "unknown"),
                "risk_profile": metadata.get("risk_profile", "n/a"),
                "source_file": metadata.get("files", {}).get("markdown", "unknown"),
                "source_path": metadata.get("source_path", "unknown"),
                "chunk_index": chunk_index,
                "total_chunks": 0,  # Will update later
                "is_table": False,
                "doc_id": self._generate_doc_id(metadata.get("source_path", ""))
            }
            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def _create_chunks(
        self,
        content: str,
        metadata: Dict
    ) -> List[Dict]:
        """
        Create chunks from content.

        Steps:
        1. Parse content into text blocks and table blocks
        2. Process tables with context
        3. Process text with sentence splitting
        4. Create chunks with overlap
        5. Merge short chunks

        Args:
            content: The markdown content.
            metadata: Document metadata.

        Returns:
            List of chunk dictionaries.
        """
        chunks = []

        # 1. Parse content into blocks (text or tables)
        blocks = self._parse_content_blocks(content)

        # 2. Separate table and text blocks
        table_blocks = []
        text_blocks = []

        for block in blocks:
            if block["type"] == "table":
                table_blocks.append(block)
            else:
                text_blocks.append(block)

        # 3. Process tables with context
        table_chunk_index = 0
        for i, block in enumerate(table_blocks):
            # Get surrounding text blocks as context
            context_blocks = self._get_surrounding_blocks(blocks, i)
            table_with_context = self._add_context_to_table_block(block, context_blocks)

            chunk = {
                "chunk_id": self._generate_chunk_id(metadata, f"table_{table_chunk_index}"),
                "content": table_with_context,
                "company": metadata.get("company", "unknown"),
                "product_type": metadata.get("product_type", "unknown"),
                "risk_profile": metadata.get("risk_profile", "n/a"),
                "source_file": metadata.get("files", {}).get("markdown", "unknown"),
                "source_path": metadata.get("source_path", "unknown"),
                "chunk_index": table_chunk_index,
                "total_chunks": 0,  # Will update later
                "is_table": True,
                "doc_id": self._generate_doc_id(metadata.get("source_path", ""))
            }
            chunks.append(chunk)
            table_chunk_index += 1

        # 4. Process text blocks
        all_text = []
        for block in text_blocks:
            text_content = block["content"].strip()
            if not text_content:
                continue

            # Keep lists together
            if self._is_list_paragraph(text_content):
                all_text.append(text_content)
            else:
                # Split into sentences
                sentences = self._split_into_sentences(text_content)
                all_text.extend(sentences)

        # Create chunks with overlap
        if all_text:
            text_chunks = self._create_overlapping_chunks(all_text, metadata, start_index=table_chunk_index)
            chunks.extend(text_chunks)

        # Update total_chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

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

    def _generate_chunk_id(self, metadata: Dict, chunk_index: Union[int, str]) -> str:
        """
        Generate a unique chunk ID.

        Format: {doc_id}_{chunk_index:04d} for int, {doc_id}_{chunk_index} for str

        Args:
            metadata: Document metadata.
            chunk_index: Index of this chunk (int or str).

        Returns:
            Unique chunk ID.
        """
        doc_id = self._generate_doc_id(metadata.get("source_path", ""))
        if isinstance(chunk_index, int):
            return f"{doc_id}_{chunk_index:04d}"
        else:
            return f"{doc_id}_{chunk_index}"

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
