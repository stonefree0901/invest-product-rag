"""Embedding generation module for RAG system."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of the sentence-transformers model.
                        Default: "all-MiniLM-L6-v2" (fast, good quality)
                        Other options:
                        - "all-mpnet-base-v2" (slower, better quality)
                        - "BAAI/bge-small-en-v1.5" (SOTA for English)
            batch_size: Batch size for embedding generation.
            device: Device to run on ("cuda", "cpu", or None for auto).
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show progress bar.

        Returns:
            List of embedding vectors.
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def load_chunks(self, chunks_path: Path) -> List[Dict]:
        """
        Load chunks from JSONL file.

        Args:
            chunks_path: Path to chunks.jsonl file.

        Returns:
            List of chunk dictionaries.
        """
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        return chunks

    def generate_chunk_embeddings(
        self,
        chunks: List[Dict],
        content_field: str = "content"
    ) -> List[Dict]:
        """
        Generate embeddings for chunks and add them to the chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries.
            content_field: Field name containing the text to embed.

        Returns:
            List of chunk dictionaries with added "embedding" field.
        """
        texts = [chunk[content_field] for chunk in chunks]
        embeddings = self.generate_embeddings(texts, show_progress=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks

    def save_chunks_with_embeddings(
        self,
        chunks: List[Dict],
        output_path: Path
    ):
        """
        Save chunks with embeddings to JSONL file.

        Args:
            chunks: List of chunk dictionaries with embeddings.
            output_path: Path to output JSONL file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                # Save without embedding to avoid huge file size
                # Embeddings will be stored in Qdrant
                chunk_to_save = {k: v for k, v in chunk.items() if k != "embedding"}
                f.write(json.dumps(chunk_to_save, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension.
        """
        return self.model.get_sentence_embedding_dimension()


def main():
    """Example usage: Generate embeddings for all chunks."""
    # Initialize embedding generator
    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        batch_size=32
    )

    # Load chunks
    chunks_path = Path("data/chunks/chunks.jsonl")
    chunks = embedder.load_chunks(chunks_path)

    # Generate embeddings
    chunks_with_embeddings = embedder.generate_chunk_embeddings(chunks)

    # Save chunks (embeddings will be stored in Qdrant, not in file)
    output_path = Path("data/chunks/chunks_with_embeddings.jsonl")
    embedder.save_chunks_with_embeddings(chunks_with_embeddings, output_path)

    # Print some info
    logger.info(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    logger.info(f"First 10 values of first embedding: {chunks_with_embeddings[0]['embedding'][:10]}")

    # Return chunks with embeddings for indexing
    return chunks_with_embeddings


if __name__ == "__main__":
    chunks_with_embeddings = main()
