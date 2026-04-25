"""Script to generate embeddings and index them in Qdrant."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.embedding import EmbeddingGenerator
from retrieval.qa_index import QdrantIndexer
from qdrant_client.models import Distance

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate embeddings and index them in Qdrant."""
    logger.info("=" * 60)
    logger.info("Starting Embedding and Indexing Pipeline")
    logger.info("=" * 60)

    # Step 1: Initialize embedding generator
    logger.info("\n[Step 1/4] Initializing embedding generator...")
    embedder = EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",  # Fast, good quality
        batch_size=32
    )
    logger.info(f"Model loaded: {embedder.model_name}")
    logger.info(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # Step 2: Load chunks
    logger.info("\n[Step 2/4] Loading chunks...")
    chunks_path = Path("data/chunks/chunks.jsonl")
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return

    chunks = embedder.load_chunks(chunks_path)
    logger.info(f"Loaded {len(chunks)} chunks")

    # Step 3: Generate embeddings
    logger.info("\n[Step 3/4] Generating embeddings...")
    chunks_with_embeddings = embedder.generate_chunk_embeddings(chunks)
    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")

    # Save chunks (without embeddings - they're in memory for Qdrant)
    output_path = Path("data/chunks/chunks_indexed.jsonl")
    embedder.save_chunks_with_embeddings(chunks_with_embeddings, output_path)

    # Step 4: Index in Qdrant
    logger.info("\n[Step 4/4] Indexing in Qdrant...")
    indexer = QdrantIndexer(
        collection_name="uk_investment_rag",
        persistence_path=Path("data/qdrant_storage"),
        in_memory=False
    )

    # Create collection
    indexer.create_collection(
        vector_size=embedder.get_embedding_dimension(),
        distance=Distance.COSINE
    )

    # Index chunks
    indexer.index_chunks(chunks_with_embeddings, batch_size=100)

    # Get collection info
    info = indexer.get_collection_info()
    logger.info(f"\n{'=' * 60}")
    logger.info("Indexing Complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Collection name: uk_investment_rag")
    logger.info(f"Vector size: {embedder.get_embedding_dimension()}")
    logger.info(f"Distance metric: COSINE")
    logger.info(f"Total chunks indexed: {len(chunks_with_embeddings)}")

    # Test search
    logger.info("\n" + "=" * 60)
    logger.info("Testing Search Functionality")
    logger.info("=" * 60)

    test_queries = [
        "What is the risk profile of Barclays Adventurous Fund?",
        "What are the ongoing charges for Lloyds Balanced Fund?",
        "What is the minimum investment amount for SIPP?"
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        query_embedding = embedder.generate_embeddings([query])[0]
        results = indexer.search(query_embedding, limit=2)

        for i, result in enumerate(results):
            logger.info(f"\n  Result {i+1} (score: {result['score']:.4f}):")
            logger.info(f"    Chunk ID: {result['chunk_id']}")
            logger.info(f"    Company: {result['payload']['company']}")
            logger.info(f"    Product: {result['payload']['product_type']}")
            logger.info(f"    Is Table: {result['payload']['is_table']}")
            logger.info(f"    Content: {result['payload']['content'][:150]}...")

    logger.info("\n" + "=" * 60)
    logger.info("Embedding and Indexing Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("\nYou can now use Jupyter notebooks for retrieval experiments.")
    logger.info("Example notebook location: notebooks/retrieval_experiments.ipynb")

    return indexer


if __name__ == "__main__":
    indexer = main()
