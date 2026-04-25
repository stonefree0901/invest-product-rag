"""Qdrant indexing module for storing chunk embeddings."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QdrantIndexer:
    """Manage Qdrant vector database for RAG system."""

    def __init__(
        self,
        collection_name: str = "uk_investment_rag",
        persistence_path: Optional[Path] = None,
        host: str = "localhost",
        port: int = 6333,
        in_memory: bool = False
    ):
        """
        Initialize Qdrant client.

        Args:
            collection_name: Name of the Qdrant collection.
            persistence_path: Path to store Qdrant data on disk (for persistent mode).
                            If provided, Qdrant will save data to this location.
            host: Qdrant server host (for remote mode).
            port: Qdrant server port (for remote mode).
            in_memory: If True, use in-memory Qdrant (no persistence).
                       If False and persistence_path is provided, use persistent storage.
                       If False and no persistence_path, connect to remote Qdrant server.
        """
        self.collection_name = collection_name

        if in_memory:
            # Use in-memory Qdrant for development (data lost on restart)
            logger.info("Initializing in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
        elif persistence_path:
            # Use persistent Qdrant on local disk
            persistence_path = Path(persistence_path)
            persistence_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing persistent Qdrant client at {persistence_path}")
            self.client = QdrantClient(path=str(persistence_path))
        else:
            # Connect to remote Qdrant server
            logger.info(f"Connecting to remote Qdrant at {host}:{port}")
            self.client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        vector_size: int,
        distance: Distance = Distance.COSINE
    ):
        """
        Create a new Qdrant collection.

        Args:
            vector_size: Dimension of the vectors.
            distance: Distance metric to use (COSINE, EUCLID, DOT).
        """
        # Check if collection already exists
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name in collection_names:
            logger.warning(f"Collection '{self.collection_name}' already exists. Recreating...")
            self.client.delete_collection(self.collection_name)

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        logger.info(f"Created collection '{self.collection_name}' with vector_size={vector_size}, distance={distance}")

    def index_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 100
    ):
        """
        Index chunks with embeddings into Qdrant.

        Args:
            chunks: List of chunk dictionaries with "embedding" field.
            batch_size: Number of chunks to upload per batch.
        """
        logger.info(f"Indexing {len(chunks)} chunks into Qdrant...")

        # Prepare points (use integer IDs for Qdrant, store chunk_id in payload)
        points = []
        for idx, chunk in enumerate(chunks):
            point = PointStruct(
                id=idx,  # Use integer ID instead of string
                vector=chunk["embedding"],
                payload={
                    "chunk_id": chunk["chunk_id"],  # Store original chunk_id in payload
                    "content": chunk["content"],
                    "company": chunk.get("company", "unknown"),
                    "product_type": chunk.get("product_type", "unknown"),
                    "risk_profile": chunk.get("risk_profile", "n/a"),
                    "source_file": chunk.get("source_file", "unknown"),
                    "source_path": chunk.get("source_path", "unknown"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 0),
                    "is_table": chunk.get("is_table", False),
                    "doc_id": chunk.get("doc_id", "")
                }
            )
            points.append(point)

        # Upload in batches
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        logger.info(f"Successfully indexed {len(points)} chunks")

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Embedding vector of the query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score (0-1).
            filters: Metadata filters (e.g., {"company": "barclays"}).

        Returns:
            List of search results with score and payload.
        """
        # Build filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        # Search
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold
        )
        results = search_result.points

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.payload.get("chunk_id", result.id),  # Use chunk_id from payload
                "score": result.score,
                "payload": result.payload
            })

        return formatted_results

    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.

        Returns:
            Collection information dictionary.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")


def main():
    """Example usage: Create index from chunks with embeddings."""
    from src.retrieval.embedding import EmbeddingGenerator

    # Initialize embedding generator
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

    # Load and generate embeddings
    chunks_path = Path("data/chunks/chunks.jsonl")
    chunks = embedder.load_chunks(chunks_path)
    chunks_with_embeddings = embedder.generate_chunk_embeddings(chunks)

    # Initialize Qdrant indexer
    indexer = QdrantIndexer(
        collection_name="uk_investment_rag",
        in_memory=True
    )

    # Create collection
    indexer.create_collection(
        vector_size=embedder.get_embedding_dimension(),
        distance=Distance.COSINE
    )

    # Index chunks
    indexer.index_chunks(chunks_with_embeddings)

    # Get collection info
    info = indexer.get_collection_info()
    logger.info(f"Collection info: {info}")

    # Test search
    logger.info("\nTesting search with sample query...")
    query = "What is the risk profile of Barclays Adventurous Fund?"
    query_embedding = embedder.generate_embeddings([query])[0]
    results = indexer.search(query_embedding, limit=3)

    logger.info(f"Search results for: '{query}'")
    for i, result in enumerate(results):
        logger.info(f"\nResult {i+1} (score: {result['score']:.4f}):")
        logger.info(f"  Chunk ID: {result['chunk_id']}")
        logger.info(f"  Company: {result['payload']['company']}")
        logger.info(f"  Content preview: {result['payload']['content'][:200]}...")

    return indexer


if __name__ == "__main__":
    indexer = main()
