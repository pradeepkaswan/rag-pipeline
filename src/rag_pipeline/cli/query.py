"""CLI entry point for querying similar movies from S3 Vectors."""

from __future__ import annotations

import argparse
import json
import time

from rag_pipeline.ingestion.embedder import Embedder
from rag_pipeline.ingestion.s3_vectors_client import S3VectorsClient
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for the query CLI command."""
    parser = argparse.ArgumentParser(
        description="Query S3 Vectors for similar movies using semantic search"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query (e.g., 'action movies with time travel')",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    parser.add_argument(
        "--min-rating",
        type=float,
        help="Minimum rating filter (e.g., 8.0 for rating >= 8.0)",
    )

    parser.add_argument(
        "--max-rating",
        type=float,
        help="Maximum rating filter (e.g., 9.0 for rating <= 9.0)",
    )

    parser.add_argument(
        "--year",
        type=int,
        help="Filter by specific year",
    )

    parser.add_argument(
        "--genre",
        type=str,
        help="Filter by genre (partial match)",
    )

    args = parser.parse_args()

    # Build metadata filter (using MongoDB-style operators)
    metadata_filter = None
    filter_conditions = []

    if args.min_rating is not None or args.max_rating is not None:
        rating_filter = {}
        if args.min_rating is not None:
            rating_filter["$gte"] = args.min_rating
        if args.max_rating is not None:
            rating_filter["$lte"] = args.max_rating
        filter_conditions.append({"rating": rating_filter})

    if args.year is not None:
        filter_conditions.append({"year": {"$eq": float(args.year)}})

    if args.genre is not None:
        # Use $eq for exact match, could also use contains logic if needed
        filter_conditions.append({"genre": {"$eq": args.genre}})

    # Combine filters with AND logic if multiple filters exist
    if filter_conditions:
        if len(filter_conditions) == 1:
            metadata_filter = filter_conditions[0]
        else:
            metadata_filter = {"$and": filter_conditions}

    logger.info("=" * 60)
    logger.info("S3 Vectors Movie Search")
    logger.info("=" * 60)
    logger.info(f"Query: {args.query}")
    logger.info(f"Top K: {args.top_k}")
    if metadata_filter:
        logger.info(f"Metadata Filter: {json.dumps(metadata_filter, indent=2)}")
    logger.info("=" * 60)

    # Generate embedding for query (cold start)
    embedder = Embedder()
    logger.info("Generating query embedding (cold start)...")

    embed_start_cold = time.perf_counter()
    query_embedding = embedder.embed_text(args.query)
    embed_time_cold = (time.perf_counter() - embed_start_cold) * 1000

    # Warm up embedding generation
    embed_start_warm = time.perf_counter()
    _ = embedder.embed_text(args.query)
    embed_time_warm = (time.perf_counter() - embed_start_warm) * 1000

    # Search S3 Vectors (cold start)
    s3_client = S3VectorsClient()
    logger.info("Searching S3 Vectors (cold start)...")

    query_start_cold = time.perf_counter()
    results = s3_client.query_vectors(
        query_vector=query_embedding,
        top_k=args.top_k,
        return_distance=True,
        return_metadata=True,
        metadata_filter=metadata_filter,
    )
    query_time_cold = (time.perf_counter() - query_start_cold) * 1000

    # Warm query
    query_start_warm = time.perf_counter()
    _ = s3_client.query_vectors(
        query_vector=query_embedding,
        top_k=args.top_k,
        return_distance=True,
        return_metadata=True,
        metadata_filter=metadata_filter,
    )
    query_time_warm = (time.perf_counter() - query_start_warm) * 1000

    total_time_cold = embed_time_cold + query_time_cold
    total_time_warm = embed_time_warm + query_time_warm

    # Display results
    logger.info("=" * 60)
    logger.info(f"Found {len(results)} matching movies:")
    logger.info("=" * 60)
    print(f"\n⏱️  Performance Metrics:")
    print(f"   Cold Start (first request):")
    print(f"      Embedding generation: {embed_time_cold:.2f}ms")
    print(f"      S3 Vectors query:     {query_time_cold:.2f}ms")
    print(f"      Total latency:        {total_time_cold:.2f}ms")
    print(f"   ")
    print(f"   Warm Query (subsequent requests):")
    print(f"      Embedding generation: {embed_time_warm:.2f}ms")
    print(f"      S3 Vectors query:     {query_time_warm:.2f}ms")
    print(f"      Total latency:        {total_time_warm:.2f}ms")
    print()

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        distance = result.get("distance", 0)
        similarity = 1 - distance  # Convert distance to similarity score

        print(f"\n{i}. {metadata.get('title', 'Unknown')} ({metadata.get('year', 'N/A')})")
        print(f"   Genre: {metadata.get('genre', 'N/A')}")
        print(f"   Director: {metadata.get('director', 'N/A')}")
        print(f"   Rating: {metadata.get('rating', 'N/A')}/10")
        print(f"   Actors: {metadata.get('actors', 'N/A')}")
        print(f"   Similarity: {similarity:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
