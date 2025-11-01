"""Benchmark and load testing for S3 Vectors."""

from __future__ import annotations

import argparse
import concurrent.futures
import random
import time
from statistics import mean, median, stdev

from rag_pipeline.ingestion.embedder import Embedder
from rag_pipeline.ingestion.s3_vectors_client import S3VectorsClient
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


# Sample queries for load testing
SAMPLE_QUERIES = [
    "action movies with explosions",
    "romantic comedy set in New York",
    "science fiction about space exploration",
    "horror movies with psychological thriller",
    "biographical drama about famous people",
    "animated family movies for children",
    "crime thriller with detective investigation",
    "historical war movies",
    "superhero action adventure",
    "mystery suspense with plot twists",
]


def single_query_benchmark(embedder: Embedder, s3_client: S3VectorsClient, query_text: str, top_k: int = 5, metadata_filter: dict | None = None) -> dict:
    """
    Run a single query and measure latency.

    Returns:
        Dict with timing metrics.
    """
    # Embedding generation
    embed_start = time.perf_counter()
    query_embedding = embedder.embed_text(query_text)
    embed_time = (time.perf_counter() - embed_start) * 1000

    # S3 Vectors query
    query_start = time.perf_counter()
    results = s3_client.query_vectors(
        query_vector=query_embedding,
        top_k=top_k,
        return_distance=True,
        return_metadata=True,
        metadata_filter=metadata_filter,
    )
    query_time = (time.perf_counter() - query_start) * 1000

    return {
        "query": query_text,
        "embed_time": embed_time,
        "query_time": query_time,
        "total_time": embed_time + query_time,
        "results_count": len(results),
    }


def main():
    """Main entry point for benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark and load test S3 Vectors performance"
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=10,
        help="Number of queries to run (default: 10)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5)",
    )

    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent queries (default: 1 for sequential)",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup queries",
    )

    parser.add_argument(
        "--table",
        action="store_true",
        help="Display detailed table of each query timing",
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
        filter_conditions.append({"genre": {"$eq": args.genre}})

    # Combine filters with AND logic if multiple filters exist
    if filter_conditions:
        if len(filter_conditions) == 1:
            metadata_filter = filter_conditions[0]
        else:
            metadata_filter = {"$and": filter_conditions}

    logger.info("=" * 60)
    logger.info("S3 Vectors Load Test & Benchmark")
    logger.info("=" * 60)
    logger.info(f"Total queries: {args.queries}")
    logger.info(f"Top-K results: {args.top_k}")
    logger.info(f"Concurrency: {args.concurrent}")
    if metadata_filter:
        import json
        logger.info(f"Metadata Filter: {json.dumps(metadata_filter, indent=2)}")
    logger.info("=" * 60)

    # Initialize clients
    embedder = Embedder()
    s3_client = S3VectorsClient()

    # Warmup phase
    if not args.skip_warmup:
        logger.info("Warming up (3 queries)...")
        for i in range(3):
            single_query_benchmark(embedder, s3_client, SAMPLE_QUERIES[0], args.top_k, metadata_filter)
        logger.info("Warmup complete\n")

    # Prepare test queries
    test_queries = []
    for i in range(args.queries):
        test_queries.append(random.choice(SAMPLE_QUERIES))

    # Run benchmark
    logger.info(f"Starting benchmark with {args.queries} queries...")
    overall_start = time.perf_counter()

    if args.concurrent == 1:
        # Sequential execution
        results = []
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Running query {i}/{args.queries}...")
            result = single_query_benchmark(embedder, s3_client, query, args.top_k, metadata_filter)
            results.append(result)
    else:
        # Concurrent execution
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = [
                executor.submit(single_query_benchmark, embedder, s3_client, query, args.top_k, metadata_filter)
                for query in test_queries
            ]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                logger.info(f"Completed query {i}/{args.queries}...")
                results.append(future.result())

    overall_time = (time.perf_counter() - overall_start) * 1000

    # Calculate statistics
    embed_times = [r["embed_time"] for r in results]
    query_times = [r["query_time"] for r in results]
    total_times = [r["total_time"] for r in results]

    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nTest Configuration:")
    print(f"  Total queries: {args.queries}")
    print(f"  Concurrency: {args.concurrent}")
    print(f"  Top-K: {args.top_k}")

    print(f"\n⏱️  Latency Statistics (milliseconds):")
    print(f"\n  Embedding Generation:")
    print(f"    Min:    {min(embed_times):.2f}ms")
    print(f"    Max:    {max(embed_times):.2f}ms")
    print(f"    Mean:   {mean(embed_times):.2f}ms")
    print(f"    Median: {median(embed_times):.2f}ms")
    if len(embed_times) > 1:
        print(f"    StdDev: {stdev(embed_times):.2f}ms")

    print(f"\n  S3 Vectors Query:")
    print(f"    Min:    {min(query_times):.2f}ms")
    print(f"    Max:    {max(query_times):.2f}ms")
    print(f"    Mean:   {mean(query_times):.2f}ms")
    print(f"    Median: {median(query_times):.2f}ms")
    if len(query_times) > 1:
        print(f"    StdDev: {stdev(query_times):.2f}ms")

    print(f"\n  Total (End-to-End):")
    print(f"    Min:    {min(total_times):.2f}ms")
    print(f"    Max:    {max(total_times):.2f}ms")
    print(f"    Mean:   {mean(total_times):.2f}ms")
    print(f"    Median: {median(total_times):.2f}ms")
    if len(total_times) > 1:
        print(f"    StdDev: {stdev(total_times):.2f}ms")

    print(f"\n📊 Throughput Metrics:")
    print(f"  Overall runtime: {overall_time:.2f}ms ({overall_time/1000:.2f}s)")
    print(f"  Queries per second: {(args.queries / (overall_time/1000)):.2f} QPS")
    print(f"  Average latency: {mean(total_times):.2f}ms")

    # Percentiles
    sorted_times = sorted(total_times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 10 else sorted_times[-1]

    print(f"\n📈 Percentiles:")
    print(f"  P50: {p50:.2f}ms")
    print(f"  P95: {p95:.2f}ms")
    print(f"  P99: {p99:.2f}ms")

    # Display detailed table if requested
    if args.table:
        print("\n" + "=" * 100)
        print("DETAILED QUERY TIMINGS")
        print("=" * 100)

        # Table header
        print(f"\n{'#':<5} {'Query':<40} {'Embed (ms)':<12} {'S3Query (ms)':<14} {'Total (ms)':<12}")
        print("-" * 100)

        # Table rows
        for i, result in enumerate(results, 1):
            query_truncated = result['query'][:37] + "..." if len(result['query']) > 40 else result['query']
            print(f"{i:<5} {query_truncated:<40} {result['embed_time']:<12.2f} {result['query_time']:<14.2f} {result['total_time']:<12.2f}")

        print("-" * 100)
        print(f"{'Total:':<5} {'':<40} {sum(embed_times):<12.2f} {sum(query_times):<14.2f} {sum(total_times):<12.2f}")
        print(f"{'Avg:':<5} {'':<40} {mean(embed_times):<12.2f} {mean(query_times):<14.2f} {mean(total_times):<12.2f}")

    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
