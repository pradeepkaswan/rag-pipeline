"""Detailed latency breakdown test."""

import time
import boto3
from rag_pipeline.config.settings import settings
from rag_pipeline.ingestion.embedder import Embedder

print("Testing S3 Vectors latency with detailed breakdown...\n")

# Test 1: Embedding generation
print("1. Testing OpenAI embedding generation...")
embedder = Embedder()
query_text = "biographical drama about a lost child"

embed_times = []
for i in range(3):
    start = time.perf_counter()
    query_embedding = embedder.embed_text(query_text)
    elapsed = (time.perf_counter() - start) * 1000
    embed_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.2f}ms")

print(f"   Average: {sum(embed_times)/len(embed_times):.2f}ms\n")

# Test 2: S3 Vectors query
print("2. Testing S3 Vectors query...")
s3vectors = boto3.client(
    "s3vectors",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)

query_times = []
for i in range(3):
    start = time.perf_counter()
    response = s3vectors.query_vectors(
        vectorBucketName=settings.s3_vector_bucket_name,
        indexName=settings.s3_vector_index_name,
        queryVector={"float32": query_embedding},
        topK=5,
        returnDistance=True,
        returnMetadata=True,
    )
    elapsed = (time.perf_counter() - start) * 1000
    query_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.2f}ms (returned {len(response.get('vectors', []))} results)")

print(f"   Average: {sum(query_times)/len(query_times):.2f}ms\n")

# Test 3: Just the HTTP round trip to S3 Vectors (list indexes - lightweight operation)
print("3. Testing S3 Vectors API round-trip (list-indexes - lightweight call)...")
roundtrip_times = []
for i in range(3):
    start = time.perf_counter()
    s3vectors.list_indexes(vectorBucketName=settings.s3_vector_bucket_name)
    elapsed = (time.perf_counter() - start) * 1000
    roundtrip_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.2f}ms")

print(f"   Average: {sum(roundtrip_times)/len(roundtrip_times):.2f}ms\n")

print("=" * 60)
print("Summary:")
print(f"  OpenAI embedding avg: {sum(embed_times)/len(embed_times):.2f}ms")
print(f"  S3 Vectors query avg: {sum(query_times)/len(query_times):.2f}ms")
print(f"  S3 API baseline avg: {sum(roundtrip_times)/len(roundtrip_times):.2f}ms")
print(f"  Total (embedding + query): {sum(embed_times)/len(embed_times) + sum(query_times)/len(query_times):.2f}ms")
