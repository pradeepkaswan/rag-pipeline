"""Debug script to test S3 Vectors query."""

import boto3
from rag_pipeline.config.settings import settings
from rag_pipeline.ingestion.embedder import Embedder

# Create clients
embedder = Embedder()
s3vectors = boto3.client(
    "s3vectors",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
)

# Generate query embedding
query_text = "biographical drama about a lost child"
print(f"Generating embedding for: {query_text}")
query_embedding = embedder.embed_text(query_text)
print(f"Embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")

# Try query
print("\nQuerying S3 Vectors...")
try:
    response = s3vectors.query_vectors(
        vectorBucketName=settings.s3_vector_bucket_name,
        indexName=settings.s3_vector_index_name,
        queryVector={"float32": query_embedding},
        topK=5,
        returnDistance=True,
        returnMetadata=True,
    )
    print(f"Response: {response}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
