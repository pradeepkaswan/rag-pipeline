"""S3 Vectors client for managing vector buckets, indexes, and vector operations."""

from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from rag_pipeline.config.settings import settings
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class S3VectorsClient:
    """Client for interacting with AWS S3 Vectors service."""

    def __init__(self):
        """Initialize S3 Vectors client with credentials from settings."""
        session_kwargs = {
            "region_name": settings.aws_region,
        }

        if settings.aws_access_key_id and settings.aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        self.client = boto3.client("s3vectors", **session_kwargs)
        self.bucket_name = settings.s3_vector_bucket_name
        self.index_name = settings.s3_vector_index_name

    def create_vector_bucket(self) -> bool:
        """
        Create a vector bucket if it doesn't exist.

        Returns:
            True if bucket was created or already exists, False otherwise.
        """
        try:
            # Check if bucket exists
            self.client.get_vector_bucket(vectorBucketName=self.bucket_name)
            logger.info(f"Vector bucket '{self.bucket_name}' already exists")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchVectorBucket":
                # Bucket doesn't exist, create it
                try:
                    self.client.create_vector_bucket(vectorBucketName=self.bucket_name)
                    logger.info(f"Created vector bucket '{self.bucket_name}'")
                    return True
                except ClientError as create_error:
                    logger.error(f"Failed to create vector bucket: {create_error}")
                    return False
            else:
                logger.error(f"Error checking vector bucket: {e}")
                return False

    def create_index(
        self,
        dimension: int = 1536,
        distance_metric: str = "cosine",
    ) -> bool:
        """
        Create a vector index if it doesn't exist.

        Args:
            dimension: Vector dimension size (1536 for text-embedding-3-small).
            distance_metric: Distance metric to use ('cosine' or 'euclidean').

        Returns:
            True if index was created or already exists, False otherwise.
        """
        try:
            # Check if index exists
            self.client.get_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
            )
            logger.info(f"Vector index '{self.index_name}' already exists")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ["NoSuchIndex", "ResourceNotFoundException", "NotFoundException"]:
                # Index doesn't exist, create it
                try:
                    self.client.create_index(
                        vectorBucketName=self.bucket_name,
                        indexName=self.index_name,
                        dimension=dimension,
                        distanceMetric=distance_metric,
                        dataType="float32",
                    )
                    logger.info(
                        f"Created vector index '{self.index_name}' with dimension {dimension} "
                        f"and {distance_metric} distance metric"
                    )
                    return True
                except ClientError as create_error:
                    logger.error(f"Failed to create vector index: {create_error}")
                    return False
            else:
                logger.error(f"Error checking vector index: {e}")
                return False

    def put_vectors(self, vectors: list[dict]) -> bool:
        """
        Insert vectors into the index.

        Args:
            vectors: List of vector dictionaries with 'key', 'data', and 'metadata' fields.
                    Example: [
                        {
                            "key": "movie_1",
                            "data": {"float32": [0.1, 0.2, ...]},
                            "metadata": {"title": "Movie Title", "genre": "Action"}
                        }
                    ]

        Returns:
            True if vectors were successfully inserted, False otherwise.
        """
        try:
            self.client.put_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                vectors=vectors,
            )
            logger.info(f"Successfully inserted {len(vectors)} vectors")
            return True
        except ClientError as e:
            logger.error(f"Failed to insert vectors: {e}")
            return False

    def query_vectors(
        self,
        query_vector: list[float],
        top_k: int = 10,
        return_distance: bool = True,
        return_metadata: bool = True,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Query vectors for similarity search.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of top results to return.
            return_distance: Whether to return distance scores.
            return_metadata: Whether to return metadata.
            metadata_filter: Optional metadata filter for the query.

        Returns:
            List of matching vectors with their keys, distances, and metadata.
        """
        try:
            query_params = {
                "vectorBucketName": self.bucket_name,
                "indexName": self.index_name,
                "queryVector": {"float32": query_vector},
                "topK": top_k,
                "returnDistance": return_distance,
                "returnMetadata": return_metadata,
            }

            if metadata_filter:
                query_params["filter"] = metadata_filter

            response = self.client.query_vectors(**query_params)
            logger.info(f"Query returned {len(response.get('vectors', []))} matches")
            return response.get("vectors", [])
        except ClientError as e:
            logger.error(f"Failed to query vectors: {e}")
            return []

    def delete_index(self) -> bool:
        """
        Delete the vector index.

        Returns:
            True if index was deleted successfully, False otherwise.
        """
        try:
            self.client.delete_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
            )
            logger.info(f"Deleted vector index '{self.index_name}'")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ["NoSuchIndex", "ResourceNotFoundException"]:
                logger.info(f"Index '{self.index_name}' does not exist, nothing to delete")
                return True
            else:
                logger.error(f"Failed to delete index: {e}")
                return False

    def delete_vector_bucket(self) -> bool:
        """
        Delete the vector bucket (and all its indexes).

        Returns:
            True if bucket was deleted successfully, False otherwise.
        """
        try:
            self.client.delete_vector_bucket(vectorBucketName=self.bucket_name)
            logger.info(f"Deleted vector bucket '{self.bucket_name}'")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchVectorBucket":
                logger.info(f"Bucket '{self.bucket_name}' does not exist, nothing to delete")
                return True
            else:
                logger.error(f"Failed to delete bucket: {e}")
                return False

    def setup(self, dimension: int = 1536) -> bool:
        """
        Setup vector bucket and index (convenience method).

        Args:
            dimension: Vector dimension size.

        Returns:
            True if setup was successful, False otherwise.
        """
        if not self.create_vector_bucket():
            return False

        if not self.create_index(dimension=dimension):
            return False

        logger.info("S3 Vectors setup completed successfully")
        return True
