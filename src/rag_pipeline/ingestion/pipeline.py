"""Ingestion pipeline for processing and uploading movie embeddings to S3 Vectors."""

from __future__ import annotations

from pathlib import Path

from rag_pipeline.ingestion.data_processor import IMDBDataProcessor
from rag_pipeline.ingestion.embedder import Embedder
from rag_pipeline.ingestion.s3_vectors_client import S3VectorsClient
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """Pipeline for ingesting movie data into S3 Vectors."""

    def __init__(self, csv_path: str | Path, batch_size: int = 50):
        """
        Initialize ingestion pipeline.

        Args:
            csv_path: Path to the IMDB CSV file.
            batch_size: Number of movies to process in each batch.
        """
        self.data_processor = IMDBDataProcessor(csv_path)
        self.embedder = Embedder()
        self.s3_client = S3VectorsClient()
        self.batch_size = batch_size

    def setup_infrastructure(self) -> bool:
        """
        Setup S3 Vectors bucket and index.

        Returns:
            True if setup was successful, False otherwise.
        """
        logger.info("Setting up S3 Vectors infrastructure...")
        return self.s3_client.setup(dimension=self.embedder.dimension)

    def run(self) -> bool:
        """
        Run the complete ingestion pipeline.

        Returns:
            True if ingestion completed successfully, False otherwise.
        """
        # Load data
        logger.info("Loading movie data...")
        if not self.data_processor.load_data():
            logger.error("Failed to load data")
            return False

        # Setup infrastructure
        if not self.setup_infrastructure():
            logger.error("Failed to setup S3 Vectors infrastructure")
            return False

        # Process and upload in batches
        total_movies = self.data_processor.get_total_count()
        logger.info(f"Starting ingestion of {total_movies} movies...")

        successful_batches = 0
        failed_batches = 0

        for start_idx in range(0, total_movies, self.batch_size):
            batch_num = start_idx // self.batch_size + 1
            total_batches = (total_movies + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}...")

            try:
                # Get batch of movies
                movie_batch = self.data_processor.get_batch(start_idx, self.batch_size)

                if not movie_batch:
                    logger.warning(f"Empty batch at index {start_idx}")
                    continue

                # Extract texts for embedding
                texts = [movie["text"] for movie in movie_batch]

                # Generate embeddings
                logger.info(f"Generating embeddings for {len(texts)} movies...")
                embeddings = self.embedder.embed_batch(texts, batch_size=len(texts))

                # Prepare vectors for S3
                vectors = []
                for movie, embedding in zip(movie_batch, embeddings):
                    vectors.append(
                        {
                            "key": movie["key"],
                            "data": {"float32": embedding},
                            "metadata": movie["metadata"],
                        }
                    )

                # Upload to S3 Vectors
                logger.info(f"Uploading {len(vectors)} vectors to S3...")
                if self.s3_client.put_vectors(vectors):
                    successful_batches += 1
                    logger.info(f"Batch {batch_num} completed successfully")
                else:
                    failed_batches += 1
                    logger.error(f"Batch {batch_num} failed to upload")

            except Exception as e:
                failed_batches += 1
                logger.error(f"Error processing batch {batch_num}: {e}")
                continue

        # Summary
        logger.info("=" * 60)
        logger.info("Ingestion Summary:")
        logger.info(f"  Total movies: {total_movies}")
        logger.info(f"  Successful batches: {successful_batches}")
        logger.info(f"  Failed batches: {failed_batches}")
        logger.info("=" * 60)

        return failed_batches == 0
