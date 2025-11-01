"""CLI entry point for ingesting movie data into S3 Vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline.ingestion.pipeline import IngestionPipeline
from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for the ingest CLI command."""
    parser = argparse.ArgumentParser(
        description="Ingest IMDB movie data into S3 Vectors for semantic search"
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/imdb_movie_dataset.csv",
        help="Path to the IMDB CSV file (default: data/imdb_movie_dataset.csv)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of movies to process in each batch (default: 50)",
    )

    args = parser.parse_args()

    # Resolve CSV path
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 1

    logger.info("=" * 60)
    logger.info("IMDB Movie Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"CSV Path: {csv_path}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info("=" * 60)

    # Run pipeline
    pipeline = IngestionPipeline(csv_path, batch_size=args.batch_size)

    try:
        success = pipeline.run()
        if success:
            logger.info("Ingestion completed successfully!")
            return 0
        else:
            logger.error("Ingestion completed with errors")
            return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
