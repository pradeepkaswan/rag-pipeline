"""Data processor for reading and preparing IMDB movie dataset for embedding."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from rag_pipeline.utils.logger import get_logger

logger = get_logger(__name__)


class IMDBDataProcessor:
    """Processor for IMDB movie dataset CSV."""

    def __init__(self, csv_path: str | Path):
        """
        Initialize data processor.

        Args:
            csv_path: Path to the IMDB CSV file.
        """
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame | None = None

    def load_data(self) -> bool:
        """
        Load CSV data into pandas DataFrame.

        Returns:
            True if data was loaded successfully, False otherwise.
        """
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} movies from {self.csv_path}")
            logger.info(f"Columns: {', '.join(self.df.columns.tolist())}")
            return True
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return False

    def prepare_text_for_embedding(self, row: pd.Series) -> str:
        """
        Create a text representation of a movie for embedding.

        Combines title, genre, and description into a simple text string
        for semantic search.

        Args:
            row: DataFrame row representing a movie.

        Returns:
            Formatted text string for embedding.
        """
        # Handle missing values
        def safe_str(val):
            return str(val) if pd.notna(val) else ""

        title = safe_str(row['Title'])
        genre = safe_str(row['Genre'])
        description = safe_str(row['Description'])

        return f"{title}. {genre}. {description}"

    def prepare_metadata(self, row: pd.Series) -> dict:
        """
        Extract metadata from a movie row.

        S3 Vectors allows a maximum of 10 metadata keys, so we select the most important fields.

        Args:
            row: DataFrame row representing a movie.

        Returns:
            Dictionary of metadata fields (max 10 keys).
        """

        def safe_value(val):
            """Convert value to JSON-serializable type."""
            if pd.isna(val):
                return None
            if isinstance(val, (int, float)):
                return float(val) if not pd.isna(val) else None
            return str(val)

        # Select top 10 most important fields for metadata
        metadata = {
            "title": safe_value(row.get("Title")),
            "genre": safe_value(row.get("Genre")),
            "director": safe_value(row.get("Director")),
            "year": safe_value(row.get("Year")),
            "rating": safe_value(row.get("Rating")),
            "actors": safe_value(row.get("Actors")),
            "runtime": safe_value(row.get("Runtime (Minutes)")),
            "votes": safe_value(row.get("Votes")),
            "revenue": safe_value(row.get("Revenue (Millions)")),
            "metascore": safe_value(row.get("Metascore")),
        }

        # Remove None values
        return {k: v for k, v in metadata.items() if v is not None}

    def get_batch(self, start_idx: int = 0, batch_size: int = 100) -> list[dict]:
        """
        Get a batch of movies for processing.

        Args:
            start_idx: Starting index for the batch.
            batch_size: Number of movies to include in the batch.

        Returns:
            List of dictionaries with 'text' and 'metadata' for each movie.
        """
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return []

        end_idx = min(start_idx + batch_size, len(self.df))
        batch = []

        for idx in range(start_idx, end_idx):
            row = self.df.iloc[idx]
            batch.append(
                {
                    "key": f"movie_{row['Rank']}",
                    "text": self.prepare_text_for_embedding(row),
                    "metadata": self.prepare_metadata(row),
                }
            )

        return batch

    def get_total_count(self) -> int:
        """
        Get total number of movies in the dataset.

        Returns:
            Number of movies.
        """
        return len(self.df) if self.df is not None else 0
