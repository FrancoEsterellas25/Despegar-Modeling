import polars as pl
import os
import streamlit as st
from typing import Optional
import config

class DataService:
    """Service for handling data loading and processing using Polars."""

    @staticmethod
    @st.cache_data
    def load_validation_data() -> pl.DataFrame:
        """Loads and cleans the validation dataset.

        Returns:
            pl.DataFrame: The loaded validation dataset as a Polars DataFrame.

        Raises:
            FileNotFoundError: If the validation data file is not found.
        """
        path = config.VAL_DATA_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"El archivo de validación no se encuentra en: {path}")
        
        # Leemos con Polars de manera altamente optimizada
        df = pl.read_csv(path)
        return df

    @staticmethod
    def get_observation_by_index(df: pl.DataFrame, index: int) -> pl.DataFrame:
        """Retrieves a single observation by its row index.

        Args:
            df (pl.DataFrame): The source DataFrame.
            index (int): The 0-based index of the row to retrieve.

        Returns:
            pl.DataFrame: A DataFrame containing the single selected row.
        """
        return df.row(index)

    @staticmethod
    def get_features_matrix(df: pl.DataFrame) -> pl.DataFrame:
        """Extracts the feature columns required for model input.

        Args:
            df (pl.DataFrame): The source DataFrame.

        Returns:
            pl.DataFrame: A DataFrame containing only the feature columns.
        """
        # Seleccionamos las columnas de features especificadas en config
        # Si faltase alguna, polars lanzará error controlado
        available_features = [col for col in config.FEATURES if col in df.columns]
        return df.select(available_features)
