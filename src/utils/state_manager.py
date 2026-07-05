import streamlit as st
import polars as pl
from typing import Optional

class StateManager:
    """State Management class encapsulating Streamlit session state access."""

    @staticmethod
    def get_selected_model() -> str:
        """Retrieves the currently selected model type.

        Returns:
            str: Either 'Stacking' or 'MoE'. Defaults to 'Stacking'.
        """
        if "selected_model" not in st.session_state:
            st.session_state["selected_model"] = "Stacking"
        return st.session_state["selected_model"]

    @staticmethod
    def set_selected_model(model_type: str) -> None:
        """Sets the selected model type.

        Args:
            model_type (str): The model type ('Stacking' or 'MoE').
        """
        st.session_state["selected_model"] = model_type

    @staticmethod
    def get_selected_index() -> int:
        """Retrieves the currently selected row index from the validation set.

        Returns:
            int: The selected row index. Defaults to 0.
        """
        if "selected_index" not in st.session_state:
            st.session_state["selected_index"] = 0
        return st.session_state["selected_index"]

    @staticmethod
    def set_selected_index(index: int) -> None:
        """Sets the selected row index from the validation set.

        Args:
            index (int): The selected row index.
        """
        st.session_state["selected_index"] = index
