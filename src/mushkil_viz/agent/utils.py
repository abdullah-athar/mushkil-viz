"""Utility functions for the multi-agent system."""

import pandas as pd
from typing import Tuple
from .router_agent import RouterAgent


def load_data(uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV with clean summary statistics.

    Args:
        uploaded_file: The uploaded CSV file

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The loaded dataframe and its summary
    """
    df = pd.read_csv(uploaded_file)

    # Create clean summary without mixed types
    summary = df.describe(include="all")
    summary_clean = summary.fillna("N/A").astype(
        str
    )  # Convert all to strings to avoid Arrow issues

    return df, summary_clean


def run_analysis(df: pd.DataFrame, user_query: str) -> dict:
    """Main entry point for multi-agent analysis.

    Args:
        df: The pandas DataFrame to analyze
        user_query: The user's analysis query

    Returns:
        dict: The analysis result
    """
    router = RouterAgent()
    router.set_dataframe(df)
    return router.process_query(user_query)
