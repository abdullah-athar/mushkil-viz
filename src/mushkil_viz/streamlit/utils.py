import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file for size and format."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (200MB limit)
    if uploaded_file.size > 200 * 1024 * 1024:
        return False, f"File too large ({uploaded_file.size / 1024**2:.1f}MB). Max 200MB."
    
    return True, ""


def load_sample_data():
    """Load available sample data from examples folder."""
    examples_path = Path(__file__).parent.parent.parent.parent / "examples"
    sample_files = {}
    
    if examples_path.exists():
        for csv_file in examples_path.glob("*.csv"):
            name = csv_file.stem.replace("_", " ").title()
            sample_files[csv_file.stem] = {
                "name": name,
                "path": csv_file,
                "size": f"{csv_file.stat().st_size / 1024**2:.1f}MB"
            }
    
    return sample_files


def safe_read_csv(file_path_or_buffer):
    """Safely read CSV with encoding fallback."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            if hasattr(file_path_or_buffer, 'seek'):
                file_path_or_buffer.seek(0)
            return pd.read_csv(file_path_or_buffer, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise e
    
    raise ValueError("Could not decode file with any supported encoding")


def display_error(message: str, details: str = None):
    """Display formatted error message with optional details."""
    st.error(f"❌ {message}")
    if details:
        with st.expander("🔧 Error Details"):
            st.code(details)


def display_success(message: str):
    """Display formatted success message."""
    st.success(f"✅ {message}")


def display_warning(message: str):
    """Display formatted warning message."""
    st.warning(f"⚠️ {message}")


def display_info(message: str):
    """Display formatted info message."""
    st.info(f"ℹ️ {message}")


def show_loading_state(message: str = "Processing..."):
    """Context manager for loading states."""
    return st.spinner(f"🔄 {message}")


def get_file_info(uploaded_file) -> dict:
    """Extract file information for display."""
    return {
        "name": uploaded_file.name,
        "size": f"{uploaded_file.size / 1024**2:.1f} MB",
        "type": uploaded_file.type or "text/csv"
    } 