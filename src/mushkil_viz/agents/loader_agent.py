"""
Loader agent for dataset inspection and schema inference.

This agent is responsible for:
- Loading datasets from various formats (CSV, Parquet, SQL, etc.)
- Inferring schema information
- Extracting sample data
- Computing basic statistics
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .base_agent import BaseAgent
from ..schema import (
    WorkflowState, DataSpec, ColumnSchema, DataType, DatasetFormat
)


class LoaderAgent(BaseAgent):
    """
    Agent responsible for loading and inspecting datasets.
    
    This agent takes a dataset URI and format, loads the data,
    infers the schema, and returns a comprehensive DataSpec object
    that describes the dataset structure and content.
    """
    
    def __init__(self, **kwargs):
        """Initialize the loader agent."""
        super().__init__(**kwargs)
        self.supported_formats = {
            DatasetFormat.CSV: self._load_csv,
            DatasetFormat.PARQUET: self._load_parquet,
            DatasetFormat.EXCEL: self._load_excel,
            DatasetFormat.JSON: self._load_json,
            DatasetFormat.SQL: self._load_sql
        }
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Load and inspect the dataset, updating the state with DataSpec.
        
        Args:
            state: Current workflow state with dataset_uri and dataset_format
            
        Returns:
            Updated workflow state with data_spec populated
        """
        self.log_info(f"Loading dataset from: {state.dataset_uri}")
        
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for loader agent")
            
            # Load the dataset
            df = self._load_dataset(state.dataset_uri, state.dataset_format)
            
            # Create DataSpec
            data_spec = self._create_data_spec(df, state.dataset_uri, state.dataset_format)
            
            # Update state
            state.data_spec = data_spec
            self.log_info(f"Successfully loaded dataset with {data_spec.row_count} rows and {data_spec.column_count} columns")
            
        except Exception as e:
            self.log_error(f"Failed to load dataset: {e}")
            raise
        
        return state
    
    def validate_state(self, state: WorkflowState) -> bool:
        """Validate that state has required dataset information."""
        return (
            hasattr(state, 'dataset_uri') and 
            state.dataset_uri and 
            hasattr(state, 'dataset_format') and 
            state.dataset_format in self.supported_formats
        )
    
    def _load_dataset(self, uri: str, format: DatasetFormat) -> pd.DataFrame:
        """
        Load dataset based on format.
        
        Args:
            uri: Dataset URI or file path
            format: Dataset format
            
        Returns:
            Loaded pandas DataFrame
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        loader_func = self.supported_formats[format]
        return loader_func(uri)
    
    def _load_csv(self, uri: str) -> pd.DataFrame:
        """Load CSV file."""
        try:
            return pd.read_csv(uri)
        except Exception as e:
            self.log_error(f"Failed to load CSV from {uri}: {e}")
            raise
    
    def _load_parquet(self, uri: str) -> pd.DataFrame:
        """Load Parquet file."""
        try:
            return pd.read_parquet(uri)
        except Exception as e:
            self.log_error(f"Failed to load Parquet from {uri}: {e}")
            raise
    
    def _load_excel(self, uri: str) -> pd.DataFrame:
        """Load Excel file."""
        try:
            return pd.read_excel(uri)
        except Exception as e:
            self.log_error(f"Failed to load Excel from {uri}: {e}")
            raise
    
    def _load_json(self, uri: str) -> pd.DataFrame:
        """Load JSON file."""
        try:
            return pd.read_json(uri)
        except Exception as e:
            self.log_error(f"Failed to load JSON from {uri}: {e}")
            raise
    
    def _load_sql(self, uri: str) -> pd.DataFrame:
        """Load data from SQL database."""
        # This is a simplified implementation
        # In practice, you'd need to parse connection strings and handle different databases
        try:
            # For now, assume it's a SQLite file
            import sqlite3
            conn = sqlite3.connect(uri)
            # You'd need to specify the table name or query
            # For now, just get the first table
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if not tables.empty:
                table_name = tables.iloc[0]['name']
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
                return df
            else:
                raise ValueError("No tables found in SQLite database")
        except Exception as e:
            self.log_error(f"Failed to load SQL from {uri}: {e}")
            raise
    
    def _create_data_spec(self, df: pd.DataFrame, uri: str, format: DatasetFormat) -> DataSpec:
        """
        Create DataSpec from loaded DataFrame.
        
        Args:
            df: Loaded pandas DataFrame
            uri: Original dataset URI
            format: Dataset format
            
        Returns:
            Complete DataSpec object
        """
        # Infer schema
        schema = self._infer_schema(df)
        
        # Get sample rows
        sample_rows = self._get_sample_rows(df)
        
        # Calculate memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Get file size if applicable
        file_size_mb = None
        if os.path.exists(uri):
            file_size_mb = os.path.getsize(uri) / (1024 * 1024)
        
        return DataSpec(
            uri=uri,
            format=format,
            row_count=len(df),
            column_count=len(df.columns),
            column_schema=schema,
            sample_rows=sample_rows,
            memory_usage_mb=memory_usage_mb,
            file_size_mb=file_size_mb
        )
    
    def _infer_schema(self, df: pd.DataFrame) -> List[ColumnSchema]:
        """
        Infer schema information for each column.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of ColumnSchema objects
        """
        schema = []
        
        for col_name in df.columns:
            col_data = df[col_name]
            
            # Determine data type
            dtype = self._infer_data_type(col_data)
            
            # Calculate null count
            null_count = col_data.isnull().sum()
            
            # Calculate unique count
            unique_count = col_data.nunique()
            
            # Get sample values
            sample_values = self._get_column_sample_values(col_data)
            
            # Generate description
            description = self._generate_column_description(col_name, dtype, null_count, unique_count)
            
            schema.append(ColumnSchema(
                name=col_name,
                dtype=dtype,
                null_count=null_count,
                unique_count=unique_count,
                sample_values=sample_values,
                description=description
            ))
        
        return schema
    
    def _infer_data_type(self, col_data: pd.Series) -> DataType:
        """
        Infer the data type of a column.
        
        Args:
            col_data: Column data as pandas Series
            
        Returns:
            Inferred DataType
        """
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return DataType.DATETIME
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(col_data):
            return DataType.BOOLEAN
        
        # Check for numeric types
        if pd.api.types.is_numeric_dtype(col_data):
            if pd.api.types.is_integer_dtype(col_data):
                return DataType.INTEGER
            else:
                return DataType.FLOAT
        
        # Check for categorical (low cardinality string)
        if pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            unique_ratio = col_data.nunique() / len(col_data)
            if unique_ratio < 0.5:  # Less than 50% unique values
                return DataType.CATEGORICAL
            else:
                return DataType.STRING
        
        return DataType.UNKNOWN
    
    def _get_column_sample_values(self, col_data: pd.Series, max_samples: int = 5) -> List[Any]:
        """
        Get sample values from a column.
        
        Args:
            col_data: Column data
            max_samples: Maximum number of samples to return
            
        Returns:
            List of sample values
        """
        # Get non-null values
        non_null_data = col_data.dropna()
        
        if len(non_null_data) == 0:
            return []
        
        # Get unique values
        unique_values = non_null_data.unique()
        
        # Take up to max_samples
        samples = unique_values[:max_samples].tolist()
        
        # Convert numpy types to Python types for JSON serialization
        return [self._convert_numpy_type(val) for val in samples]
    
    def _convert_numpy_type(self, val: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif isinstance(val, np.bool_):
            return bool(val)
        elif pd.isna(val):
            return None
        else:
            return val
    
    def _get_sample_rows(self, df: pd.DataFrame, max_rows: int = 5) -> List[Dict[str, Any]]:
        """
        Get sample rows from the DataFrame.
        
        Args:
            df: DataFrame to sample
            max_rows: Maximum number of rows to return
            
        Returns:
            List of dictionaries representing sample rows
        """
        sample_df = df.head(max_rows)
        return sample_df.to_dict('records')
    
    def _generate_column_description(self, col_name: str, dtype: DataType, null_count: int, unique_count: int) -> str:
        """
        Generate a human-readable description of a column.
        
        Args:
            col_name: Column name
            dtype: Data type
            null_count: Number of null values
            unique_count: Number of unique values
            
        Returns:
            Column description
        """
        description_parts = [f"Column '{col_name}' with {dtype.value} data type"]
        
        if null_count > 0:
            description_parts.append(f"contains {null_count} null values")
        
        description_parts.append(f"has {unique_count} unique values")
        
        return ", ".join(description_parts) + "." 