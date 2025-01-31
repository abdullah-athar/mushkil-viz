from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy import stats

class BaseAnalyzer:
    """Base class for data analysis with common functionality."""
    
    def __init__(self):
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.temporal_columns: List[str] = []
        self.text_columns: List[str] = []
        
    def _detect_column_types(self, df: pd.DataFrame) -> None:
        """Detect and categorize column types."""
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                self.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.temporal_columns.append(col)
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                if df[col].nunique() / len(df) < 0.05:  # Less than 5% unique values
                    self.categorical_columns.append(col)
                else:
                    self.text_columns.append(col)
    
    def _compute_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Compute basic statistical measures for numeric columns."""
        stats_dict = {}
        
        for col in self.numeric_columns:
            stats_dict[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "skew": stats.skew(df[col].dropna()),
                "kurtosis": stats.kurtosis(df[col].dropna()),
                "missing_pct": (df[col].isna().sum() / len(df)) * 100
            }
            
        return stats_dict
    
    def _detect_outliers(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = "zscore",
        threshold: float = 3.0
    ) -> Dict[str, np.ndarray]:
        """Detect outliers in numeric columns."""
        columns = columns or self.numeric_columns
        outliers = {}
        
        for col in columns:
            if method == "zscore":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers[col] = z_scores > threshold
            elif method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                
        return outliers
    
    def _compute_correlations(
        self, 
        df: pd.DataFrame, 
        method: str = "pearson"
    ) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        return df[self.numeric_columns].corr(method=method)
    
    def _reduce_dimensions(
        self, 
        df: pd.DataFrame,
        method: str = "pca",
        n_components: int = 2
    ) -> Tuple[np.ndarray, float]:
        """Reduce dimensionality of numeric features."""
        X = df[self.numeric_columns]
        X_scaled = StandardScaler().fit_transform(X)
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(X_scaled)
            explained_variance = reducer.explained_variance_ratio_.sum()
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(X_scaled)
            explained_variance = None
        elif method == "umap":
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(X_scaled)
            explained_variance = None
            
        return reduced_data, explained_variance
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive analysis of the dataset."""
        # Detect column types
        self._detect_column_types(df)
        
        # Basic dataset info
        analysis_results = {
            "dataset_info": {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "column_types": {
                    "numeric": self.numeric_columns,
                    "categorical": self.categorical_columns,
                    "temporal": self.temporal_columns,
                    "text": self.text_columns
                }
            },
            "basic_stats": self._compute_basic_stats(df),
            "outliers": self._detect_outliers(df),
            "correlations": self._compute_correlations(df).to_dict()
        }
        
        # Only perform dimensionality reduction if we have enough numeric columns
        if len(self.numeric_columns) > 2:
            reduced_data, explained_variance = self._reduce_dimensions(df)
            analysis_results["dimensionality_reduction"] = {
                "coordinates": reduced_data.tolist(),
                "explained_variance": explained_variance
            }
            
        return analysis_results 