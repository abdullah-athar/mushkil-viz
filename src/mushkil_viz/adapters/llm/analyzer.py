"""LLM-based analyzer module."""
import json
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats

from mushkil_viz.adapters.llm.constants import DEFAULT_MODEL, LOG_FORMAT
from mushkil_viz.core.analyzers.base import BaseAnalyzer
from mushkil_viz.core.client_init import init_openai_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

SYSTEM_PROMPT = """You are a data visualization expert. Your task is to analyze the provided dataset and suggest meaningful visualizations.
Focus on insights that would be valuable to visualize. For each visualization:
1. Choose an appropriate plot type based on the data characteristics
2. Provide clear titles and axis labels
3. Include any necessary data transformations
4. Explain why this visualization is useful

Available Plot Types:
- Bar/Column Charts (for categorical comparisons)
- Line Charts (for trends over time or sequences)
- Scatter Plots (for relationships between variables)
- Box Plots (for distribution and outliers)
- Histograms (for distribution of numeric data)
- Heatmaps (for correlation matrices or 2D distributions)
- Area Charts (for cumulative or stacked trends)
- Pie Charts (for part-to-whole relationships)
- Violin Plots (for probability density)
- Bubble Charts (for 3 dimensions of data)

Return your response as a JSON object with the following structure:
{
    "visualizations": [
        {
            "id": "unique_id",
            "type": "plot_type",
            "title": "Descriptive Title",
            "description": "Why this visualization is useful",
            "data": {
                "x": "column_name or transformation",
                "y": "column_name or transformation",
                "color": "optional_column",
                "size": "optional_column",
                "transformations": []
            },
            "layout": {
                "title": "Plot Title",
                "xaxis_title": "X Axis Label",
                "yaxis_title": "Y Axis Label"
            }
        }
    ]
}"""

class LLMAnalyzer(BaseAnalyzer):
    """Analyzer that uses LLMs to generate and execute data analysis functions."""
    
    def __init__(self, api_key: str = None, api_base: str = None):
        super().__init__()
        self.client = init_openai_client(api_key, api_base)
        self.model = DEFAULT_MODEL
        self.system_prompt = SYSTEM_PROMPT
        self.max_retries = 3
        self.max_plots = 15
        logger.debug("LLMAnalyzer initialized with model: %s", self.model)

    def _build_dataset_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build rich dataset context for LLM."""
        def convert_to_native(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj

        context = {
            "dataset_shape": tuple(df.shape),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "columns": {}
        }
        
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
                "n_missing": int(df[col].isna().sum()),
                "sample_values": [convert_to_native(v) for v in df[col].head(5)],
                "distribution_type": "categorical"
            }
            
            if pd.api.types.is_numeric_dtype(df[col].dtype):
                stats = df[col].describe()
                col_info.update({
                    "distribution_type": "numeric",
                    "basic_stats": {
                        "mean": convert_to_native(stats["mean"]),
                        "std": convert_to_native(stats["std"]),
                        "min": convert_to_native(stats["min"]),
                        "max": convert_to_native(stats["max"])
                    }
                })
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["distribution_type"] = "temporal"
            else:
                if col_info["n_unique"] < 10:
                    col_info["value_counts"] = {
                        convert_to_native(k): convert_to_native(v) 
                        for k, v in df[col].value_counts().head().items()
                    }
                    
            context["columns"][col] = col_info
            
        # Add correlation information for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            strong_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            "col1": corr.columns[i],
                            "col2": corr.columns[j],
                            "correlation": convert_to_native(corr.iloc[i, j])
                        })
            if strong_corr:
                context["strong_correlations"] = strong_corr
            
        return context

    def _generate_default_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """Generate default visualizations for Netflix dataset."""
        specs = []
        
        # Type Distribution
        if "type" in df.columns:
            specs.append({
                "id": "type_distribution",
                "type": "bar",
                "title": "Distribution of Content Types",
                "description": "Shows the distribution of movies vs TV shows",
                "data": {
                    "transformations": [{
                        "operation": "groupby",
                        "columns": ["type"],
                        "aggregation": "count"
                    }],
                    "x": "type",
                    "y": "count"
                },
                "layout": {
                    "title": "Distribution of Content Types",
                    "xaxis_title": "Content Type",
                    "yaxis_title": "Count"
                }
            })
            
        # Rating Distribution
        if "rating" in df.columns:
            specs.append({
                "id": "rating_distribution",
                "type": "bar",
                "title": "Distribution of Content Ratings",
                "description": "Shows the distribution of content ratings",
                "data": {
                    "transformations": [{
                        "operation": "groupby",
                        "columns": ["rating"],
                        "aggregation": "count"
                    }],
                    "x": "rating",
                    "y": "count"
                },
                "layout": {
                    "title": "Distribution of Content Ratings",
                    "xaxis_title": "Rating",
                    "yaxis_title": "Count"
                }
            })
            
        # Release Year Distribution
        if "release_year" in df.columns:
            specs.append({
                "id": "release_year_distribution",
                "type": "histogram",
                "title": "Distribution of Release Years",
                "description": "Shows the distribution of content release years",
                "data": {
                    "x": "release_year"
                },
                "layout": {
                    "title": "Distribution of Release Years",
                    "xaxis_title": "Release Year",
                    "yaxis_title": "Count"
                }
            })
            
        # Date Added Distribution
        if "date_added" in df.columns:
            specs.append({
                "id": "date_added_distribution",
                "type": "histogram",
                "title": "Distribution of Date Added",
                "description": "Shows when content was added to Netflix",
                "data": {
                    "x": "date_added"
                },
                "layout": {
                    "title": "Distribution of Date Added",
                    "xaxis_title": "Date Added",
                    "yaxis_title": "Count"
                }
            })
            
        # Country Distribution
        if "country" in df.columns:
            specs.append({
                "id": "country_distribution",
                "type": "bar",
                "title": "Top Countries by Content",
                "description": "Shows the distribution of content by country",
                "data": {
                    "transformations": [{
                        "operation": "groupby",
                        "columns": ["country"],
                        "aggregation": "count"
                    }, {
                        "operation": "sort",
                        "column": "count",
                        "ascending": False
                    }],
                    "x": "country",
                    "y": "count"
                },
                "layout": {
                    "title": "Top Countries by Content",
                    "xaxis_title": "Country",
                    "yaxis_title": "Count"
                }
            })
            
        # Duration Distribution
        if "duration" in df.columns:
            specs.append({
                "id": "duration_distribution",
                "type": "histogram",
                "title": "Distribution of Content Duration",
                "description": "Shows the distribution of content duration",
                "data": {
                    "x": "duration"
                },
                "layout": {
                    "title": "Distribution of Content Duration",
                    "xaxis_title": "Duration",
                    "yaxis_title": "Count"
                }
            })
            
        return specs

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive analysis of the dataset using LLM suggestions."""
        logger.debug("Starting analysis on DataFrame with shape: %s", str(df.shape))
        
        # Get base analysis results
        base_analysis = super().analyze(df)
        
        # Generate default visualizations for Netflix dataset
        visualization_specs = self._generate_default_visualizations(df)
        
        # Add visualization specs to results
        results = base_analysis
        results["visualization_specs"] = visualization_specs
        
        return results


if __name__ == "__main__":
    analyzer = LLMAnalyzer()
    df = pd.read_csv("examples/financial_data.csv")
    results = analyzer.analyze(df)
    
    for analysis in results.get("visualization_specs", []):
        print(f"Function Name: {analysis['id']}")
        print(f"Description: {analysis['description']}")
        print("Result:")
        print(analysis['data'])
        print("-" * 50)

