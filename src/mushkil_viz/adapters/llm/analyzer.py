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
from mushkil_viz.adapters.llm.schemas import (
    validate_visualization_response,
    validate_visualization_spec,
    VisualizationSpec
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

SYSTEM_PROMPT = """You are a data visualization expert. Your task is to analyze the provided dataset context and suggest meaningful visualizations based on the actual data characteristics found in the dataset.

For each column type found in the dataset:

1. For Numeric Columns:
   - Create distribution plots (histogram/box) for continuous variables
   - Show correlations between numeric variables if multiple exist
   - Identify and visualize any outliers
   - If time series data exists, show trends over time

2. For Categorical Columns:
   - Show value distributions using appropriate charts (bar/pie)
   - If few unique values (<10), show detailed breakdowns
   - Create cross-tabulations with other categorical columns
   - Show relationships with numeric columns using box plots

3. For Temporal Columns:
   - Show time-based patterns and trends
   - Create time series visualizations
   - Show seasonal patterns if they exist
   - Aggregate by different time periods (day, month, year)

4. For Text Columns:
   - Show length distributions
   - Display frequency of common values
   - Create meaningful groupings if possible

Look for and visualize:
1. Strong correlations between variables
2. Unusual patterns or outliers
3. Time-based trends if temporal data exists
4. Group differences and comparisons
5. Key summary statistics and distributions

For each visualization, provide:
1. A clear title and description of what it shows
2. Appropriate axis labels and scales
3. Any necessary data transformations
4. Why this visualization is useful for understanding the data

Return your response as a JSON object with this structure:
{
    "visualizations": [
        {
            "id": "unique_id",  # Make this descriptive of what's being shown
            "type": "plot_type",
            "title": "Descriptive Title",
            "description": "Why this visualization is useful",
            "data": {
                "x": "column_name",
                "y": "column_name",
                "color": "optional_column",
                "size": "optional_column",
                "transformations": [
                    {
                        "operation": "groupby/sort/filter/datetime",
                        "columns": ["column_names"],
                        "aggregation": "count/sum/mean",
                        "column": "column_name",
                        "ascending": true/false,
                        "condition": "filter_condition",
                        "unit": "time_unit"
                    }
                ]
            },
            "layout": {
                "title": "Plot Title",
                "xaxis_title": "X Axis Label",
                "yaxis_title": "Y Axis Label"
            }
        }
    ]
}

Available Plot Types:
- Bar/Column Charts: For categorical comparisons
- Line Charts: For trends over time or sequences
- Scatter Plots: For relationships between variables
- Box Plots: For distribution and outliers
- Histograms: For distribution of numeric data
- Heatmaps: For correlation matrices or 2D distributions
- Area Charts: For cumulative or stacked trends
- Pie Charts: For part-to-whole relationships (use sparingly)
- Violin Plots: For probability density
- Bubble Charts: For 3 dimensions of data

Guidelines:
1. Focus on the most insightful visualizations (max 10)
2. Ensure each visualization adds unique value
3. Use appropriate plot types for the data
4. Include clear titles and labels
5. Apply transformations when needed (grouping, filtering, etc.)
6. Consider the number of unique values when choosing plot types
7. Handle missing values appropriately"""

class LLMAnalyzer(BaseAnalyzer):
    """Analyzer that uses LLMs to generate and execute data analysis functions."""
    
    def __init__(self, api_key: str = None, api_base: str = None):
        super().__init__()
        self.client = init_openai_client(api_key, api_base)
        self.model = DEFAULT_MODEL
        self.system_prompt = SYSTEM_PROMPT
        self.max_retries = 3
        self.max_plots = 10
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

    def _get_visualization_suggestions(self, context: Dict[str, Any]) -> List[VisualizationSpec]:
        """Get visualization suggestions from LLM based on dataset context."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Suggest visualizations for this dataset context: {json.dumps(context)}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            # Parse and validate response
            suggestions = json.loads(response.choices[0].message.content)
            validated_response = validate_visualization_response(suggestions)
            
            # Limit number of visualizations
            return validated_response.visualizations[:self.max_plots]
            
        except Exception as e:
            logger.error(f"Error getting visualization suggestions: {str(e)}")
            return []

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive analysis of the dataset using LLM suggestions."""
        logger.debug("Starting analysis on DataFrame with shape: %s", str(df.shape))
        
        # Get base analysis results
        base_analysis = super().analyze(df)
        
        # Build rich dataset context
        context = self._build_dataset_context(df)
        
        # Get visualization suggestions from LLM
        visualization_specs = self._get_visualization_suggestions(context)
        
        # Add visualization specs to results
        results = base_analysis
        results["visualization_specs"] = [spec.dict() for spec in visualization_specs]
        
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

