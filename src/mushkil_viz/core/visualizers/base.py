from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

class BaseVisualizer:
    """Base class for data visualization with intelligent plot selection."""
    
    def __init__(self):
        self.plot_registry = {
            "numeric_distribution": self._plot_numeric_distribution,
            "categorical_distribution": self._plot_categorical_distribution,
            "correlation_matrix": self._plot_correlation_matrix,
            "scatter_matrix": self._plot_scatter_matrix,
            "dimension_reduction": self._plot_dimension_reduction
        }
        
    def _select_plot_type(self, data: pd.DataFrame, column: str) -> str:
        """Intelligently select the most appropriate plot type."""
        dtype = data[column].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric_distribution"
        else:
            return "categorical_distribution"
            
    def _plot_numeric_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        **kwargs
    ) -> go.Figure:
        """Create distribution plot for numeric data."""
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data[column],
            name="Distribution",
            nbinsx=30,
            showlegend=True
        ))
        
        # Add KDE if sample size is large enough
        if len(data) > 100:
            kde = sns.kdeplot(data=data[column].dropna())
            line_data = kde.get_lines()[0].get_data()
            fig.add_trace(go.Scatter(
                x=line_data[0],
                y=line_data[1],
                name="KDE",
                line=dict(color="red")
            ))
            plt.close()  # Clean up matplotlib figure
            
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white"
        )
        
        return fig
        
    def _plot_categorical_distribution(
        self,
        data: pd.DataFrame,
        column: str,
        **kwargs
    ) -> go.Figure:
        """Create bar plot for categorical data."""
        value_counts = data[column].value_counts()
        
        fig = go.Figure(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            text=value_counts.values,
            textposition="auto"
        ))
        
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            showlegend=False
        )
        
        # Rotate labels if there are many categories
        if len(value_counts) > 10:
            fig.update_layout(xaxis_tickangle=-45)
            
        return fig
        
    def _plot_correlation_matrix(
        self,
        correlation_matrix: pd.DataFrame,
        **kwargs
    ) -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            template="plotly_white",
            height=600,
            width=800
        )
        
        return fig
        
    def _plot_time_series(
        self,
        data: pd.DataFrame,
        column: str,
        time_column: str,
        **kwargs
    ) -> go.Figure:
        """Create time series plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[time_column],
            y=data[column],
            mode="lines+markers",
            name=column
        ))
        
        fig.update_layout(
            title=f"Time Series of {column}",
            xaxis_title=time_column,
            yaxis_title=column,
            template="plotly_white"
        )
        
        return fig
        
    def _plot_scatter_matrix(
        self,
        data: pd.DataFrame,
        columns: List[str],
        **kwargs
    ) -> go.Figure:
        """Create scatter plot matrix."""
        fig = px.scatter_matrix(
            data,
            dimensions=columns,
            title="Scatter Plot Matrix"
        )
        
        fig.update_layout(
            template="plotly_white",
            height=800,
            width=800
        )
        
        return fig
        
    def _plot_dimension_reduction(
        self,
        coordinates: np.ndarray,
        labels: Optional[pd.Series] = None,
        method: str = "PCA",
        **kwargs
    ) -> go.Figure:
        """Create dimension reduction visualization."""
        fig = go.Figure()
        
        if labels is not None:
            # Color points by label if available
            for label in labels.unique():
                mask = labels == label
                fig.add_trace(go.Scatter(
                    x=coordinates[mask, 0],
                    y=coordinates[mask, 1],
                    mode="markers",
                    name=str(label),
                    marker=dict(size=8)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                mode="markers",
                marker=dict(size=8)
            ))
            
        fig.update_layout(
            title=f"{method} Visualization",
            xaxis_title=f"{method} 1",
            yaxis_title=f"{method} 2",
            template="plotly_white",
            height=600,
            width=800
        )
        
        return fig
        
    
    
    def visualize(
        self,
        data: pd.DataFrame,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive visualizations based on analysis results."""
        visualizations = {}
        
        # Get column types from analysis results
        numeric_cols = analysis_results["dataset_info"]["column_types"]["numeric"]
        temporal_cols = analysis_results["dataset_info"]["column_types"]["temporal"]
        
        # Generate distribution plots for each column type
        for col_type, columns in analysis_results["dataset_info"]["column_types"].items():
            for col in columns:
                plot_type = self._select_plot_type(data, col)
                fig = self.plot_registry[plot_type](data=data, column=col)
                visualizations[f"{col}_distribution"] = fig.to_json()
        
        # Generate time series plots for each numerical column against temporal columns
        if temporal_cols and numeric_cols:
            for temporal_col in temporal_cols:
                for numeric_col in numeric_cols:
                    fig = self._plot_time_series(
                        data=data,
                        column=numeric_col,
                        time_column=temporal_col
                    )
                    visualizations[f"timeseries_{temporal_col}_{numeric_col}"] = fig.to_json()
        
        # Generate scatter matrix for numerical columns
        if len(numeric_cols) >= 2:
            scatter_fig = self._plot_scatter_matrix(
                data=data,
                columns=numeric_cols
            )
            visualizations["scatter_matrix"] = scatter_fig.to_json()
        
        # Generate correlation matrix if numeric columns exist
        if numeric_cols:
            corr_fig = self._plot_correlation_matrix(
                pd.DataFrame(analysis_results["correlations"])
            )
            visualizations["correlation_matrix"] = corr_fig.to_json()
        
        # Generate dimension reduction plot if available
        if "dimensionality_reduction" in analysis_results:
            dim_red_fig = self._plot_dimension_reduction(
                np.array(analysis_results["dimensionality_reduction"]["coordinates"]),
                method="PCA"
            )
            visualizations["dimension_reduction"] = dim_red_fig.to_json()
        
        return visualizations 