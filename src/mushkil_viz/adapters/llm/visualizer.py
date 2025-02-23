from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
from mushkil_viz.core.client_init import init_openai_client

class VisualizationError(Exception):
    """Custom exception for visualization errors with fallback options."""
    def __init__(self, spec: Dict, error: str, fallback: Optional[Dict] = None):
        self.spec = spec
        self.error = error
        self.fallback = fallback
        super().__init__(f"Error creating visualization {spec.get('id')}: {error}")

class LLMVisualizer:
    """LLM domain-specific visualizer that dynamically generates visualizations."""

    def __init__(self, api_key: str = None, api_base: str = None, config: Dict = None):
        self.client = init_openai_client(api_key, api_base)
        self.config = config or {}
        self.system_prompt = """You are a data visualization expert. Your task is to suggest the best visualization 
        type and parameters for the given data and analysis results. Return a JSON object with visualization details."""
        self.fallback_types = {
            "bubble": "scatter",
            "violin": "box",
            "area": "line",
            "heatmap": "scatter"
        }

    def visualize(self, analysis_results: Dict[str, Any], data: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate visualizations based on LLM suggestions."""
        visualizations = {}
        
        for spec in analysis_results.get("visualization_specs", []):
            try:
                viz = self._create_visualization(data, spec)
                if viz:
                    visualizations[spec["id"]] = viz
            except VisualizationError as ve:
                if ve.fallback:
                    try:
                        viz = self._create_visualization(data, ve.fallback)
                        if viz:
                            visualizations[spec["id"]] = viz
                    except Exception as e:
                        print(f"Fallback visualization failed for {spec['id']}: {str(e)}")
            except Exception as e:
                print(f"Error creating visualization for {spec['id']}: {str(e)}")
                
        return visualizations

    def _apply_transformations(self, data: pd.DataFrame, transformations: List[Dict]) -> pd.DataFrame:
        """Apply data transformations specified in the visualization spec."""
        df = data.copy()
        
        for transform in transformations:
            try:
                operation = transform.get("operation")
                if operation == "groupby":
                    groupby_cols = transform.get("columns", [])
                    agg_func = transform.get("aggregation", "count")
                    df = df.groupby(groupby_cols).agg(agg_func).reset_index()
                elif operation == "sort":
                    sort_col = transform.get("column")
                    ascending = transform.get("ascending", True)
                    df = df.sort_values(sort_col, ascending=ascending)
                elif operation == "filter":
                    condition = transform.get("condition")
                    df = df.query(condition)
                elif operation == "datetime":
                    col = transform.get("column")
                    unit = transform.get("unit", "D")
                    df[col] = pd.to_datetime(df[col]).dt.to_period(unit)
            except Exception as e:
                print(f"Error applying transformation {operation}: {str(e)}")
                
        return df

    def _create_visualization(self, data: pd.DataFrame, spec: Dict) -> Optional[str]:
        """Create visualization based on specification."""
        try:
            # Apply any data transformations
            if spec.get("data", {}).get("transformations"):
                data = self._apply_transformations(data, spec["data"]["transformations"])
            
            # Get chart configuration
            chart_config = self.config.get("visualization", {}).get("chart_defaults", {})
            layout_config = self.config.get("visualization", {}).get("layout", {})
            colors = self.config.get("visualization", {}).get("color_scheme", {})
            
            # Create figure based on plot type
            fig = None
            plot_type = spec["type"]
            
            if plot_type == "bar":
                fig = go.Figure(data=[
                    go.Bar(
                        x=data[spec["data"]["x"]],
                        y=data[spec["data"]["y"]],
                        marker_color=colors.get("primary"),
                        **chart_config.get("bar", {})
                    )
                ])
            elif plot_type == "line":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data[spec["data"]["x"]],
                        y=data[spec["data"]["y"]],
                        mode="lines+markers",
                        line=dict(color=colors.get("primary")),
                        **chart_config.get("line", {})
                    )
                ])
            elif plot_type == "scatter":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data[spec["data"]["x"]],
                        y=data[spec["data"]["y"]],
                        mode="markers",
                        marker=dict(
                            color=data[spec["data"].get("color")] if spec["data"].get("color") else colors.get("primary"),
                            size=data[spec["data"].get("size")] if spec["data"].get("size") else 8
                        ),
                        **chart_config.get("scatter", {})
                    )
                ])
            elif plot_type == "box":
                fig = go.Figure(data=[
                    go.Box(
                        y=data[spec["data"]["y"]],
                        x=data[spec["data"].get("x")] if spec["data"].get("x") else None,
                        marker_color=colors.get("primary"),
                        **chart_config.get("box", {})
                    )
                ])
            elif plot_type == "histogram":
                fig = go.Figure(data=[
                    go.Histogram(
                        x=data[spec["data"]["x"]],
                        marker_color=colors.get("primary"),
                        **chart_config.get("histogram", {})
                    )
                ])
            elif plot_type == "heatmap":
                pivot_data = data.pivot(
                    index=spec["data"]["y"],
                    columns=spec["data"]["x"],
                    values=spec["data"]["z"]
                )
                fig = go.Figure(data=[
                    go.Heatmap(
                        z=pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        **chart_config.get("heatmap", {})
                    )
                ])
            elif plot_type == "pie":
                fig = go.Figure(data=[
                    go.Pie(
                        labels=data[spec["data"]["x"]],
                        values=data[spec["data"]["y"]],
                        marker=dict(colors=[colors.get("primary"), colors.get("secondary"), colors.get("accent")]),
                        **chart_config.get("pie", {})
                    )
                ])
            elif plot_type == "violin":
                fig = go.Figure(data=[
                    go.Violin(
                        y=data[spec["data"]["y"]],
                        x=data[spec["data"].get("x")] if spec["data"].get("x") else None,
                        marker_color=colors.get("primary"),
                        **chart_config.get("violin", {})
                    )
                ])
            elif plot_type == "bubble":
                fig = go.Figure(data=[
                    go.Scatter(
                        x=data[spec["data"]["x"]],
                        y=data[spec["data"]["y"]],
                        mode="markers",
                        marker=dict(
                            size=data[spec["data"]["size"]],
                            color=data[spec["data"].get("color")] if spec["data"].get("color") else colors.get("primary"),
                            sizemode="area",
                            sizeref=2.*max(data[spec["data"]["size"]])/(40.**2),
                            sizemin=4
                        ),
                        **chart_config.get("bubble", {})
                    )
                ])
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
                
            # Update layout
            fig.update_layout(
                title=spec["layout"]["title"],
                xaxis_title=spec["layout"]["xaxis_title"],
                yaxis_title=spec["layout"]["yaxis_title"],
                **layout_config
            )
            
            return fig.to_json()
            
        except Exception as e:
            # Try fallback if available
            if plot_type in self.fallback_types:
                fallback_spec = spec.copy()
                fallback_spec["type"] = self.fallback_types[plot_type]
                raise VisualizationError(spec, str(e), fallback_spec)
            else:
                raise VisualizationError(spec, str(e))

if __name__ == "__main__":
    # Example usage
    example_config = {
        "visualization": {
            "color_scheme": {
                "primary": "#3498db",
                "secondary": "#2ecc71",
                "accent": "#e74c3c",
                "neutral": "#95a5a6"
            },
            "chart_defaults": {
                "bar": {"opacity": 0.8},
                "line": {"mode": "lines+markers"},
                "scatter": {"mode": "markers"},
                "histogram": {"nbins": 30},
                "heatmap": {"colorscale": "Viridis"}
            },
            "layout": {
                "font_family": "Arial, sans-serif",
                "showlegend": True,
                "template": "plotly_white"
            }
        }
    }
    
    # Create sample data
    data = pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "values": [10, 20, 15, 25]
    })
    
    # Create sample analysis results
    analysis_results = {
        "visualization_specs": [{
            "id": "sample_plot",
            "type": "bar",
            "title": "Sample Bar Plot",
            "description": "A simple bar plot example",
            "data": {
                "x": "category",
                "y": "values"
            },
            "layout": {
                "title": "Sample Plot",
                "xaxis_title": "Categories",
                "yaxis_title": "Values"
            }
        }]
    }
    
    # Create visualizer and generate plot
    visualizer = LLMVisualizer(config=example_config)
    visualizations = visualizer.visualize(analysis_results, data)
    print("Generated visualizations:", list(visualizations.keys()))