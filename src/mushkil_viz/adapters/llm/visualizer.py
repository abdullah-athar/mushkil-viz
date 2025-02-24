from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
from mushkil_viz.core.client_init import init_openai_client
import plotly.utils

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
        if not transformations:
            return data
            
        df = data.copy()
        
        for transform in transformations:
            try:
                operation = transform.get("operation")
                
                if operation == "groupby":
                    groupby_cols = transform.get("columns", [])
                    if not groupby_cols:
                        continue
                        
                    agg_func = transform.get("aggregation", "count")
                    # Handle multiple aggregation functions
                    if isinstance(agg_func, dict):
                        df = df.groupby(groupby_cols).agg(agg_func).reset_index()
                    else:
                        df = df.groupby(groupby_cols).agg(agg_func).reset_index()
                    
                elif operation == "sort":
                    sort_col = transform.get("column")
                    if not sort_col:
                        continue
                        
                    ascending = transform.get("ascending", True)
                    df = df.sort_values(sort_col, ascending=ascending)
                    
                    # Limit rows if specified
                    if transform.get("limit"):
                        df = df.head(transform["limit"])
                    
                elif operation == "filter":
                    condition = transform.get("condition")
                    if not condition:
                        continue
                        
                    # Handle different filter conditions
                    if isinstance(condition, str):
                        df = df.query(condition)
                    elif isinstance(condition, dict):
                        for col, value in condition.items():
                            if isinstance(value, (list, tuple)):
                                df = df[df[col].isin(value)]
                            else:
                                df = df[df[col] == value]
                    
                elif operation == "datetime":
                    col = transform.get("column")
                    if not col:
                        continue
                        
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # Apply datetime transformations
                    unit = transform.get("unit", "D")
                    if transform.get("extract"):
                        # Extract specific datetime component
                        component = transform["extract"].lower()
                        if component == "year":
                            df[f"{col}_year"] = df[col].dt.year
                        elif component == "month":
                            df[f"{col}_month"] = df[col].dt.month
                        elif component == "day":
                            df[f"{col}_day"] = df[col].dt.day
                        elif component == "hour":
                            df[f"{col}_hour"] = df[col].dt.hour
                    else:
                        # Convert to period
                        df[col] = df[col].dt.to_period(unit)
                
                elif operation == "calculate":
                    # Handle calculated columns
                    new_col = transform.get("new_column")
                    formula = transform.get("formula")
                    if new_col and formula:
                        df[new_col] = df.eval(formula)
                
                elif operation == "bin":
                    # Handle numeric binning
                    col = transform.get("column")
                    bins = transform.get("bins")
                    if col and bins:
                        if isinstance(bins, int):
                            df[f"{col}_binned"] = pd.qcut(df[col], bins, labels=False)
                        elif isinstance(bins, list):
                            df[f"{col}_binned"] = pd.cut(df[col], bins, labels=False)
                
                elif operation == "text":
                    # Handle text operations
                    col = transform.get("column")
                    if not col:
                        continue
                        
                    text_op = transform.get("text_operation")
                    if text_op == "length":
                        df[f"{col}_length"] = df[col].str.len()
                    elif text_op == "lower":
                        df[col] = df[col].str.lower()
                    elif text_op == "upper":
                        df[col] = df[col].str.upper()
                    elif text_op == "extract":
                        pattern = transform.get("pattern")
                        if pattern:
                            df[f"{col}_extracted"] = df[col].str.extract(pattern)
                
            except Exception as e:
                print(f"Error applying transformation {operation}: {str(e)}")
                continue
                
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
            plot_data = spec.get("data", {})
            
            # Ensure required columns exist
            for col in [plot_data.get("x"), plot_data.get("y")]:
                if col and col not in data.columns:
                    raise ValueError(f"Column {col} not found in dataset")
            
            if plot_type == "bar":
                x_data = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                y_data = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                fig = go.Figure(data=[
                    go.Bar(
                        x=x_data,
                        y=y_data,
                        marker_color=colors.get("primary"),
                        **chart_config.get("bar", {})
                    )
                ])
            elif plot_type == "line":
                x_data = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                y_data = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                fig = go.Figure(data=[
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines+markers",
                        line=dict(color=colors.get("primary")),
                        **chart_config.get("line", {})
                    )
                ])
            elif plot_type == "scatter":
                x_data = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                y_data = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                color_data = data[plot_data["color"]].tolist() if plot_data.get("color") else None
                size_data = data[plot_data["size"]].tolist() if plot_data.get("size") else None
                
                marker_dict = {
                    "color": color_data if color_data else colors.get("primary"),
                    "size": size_data if size_data else 8
                }
                fig = go.Figure(data=[
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="markers",
                        marker=marker_dict,
                        **chart_config.get("scatter", {})
                    )
                ])
            elif plot_type == "box":
                y_data = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                x_data = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                fig = go.Figure(data=[
                    go.Box(
                        y=y_data,
                        x=x_data,
                        marker_color=colors.get("primary"),
                        **chart_config.get("box", {})
                    )
                ])
            elif plot_type == "histogram":
                x_data = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                y_data = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                fig = go.Figure(data=[
                    go.Histogram(
                        x=x_data,
                        y=y_data,
                        marker_color=colors.get("primary"),
                        **chart_config.get("histogram", {})
                    )
                ])
            elif plot_type == "heatmap":
                if all(k in plot_data for k in ["x", "y", "z"]):
                    pivot_data = data.pivot(
                        index=plot_data["y"],
                        columns=plot_data["x"],
                        values=plot_data["z"]
                    )
                    fig = go.Figure(data=[
                        go.Heatmap(
                            z=pivot_data.values.tolist(),
                            x=pivot_data.columns.tolist(),
                            y=pivot_data.index.tolist(),
                            colorscale=chart_config.get("heatmap", {}).get("colorscale", "Viridis")
                        )
                    ])
            elif plot_type == "pie":
                labels = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                values = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=[colors.get("primary"), colors.get("secondary"), colors.get("accent")]),
                        **chart_config.get("pie", {})
                    )
                ])
            elif plot_type == "violin":
                y_data = data[plot_data["y"]].tolist() if plot_data.get("y") else None
                x_data = data[plot_data["x"]].tolist() if plot_data.get("x") else None
                fig = go.Figure(data=[
                    go.Violin(
                        y=y_data,
                        x=x_data,
                        marker_color=colors.get("primary"),
                        **chart_config.get("violin", {})
                    )
                ])
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            if fig is None:
                raise ValueError(f"Failed to create figure for plot type: {plot_type}")
                
            # Update layout with both config and spec settings
            layout_updates = {
                "title": spec.get("layout", {}).get("title"),
                "xaxis_title": spec.get("layout", {}).get("xaxis_title"),
                "yaxis_title": spec.get("layout", {}).get("yaxis_title"),
                **layout_config
            }
            fig.update_layout(**{k: v for k, v in layout_updates.items() if v is not None})
            
            # Convert figure to dict first
            fig_dict = fig.to_dict()
            
            # Create the final visualization data
            viz_data = {
                "data": fig_dict["data"],
                "layout": fig_dict["layout"],
                "description": spec.get("description", "")
            }
            
            # Convert to JSON string
            return json.dumps(viz_data)
            
        except Exception as e:
            print(f"Error in _create_visualization: {str(e)}")
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