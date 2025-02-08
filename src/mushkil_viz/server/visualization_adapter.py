"""Adapter module to handle visualization format conversion between CLI and frontend."""
from typing import Dict, Any, Union
import plotly.graph_objects as go
import json

def convert_plotly_to_frontend_format(fig: Union[go.Figure, str, Dict]) -> Dict:
    """Convert a Plotly figure to frontend-compatible format.
    
    Args:
        fig: Either a Plotly Figure object, JSON string, or dictionary
        
    Returns:
        List containing the plot data traces in frontend format
    """
    # If already a string (JSON), parse it
    if isinstance(fig, str):
        try:
            fig_dict = json.loads(fig)
        except json.JSONDecodeError:
            return None
    # If already a dict, use as is
    elif isinstance(fig, dict):
        fig_dict = fig
    # If Plotly figure, convert to dict
    elif isinstance(fig, go.Figure):
        fig_dict = json.loads(fig.to_json())
    else:
        return None
        
    # Extract just the data array from the figure
    if isinstance(fig_dict, dict) and 'data' in fig_dict:
        return fig_dict['data']
    
    return None

def adapt_visualizations_for_frontend(visualizations: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all visualizations to frontend-compatible format.
    
    Args:
        visualizations: Dictionary of visualization results
        
    Returns:
        Dictionary with all visualizations converted to frontend format
    """
    adapted_visualizations = {}
    
    for key, viz in visualizations.items():
        if viz is not None:
            adapted_viz = convert_plotly_to_frontend_format(viz)
            if adapted_viz is not None:
                adapted_visualizations[key] = adapted_viz
                
    return adapted_visualizations 