"""Multi-agent system for data analysis and visualization."""

from .base_agent import BaseAgent
from .analysis_agent import AnalysisAgent
from .visualization_agent import VisualizationAgent
from .sample_prompts_agent import SamplePromptsAgent
from .router_agent import RouterAgent
from .utils import load_data, run_analysis


# For backward compatibility, create a MultiAgentSystem class that wraps RouterAgent
# TODO Abdullah: I will Remove this soon!
class MultiAgentSystem:
    """Backward compatible wrapper around the new multi-agent system."""

    def __init__(self):
        """Initialize the multi-agent system."""
        self.router = RouterAgent()

    def set_dataframe(self, df):
        """Set the dataframe for analysis."""
        self.router.set_dataframe(df)

    def process_query(self, query: str):
        """Process user query with smart routing."""
        return self.router.process_query(query)

    def create_router_agent(self):
        """Create router agent (for backward compatibility)."""
        return self.router.create_agent()


__all__ = [
    "BaseAgent",
    "AnalysisAgent",
    "VisualizationAgent",
    "SamplePromptsAgent",
    "RouterAgent",
    "MultiAgentSystem",
    "load_data",
    "run_analysis",
]
