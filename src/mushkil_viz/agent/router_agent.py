"""Router agent that coordinates all other agents."""

import pandas as pd
import json
from langchain.agents import AgentType, initialize_agent
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from .analysis_agent import AnalysisAgent
from .visualization_agent import VisualizationAgent
from .sample_prompts_agent import SamplePromptsAgent


class RouterAgent(BaseAgent):
    """Router agent that coordinates all specialized agents."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize the router agent.

        Args:
            df: Optional pandas DataFrame to work with
        """
        super().__init__(df)
        self.analysis_agent = AnalysisAgent(df)
        self.visualization_agent = VisualizationAgent(df)
        self.sample_prompts_agent = SamplePromptsAgent(df)

    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe for all agents.

        Args:
            df: The pandas DataFrame to analyze
        """
        super().set_dataframe(df)
        self.analysis_agent.set_dataframe(df)
        self.visualization_agent.set_dataframe(df)
        self.sample_prompts_agent.set_dataframe(df)

    def create_tool(self):
        """Router doesn't have its own tool."""
        raise NotImplementedError("Router agent doesn't have its own tool")

    def get_tool_name(self) -> str:
        """Router doesn't have its own tool name."""
        raise NotImplementedError("Router agent doesn't have its own tool name")

    def get_tool_description(self) -> str:
        """Router doesn't have its own tool description."""
        raise NotImplementedError("Router agent doesn't have its own tool description")

    def create_agent(self):
        """Create the router agent with all specialized tools."""
        tools = [
            self.analysis_agent.create_tool(),
            self.visualization_agent.create_tool(),
            self.sample_prompts_agent.create_tool(),
        ]

        # Create simple agent
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,  # Enable verbose for debugging
            max_iterations=5,  # Increase iterations
            handle_parsing_errors=True,  # Handle errors gracefully
            agent_kwargs={"prefix": self._get_agent_prefix()},
        )

        return agent

    def _get_agent_prefix(self) -> str:
        """Get the prefix for the agent prompt.

        Returns:
            str: The agent prefix
        """
        rows = self.df.shape[0] if self.df is not None else 0
        cols = self.df.shape[1] if self.df is not None else 0
        columns = ", ".join(self.df.columns.tolist()) if self.df is not None else "None"

        return f"""Data analysis assistant. Dataset: {rows} rows, {cols} columns.
Columns: {columns}

Answer simple questions directly. For complex tasks, use tools once:
- analyze_data: statistical analysis (always returns formatted tables)
- create_plot: visualizations
- get_sample_prompts: generate relevant analysis suggestions

Tools:"""

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query with smart routing.

        Args:
            query: The user's query

        Returns:
            Dict[str, Any]: The processed result
        """
        try:
            agent = self.create_agent()
            result = agent.invoke({"input": query})

            # Get both the full result and the final output
            full_response = str(result)  # This includes observations
            final_output = result["output"]

            # Check for table data in BOTH full response and final output
            table_result = self._extract_table_data(full_response + " " + final_output)
            if table_result:
                return table_result

            # Check for chart data in BOTH full response and final output
            chart_result = self._extract_chart_data(full_response + " " + final_output)
            if chart_result:
                return chart_result

            # Default response
            return {
                "output": final_output,
                "chart": None,
                "table": None,
                "error": False,
            }

        except Exception as e:
            return {
                "output": f"Error: {str(e)}",
                "chart": None,
                "table": None,
                "error": True,
            }

    def _extract_table_data(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract table data from the response text.

        Args:
            text: The response text to search

        Returns:
            Optional[Dict[str, Any]]: The table result if found, None otherwise
        """
        if "DATAFRAME_TABLE:" not in text:
            return None

        # Find the start of DATAFRAME_TABLE
        table_start = text.find("DATAFRAME_TABLE:")
        if table_start == -1:
            return None

        # Get everything after "DATAFRAME_TABLE:"
        table_part = text[table_start + len("DATAFRAME_TABLE:") :]

        # Split by first colon to get title and JSON
        colon_pos = table_part.find(":")
        if colon_pos == -1:
            return None

        table_title = table_part[:colon_pos].strip()
        json_part = table_part[colon_pos + 1 :]

        # Find the JSON object - look for the first { and match braces
        json_data = self._extract_json_from_text(json_part)
        if not json_data:
            return None

        # Validate JSON
        try:
            json.loads(json_data)  # Test if valid JSON

            return {
                "output": f"📊 {table_title}",
                "chart": None,
                "table": {"title": table_title, "data": json_data},
                "error": False,
            }
        except json.JSONDecodeError as e:
            return {
                "output": f"⚠️ Table generated but JSON parsing failed: {str(e)}",
                "chart": None,
                "table": None,
                "error": False,
            }

    def _extract_chart_data(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract chart data from the response text.

        Args:
            text: The response text to search

        Returns:
            Optional[Dict[str, Any]]: The chart result if found, None otherwise
        """
        if "PLOTLY_CHART:" not in text:
            return None

        # Find the start of PLOTLY_CHART and extract everything after it
        chart_start = text.find("PLOTLY_CHART:")
        if chart_start == -1:
            return None

        # Get everything after "PLOTLY_CHART:"
        chart_part = text[chart_start + len("PLOTLY_CHART:") :]

        # Find the JSON object
        json_data = self._extract_json_from_text(chart_part)
        if not json_data:
            return None

        # Validate JSON
        try:
            json.loads(json_data)  # Test if valid JSON

            return {
                "output": "📈 Interactive chart created successfully!",
                "chart": json_data,
                "table": None,
                "error": False,
            }
        except json.JSONDecodeError as e:
            return {
                "output": f"⚠️ Chart generated but JSON parsing failed: {str(e)}",
                "chart": None,
                "table": None,
                "error": False,
            }

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text by matching braces.

        Args:
            text: The text to search for JSON

        Returns:
            Optional[str]: The JSON string if found, None otherwise
        """
        brace_count = 0
        json_start = -1
        json_end = -1

        for i, char in enumerate(text):
            if char == "{":
                if json_start == -1:
                    json_start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i + 1
                    break

        if json_start != -1 and json_end != -1:
            return text[json_start:json_end].strip()

        return None
