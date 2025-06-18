"""Visualization agent for creating charts and plots."""

import pandas as pd
import plotly.express as px
import plotly.io as pio
import re
import json
from langchain.agents import Tool
from .base_agent import BaseAgent


class VisualizationAgent(BaseAgent):
    """Agent responsible for creating visualizations."""

    def get_tool_name(self) -> str:
        """Get the name of the tool."""
        return "create_plot"

    def get_tool_description(self) -> str:
        """Get the description of the tool."""
        return "Create any type of visualization using dynamic Plotly code generation"

    def create_tool(self) -> Tool:
        """Create the visualization tool."""
        return Tool(
            name=self.get_tool_name(),
            description=self.get_tool_description(),
            func=self._create_plot,
        )

    def _create_plot(self, query: str) -> str:
        """Create any visualization based on query using LLM-generated Plotly code.

        Args:
            query: The visualization query from the user

        Returns:
            str: The visualization result or error message
        """
        if self.df is None:
            return "No data loaded."

        try:
            # Simple visualization prompt
            viz_prompt = f"""
            Dataset columns: {list(self.df.columns)}
            Request: "{query}"

            Write ONE line of plotly express code that creates a figure and stores it in 'fig'.

            Examples:
            fig = px.histogram(df, x='sepal.length')
            fig = px.scatter(df, x='sepal.length', y='sepal.width')
            fig = px.box(df, x='variety', y='sepal.length')
            fig = px.imshow(df.corr(), text_auto=True)

            Your code (one line only):
            """

            code_response = self.llm.invoke(viz_prompt)
            raw_response = code_response.content.strip()

            # Extract just the code line
            code = self._extract_code_from_response(raw_response)

            # Fix column names with dots
            code = self._fix_column_names_in_code(code)

            # Execute visualization code
            return self._execute_visualization_code(code)

        except Exception as e:
            return f"Error: {str(e)}"

    def _extract_code_from_response(self, raw_response: str) -> str:
        """Extract the code line from LLM response.

        Args:
            raw_response: The raw response from the LLM

        Returns:
            str: The extracted code line
        """
        # Look for lines that start with "fig ="
        fig_lines = re.findall(r"^fig\s*=.*$", raw_response, re.MULTILINE)
        if fig_lines:
            return fig_lines[0].strip()

        # Fallback: look for any line with "fig ="
        fig_match = re.search(r"fig\s*=.*", raw_response)
        if fig_match:
            return fig_match.group(0).strip()

        # Last resort: try to find any px. operation
        lines = raw_response.split("\n")
        for line in lines:
            line = line.strip()
            if line and "px." in line and not line.startswith("#"):
                return f"fig = {line}" if not line.startswith("fig") else line

        # Default fallback
        return "fig = px.scatter(df, x=df.columns[0], y=df.columns[1])"

    def _fix_column_names_in_code(self, code: str) -> str:
        """Fix problematic column names in the code.

        Args:
            code: The code to fix

        Returns:
            str: The fixed code
        """
        for col in self.df.columns:
            if "." in col or " " in col:
                wrong_pattern = f"df.{col}"
                if wrong_pattern in code:
                    code = code.replace(wrong_pattern, f"df['{col}']")
        return code

    def _execute_visualization_code(self, code: str) -> str:
        """Execute the visualization code safely.

        Args:
            code: The code to execute

        Returns:
            str: The formatted result
        """
        # Safe execution environment
        safe_globals = {
            "__builtins__": {
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
            },
            "px": px,
            "pd": pd,
            "pio": pio,
        }

        local_vars = {"df": self.df}

        try:
            exec(code, safe_globals, local_vars)

            if "fig" in local_vars:
                fig = local_vars["fig"]
                # Use multiple methods to serialize Plotly figure
                try:
                    # Method 1: Use plotly.io.to_json for best compatibility
                    fig_json = pio.to_json(fig)
                    return f"PLOTLY_CHART:{fig_json}"
                except Exception as pio_error:
                    # Method 2: Use Plotly's native to_json method
                    try:
                        fig_json = fig.to_json()
                        return f"PLOTLY_CHART:{fig_json}"
                    except Exception as json_error:
                        # Method 3: Fallback to dict method
                        try:
                            fig_dict = fig.to_dict()
                            fig_json = json.dumps(
                                fig_dict, default=str
                            )  # Handle non-serializable objects
                            return f"PLOTLY_CHART:{fig_json}"
                        except Exception as dict_error:
                            return f"JSON serialization failed - pio: {str(pio_error)} | json: {str(json_error)} | dict: {str(dict_error)}"
            else:
                return "No visualization created. Try a different request."

        except Exception as exec_error:
            return f"Visualization failed: {str(exec_error)}. Try a simpler request."
