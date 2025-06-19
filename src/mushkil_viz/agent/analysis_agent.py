"""Analysis agent for statistical analysis tasks."""

import pandas as pd
import re
from langchain.agents import Tool
from typing import Any
from .base_agent import BaseAgent


class AnalysisAgent(BaseAgent):
    """Agent responsible for statistical analysis tasks."""

    def get_tool_name(self) -> str:
        """Get the name of the tool."""
        return "analyze_data"

    def get_tool_description(self) -> str:
        """Get the description of the tool."""
        return "Perform any statistical analysis using dynamic Python code generation based on the query"

    def create_tool(self) -> Tool:
        """Create the analysis tool."""
        return Tool(
            name=self.get_tool_name(),
            description=self.get_tool_description(),
            func=self._analyze_data,
        )

    def _analyze_data(self, query: str) -> str:
        """Perform statistical analysis based on the query and available data.

        Args:
            query: The analysis query from the user

        Returns:
            str: The analysis result or error message
        """
        if self.df is None:
            return "No data loaded."

        try:
            # Simple analysis prompt
            analysis_prompt = f"""
            Dataset columns: {list(self.df.columns)}
            Request: "{query}"

            Write ONE line of pandas code that stores the result in a variable called 'result'.

            Examples:
            result = df.describe()
            result = df.groupby('variety').mean()
            result = df['sepal.length'].value_counts()
            result = df.corr()

            Your code (one line only):
            """

            # Get code from LLM
            code_response = self.llm.invoke(analysis_prompt)
            raw_response = code_response.content.strip()

            # Extract just the code line
            code = self._extract_code_from_response(raw_response)

            # Validate column names in the code
            code = self._fix_column_names_in_code(code)

            # Execute the code safely
            return self._execute_analysis_code(code)

        except Exception as e:
            return f"Analysis error: {str(e)}"

    def _extract_code_from_response(self, raw_response: str) -> str:
        """Extract the code line from LLM response.

        Args:
            raw_response: The raw response from the LLM

        Returns:
            str: The extracted code line
        """
        # Look for lines that start with "result ="
        result_lines = re.findall(r"^result\s*=.*$", raw_response, re.MULTILINE)
        if result_lines:
            return result_lines[0].strip()

        # Fallback: look for any line with "result ="
        result_match = re.search(r"result\s*=.*", raw_response)
        if result_match:
            return result_match.group(0).strip()

        # Last resort: try to find any pandas operation
        lines = raw_response.split("\n")
        for line in lines:
            line = line.strip()
            if line and ("df." in line or "pd." in line) and not line.startswith("#"):
                return f"result = {line}" if not line.startswith("result") else line

        # Default fallback
        return "result = df.describe()"

    def _fix_column_names_in_code(self, code: str) -> str:
        """Fix problematic column names in the code.

        Args:
            code: The code to fix

        Returns:
            str: The fixed code
        """
        for col in self.df.columns:
            if "." in col or " " in col:
                # Ensure problematic column names are properly quoted
                wrong_pattern = f"df.{col}"
                if wrong_pattern in code:
                    code = code.replace(wrong_pattern, f"df['{col}']")
        return code

    def _execute_analysis_code(self, code: str) -> str:
        """Execute the analysis code safely.

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
            "pd": pd,
        }

        local_vars = {"df": self.df}

        try:
            # Execute the cleaned code
            exec(code, safe_globals, local_vars)

            # Look for the result
            if "result" in local_vars:
                result = local_vars["result"]
                return self._format_analysis_result(result)
            else:
                # Fallback: look for any DataFrame or Series
                for var_name, var_value in local_vars.items():
                    if (
                        isinstance(var_value, (pd.DataFrame, pd.Series))
                        and var_name != "df"
                    ):
                        return self._format_analysis_result(var_value, var_name)
                return "No results found in analysis."

        except Exception as exec_error:
            # Fallback to basic analysis if code fails
            return f"Code execution failed: {str(exec_error)}. Try a simpler request."

    def _format_analysis_result(
        self, result: Any, title: str = "Analysis Results"
    ) -> str:
        """Format the analysis result for display.

        Args:
            result: The analysis result
            title: The title for the result

        Returns:
            str: The formatted result
        """
        if isinstance(result, pd.DataFrame):
            # Always format DataFrames as tables
            return f"DATAFRAME_TABLE:{title}:{result.to_json()}"
        elif isinstance(result, pd.Series):
            # Convert Series to DataFrame for better display
            result_df = result.to_frame().reset_index()
            if len(result_df.columns) == 2:
                result_df.columns = ["Category", "Value"]
            return f"DATAFRAME_TABLE:{title}:{result_df.to_json()}"
        else:
            # For other results, try to create a summary table
            try:
                if hasattr(result, "to_frame"):
                    result_df = result.to_frame()
                    return f"DATAFRAME_TABLE:{title}:{result_df.to_json()}"
                else:
                    return str(result)
            except Exception as e:
                return f"Error formatting result: {str(e)}"
