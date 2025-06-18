"""Sample prompts agent for generating analysis suggestions."""

from langchain.agents import Tool
from typing import List
from .base_agent import BaseAgent


class SamplePromptsAgent(BaseAgent):
    """Agent responsible for generating sample analysis prompts."""

    def get_tool_name(self) -> str:
        """Get the name of the tool."""
        return "get_sample_prompts"

    def get_tool_description(self) -> str:
        """Get the description of the tool."""
        return "Generate relevant sample analysis prompts based on the current dataset structure"

    def create_tool(self) -> Tool:
        """Create the sample prompts tool."""
        return Tool(
            name=self.get_tool_name(),
            description=self.get_tool_description(),
            func=self._get_sample_prompts,
        )

    def _get_sample_prompts(self, query: str) -> str:
        """Generate relevant sample prompts for the current dataset.

        Args:
            query: The query (not used, but required by tool interface)

        Returns:
            str: The generated sample prompts
        """
        if self.df is None:
            return "No data loaded."

        try:
            # Analyze dataset structure
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = self.df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            prompts = []

            # Basic analysis prompts
            prompts.extend(self._get_basic_analysis_prompts())

            # Column-specific prompts
            if numeric_cols:
                prompts.extend(self._get_numerical_analysis_prompts(numeric_cols))

            if categorical_cols:
                prompts.extend(self._get_categorical_analysis_prompts(categorical_cols))

            # Visualization prompts
            prompts.extend(
                self._get_visualization_prompts(numeric_cols, categorical_cols)
            )

            return "\n".join(prompts)

        except Exception as e:
            return f"Error generating prompts: {str(e)}"

    def _get_basic_analysis_prompts(self) -> List[str]:
        """Get basic analysis prompts.

        Returns:
            List[str]: List of basic analysis prompts
        """
        return [
            "📊 **Basic Analysis:**",
            "• Show me summary statistics",
            "• Analyze missing values",
            "• What are the data types?",
        ]

    def _get_numerical_analysis_prompts(self, numeric_cols: List[str]) -> List[str]:
        """Get numerical analysis prompts.

        Args:
            numeric_cols: List of numerical column names

        Returns:
            List[str]: List of numerical analysis prompts
        """
        prompts = ["\n📈 **Numerical Analysis:**"]

        if numeric_cols:
            prompts.append(
                f"• Show correlation between {numeric_cols[0]} and other columns"
            )
            prompts.append(f"• What's the distribution of {numeric_cols[0]}?")

            if len(numeric_cols) > 1:
                prompts.append(f"• Compare {numeric_cols[0]} vs {numeric_cols[1]}")

        return prompts

    def _get_categorical_analysis_prompts(
        self, categorical_cols: List[str]
    ) -> List[str]:
        """Get categorical analysis prompts.

        Args:
            categorical_cols: List of categorical column names

        Returns:
            List[str]: List of categorical analysis prompts
        """
        prompts = ["\n🏷️ **Categorical Analysis:**"]

        if categorical_cols:
            prompts.append(f"• Show value counts for {categorical_cols[0]}")

            if len(categorical_cols) > 1:
                prompts.append(
                    f"• Cross-tabulate {categorical_cols[0]} and {categorical_cols[1]}"
                )

        return prompts

    def _get_visualization_prompts(
        self, numeric_cols: List[str], categorical_cols: List[str]
    ) -> List[str]:
        """Get visualization prompts.

        Args:
            numeric_cols: List of numerical column names
            categorical_cols: List of categorical column names

        Returns:
            List[str]: List of visualization prompts
        """
        prompts = ["\n📊 **Visualizations:**"]

        if numeric_cols:
            prompts.append(f"• Create histogram of {numeric_cols[0]}")

            if len(numeric_cols) > 1:
                prompts.append(
                    f"• Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"
                )

        if categorical_cols and numeric_cols:
            prompts.append(f"• Box plot of {numeric_cols[0]} by {categorical_cols[0]}")

        return prompts
