"""Base agent class for the multi-agent system."""

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool
import os
from abc import ABC, abstractmethod
from typing import Optional


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize the base agent.

        Args:
            df: Optional pandas DataFrame to work with
        """
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_GEMINI_KEY"),
            temperature=0.1,
        )
        self.df = df

    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe for analysis.

        Args:
            df: The pandas DataFrame to analyze
        """
        self.df = df

    @abstractmethod
    def create_tool(self) -> Tool:
        """Create and return the agent's tool.

        Returns:
            Tool: The LangChain tool for this agent
        """
        pass

    @abstractmethod
    def get_tool_name(self) -> str:
        """Get the name of the tool.

        Returns:
            str: The tool name
        """
        pass

    @abstractmethod
    def get_tool_description(self) -> str:
        """Get the description of the tool.

        Returns:
            str: The tool description
        """
        pass
