"""
Base agent class for LLM-powered analysis agents.

This module provides a common interface and utilities for all agents
in the analysis pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import os
from google import genai
from google.genai import types

from ..schema import WorkflowState


class BaseAgent(ABC):
    """
    Base class for all LLM agents in the analysis pipeline.
    
    Provides common functionality for:
    - LLM client management
    - Message formatting
    - Error handling
    - Logging
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            api_key: Gemini API key (if not provided, will use environment)
            **kwargs: Additional arguments passed to the LLM
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Gemini client
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable or api_key parameter is required")
        
        # Create client - API key will be picked up from environment automatically
        self.client = genai.Client()
        
        # Store generation config for reuse
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
    @abstractmethod
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Process the workflow state and return updated state.
        
        This is the main method that each agent must implement.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> str:
        """Format system and user prompts into a single prompt for Gemini."""
        return f"{system_prompt}\n\n{user_prompt}"
    
    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM with formatted prompt.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            LLM response as string
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config
            )
            return response.text
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = True):
        """Log an error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def validate_state(self, state: WorkflowState) -> bool:
        """
        Validate that the state has required data for this agent.
        
        Args:
            state: Workflow state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        # Base implementation - subclasses should override
        return True
    
    def get_agent_name(self) -> str:
        """Get the name of this agent."""
        return self.__class__.__name__ 