"""
Base agent class for LLM-powered analysis agents.

This module provides a common interface and utilities for all agents
in the analysis pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import os
import json
import re
from json import JSONDecoder, JSONDecodeError
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
    - JSON parsing utilities
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

    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Robustly extract and parse JSON from an LLM response.
        
        Args:
            response: LLM response string that may contain JSON
            
        Returns:
            Dict[str, Any]: Parsed JSON object
            
        Raises:
            ValueError: If no valid JSON found in response
        """
        try:
            # 1) Strip any leading/trailing markdown fences
            clean = re.sub(r'```(?:json|python)?\s*|\s*```$', '', response).strip()

            # 2) Try to locate the first JSON object using the JSONDecoder.raw_decode
            decoder = JSONDecoder()
            obj, end = decoder.raw_decode(clean)

            # 3) Return the parsed object
            return obj
            
        except (JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, try to find JSON object manually
            self.log_warning(f"Initial JSON parsing failed: {e}, trying manual extraction")
            
            try:
                # Strategy 1: Find the first { and last } to extract JSON manually
                start_idx = clean.find('{')
                if start_idx == -1:
                    raise ValueError("No JSON object found in response")
                
                # Count braces to find the end of the JSON object
                brace_count = 0
                end_idx = -1
                
                for i in range(start_idx, len(clean)):
                    if clean[i] == '{':
                        brace_count += 1
                    elif clean[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx == -1:
                    # Strategy 2: If JSON is incomplete, try to find a reasonable truncation point
                    self.log_warning("JSON appears incomplete, trying truncation strategy")
                    
                    # Look for the last complete field before truncation
                    for i in range(len(clean) - 1, start_idx, -1):
                        if clean[i] == '"' and i > 0 and clean[i-1] != '\\':
                            # Found a potential end of a string field
                            # Try to close the JSON properly
                            truncated = clean[start_idx:i+1]
                            
                            # Add closing braces if needed
                            open_braces = truncated.count('{') - truncated.count('}')
                            if open_braces > 0:
                                truncated += '}' * open_braces
                            
                            try:
                                return json.loads(truncated)
                            except json.JSONDecodeError:
                                continue
                    
                    raise ValueError("Cannot parse incomplete JSON object")
                
                json_str = clean[start_idx:end_idx]
                
                # Validate the extracted JSON
                return json.loads(json_str)
                
            except json.JSONDecodeError as parse_error:
                # Final fallback: try to extract just the essential fields
                self.log_error(f"All JSON parsing strategies failed: {parse_error}")
                
                # Try to extract code field manually as a last resort
                code_match = re.search(r'"code"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', clean)
                if code_match:
                    code_content = code_match.group(1)
                    # Unescape the code
                    code_content = code_content.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    
                    # Return minimal valid structure
                    return {
                        "code": code_content,
                        "imports": ["pandas", "numpy", "matplotlib"],
                        "expected_artifacts": [],
                        "execution_timeout_seconds": 300,
                        "memory_limit_mb": 512,
                        "safety_level": "medium"
                    }
                
                raise json.JSONDecodeError(f"Cannot extract any valid JSON from response: {parse_error}", "", 0) 