"""
Coder agent for generating Python code.

This agent is responsible for:
- Generating Python code for each analysis step
- Ensuring code safety and best practices
- Creating appropriate visualizations
- Handling different data types and analysis types
"""

from typing import List, Dict, Any, Optional
import logging
import ast
import re
import json

from .base_agent import BaseAgent
from ..schema import (
    WorkflowState, CodeBundle, CodeArtifact, AnalysisStep, DataSpec, DataType
)

class CoderAgent(BaseAgent):
    """
    Agent responsible for generating Python code for analysis steps.
    
    This agent takes an AnalysisStep and generates safe, executable
    Python code that performs the specified analysis. The code includes
    proper imports, error handling, and artifact generation.
    """
    
    def __init__(self, **kwargs):
        """Initialize the coder agent with appropriate token limits for code generation."""
        # Set higher token limit for code generation while still keeping it manageable
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 6000
        super().__init__(**kwargs)
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Generate Python code for the current analysis step.
        
        Args:
            state: Current workflow state with analysis_plan and current_step
            
        Returns:
            Updated workflow state with code_bundle for current step
        """
        self.log_info(f"Generating code for step {state.current_step}")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Validate state
                if not self.validate_state(state):
                    raise ValueError("Invalid state for coder agent")
                
                # Get current analysis step
                current_step = self._get_current_step(state)
                
                # Generate code using LLM
                code_bundle = self._generate_code(current_step, state.data_spec)
                
                # Update state
                state.code_bundles[state.current_step] = code_bundle
                self.log_info(f"Generated code for step {state.current_step}: {current_step.title}")
                return state
             
            except (ValueError, json.JSONDecodeError) as e:
                retry_count += 1
                self.log_warning(f"JSON parsing failed for step {state.current_step} (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    self.log_error(f"Failed to generate valid code after {max_retries} attempts for step {state.current_step}")
                    raise ValueError(f"Failed to generate valid code after {max_retries} attempts: {e}")
                
                # Continue to retry
                continue
                
            except Exception as e:
                self.log_error(f"Failed to generate code for step {state.current_step}: {e}")
                raise
        
        return state
    
    def validate_state(self, state: WorkflowState) -> bool:
        """Validate that state has required analysis plan and current step."""
        return (
            hasattr(state, 'analysis_plan') and 
            state.analysis_plan is not None and
            hasattr(state, 'current_step') and
            state.current_step >= 0 and
            hasattr(state, 'data_spec') and
            state.data_spec is not None
        )
    
    def _get_current_step(self, state: WorkflowState) -> AnalysisStep:
        """
        Get the current analysis step.
        
        Args:
            state: Current workflow state
            
        Returns:
            Current analysis step
        """
        for step in state.analysis_plan.steps:
            if step.step_id == state.current_step:
                return step
        
        raise ValueError(f"Step {state.current_step} not found in analysis plan")
    
    def _generate_code(self, step: AnalysisStep, data_spec: DataSpec) -> CodeBundle:
        """
        Generate Python code for the analysis step.
        
        Args:
            step: Analysis step to generate code for
            data_spec: Dataset specification
            
        Returns:
            CodeBundle with generated code and metadata
        """
        # Prepare context for LLM
        context = self._prepare_coding_context(step, data_spec)
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt
        user_prompt = self._create_user_prompt(context)
        
        # Call LLM
        messages = self.format_messages(system_prompt, user_prompt)
        response = self.call_llm(messages)
        
        # Parse response and validate code
        code_bundle = self._parse_and_validate_response(response, step, data_spec)
        
        return code_bundle
    
    def _prepare_coding_context(self, step: AnalysisStep, data_spec: DataSpec) -> Dict[str, Any]:
        """
        Prepare context for code generation.
        
        Args:
            step: Analysis step
            data_spec: Dataset specification
            
        Returns:
            Context dictionary
        """
        # Get target columns info
        target_columns_info = []
        if step.target_columns:
            for col_name in step.target_columns:
                col_schema = next((col for col in data_spec.column_schema if col.name == col_name), None)
                if col_schema:
                    target_columns_info.append({
                        "name": col_name,
                        "dtype": col_schema.dtype.value,
                        "null_count": col_schema.null_count,
                        "unique_count": col_schema.unique_count
                    })
        
        return {
            "step_id": step.step_id,
            "title": step.title,
            "description": step.description,
            "analysis_type": step.analysis_type,
            "target_columns": step.target_columns,
            "target_columns_info": target_columns_info,
            "expected_artifacts": step.expected_artifacts,
            "dependencies": step.dependencies,
            "priority": step.priority,
            "dataset_uri": data_spec.uri,
            "dataset_info": {
                "row_count": data_spec.row_count,
                "column_count": data_spec.column_count,
                "schema": [
                    {
                        "name": col.name,
                        "dtype": col.dtype.value,
                        "null_count": col.null_count
                    }
                    for col in data_spec.column_schema
                ]
            }
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the coder agent."""
        return """You are an expert Python data scientist and analyst. Your task is to generate safe, efficient, and well-documented Python code for data analysis tasks.

IMPORTANT SAFETY REQUIREMENTS:
1. NEVER use eval(), exec(), or any code execution functions
2. NEVER use os.system(), subprocess, or any system commands
3. NEVER use file operations outside of DATA_ARTIFACTS_DIR
4. NEVER use network requests or external API calls
5. NEVER use any potentially dangerous libraries
6. ALWAYS use try-except blocks for error handling
7. ALWAYS validate data before processing

ALLOWED LIBRARIES:
- pandas, numpy, matplotlib, seaborn, plotly
- scipy, sklearn (for analysis only)
- json, csv, pathlib (for file operations in artifacts directory only)

CRITICAL FILE SAVING REQUIREMENTS:
1. ALWAYS use the DATA_ARTIFACTS_DIR variable for saving files - this is a global variable provided by the runtime
2. NEVER use hardcoded paths or relative paths for saving files
3. Use DATA_ARTIFACTS_DIR for ALL file operations (plots, CSV files, JSON files, etc.)
4. Example correct usage:
   - plt.savefig(os.path.join(DATA_ARTIFACTS_DIR, 'plot.png'))
   - df.to_csv(os.path.join(DATA_ARTIFACTS_DIR, 'results.csv'))
   - with open(os.path.join(DATA_ARTIFACTS_DIR, 'summary.txt'), 'w') as f: f.write(text)
5. Always ensure DATA_ARTIFACTS_DIR exists before saving files

CODE REQUIREMENTS:
1. Keep code CONCISE and FOCUSED - avoid overly verbose or repetitive code
2. Write efficient, readable code with clear purpose
3. Load data from the DATASET_URI variable (provided by runtime)
4. Perform the specified analysis efficiently
5. Generate visualizations and save them to DATA_ARTIFACTS_DIR
6. Save any data outputs to CSV files in DATA_ARTIFACTS_DIR
7. Include essential error handling (not excessive)
8. Add brief, clear comments explaining key steps
9. Return a dictionary with results and artifact paths
10. Ensure all Python code is syntactically correct
11. Use proper indentation (4 spaces per level)
12. Keep the main() function concise and focused
13. Aim for code under 100 lines when possible

SYNTAX REQUIREMENTS:
- All code must be valid Python 3.x syntax
- All try-except blocks must have proper except clauses
- All function definitions must be complete
- All string literals must be properly quoted
- All parentheses, brackets, and braces must be balanced
- Use consistent indentation (4 spaces per level)

CRITICAL JSON FORMATTING REQUIREMENTS:
1. Return ONLY valid JSON - no markdown formatting or extra text
2. Use double quotes for all strings
3. Keep the total JSON response under 8000 characters to avoid truncation
4. Properly escape special characters in the code field:
   - Newlines: \\n
   - Tabs: \\t  
   - Double quotes: \\"
   - Backslashes: \\\\
5. Ensure the JSON is complete and well-formed
6. Do not include any text before or after the JSON
7. Make sure all brackets and braces are properly balanced
8. Do not include trailing commas in arrays or objects
9. The "code" field must contain complete, syntactically correct Python code
10. Write CONCISE code to keep JSON response size manageable

RESPONSE FORMAT:
Return your response as a JSON object with the following structure:
{
    "code": "string (concise Python code with proper escaping - keep under 6000 chars)",
    "imports": ["string (list of required imports)"],
    "expected_artifacts": [
        {
            "name": "string",
            "type": "string (figure, table, text, etc.)",
            "path": "string (expected file path)",
            "description": "string"
        }
    ],
    "execution_timeout_seconds": number,
    "memory_limit_mb": number,
    "safety_level": "string (low, medium, high)"
}

IMPORTANT: Write CONCISE, FOCUSED code. Avoid verbose comments and repetitive operations. The "code" field should contain efficient Python code as a string.

CODE TEMPLATE EXAMPLE:
```python
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load data from DATASET_URI (provided by runtime)
    df = pd.read_csv(DATASET_URI)
    
    # Perform analysis
    # ... your analysis code here ...
    
    # Save visualizations to DATA_ARTIFACTS_DIR
    plt.figure(figsize=(10, 6))
    # ... your plotting code here ...
    plt.savefig(os.path.join(DATA_ARTIFACTS_DIR, 'analysis_plot.png'))
    plt.close()
    
    # Save data to DATA_ARTIFACTS_DIR
    results_df.to_csv(os.path.join(DATA_ARTIFACTS_DIR, 'results.csv'))
    
    # Return results
    return {
        "status": "success",
        "artifacts": [
            {"name": "analysis_plot.png", "path": os.path.join(DATA_ARTIFACTS_DIR, 'analysis_plot.png')},
            {"name": "results.csv", "path": os.path.join(DATA_ARTIFACTS_DIR, 'results.csv')}
        ],
        "summary": "Analysis completed successfully"
    }
    
```"""
    
    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create user prompt for code generation.
        
        Args:
            context: Coding context dictionary
            
        Returns:
            Formatted user prompt
        """
        prompt = f"""Please generate Python code for the following analysis step:

Step ID: {context['step_id']}
Title: {context['title']}
Description: {context['description']}
Analysis Type: {context['analysis_type']}

Dataset Information:
- Dataset URI: {context.get('dataset_uri', 'dataset.csv')}
- Rows: {context['dataset_info']['row_count']:,}
- Columns: {context['dataset_info']['column_count']}

Target Columns: {context['target_columns'] if context['target_columns'] else 'All columns'}

Expected Artifacts: {', '.join(context['expected_artifacts'])}

Target Columns Information:
"""
        
        for col_info in context['target_columns_info']:
            prompt += f"- {col_info['name']}: {col_info['dtype']} ({col_info['null_count']} nulls, {col_info['unique_count']} unique values)\n"
        
        prompt += f"""

Dataset Schema:
"""
        
        for col_info in context['dataset_info']['schema']:
            prompt += f"- {col_info['name']}: {col_info['dtype']} ({col_info['null_count']} nulls)\n"
        
        prompt += f"""

Please generate Python code that:
1. Loads the dataset from the URI: {context.get('dataset_uri', 'dataset.csv')}
2. Performs the specified analysis: {context['analysis_type']}
3. Creates appropriate visualizations and saves them to DATA_ARTIFACTS_DIR
4. Saves any data outputs to CSV files in DATA_ARTIFACTS_DIR
5. Returns a dictionary with results and artifact paths
6. Includes comprehensive error handling and logging

IMPORTANT: The runtime provides these global variables:
- DATA_ARTIFACTS_DIR: Directory where all files should be saved
- DATASET_URI: Path to the dataset file

Use these variables for all file operations. Example:
- plt.savefig(os.path.join(DATA_ARTIFACTS_DIR, 'plot.png'))
- df.to_csv(os.path.join(DATA_ARTIFACTS_DIR, 'results.csv'))

The code should be production-ready and follow best practices for data analysis.

Return your response as a valid JSON object with the specified structure."""
        
        return prompt
    
    def _parse_and_validate_response(self, response: str, step: AnalysisStep, data_spec: DataSpec) -> CodeBundle:
        """
        Parse LLM response and validate the generated code.
        
        Args:
            response: LLM response string
            step: Original analysis step
            data_spec: Dataset specification
            
        Returns:
            Parsed and validated CodeBundle
        """
        # Clean the response
        cleaned_response = response.strip()
        
        # Remove markdown formatting if present
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Try to extract JSON from the response
        try:
            code_data = self.extract_json_from_response(cleaned_response)
        except (ValueError, json.JSONDecodeError) as e:
            # Re-raise as JsonDecodeError for consistent error handling
            raise json.JSONDecodeError(f"Failed to parse JSON from LLM response: {e}")
        
        # Validate required fields
        if 'code' not in code_data:
            raise ValueError("Missing 'code' field in LLM response")
        
        # Validate Python syntax
        #self._validate_python_syntax(code_data['code'])
        
        # Create CodeArtifact objects
        expected_artifacts = []
        for artifact_data in code_data.get('expected_artifacts', []):
            artifact = CodeArtifact(
                name=artifact_data['name'],
                type=artifact_data['type'],
                path=artifact_data['path'],
                description=artifact_data['description']
            )
            expected_artifacts.append(artifact)
        
        # Create CodeBundle
        code_bundle = CodeBundle(
            step_id=step.step_id,
            code=code_data['code'],
            imports=code_data.get('imports', []),
            expected_artifacts=expected_artifacts,
            execution_timeout_seconds=code_data.get('execution_timeout_seconds', 300),
            memory_limit_mb=code_data.get('memory_limit_mb', 512),
            safety_level=code_data.get('safety_level', 'medium')
        )
        
        return code_bundle
    
    def _validate_python_syntax(self, code: str) -> None:
        """
        Validate that the generated Python code has correct syntax.
        
        Args:
            code: Python code string to validate
            
        Raises:
            ValueError: If code has syntax errors
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.log_error(f"Python syntax error: {e}")
            raise ValueError(f"Generated code has syntax errors: {e}")
    
    def _infer_artifact_type(self, artifact_name: str) -> str:
        """Infer artifact type from filename."""
        if any(ext in artifact_name.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']):
            return 'figure'
        elif any(ext in artifact_name.lower() for ext in ['.csv', '.xlsx', '.json']):
            return 'table'
        elif any(ext in artifact_name.lower() for ext in ['.txt', '.md']):
            return 'text'
        else:
            return 'data' 