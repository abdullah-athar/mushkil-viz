"""
Planner agent for creating analysis plans.

This agent is responsible for:
- Analyzing dataset characteristics
- Creating a comprehensive analysis plan
- Determining appropriate analysis steps
- Prioritizing analysis tasks
"""

import json
from typing import List, Dict, Any
import logging

from .base_agent import BaseAgent
from ..schema import (
    WorkflowState, AnalysisPlan, AnalysisStep, DataSpec, DataType
)


class PlannerAgent(BaseAgent):
    """
    Agent responsible for creating comprehensive analysis plans.
    
    This agent takes a DataSpec and creates a detailed plan of analysis
    steps that should be performed on the dataset. The plan includes
    statistical analysis, data quality checks, visualizations, and
    correlation analysis based on the dataset characteristics.
    """
    
    def __init__(self, **kwargs):
        """Initialize the planner agent."""
        super().__init__(**kwargs)
        self.analysis_templates = self._load_analysis_templates()
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Create an analysis plan based on the dataset specification.
        
        Args:
            state: Current workflow state with data_spec populated
            
        Returns:
            Updated workflow state with analysis_plan populated
        """
        self.log_info("Creating analysis plan")
        
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for planner agent")
            
            # Create analysis plan using LLM
            analysis_plan = self._create_analysis_plan(state.data_spec)
            
            # Update state
            state.analysis_plan = analysis_plan
            
            # Set current_step to the first step in the plan
            if analysis_plan.steps:
                state.current_step = analysis_plan.steps[0].step_id
                self.log_info(f"Set current_step to {state.current_step}")
            
            self.log_info(f"Created analysis plan with {analysis_plan.total_steps} steps")
            
        except Exception as e:
            self.log_error(f"Failed to create analysis plan: {e}")
            raise
        
        return state
    
    def validate_state(self, state: WorkflowState) -> bool:
        """Validate that state has required data_spec."""
        return (
            hasattr(state, 'data_spec') and 
            state.data_spec is not None
        )
    
    def _create_analysis_plan(self, data_spec: DataSpec) -> AnalysisPlan:
        """
        Create a comprehensive analysis plan using LLM.
        
        Args:
            data_spec: Dataset specification
            
        Returns:
            Complete analysis plan
        """
        # Prepare context for LLM
        context = self._prepare_dataset_context(data_spec)
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt
        user_prompt = self._create_user_prompt(context)
        
        # Call LLM
        messages = self.format_messages(system_prompt, user_prompt)
        response = self.call_llm(messages)
        
        # Parse response
        analysis_plan = self._parse_llm_response(response, data_spec)
        
        return analysis_plan
    
    def _prepare_dataset_context(self, data_spec: DataSpec) -> Dict[str, Any]:
        """
        Prepare dataset context for LLM prompt.
        
        Args:
            data_spec: Dataset specification
            
        Returns:
            Context dictionary
        """
        # Get column information
        column_names = [
            col.name for col in data_spec.column_schema
        ]
        
        numeric_columns = [
            col.name for col in data_spec.column_schema
        ]
        
        categorical_columns = [
            col.name for col in data_spec.column_schema
        ]
        
        datetime_columns = [
            col.name for col in data_spec.column_schema
        ]
        
        # Get detailed column info
        column_details = []
        for col in data_spec.column_schema:
            column_details.append({
                "name": col.name,
                "dtype": col.dtype.value,
                "null_count": col.null_count,
                "unique_count": col.unique_count,
                "sample_values": col.sample_values[:5] if col.sample_values else []
            })
        
        return {
            "dataset_name": data_spec.uri.split('/')[-1] if '/' in data_spec.uri else data_spec.uri,
            "row_count": data_spec.row_count,
            "column_count": data_spec.column_count,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "columns_with_nulls": [col.name for col in data_spec.column_schema if col.null_count > 0],
            "schema_summary": column_details,
            "sample_data": data_spec.sample_rows[:3]  # First 3 rows
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the planner agent."""
        return """You are an expert data analyst and data scientist. Your task is to create a simple analysis plan with exactly 3 basic plotting steps for a given dataset.

Your analysis plan should include exactly 3 steps:
1. Basic data overview and summary statistics
2. Distribution plots for numeric variables (histograms, box plots)
3. Simple visualizations for categorical variables (bar charts, pie charts)

For each analysis step, provide:
- A clear title and description
- The type of analysis
- Target columns (if applicable)
- Expected artifacts (figures, tables, etc.)
- Priority level (1=highest, 3=lowest)

Return your response as a JSON object with the following structure:
{
    "dataset_name": "string",
    "total_steps": 3,
    "estimated_duration_minutes": number,
    "summary": "string",
    "steps": [
        {
            "step_id": 1,
            "title": "string",
            "description": "string",
            "analysis_type": "string",
            "target_columns": ["string"],
            "expected_artifacts": ["string"],
            "dependencies": [],
            "priority": 1
        }
    ]
}

Keep it simple and focused on basic plotting. No more than 3 steps total."""
    
    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create user prompt with dataset context.
        
        Args:
            context: Dataset context dictionary
            
        Returns:
            Formatted user prompt
        """
        prompt = f"""Please create a simple analysis plan with exactly 3 basic plotting steps for the following dataset:

Dataset: {context['dataset_name']}
Rows: {context['row_count']:,}
Columns: {context['column_count']}

Column Information:
"""
        
        for col_info in context['schema_summary']:
            prompt += f"- {col_info['name']}: {col_info['dtype']} ({col_info['null_count']} nulls, {col_info['unique_count']} unique values)\n"
        
        prompt += f"""

Column Types:
- Numeric columns: {', '.join(context['numeric_columns']) if context['numeric_columns'] else 'None'}
- Categorical columns: {', '.join(context['categorical_columns']) if context['categorical_columns'] else 'None'}
- Datetime columns: {', '.join(context['datetime_columns']) if context['datetime_columns'] else 'None'}

Sample Data (first 3 rows):
{json.dumps(context['sample_data'], indent=2)}

Please create exactly 3 simple analysis steps focused on basic plotting:
1. Data overview and summary statistics
2. Distribution plots for numeric variables (if any)
3. Simple visualizations for categorical variables (if any)

Keep it simple and focused on basic plotting. No more than 3 steps total."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, data_spec: DataSpec) -> AnalysisPlan:
        """
        Parse LLM response into AnalysisPlan object.
        
        Args:
            response: LLM response string
            data_spec: Original dataset specification
            
        Returns:
            Parsed AnalysisPlan
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[json_start:json_end]
            plan_data = json.loads(json_str)
            
            # Create AnalysisStep objects
            steps = []
            for step_data in plan_data.get('steps', []):
                step = AnalysisStep(
                    step_id=step_data['step_id'],
                    title=step_data['title'],
                    description=step_data['description'],
                    analysis_type=step_data['analysis_type'],
                    target_columns=step_data.get('target_columns', []),
                    expected_artifacts=step_data.get('expected_artifacts', []),
                    dependencies=step_data.get('dependencies', []),
                    priority=step_data.get('priority', 3)
                )
                steps.append(step)
            
            # Create AnalysisPlan
            analysis_plan = AnalysisPlan(
                dataset_name=plan_data.get('dataset_name', data_spec.uri.split('/')[-1]),
                total_steps=len(steps),
                steps=steps,
                estimated_duration_minutes=plan_data.get('estimated_duration_minutes', 30),
                summary=plan_data.get('summary', 'Comprehensive dataset analysis')
            )
            
            return analysis_plan
            
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
            # Fallback to template-based plan
            return self._create_fallback_plan(data_spec)
    
    def _create_fallback_plan(self, data_spec: DataSpec) -> AnalysisPlan:
        """
        Create a fallback analysis plan with exactly 3 basic plotting steps.
        
        Args:
            data_spec: Dataset specification
            
        Returns:
            Fallback analysis plan with 3 steps
        """
        self.log_warning("Using fallback analysis plan due to LLM parsing failure")
        
        steps = []
        step_id = 1
        
        # Step 1: Basic data overview and summary statistics
        steps.append(AnalysisStep(
            step_id=step_id,
            title="Data Overview and Summary Statistics",
            description="Generate basic summary statistics and overview of the dataset",
            analysis_type="overview",
            expected_artifacts=["summary_statistics.csv", "data_overview.txt"],
            priority=1
        ))
        step_id += 1
        
        # Step 2: Distribution plots for numeric variables
        numeric_columns = [col.name for col in data_spec.column_schema if col.dtype in [DataType.INTEGER, DataType.FLOAT]]
        if numeric_columns:
            steps.append(AnalysisStep(
                step_id=step_id,
                title="Numeric Variable Distributions",
                description="Create histograms and box plots for numeric variables",
                analysis_type="visualization",
                target_columns=numeric_columns,
                expected_artifacts=["numeric_distributions.png", "box_plots.png"],
                priority=2
            ))
        else:
            # If no numeric columns, create a general data visualization step
            steps.append(AnalysisStep(
                step_id=step_id,
                title="Data Visualization",
                description="Create basic visualizations of the dataset",
                analysis_type="visualization",
                expected_artifacts=["data_visualization.png"],
                priority=2
            ))
        step_id += 1
        
        # Step 3: Categorical variable analysis
        categorical_columns = [col.name for col in data_spec.column_schema if col.dtype == DataType.CATEGORICAL]
        if categorical_columns:
            steps.append(AnalysisStep(
                step_id=step_id,
                title="Categorical Variable Analysis",
                description="Create bar charts and pie charts for categorical variables",
                analysis_type="categorical_analysis",
                target_columns=categorical_columns,
                expected_artifacts=["categorical_analysis.png", "frequency_charts.png"],
                priority=3
            ))
        else:
            # If no categorical columns, create a general analysis step
            steps.append(AnalysisStep(
                step_id=step_id,
                title="Data Analysis Summary",
                description="Generate a summary analysis of the dataset",
                analysis_type="summary",
                expected_artifacts=["analysis_summary.txt"],
                priority=3
            ))
        
        return AnalysisPlan(
            dataset_name=data_spec.uri.split('/')[-1] if '/' in data_spec.uri else data_spec.uri,
            total_steps=3,
            steps=steps,
            estimated_duration_minutes=15,
            summary="Basic 3-step analysis: overview, numeric distributions, and categorical analysis"
        )
    
    def _load_analysis_templates(self) -> Dict[str, Any]:
        """Load predefined analysis templates."""
        # This could be loaded from a file in a real implementation
        return {
            "overview": {
                "title": "Data Overview and Summary Statistics",
                "description": "Generate basic summary statistics and overview of the dataset",
                "analysis_type": "overview"
            },
            "visualization": {
                "title": "Data Visualization", 
                "description": "Create basic visualizations of the dataset",
                "analysis_type": "visualization"
            },
            "categorical_analysis": {
                "title": "Categorical Variable Analysis",
                "description": "Create bar charts and pie charts for categorical variables",
                "analysis_type": "categorical_analysis"
            }
        } 