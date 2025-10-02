"""
Data structures for the LLM-powered agentic analysis framework.

This module defines the core Pydantic models that represent the state
and data flow between different agents in the analysis pipeline.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
import pandas as pd


class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    CSV = "csv"
    PARQUET = "parquet"
    SQL = "sql"
    EXCEL = "excel"
    JSON = "json"


class DataType(str, Enum):
    """Common data types for schema inference."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class ColumnSchema(BaseModel):
    """Schema information for a single column."""
    name: str
    dtype: DataType
    null_count: int = 0
    unique_count: Optional[int] = None
    sample_values: List[Any] = Field(default_factory=list)
    description: Optional[str] = None


class DataSpec(BaseModel):
    """
    Complete specification of a dataset including metadata and sample data.
    
    This is the output of the loader agent and serves as input to the planner.
    """
    uri: str = Field(..., description="Path or connection string to the dataset")
    format: DatasetFormat = Field(..., description="Format of the dataset")
    row_count: int = Field(..., description="Total number of rows in the dataset")
    column_count: int = Field(..., description="Total number of columns")
    column_schema: List[ColumnSchema] = Field(..., description="Schema information for each column")
    sample_rows: List[Dict[str, Any]] = Field(..., description="First few rows as dictionaries")
    memory_usage_mb: Optional[float] = Field(None, description="Approximate memory usage")
    file_size_mb: Optional[float] = Field(None, description="File size if applicable")
    
    class Config:
        arbitrary_types_allowed = True


class AnalysisStep(BaseModel):
    """
    A single step in the analysis plan.
    
    Each step represents a specific analysis task that will be executed
    by the coder and runtime agents.
    """
    step_id: int = Field(..., description="Sequential identifier for this step")
    title: str = Field(..., description="Human-readable title for this analysis step")
    description: str = Field(..., description="Detailed description of what this step will do")
    analysis_type: str = Field(..., description="Type of analysis (e.g., 'statistics', 'visualization', 'correlation')")
    target_columns: Optional[List[str]] = Field(None, description="Specific columns to analyze, if applicable")
    expected_artifacts: List[str] = Field(default_factory=list, description="Expected output files/figures")
    dependencies: List[int] = Field(default_factory=list, description="Step IDs this step depends on")
    priority: int = Field(default=1, description="Priority level (1=highest, 5=lowest)")


class AnalysisPlan(BaseModel):
    """
    Complete analysis plan with ordered steps.
    
    This is the output of the planner agent and guides the execution
    of the coder and runtime agents.
    """
    dataset_name: str = Field(..., description="Name of the dataset being analyzed")
    total_steps: int = Field(..., description="Total number of analysis steps")
    steps: List[AnalysisStep] = Field(..., description="Ordered list of analysis steps")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated time to complete all steps")
    summary: str = Field(..., description="High-level summary of the analysis plan")
    
    class Config:
        json_encoders = {
            AnalysisStep: lambda v: v.dict()
        }


class CodeArtifact(BaseModel):
    """
    Expected output artifact from code execution.
    """
    name: str = Field(..., description="Name of the artifact")
    type: str = Field(..., description="Type (figure, table, text, etc.)")
    path: str = Field(..., description="Expected file path")
    description: str = Field(..., description="What this artifact represents")


class CodeBundle(BaseModel):
    """
    Generated code along with expected outputs and metadata.
    
    This is the output of the coder agent and input to the runtime agent.
    """
    step_id: int = Field(..., description="Analysis step this code corresponds to")
    code: str = Field(..., description="Python code to execute")
    imports: List[str] = Field(default_factory=list, description="Required Python imports")
    expected_artifacts: List[CodeArtifact] = Field(default_factory=list, description="Expected output artifacts")
    execution_timeout_seconds: int = Field(default=300, description="Maximum execution time")
    memory_limit_mb: int = Field(default=512, description="Memory limit for execution")
    safety_level: str = Field(default="medium", description="Safety level for code execution")


class ExecutionStatus(str, Enum):
    """Status of code execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    SAFETY_VIOLATION = "safety_violation"


class ExecutionResult(BaseModel):
    """
    Results from code execution including outputs and artifacts.
    
    This is the output of the runtime agent and input to the grader agent.
    """
    step_id: int = Field(..., description="Analysis step that was executed")
    status: ExecutionStatus = Field(..., description="Execution status")
    stdout: str = Field(default="", description="Standard output from execution")
    stderr: str = Field(default="", description="Standard error from execution")
    execution_time_seconds: float = Field(..., description="Actual execution time")
    memory_used_mb: Optional[float] = Field(None, description="Memory used during execution")
    artifacts: List[Dict[str, Any]] = Field(default_factory=list, description="Generated artifacts")
    return_value: Optional[Any] = Field(None, description="Return value from the executed code")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    
    class Config:
        arbitrary_types_allowed = True


class GradeVerdict(str, Enum):
    """Possible grading verdicts."""
    PASS = "pass"
    REPLAN = "replan"
    REGENERATE = "regenerate"
    FAIL = "fail"


class GradeReport(BaseModel):
    """
    Evaluation of code execution results and quality.
    
    This is the output of the grader agent and determines the next
    action in the workflow.
    """
    step_id: int = Field(..., description="Analysis step being graded")
    verdict: GradeVerdict = Field(..., description="Overall verdict on the execution")
    score: float = Field(..., description="Numerical score from 0.0 to 1.0")
    comments: List[str] = Field(default_factory=list, description="Detailed comments about the execution")
    code_quality_score: float = Field(..., description="Code quality score from 0.0 to 1.0")
    output_quality_score: float = Field(..., description="Output quality score from 0.0 to 1.0")
    safety_score: float = Field(..., description="Safety score from 0.0 to 1.0")
    suggested_improvements: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    can_proceed: bool = Field(..., description="Whether the workflow can proceed to the next step")
    requires_regeneration: bool = Field(..., description="Whether the code needs to be regenerated")
    requires_replanning: bool = Field(..., description="Whether the entire plan needs to be revised")


class AnalysisReport(BaseModel):
    """
    Final comprehensive report with all findings and artifacts.
    
    This is the output of the reporter agent and represents the final
    deliverable of the analysis pipeline.
    """
    dataset_name: str = Field(..., description="Name of the analyzed dataset")
    analysis_summary: str = Field(..., description="High-level summary of findings")
    key_insights: List[str] = Field(..., description="Key insights discovered")
    statistics_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    visualizations: List[Dict[str, Any]] = Field(default_factory=list, description="Generated visualizations")
    data_quality_issues: List[str] = Field(default_factory=list, description="Data quality issues found")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on analysis")
    markdown_content: str = Field(..., description="Complete markdown report")
    execution_summary: Dict[str, Any] = Field(default_factory=dict, description="Execution statistics")
    artifacts_directory: str = Field(..., description="Directory containing all generated artifacts")
    
    class Config:
        arbitrary_types_allowed = True


class WorkflowState(BaseModel):
    """
    Complete state of the analysis workflow.
    
    This is the main state object that flows through all agents
    in the LangGraph workflow.
    """
    # Input data
    dataset_uri: str = Field(..., description="URI of the dataset to analyze")
    dataset_format: DatasetFormat = Field(..., description="Format of the dataset")
    
    # Agent outputs
    data_spec: Optional[DataSpec] = Field(None, description="Dataset specification from loader")
    analysis_plan: Optional[AnalysisPlan] = Field(None, description="Analysis plan from planner")
    code_bundles: Dict[int, CodeBundle] = Field(default_factory=dict, description="Generated code for each step")
    execution_results: Dict[int, ExecutionResult] = Field(default_factory=dict, description="Execution results for each step")
    grade_reports: Dict[int, GradeReport] = Field(default_factory=dict, description="Grading reports for each step")
    final_report: Optional[AnalysisReport] = Field(None, description="Final analysis report")
    
    # Workflow control
    current_step: int = Field(default=0, description="Current analysis step being processed")
    completed_steps: List[int] = Field(default_factory=list, description="Successfully completed steps")
    failed_steps: List[int] = Field(default_factory=list, description="Steps that failed execution")
    max_iterations: int = Field(default=3, description="Maximum iterations per step")
    current_iteration: int = Field(default=0, description="Current iteration for the current step")
    next_step: Optional[str] = Field(None, description="Next step in the workflow routing")
    
    # Metadata
    workflow_id: str = Field(..., description="Unique identifier for this workflow")
    start_time: Optional[str] = Field(None, description="Workflow start timestamp")
    end_time: Optional[str] = Field(None, description="Workflow end timestamp")
    total_execution_time: Optional[float] = Field(None, description="Total execution time in seconds")
    
    class Config:
        arbitrary_types_allowed = True 