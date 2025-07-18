"""
Reporter agent for synthesizing final analysis reports.

This agent is responsible for:
- Collecting all analysis results and artifacts
- Synthesizing findings into a comprehensive report
- Generating markdown with embedded visualizations
- Providing key insights and recommendations
"""

import json
import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .base_agent import BaseAgent
from ..schema import (
    WorkflowState, AnalysisReport, ExecutionResult, GradeReport, DataSpec
)


class ReporterAgent(BaseAgent):
    """
    Agent responsible for creating comprehensive final reports.
    
    This agent collects all analysis results, artifacts, and findings
    to create a complete markdown report with embedded visualizations
    and insights.
    """
    
    def __init__(self, **kwargs):
        """Initialize the reporter agent."""
        super().__init__(**kwargs)
        self.report_templates = self._load_report_templates()
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Create comprehensive final report from all analysis results.
        
        Args:
            state: Current workflow state with all analysis results
            
        Returns:
            Updated workflow state with final_report populated
        """
        self.log_info("Creating comprehensive final report")
        
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for reporter agent")
            
            # Create comprehensive report
            final_report = self._create_final_report(state)
            
            # Update state
            state.final_report = final_report
            self.log_info("Successfully created final report")
            
        except Exception as e:
            self.log_error(f"Failed to create final report: {e}")
            raise
        
        return state
    
    def validate_state(self, state: WorkflowState) -> bool:
        """Validate that state has required analysis results."""
        return (
            hasattr(state, 'data_spec') and 
            state.data_spec is not None and
            hasattr(state, 'analysis_plan') and 
            state.analysis_plan is not None and
            hasattr(state, 'execution_results') and 
            len(state.execution_results) > 0
        )
    
    def _create_final_report(self, state: WorkflowState) -> AnalysisReport:
        """
        Create comprehensive final report.
        
        Args:
            state: Complete workflow state
            
        Returns:
            Complete AnalysisReport
        """
        # Prepare context for LLM
        context = self._prepare_report_context(state)
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt
        user_prompt = self._create_user_prompt(context)
        
        # Call LLM
        messages = self.format_messages(system_prompt, user_prompt)
        response = self.call_llm(messages)
        
        # Parse response and create report
        analysis_report = self._parse_llm_response(response, state)
        
        return analysis_report
    
    def _prepare_report_context(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Prepare context for report generation.
        
        Args:
            state: Workflow state
            
        Returns:
            Context dictionary
        """
        # Collect all execution results
        execution_summary = {}
        for step_id, result in state.execution_results.items():
            execution_summary[step_id] = {
                "status": result.status.value,
                "execution_time": result.execution_time_seconds,
                "artifacts": result.artifacts,
                "error_message": result.error_message
            }
        
        # Collect all grade reports
        grade_summary = {}
        for step_id, grade in state.grade_reports.items():
            grade_summary[step_id] = {
                "verdict": grade.verdict.value,
                "score": grade.score,
                "comments": grade.comments
            }
        
        # Collect key insights from successful executions
        key_insights = []
        for step_id, result in state.execution_results.items():
            if result.status.value == "success" and result.return_value:
                if isinstance(result.return_value, dict):
                    # Extract insights from return value
                    if "insights" in result.return_value:
                        key_insights.extend(result.return_value["insights"])
                    elif "key_findings" in result.return_value:
                        key_insights.extend(result.return_value["key_findings"])
        
        # Collect data quality issues
        data_quality_issues = []
        if state.data_spec:
            for col in state.data_spec.column_schema:
                if col.null_count > 0:
                    data_quality_issues.append(f"Column '{col.name}' has {col.null_count} null values")
        
        return {
            "dataset_name": state.data_spec.uri.split('/')[-1] if '/' in state.data_spec.uri else state.data_spec.uri,
            "dataset_info": {
                "row_count": state.data_spec.row_count,
                "column_count": state.data_spec.column_count,
                "format": state.data_spec.format.value,
                "memory_usage_mb": state.data_spec.memory_usage_mb
            },
            "analysis_plan": {
                "total_steps": state.analysis_plan.total_steps,
                "summary": state.analysis_plan.summary,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "title": step.title,
                        "description": step.description,
                        "analysis_type": step.analysis_type
                    }
                    for step in state.analysis_plan.steps
                ]
            },
            "execution_summary": execution_summary,
            "grade_summary": grade_summary,
            "key_insights": key_insights,
            "data_quality_issues": data_quality_issues,
            "completed_steps": state.completed_steps,
            "failed_steps": state.failed_steps,
            "total_execution_time": state.total_execution_time
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the reporter agent."""
        return """You are an expert data analyst and technical writer. Your task is to create a comprehensive, professional analysis report based on the results of automated data analysis.

REPORT REQUIREMENTS:
1. Create a clear, well-structured markdown report
2. Include an executive summary with key findings
3. Provide detailed analysis sections for each completed step
4. Include visualizations and artifacts where available
5. Highlight data quality issues and recommendations
6. Provide actionable insights and next steps
7. Use professional, clear language suitable for stakeholders

REPORT STRUCTURE:
1. Executive Summary
2. Dataset Overview
3. Analysis Results (by step)
4. Key Insights
5. Data Quality Assessment
6. Recommendations
7. Technical Details

Return your response as a JSON object with the following structure:
{
    "dataset_name": "string",
    "analysis_summary": "string (executive summary)",
    "key_insights": ["string (list of key insights)"],
    "statistics_summary": {"object (summary statistics)"},
    "visualizations": ["string (list of visualization descriptions)"],
    "data_quality_issues": ["string (list of data quality issues)"],
    "recommendations": ["string (list of recommendations)"],
    "markdown_content": "string (complete markdown report)",
    "execution_summary": {"object (execution statistics)"},
    "artifacts_directory": "string (path to artifacts)"
}

The markdown_content should be a complete, professional report that can be directly used."""
    
    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create user prompt for report generation.
        
        Args:
            context: Report context dictionary
            
        Returns:
            Formatted user prompt
        """
        prompt = f"""Please create a comprehensive analysis report for the following dataset and analysis results:

DATASET: {context['dataset_name']}
- Rows: {context['dataset_info']['row_count']:,}
- Columns: {context['dataset_info']['column_count']}
- Format: {context['dataset_info']['format']}
- Memory Usage: {context['dataset_info']['memory_usage_mb']:.2f} MB

ANALYSIS PLAN:
{context['analysis_plan']['summary']}
Total Steps: {context['analysis_plan']['total_steps']}

COMPLETED STEPS:
"""
        
        for step in context['analysis_plan']['steps']:
            if step['step_id'] in context['completed_steps']:
                prompt += f"- Step {step['step_id']}: {step['title']} ({step['analysis_type']})\n"
        
        prompt += f"""

FAILED STEPS:
"""
        
        for step_id in context['failed_steps']:
            if step_id in context['execution_summary']:
                error = context['execution_summary'][step_id].get('error_message', 'Unknown error')
                prompt += f"- Step {step_id}: {error}\n"
        
        prompt += f"""

EXECUTION SUMMARY:
"""
        
        for step_id, summary in context['execution_summary'].items():
            prompt += f"- Step {step_id}: {summary['status']} ({summary['execution_time']:.2f}s)\n"
        
        prompt += f"""

KEY INSIGHTS:
"""
        
        for insight in context['key_insights']:
            prompt += f"- {insight}\n"
        
        prompt += f"""

DATA QUALITY ISSUES:
"""
        
        for issue in context['data_quality_issues']:
            prompt += f"- {issue}\n"
        
        prompt += f"""

Please create a comprehensive, professional report that:
1. Provides clear insights about the dataset
2. Summarizes the analysis results
3. Highlights key findings and patterns
4. Identifies data quality issues
5. Provides actionable recommendations
6. Includes technical details for reproducibility

The report should be suitable for both technical and non-technical stakeholders."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, state: WorkflowState) -> AnalysisReport:
        """
        Parse LLM response into AnalysisReport object.
        
        Args:
            response: LLM response string
            state: Original workflow state
            
        Returns:
            Parsed AnalysisReport
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_str = response[json_start:json_end]
            report_data = json.loads(json_str)
            
            # Create AnalysisReport
            analysis_report = AnalysisReport(
                dataset_name=report_data.get('dataset_name', state.data_spec.uri.split('/')[-1]),
                analysis_summary=report_data.get('analysis_summary', 'Analysis completed'),
                key_insights=report_data.get('key_insights', []),
                statistics_summary=report_data.get('statistics_summary', {}),
                visualizations=report_data.get('visualizations', []),
                data_quality_issues=report_data.get('data_quality_issues', []),
                recommendations=report_data.get('recommendations', []),
                markdown_content=report_data.get('markdown_content', ''),
                execution_summary=report_data.get('execution_summary', {}),
                artifacts_directory="data/artifacts"
            )
            
            return analysis_report
            
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
            # Fallback to template-based report
            return self._create_fallback_report(state)
    
    def _create_fallback_report(self, state: WorkflowState) -> AnalysisReport:
        """
        Create fallback report using templates.
        
        Args:
            state: Workflow state
            
        Returns:
            Fallback AnalysisReport
        """
        self.log_warning("Using fallback report due to LLM parsing failure")
        
        # Generate basic markdown report
        markdown_content = self._generate_basic_markdown(state)
        
        # Collect basic insights
        key_insights = []
        data_quality_issues = []
        
        if state.data_spec:
            for col in state.data_spec.column_schema:
                if col.null_count > 0:
                    data_quality_issues.append(f"Column '{col.name}' has {col.null_count} null values")
        
        # Collect execution statistics
        execution_summary = {
            "total_steps": len(state.analysis_plan.steps),
            "completed_steps": len(state.completed_steps),
            "failed_steps": len(state.failed_steps),
            "total_time": state.total_execution_time or 0.0
        }
        
        return AnalysisReport(
            dataset_name=state.data_spec.uri.split('/')[-1] if '/' in state.data_spec.uri else state.data_spec.uri,
            analysis_summary="Automated analysis completed with fallback report generation",
            key_insights=key_insights,
            statistics_summary={},
            visualizations=[],
            data_quality_issues=data_quality_issues,
            recommendations=["Review generated artifacts manually for detailed insights"],
            markdown_content=markdown_content,
            execution_summary=execution_summary,
            artifacts_directory="data/artifacts"
        )
    
    def _generate_basic_markdown(self, state: WorkflowState) -> str:
        """
        Generate basic markdown report from workflow state.
        
        Args:
            state: Workflow state
            
        Returns:
            Basic markdown content
        """
        markdown = f"""# Data Analysis Report: {state.data_spec.uri.split('/')[-1] if '/' in state.data_spec.uri else state.data_spec.uri}

## Executive Summary

This report presents the results of automated data analysis performed on the dataset. The analysis included {len(state.analysis_plan.steps)} steps covering data quality assessment, statistical analysis, and visualization.

## Dataset Overview

- **Dataset**: {state.data_spec.uri}
- **Format**: {state.data_spec.format.value}
- **Rows**: {state.data_spec.row_count:,}
- **Columns**: {state.data_spec.column_count}
- **Memory Usage**: {state.data_spec.memory_usage_mb:.2f} MB

## Analysis Results

### Completed Steps ({len(state.completed_steps)}/{len(state.analysis_plan.steps)})

"""
        
        for step in state.analysis_plan.steps:
            if step.step_id in state.completed_steps:
                markdown += f"#### Step {step.step_id}: {step.title}\n\n"
                markdown += f"{step.description}\n\n"
                
                if step.step_id in state.execution_results:
                    result = state.execution_results[step.step_id]
                    markdown += f"- **Status**: {result.status.value}\n"
                    markdown += f"- **Execution Time**: {result.execution_time_seconds:.2f} seconds\n"
                    markdown += f"- **Artifacts**: {len(result.artifacts)} generated\n\n"
        
        markdown += "### Failed Steps\n\n"
        
        for step_id in state.failed_steps:
            if step_id in state.execution_results:
                result = state.execution_results[step_id]
                markdown += f"- **Step {step_id}**: {result.error_message or 'Unknown error'}\n\n"
        
        markdown += "## Data Quality Assessment\n\n"
        
        if state.data_spec:
            for col in state.data_spec.column_schema:
                if col.null_count > 0:
                    markdown += f"- Column '{col.name}' has {col.null_count} null values\n"
        
        markdown += "\n## Recommendations\n\n"
        markdown += "- Review generated artifacts in `data/artifacts/` for detailed insights\n"
        markdown += "- Consider addressing data quality issues identified above\n"
        markdown += "- Validate key findings with domain experts\n\n"
        
        markdown += "## Technical Details\n\n"
        markdown += f"- **Total Execution Time**: {state.total_execution_time or 0.0:.2f} seconds\n"
        markdown += f"- **Artifacts Directory**: `data/artifacts/`\n"
        markdown += f"- **Analysis Framework**: LLM-powered agentic analysis\n\n"
        
        markdown += "---\n*Report generated automatically by LLM-powered analysis framework*"
        
        return markdown
    
    def _load_report_templates(self) -> Dict[str, Any]:
        """Load report templates."""
        return {
            "executive_summary": "Analysis of {dataset_name} revealed key insights about data quality and patterns.",
            "recommendations": [
                "Review data quality issues",
                "Validate findings with domain experts",
                "Consider additional analysis based on insights"
            ]
        } 