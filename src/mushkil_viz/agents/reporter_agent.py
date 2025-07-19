import json
from typing import List, Dict, Any
from .base_agent import BaseAgent
from ..schema import WorkflowState, AnalysisReport


class ReporterAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.report_templates = self._load_report_templates()

    def process(self, state: WorkflowState) -> WorkflowState:
        self.log_info("Creating comprehensive final report")
        try:
            if not self.validate_state(state):
                raise ValueError("Invalid state for reporter agent")
            final_report = self._create_final_report(state)
            state.final_report = final_report
            self.log_info("Successfully created final report")
        except Exception as e:
            self.log_error(f"Failed to create final report: {e}")
            raise
        return state

    def validate_state(self, state: WorkflowState) -> bool:
        return (
            hasattr(state, 'data_spec') and state.data_spec is not None and
            hasattr(state, 'analysis_plan') and state.analysis_plan is not None and
            hasattr(state, 'execution_results') and len(state.execution_results) > 0
        )

    def _create_final_report(self, state: WorkflowState) -> AnalysisReport:
        context = self._prepare_report_context(state)
        system_prompt = self._get_system_prompt()
        user_prompt = self._create_user_prompt(context)
        messages = self.format_messages(system_prompt, user_prompt)
        response = self.call_llm(messages)
        return self._parse_llm_response(response, state)

    def _prepare_report_context(self, state: WorkflowState) -> Dict[str, Any]:
        execution_summary = {}
        artifacts_by_step = {}
        for step_id, result in state.execution_results.items():
            execution_summary[step_id] = {
                "status": result.status.value,
                "execution_time": result.execution_time_seconds,
                "artifacts": result.artifacts,
                "error_message": result.error_message
            }
            artifacts_by_step[step_id] = {}
            for art in result.artifacts:
                t = getattr(art, "type", "unknown")
                artifacts_by_step[step_id].setdefault(t, []).append({
                    "path": getattr(art, "uri", getattr(art, "path", None)),
                    "desc": getattr(art, "description", "")
                })

        grade_summary = {
            step_id: {
                "verdict": grade.verdict.value,
                "score": grade.score,
                "comments": grade.comments
            }
            for step_id, grade in state.grade_reports.items()
        }

        key_insights = []
        for result in state.execution_results.values():
            if result.status.value == "success" and result.return_value:
                if isinstance(result.return_value, dict):
                    key_insights.extend(result.return_value.get("insights", []))
                    key_insights.extend(result.return_value.get("key_findings", []))

        data_quality_issues = []
        if state.data_spec:
            for col in state.data_spec.column_schema:
                if col.null_count > 0:
                    data_quality_issues.append(f"Column '{col.name}' has {col.null_count} null values")

        return {
            "dataset_name": state.data_spec.uri.split('/')[-1],
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
            "artifacts_by_step": artifacts_by_step,
            "completed_steps": state.completed_steps,
            "failed_steps": state.failed_steps,
            "total_execution_time": state.total_execution_time
        }

    def _format_artifact_markdown(self, step_id: str, artifacts: Dict[str, List[Dict[str, str]]]) -> str:
        md = ""
        for art_type, items in artifacts.items():
            md += f"\n**{art_type.title()}s for Step {step_id}:**\n"
            for item in items:
                path = item["path"]
                desc = item["desc"]
                if art_type.lower() == "figure":
                    md += f"![Figure – Step {step_id}]({path})  \n*{desc}*\n"
                elif art_type.lower() == "table":
                    md += f"<details>\n<summary>Table – Step {step_id}: {desc}</summary>\n\n```csv\n(path: {path})\n```\n</details>\n"
                else:
                    md += f"- {art_type.title()}: `{path}` – {desc}\n"
        return md

    def _get_system_prompt(self) -> str:
        return """You are an expert data analyst and technical writer. Your task is to create a comprehensive, professional analysis report based on the results of automated data analysis.

REPORT REQUIREMENTS:
1. Create a clear, well-structured markdown report
2. Include an executive summary with key findings
3. Provide detailed analysis sections for each completed step
4. Include visualizations and artifacts where available (embed figures as ![alt](path) and wrap large tables in collapsible <details>)
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
  "analysis_summary": "string",
  "key_insights": ["string"],
  "statistics_summary": { "object" },
  "visualizations": ["string"],
  "data_quality_issues": ["string"],
  "recommendations": ["string"],
  "markdown_content": "string",
  "execution_summary": { "object" },
  "artifacts_directory": "string"
}"""

    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
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
                artifacts = context['artifacts_by_step'].get(step['step_id'], {})
                if artifacts:
                    prompt += self._format_artifact_markdown(step['step_id'], artifacts)

        prompt += f"\nFAILED STEPS:\n"
        for step_id in context['failed_steps']:
            err = context['execution_summary'].get(step_id, {}).get('error_message', 'Unknown error')
            prompt += f"- Step {step_id}: {err}\n"

        prompt += "\nEXECUTION SUMMARY:\n"
        for step_id, summary in context['execution_summary'].items():
            prompt += f"- Step {step_id}: {summary['status']} ({summary['execution_time']:.2f}s)\n"

        prompt += "\nKEY INSIGHTS:\n"
        for insight in context['key_insights']:
            prompt += f"- {insight}\n"

        prompt += "\nDATA QUALITY ISSUES:\n"
        for issue in context['data_quality_issues']:
            prompt += f"- {issue}\n"

        prompt += "\nPlease create a comprehensive, professional report suitable for both technical and non-technical stakeholders."
        return prompt

    def _parse_llm_response(self, response: str, state: WorkflowState) -> AnalysisReport:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            report_data = json.loads(json_str)

            return AnalysisReport(
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
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
            return self._create_fallback_report(state)

    def _create_fallback_report(self, state: WorkflowState) -> AnalysisReport:
        self.log_warning("Using fallback report due to LLM parsing failure")
        markdown = self._generate_basic_markdown(state)
        data_quality_issues = [
            f"Column '{col.name}' has {col.null_count} null values"
            for col in state.data_spec.column_schema if col.null_count > 0
        ]
        execution_summary = {
            "total_steps": len(state.analysis_plan.steps),
            "completed_steps": len(state.completed_steps),
            "failed_steps": len(state.failed_steps),
            "total_time": state.total_execution_time or 0.0
        }
        return AnalysisReport(
            dataset_name=state.data_spec.uri.split('/')[-1],
            analysis_summary="Automated analysis completed with fallback report generation",
            key_insights=[],
            statistics_summary={},
            visualizations=[],
            data_quality_issues=data_quality_issues,
            recommendations=["Review generated artifacts manually for detailed insights"],
            markdown_content=markdown,
            execution_summary=execution_summary,
            artifacts_directory="data/artifacts"
        )

    def _generate_basic_markdown(self, state: WorkflowState) -> str:
        markdown = f"""# Data Analysis Report: {state.data_spec.uri.split('/')[-1]}

## Executive Summary

This report presents the results of automated data analysis including {len(state.analysis_plan.steps)} steps.

## Dataset Overview

- Dataset: {state.data_spec.uri}
- Format: {state.data_spec.format.value}
- Rows: {state.data_spec.row_count:,}
- Columns: {state.data_spec.column_count}
- Memory Usage: {state.data_spec.memory_usage_mb:.2f} MB

## Analysis Results

### Completed Steps
"""
        for step in state.analysis_plan.steps:
            if step.step_id in state.completed_steps:
                markdown += f"#### Step {step.step_id}: {step.title}\n\n"
                markdown += f"{step.description}\n\n"

        markdown += "### Failed Steps\n"
        for step_id in state.failed_steps:
            err = state.execution_results[step_id].error_message or 'Unknown error'
            markdown += f"- Step {step_id}: {err}\n"

        markdown += "\n## Data Quality Assessment\n"
        for col in state.data_spec.column_schema:
            if col.null_count > 0:
                markdown += f"- Column '{col.name}' has {col.null_count} null values\n"

        markdown += "\n## Recommendations\n"
        markdown += "- Review generated artifacts in `data/artifacts/`\n"
        markdown += "- Address data quality issues\n"
        markdown += "- Validate key findings with domain experts\n"

        markdown += "\n## Technical Details\n"
        markdown += f"- Total Execution Time: {state.total_execution_time or 0.0:.2f} seconds\n"
        markdown += f"- Artifacts Directory: `data/artifacts/`\n"
        markdown += "---\n*Generated by automated analysis framework*"
        return markdown

    def _load_report_templates(self) -> Dict[str, Any]:
        return {
            "executive_summary": "Analysis of {dataset_name} revealed key insights.",
            "recommendations": [
                "Review data quality issues",
                "Validate findings with domain experts",
                "Consider additional analysis"
            ]
        }