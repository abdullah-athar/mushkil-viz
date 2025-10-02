import json
import os
import pandas as pd
import base64
from pathlib import Path
from typing import List, Dict, Any
from .base_agent import BaseAgent
from ..schema import WorkflowState, AnalysisReport


class ReporterAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        

    def process(self, state: WorkflowState) -> WorkflowState:
        self.log_info("Creating HTML report with CSV tables and figures")
        self.dataset_name = self._extract_dataset_name(state.data_spec.uri)
        
        try:
            if not self.validate_state(state):
                raise ValueError("Invalid state for reporter agent")
            final_report = self._create_html_report(state)
            state.final_report = final_report
            self.log_info("Successfully created HTML report")
        except Exception as e:
            self.log_error(f"Failed to create HTML report: {e}")
            raise
        return state

    def validate_state(self, state: WorkflowState) -> bool:
        return (
            hasattr(state, 'data_spec') and state.data_spec is not None and
            hasattr(state, 'analysis_plan') and state.analysis_plan is not None and
            hasattr(state, 'execution_results') and len(state.execution_results) > 0
        )


    def _create_html_report(self, state: WorkflowState) -> AnalysisReport:
        """Create a simple HTML report with CSV tables and figures."""
        # Collect all CSV and figure files
        csv_files = self._find_csv_files()
        figure_files = self._find_figure_files()
        
        # Generate LLM-powered analysis
        analysis_content = self._generate_intelligent_analysis(state, csv_files, figure_files)
        
        # Generate HTML content
        html_content = self._generate_html_content(state, csv_files, figure_files, analysis_content)
        
        # Save HTML report
        html_path = self._save_html_report(html_content, state)
        
        return AnalysisReport(
            dataset_name=self.dataset_name,
            analysis_summary="HTML report generated with CSV tables and figures",
            key_insights=analysis_content.get("key_insights", ["Report includes all generated tables and visualizations"]),
            statistics_summary={},
            visualizations=[{"path": str(f), "type": f.suffix[1:], "name": f.stem} for f in figure_files],
            data_quality_issues=[],
            recommendations=analysis_content.get("recommendations", ["Review the HTML report for detailed analysis results"]),
            markdown_content=analysis_content.get("executive_summary", "Analysis completed successfully."),
            execution_summary={
                "total_steps": len(state.analysis_plan.steps),
                "completed_steps": len(state.completed_steps),
                "failed_steps": len(state.failed_steps)
            },
            artifacts_directory="data/artifacts"
        )

    def _generate_intelligent_analysis(self, state: WorkflowState, csv_files: List[Path], figure_files: List[Path]) -> Dict[str, Any]:
        """Use LLM to generate intelligent analysis of the data and visualizations."""
        try:
            # Prepare context for LLM
            context = self._prepare_analysis_context(state, csv_files, figure_files)
            
            # Create prompts
            system_prompt = self._get_analysis_system_prompt()
            user_prompt = self._create_analysis_user_prompt(context)
            
            # Make LLM call
            messages = self.format_messages(system_prompt, user_prompt)
            response = self.call_llm(messages)
            
            # Parse response
            return self._parse_analysis_response(response)
            
        except Exception as e:
            self.log_error(f"Failed to generate intelligent analysis: {e}")
            return self._get_fallback_analysis()

    def _prepare_analysis_context(self, state: WorkflowState, csv_files: List[Path], figure_files: List[Path]) -> Dict[str, Any]:
        """Prepare context for LLM analysis."""
        # Get dataset info
        dataset_info = {
            "name": state.data_spec.uri.split('/')[-1],
            "rows": state.data_spec.row_count,
            "columns": state.data_spec.column_count,
            "format": state.data_spec.format.value,
            "memory_mb": state.data_spec.memory_usage_mb
        }
        
        # Get column information
        columns_info = []
        for col in state.data_spec.column_schema:
            columns_info.append({
                "name": col.name,
                "dtype": col.dtype,
                "null_count": col.null_count,
                "unique_count": col.unique_count,
                "null_percentage": (col.null_count / state.data_spec.row_count * 100) if state.data_spec.row_count > 0 else 0
            })
        
        # Get analysis steps info
        analysis_steps = []
        for step in state.analysis_plan.steps:
            step_info = {
                "step_id": step.step_id,
                "title": step.title,
                "description": step.description,
                "analysis_type": step.analysis_type,
                "status": "completed" if step.step_id in state.completed_steps else "failed"
            }
            
            # Add execution results if available
            if step.step_id in state.execution_results:
                result = state.execution_results[step.step_id]
                step_info["execution_time"] = result.execution_time_seconds
                step_info["artifacts_count"] = len(result.artifacts)
                if result.return_value and isinstance(result.return_value, dict):
                    step_info["insights"] = result.return_value.get("insights", [])
                    step_info["key_findings"] = result.return_value.get("key_findings", [])
            
            analysis_steps.append(step_info)
        
        # Get file information
        csv_info = []
        for csv_file in csv_files:
            try:
                # Get basic info about CSV without reading full file
                df_sample = pd.read_csv(csv_file, nrows=5)  # Just read first 5 rows
                csv_info.append({
                    "filename": csv_file.name,
                    "path": str(csv_file),
                    "columns": list(df_sample.columns),
                    "sample_data": df_sample.to_dict('records')
                })
            except Exception as e:
                csv_info.append({
                    "filename": csv_file.name,
                    "path": str(csv_file),
                    "error": str(e)
                })
        
        figure_info = [{"filename": f.name, "path": str(f)} for f in figure_files]
        
        return {
            "dataset_info": dataset_info,
            "columns_info": columns_info,
            "analysis_steps": analysis_steps,
            "csv_files": csv_info,
            "figure_files": figure_info,
            "execution_summary": {
                "total_steps": len(state.analysis_plan.steps),
                "completed_steps": len(state.completed_steps),
                "failed_steps": len(state.failed_steps)
            }
        }

    def _get_analysis_system_prompt(self) -> str:
        """Get system prompt for intelligent analysis."""
        return """You are an expert data analyst and report writer. Your task is to provide intelligent analysis and insights for a data analysis report.

You will be given information about:
- The dataset and its characteristics
- Analysis steps that were performed
- Generated CSV files and visualizations

Your job is to:
1. Provide key insights about the data and analysis results
2. Explain what the visualizations likely show and their significance
3. Identify interesting patterns or findings
4. Give actionable recommendations based on the analysis
5. Highlight any data quality issues or limitations

Write in a professional but accessible tone suitable for both technical and business stakeholders.

Return your response as a JSON object with this structure:
{
  "executive_summary": "string - 2-3 sentence overview of the analysis",
  "key_insights": ["string array - 3-5 key findings from the analysis"],
  "visualization_analysis": "string - discussion of what the visualizations reveal",
  "data_quality_assessment": "string - assessment of data quality and limitations",
  "recommendations": ["string array - 3-5 actionable recommendations"],
  "methodology_notes": "string - brief explanation of analysis approach"
}"""

    def _create_analysis_user_prompt(self, context: Dict[str, Any]) -> str:
        """Create user prompt for analysis."""
        dataset_info = context["dataset_info"]
        
        prompt = f"""Please analyze the following data analysis results and provide intelligent insights:

DATASET OVERVIEW:
- Name: {dataset_info['name']}
- Size: {dataset_info['rows']:,} rows × {dataset_info['columns']} columns
- Format: {dataset_info['format']}
- Memory Usage: {dataset_info['memory_mb']:.2f} MB

COLUMN INFORMATION:
"""
        
        for col in context["columns_info"]:
            null_pct = col["null_percentage"]
            prompt += f"- {col['name']} ({col['dtype']}): {col['unique_count']} unique values, {null_pct:.1f}% null\n"
        
        prompt += f"\nANALYSIS STEPS PERFORMED:\n"
        for step in context["analysis_steps"]:
            status_emoji = "✅" if step["status"] == "completed" else "❌"
            prompt += f"{status_emoji} Step {step['step_id']}: {step['title']} ({step['analysis_type']})\n"
            prompt += f"   Description: {step['description']}\n"
            if step.get("execution_time"):
                prompt += f"   Execution time: {step['execution_time']:.2f}s\n"
            if step.get("insights"):
                prompt += f"   Insights: {', '.join(step['insights'][:3])}\n"
        
        prompt += f"\nGENERATED ARTIFACTS:\n"
        
        if context["csv_files"]:
            prompt += "CSV Files:\n"
            for csv in context["csv_files"]:
                if "error" not in csv:
                    prompt += f"- {csv['filename']}: {len(csv['columns'])} columns ({', '.join(csv['columns'][:5])}{'...' if len(csv['columns']) > 5 else ''})\n"
                    if csv["sample_data"]:
                        prompt += f"  Sample data: {csv['sample_data'][0] if csv['sample_data'] else 'N/A'}\n"
                else:
                    prompt += f"- {csv['filename']}: Error reading file\n"
        
        if context["figure_files"]:
            prompt += "\nVisualizations:\n"
            for fig in context["figure_files"]:
                prompt += f"- {fig['filename']}\n"
        
        prompt += f"\nEXECUTION SUMMARY:\n"
        prompt += f"- Completed: {context['execution_summary']['completed_steps']}/{context['execution_summary']['total_steps']} steps\n"
        prompt += f"- Failed: {context['execution_summary']['failed_steps']} steps\n"
        
        prompt += "\nPlease provide intelligent analysis covering key insights, visualization interpretation, data quality assessment, and actionable recommendations."
        
        return prompt

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        try:
            return self.extract_json_from_response(response)
        except Exception as e:
            self.log_error(f"Failed to parse analysis response: {e}")
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Get fallback analysis if LLM call fails."""
        return {
            "executive_summary": "Data analysis completed with generated visualizations and tables.",
            "key_insights": [
                "Analysis results are available in the generated artifacts",
                "Multiple analysis steps were executed on the dataset",
                "Visualizations and data tables provide detailed insights"
            ],
            "visualization_analysis": "The generated visualizations provide insights into data patterns, distributions, and relationships between variables.",
            "data_quality_assessment": "Data quality assessment is available in the detailed analysis results.",
            "recommendations": [
                "Review all generated visualizations carefully",
                "Examine the data tables for detailed findings",
                "Consider additional analysis based on initial results"
            ],
            "methodology_notes": "Analysis was performed using automated data science techniques with multiple analytical approaches."
        }

    def _find_csv_files(self) -> List[Path]:
        """Find all CSV files only for the current dataset."""
        csv_files = []
        data_dir = Path("data/artifacts") / self.dataset_name

        if data_dir.exists():
            csv_files = list(data_dir.rglob("*.csv"))

        return sorted(csv_files)

    def _find_figure_files(self) -> List[Path]:
        """Find all figure files only for the current dataset."""
        figure_files = []
        data_dir = Path("data/artifacts") / self.dataset_name

        if data_dir.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"]:
                figure_files.extend(data_dir.rglob(ext))

        return sorted(figure_files)

    def _generate_html_content(self, state: WorkflowState, csv_files: List[Path], figure_files: List[Path], analysis_content: Dict[str, Any]) -> str:
        """Generate the complete HTML content."""
        dataset_name = self.dataset_name
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Report: {dataset_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        .dataset-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .dataset-info p {{
            margin: 5px 0;
        }}
        .analysis-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #28a745;
        }}
        .insight-list {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}
        .insight-list ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .insight-list li {{
            margin: 8px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 12px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .figure-container {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fafafa;
        }}
        .figure-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .figure-description {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        .csv-section {{
            margin: 30px 0;
        }}
        .table-wrapper {{
            overflow-x: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .no-data {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            padding: 20px;
        }}
        .recommendations {{
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #17a2b8;
        }}
        .download-link {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 8px 15px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 14px;
            transition: background-color 0.3s;
        }}
        .download-link:hover {{
            background-color: #2980b9;
            color: white;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Analysis Report: {dataset_name}</h1>
        
        <div class="dataset-info">
            <h3>Dataset Information</h3>
            <p><strong>Dataset:</strong> {dataset_name}</p>
            <p><strong>Rows:</strong> {state.data_spec.row_count:,}</p>
            <p><strong>Columns:</strong> {state.data_spec.column_count}</p>
            <p><strong>Format:</strong> {state.data_spec.format.value}</p>
            <p><strong>Memory Usage:</strong> {state.data_spec.memory_usage_mb:.2f} MB</p>
        </div>

        <div class="analysis-section">
            <h2>🔍 Executive Summary</h2>
            <p>{analysis_content.get('executive_summary', 'Analysis completed successfully.')}</p>
        </div>

        <div class="insight-list">
            <h2>💡 Key Insights</h2>
            <ul>
"""
        
        for insight in analysis_content.get('key_insights', []):
            html += f"                <li>{insight}</li>\n"
        
        html += """            </ul>
        </div>

        <div class="analysis-section">
            <h2>📊 Visualization Analysis</h2>
            <p>{}</p>
        </div>""".format(analysis_content.get('visualization_analysis', 'Generated visualizations provide insights into the data.'))

        # Add figures section
        if figure_files:
            html += "\n        <h2>📊 Generated Visualizations</h2>\n"
            for i, figure_file in enumerate(figure_files):
                # Generate a description for the figure
                description = self._generate_figure_description(figure_file, state)
                # Convert image to base64 for embedding
                img_data_uri = self._image_to_base64(figure_file)
                
                if img_data_uri:
                    html += f"""
        <div class="figure-container">
            <h3>Figure {i+1}: {figure_file.stem.replace('_', ' ').title()}</h3>
            <img src="{img_data_uri}" alt="Figure {i+1}" />
            <div class="figure-description">
                {description}
            </div>
        </div>"""
                else:
                    html += f"""
        <div class="figure-container">
            <h3>Figure {i+1}: {figure_file.stem.replace('_', ' ').title()}</h3>
            <div class="no-data">Failed to load image: {figure_file.name}</div>
            <div class="figure-description">
                {description}
            </div>
        </div>"""
        else:
            html += '\n        <h2>📊 Generated Visualizations</h2>\n        <div class="no-data">No visualizations were generated during analysis.</div>\n'

        # Add CSV tables section
        if csv_files:
            html += "\n        <h2>📋 Generated Data Tables</h2>\n"
            for i, csv_file in enumerate(csv_files):
                html += f"\n        <div class='csv-section'>\n"
                html += f"            <h3>Table {i+1}: {csv_file.stem.replace('_', ' ').title()}</h3>\n"
                
                # Add download link
                csv_data_uri = self._csv_to_base64(csv_file)
                if csv_data_uri:
                    html += f"            <p><a href='{csv_data_uri}' download='{csv_file.name}' class='download-link'>📥 Download CSV: {csv_file.name}</a></p>\n"
                
                try:
                    # Read CSV and convert to HTML table
                    df = pd.read_csv(csv_file)
                    # Limit to first 100 rows to avoid huge tables
                    if len(df) > 100:
                        table_html = df.head(100).to_html(classes='data-table', index=False, escape=False)
                        html += f"            <p><em>Showing first 100 rows of {len(df)} total rows</em></p>\n"
                    else:
                        table_html = df.to_html(classes='data-table', index=False, escape=False)
                    
                    html += f"            <div class='table-wrapper'>\n{table_html}\n            </div>\n"
                except Exception as e:
                    html += f"            <p class='no-data'>Error loading table: {str(e)}</p>\n"
                html += "        </div>\n"
        else:
            html += '\n        <h2>📋 Generated Data Tables</h2>\n        <div class="no-data">No CSV tables were generated during analysis.</div>\n'

        # Add data quality and recommendations
        html += f"""
        <div class="analysis-section">
            <h2>🔍 Data Quality Assessment</h2>
            <p>{analysis_content.get('data_quality_assessment', 'Data quality assessment completed.')}</p>
        </div>

        <div class="recommendations">
            <h2>🎯 Recommendations</h2>
            <ul>
"""
        
        for rec in analysis_content.get('recommendations', []):
            html += f"                <li>{rec}</li>\n"
        
        html += f"""            </ul>
        </div>

        <div class="analysis-section">
            <h2>🔬 Methodology Notes</h2>
            <p>{analysis_content.get('methodology_notes', 'Analysis performed using automated data science techniques.')}</p>
        </div>

        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px;">
            <p><em>Report generated automatically by MushkilViz Analysis Framework</em></p>
            <p><em>All visualizations and data tables are embedded in this HTML file for local viewing. CSV files can be downloaded using the provided links.</em></p>
        </div>
    </div>
</body>
</html>"""
        
        return html

    def _generate_figure_description(self, figure_file: Path, state: WorkflowState) -> str:
        """Generate a description for a figure based on its filename and context."""
        filename = figure_file.stem.lower()
        
        # Try to extract step information from path
        step_info = ""
        if "step_" in str(figure_file):
            step_match = str(figure_file).split("step_")
            if len(step_match) > 1:
                step_num = step_match[1].split("/")[0] if "/" in step_match[1] else step_match[1].split("\\")[0]
                try:
                    step_id = int(step_num)
                    for step in state.analysis_plan.steps:
                        if step.step_id == step_id:
                            step_info = f"from {step.title}: {step.description}"
                            break
                except:
                    pass
        
        # Generate description based on filename patterns
        if "histogram" in filename or "hist" in filename:
            return f"This histogram shows the distribution of values {step_info}"
        elif "scatter" in filename:
            return f"This scatter plot shows the relationship between variables {step_info}"
        elif "correlation" in filename or "corr" in filename:
            return f"This correlation plot shows relationships between different variables {step_info}"
        elif "boxplot" in filename or "box" in filename:
            return f"This box plot shows the distribution and outliers {step_info}"
        elif "line" in filename:
            return f"This line plot shows trends over time or sequence {step_info}"
        elif "bar" in filename:
            return f"This bar chart compares values across categories {step_info}"
        elif "heatmap" in filename:
            return f"This heatmap visualizes patterns in the data {step_info}"
        else:
            return f"This visualization presents analysis results {step_info}".strip()

    def _save_html_report(self, html_content: str, state: WorkflowState) -> str:
        """Save the HTML report to disk."""
        # Extract dataset name from URI for directory organization
        dataset_name = self._extract_dataset_name(state.dataset_uri)
        
        # Create dataset-specific output directory
        output_dir = Path("data/artifacts") / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save HTML file
        html_file = output_dir / f"analysis_report_{state.workflow_id}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log_info(f"HTML report saved to: {html_file}")
        return str(html_file)
    
    def _extract_dataset_name(self, dataset_uri: str) -> str:
        """
        Extract dataset name from URI for directory organization.
        
        Args:
            dataset_uri: URI of the dataset
            
        Returns:
            Clean dataset name for directory
        """
        # Extract filename from URI
        if '/' in dataset_uri:
            filename = dataset_uri.split('/')[-1]
        else:
            filename = dataset_uri
        
        # Remove file extension
        if '.' in filename:
            name = filename.rsplit('.', 1)[0]
        else:
            name = filename
        
        # Clean the name for directory use (remove special characters)
        import re
        clean_name = re.sub(r'[^\w\-_]', '_', name)
        
        return clean_name or "dataset"

    def _image_to_base64(self, image_path: Path) -> str:
        """
        Convert an image file to base64 data URI for embedding in HTML.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 data URI string
        """
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Determine MIME type based on file extension
                ext = image_path.suffix.lower()
                if ext == '.png':
                    mime_type = 'image/png'
                elif ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.svg':
                    mime_type = 'image/svg+xml'
                elif ext == '.pdf':
                    mime_type = 'application/pdf'
                else:
                    mime_type = 'image/png'  # default
                
                return f"data:{mime_type};base64,{img_base64}"
        except Exception as e:
            self.log_error(f"Failed to convert image {image_path} to base64: {e}")
            return ""

    def _csv_to_base64(self, csv_path: Path) -> str:
        """
        Convert a CSV file to base64 data URI for embedding as downloadable link.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Base64 data URI string
        """
        try:
            with open(csv_path, 'rb') as csv_file:
                csv_data = csv_file.read()
                csv_base64 = base64.b64encode(csv_data).decode('utf-8')
                return f"data:text/csv;base64,{csv_base64}"
        except Exception as e:
            self.log_error(f"Failed to convert CSV {csv_path} to base64: {e}")
            return ""