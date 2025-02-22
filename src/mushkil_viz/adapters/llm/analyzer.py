"""LLM-based analyzer module."""
import json
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from mushkil_viz.adapters.llm.constants import DEFAULT_MODEL, LOG_FORMAT, SYSTEM_PROMPT
from mushkil_viz.adapters.llm.schemas import (
    AnalysisFunction,
    AnalysisResult,
    DatasetContext,
    FUNCTION_SCHEMA,
)
from mushkil_viz.core.analyzers.base import BaseAnalyzer
from mushkil_viz.core.client_init import init_openai_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class LLMAnalyzer(BaseAnalyzer):
    """Analyzer that uses LLMs to generate and execute data analysis functions."""
    
    def __init__(self, api_key: str = None, api_base: str = None):
        super().__init__()
        self.client = init_openai_client(api_key, api_base)
        self.model = DEFAULT_MODEL
        self.function_schema = FUNCTION_SCHEMA
        self.system_prompt = SYSTEM_PROMPT
        logger.debug("LLMAnalyzer initialized with model: %s", self.model)

    def _build_dataset_context(self, df: pd.DataFrame, base_analysis: Dict) -> DatasetContext:
        """Build the dataset context from DataFrame and base analysis."""
        return DatasetContext(
            column_types=base_analysis["dataset_info"]["column_types"],
            sample_data={col: df[col].head(5).tolist() for col in df.columns},
            basic_stats=base_analysis["basic_stats"],
            column_descriptions={col: f"Column containing {col.replace('_', ' ')}" for col in df.columns}
        )

    def _build_user_prompt(self, context: DatasetContext) -> str:
        """Build the user prompt from dataset context."""
        return (
            f"Dataset Context:\n"
            f"Column Types: {json.dumps(context.column_types, indent=2)}\n"
            f"Sample Data: {json.dumps(context.sample_data, indent=2)}\n"
            f"Basic Stats: {json.dumps(context.basic_stats, indent=2)}\n"
            f"Column Descriptions: {json.dumps(context.column_descriptions, indent=2)}\n\n"
            "Please generate appropriate analysis functions for this dataset."
        )

    def _get_required_columns(self, func: AnalysisFunction) -> List[str]:
        """Get list of required columns for an analysis function."""
        return [
            param for param, details in func.parameters.items() 
            if details.get('type', '').startswith('pd.Series')
        ]

    def _generate_analysis_functions(self, context: DatasetContext) -> List[AnalysisFunction]:
        """Generate analysis functions based on the dataset context."""
        logger.debug("Generating analysis functions with context: %s", context.model_dump())
        
        user_prompt = self._build_user_prompt(context)
        logger.debug("Sending prompt to LLM with length: %d", len(user_prompt))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[self.function_schema],
            tool_choice={"type": "function", "function": {"name": "provide_analysis_plan"}}
        )
       
        function_call = response.choices[0].message.tool_calls
        logger.debug("Received function call: %s", function_call)
        
        function_call_data = json.loads(function_call[0].function.arguments)
        logger.debug("Received analysis plan with %d functions", len(function_call_data["functions"]))
        return [AnalysisFunction.model_validate(func) for func in function_call_data["functions"]]

    def _execute_analysis_function(self, df: pd.DataFrame, analysis: AnalysisFunction) -> AnalysisResult:
        """Execute a single analysis function and return its results."""
        logger.debug("Executing analysis function: %s", analysis.name)
        try:
            # Create the function namespace with necessary imports
            namespace = {
                'pd': pd,
                'np': np,
                'stats': stats,
                'df': df
            }
            
            exec(analysis.code, namespace)
            func = namespace.get(analysis.name)
            if not func:
                raise ValueError(f"Function {analysis.name} not found in the executed code.")

            result = func(df, **analysis.parameters)
            if result is None:
                raise ValueError("Analysis function did not produce a result")
            
            logger.debug("Successfully executed analysis: %s", analysis.name)
            return AnalysisResult(
                name=analysis.name,
                description=analysis.description,
                result=result if isinstance(result, dict) else result.to_dict(),
                visualization_hints=namespace.get('visualization_hints')
            )
        except Exception as e:
            logger.error("Error executing analysis %s: %s", analysis.name, str(e))
            return AnalysisResult(
                name=analysis.name,
                description=analysis.description,
                result={'error': str(e)},
                visualization_hints=None
            )

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive analysis of the dataset using LLM-generated functions."""
        logger.debug("Starting analysis on DataFrame with shape: %s", str(df.shape))
        
        base_analysis = super().analyze(df)
        logger.debug("Completed base analysis")
        
        context = self._build_dataset_context(df, base_analysis)
        analysis_functions = self._generate_analysis_functions(context)
        logger.debug("Generated %d analysis functions", len(analysis_functions))
        
        analysis_results = []
        for func in analysis_functions:
            logger.info("Executing function: %s", func)
            required_columns = self._get_required_columns(func)
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if not missing_columns:
                logger.debug("Running analysis function: %s (requires columns: %s)", func.name, required_columns)
                result = self._execute_analysis_function(df, func)
                analysis_results.append(result.model_dump())
                
            else:
                logger.warning(
                    "Skipping analysis %s: missing required columns %s", 
                    func.name, missing_columns
                )
        
        logger.debug("Completed %d analyses", len(analysis_results))
        
        results = base_analysis
        results["llm_analysis_results"] = analysis_results
        return results


if __name__ == "__main__":
    analyzer = LLMAnalyzer()
    df = pd.read_csv("examples/financial_data.csv")
    results = analyzer.analyze(df)
    
    for analysis in results.get("llm_analysis_results", []):
        print(f"Function Name: {analysis['name']}")
        print(f"Description: {analysis['description']}")
        print("Result:")
        print(analysis['result'])
        print("-" * 50)

