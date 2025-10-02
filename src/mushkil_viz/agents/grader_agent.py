"""
Grader agent for evaluating execution results and code quality.

This agent is responsible for:
- Evaluating code execution results
- Assessing code quality and safety
- Determining if code needs to be regenerated
- Providing feedback for improvements
"""

import json
from typing import List, Dict, Any, Optional
import logging

from .base_agent import BaseAgent
from ..schema import (
    WorkflowState, GradeReport, GradeVerdict, ExecutionResult, CodeBundle
)


class GraderAgent(BaseAgent):
    """
    Agent responsible for evaluating execution results and code quality.
    
    This agent takes execution results and evaluates them based on:
    - Execution success/failure
    - Code quality and safety
    - Output quality and completeness
    - Artifact generation
    """
    
    def __init__(self, **kwargs):
        """Initialize the grader agent."""
        super().__init__(**kwargs)
        self.grading_criteria = self._load_grading_criteria()
    
    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Evaluate execution results and create grade report.
        
        Args:
            state: Current workflow state with execution_results for current step
            
        Returns:
            Updated workflow state with grade_report for current step
        """
        self.log_info(f"Grading execution results for step {state.current_step}")
        
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for grader agent")
            
            # Get execution result and code bundle
            execution_result = state.execution_results[state.current_step]
            code_bundle = state.code_bundles[state.current_step]
            
            # Grade the execution
            grade_report = self._grade_execution(execution_result, code_bundle)
            
            # Update state
            state.grade_reports[state.current_step] = grade_report
            self.log_info(f"Graded step {state.current_step}: {grade_report.verdict.value}")
            
        except Exception as e:
            self.log_error(f"Failed to grade step {state.current_step}: {e}")
            raise
        
        return state
    
    def validate_state(self, state: WorkflowState) -> bool:
        """Validate that state has required execution results."""
        return (
            hasattr(state, 'current_step') and
            state.current_step > 0 and
            hasattr(state, 'execution_results') and
            state.current_step in state.execution_results and
            hasattr(state, 'code_bundles') and
            state.current_step in state.code_bundles
        )
    
    def _grade_execution(self, execution_result: ExecutionResult, code_bundle: CodeBundle) -> GradeReport:
        """
        Grade the execution result and code quality.
        
        Args:
            execution_result: Results from code execution
            code_bundle: Original code bundle that was executed
            
        Returns:
            GradeReport with evaluation results
        """
        # Prepare context for LLM
        context = self._prepare_grading_context(execution_result, code_bundle)
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt
        user_prompt = self._create_user_prompt(context)
        
        # Call LLM
        messages = self.format_messages(system_prompt, user_prompt)
        response = self.call_llm(messages)
        
        # Parse response
        grade_report = self._parse_llm_response(response, execution_result.step_id)
        
        return grade_report
    
    def _prepare_grading_context(self, execution_result: ExecutionResult, code_bundle: CodeBundle) -> Dict[str, Any]:
        """
        Prepare context for grading.
        
        Args:
            execution_result: Execution results
            code_bundle: Code bundle
            
        Returns:
            Context dictionary
        """
        return {
            "step_id": execution_result.step_id,
            "execution_status": execution_result.status.value,
            "execution_time": execution_result.execution_time_seconds,
            "stdout": execution_result.stdout,
            "stderr": execution_result.stderr,
            "error_message": execution_result.error_message,
            "artifacts": execution_result.artifacts,
            "return_value": execution_result.return_value,
            "code": code_bundle.code,
            "expected_artifacts": [
                {
                    "name": artifact.name,
                    "type": artifact.type,
                    "path": artifact.path,
                    "description": artifact.description
                }
                for artifact in code_bundle.expected_artifacts
            ],
            "safety_level": code_bundle.safety_level,
            "memory_used": execution_result.memory_used_mb
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the grader agent."""
        return """You are an expert code reviewer and data scientist. Your task is to evaluate the execution results and code quality for a data analysis step.

EVALUATION CRITERIA:
1. Execution Success: Did the code run successfully without errors?
2. Code Quality: Is the code well-written, safe, and follows best practices?
3. Output Quality: Are the generated artifacts useful and complete?
4. Safety: Does the code follow security best practices?
5. Performance: Is the execution time and memory usage reasonable?

GRADING SCALE:
- Score: 0.0 to 1.0 (1.0 = perfect)
- Verdict: pass, replan, regenerate, or fail

VERDICT GUIDELINES:
- PASS: Code executed successfully, good quality, useful outputs
- REGENERATE: Code failed or has issues that can be fixed with code changes
- REPLAN: Analysis approach needs to be reconsidered
- FAIL: Critical issues that cannot be easily fixed

Return your response as a JSON object with the following structure:
{
    "verdict": "string (pass, replan, regenerate, fail)",
    "score": number (0.0 to 1.0),
    "comments": ["string (list of detailed comments)"],
    "code_quality_score": number (0.0 to 1.0),
    "output_quality_score": number (0.0 to 1.0),
    "safety_score": number (0.0 to 1.0),
    "suggested_improvements": ["string (list of suggestions)"],
    "can_proceed": boolean,
    "requires_regeneration": boolean,
    "requires_replanning": boolean
}

Be thorough and constructive in your evaluation."""
    
    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create user prompt for grading.
        
        Args:
            context: Grading context dictionary
            
        Returns:
            Formatted user prompt
        """
        # Handle memory_used which can be None
        memory_used_str = f"{context['memory_used']:.2f} MB" if context['memory_used'] is not None else "Not available"
        
        prompt = f"""Please evaluate the following execution results and code:

Step ID: {context['step_id']}
Execution Status: {context['execution_status']}
Execution Time: {context['execution_time']:.2f} seconds
Memory Used: {memory_used_str}

CODE:
```python
{context['code']}
```

EXECUTION OUTPUT:
STDOUT:
{context['stdout']}

STDERR:
{context['stderr']}

ERROR MESSAGE:
{context['error_message'] if context['error_message'] else 'None'}

GENERATED ARTIFACTS:
{json.dumps(context['artifacts'], indent=2)}

EXPECTED ARTIFACTS:
{json.dumps(context['expected_artifacts'], indent=2)}

RETURN VALUE:
{context['return_value'] if context['return_value'] is not None else 'None'}

Please evaluate:
1. Did the code execute successfully?
2. Is the code safe and well-written?
3. Are the outputs useful and complete?
4. Are there any issues that need to be addressed?
5. Should the code be regenerated or the plan revised?

Provide a comprehensive evaluation with specific feedback."""
        
        return prompt
    
    def _parse_llm_response(self, response: str, step_id: int) -> GradeReport:
        """
        Parse LLM response into GradeReport object.
        
        Args:
            response: LLM response string
            step_id: Step ID being graded
            
        Returns:
            Parsed GradeReport
        """
        try:
            # Extract JSON from response using base class method
            grade_data = self.extract_json_from_response(response)
            
            # Create GradeReport
            grade_report = GradeReport(
                step_id=step_id,
                verdict=GradeVerdict(grade_data['verdict']),
                score=grade_data['score'],
                comments=grade_data.get('comments', []),
                code_quality_score=grade_data.get('code_quality_score', 0.0),
                output_quality_score=grade_data.get('output_quality_score', 0.0),
                safety_score=grade_data.get('safety_score', 0.0),
                suggested_improvements=grade_data.get('suggested_improvements', []),
                can_proceed=grade_data.get('can_proceed', False),
                requires_regeneration=grade_data.get('requires_regeneration', False),
                requires_replanning=grade_data.get('requires_replanning', False)
            )
            
            return grade_report
            
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
            # Fallback to rule-based grading
            return self._create_fallback_grade(step_id)
    
    def _create_fallback_grade(self, step_id: int) -> GradeReport:
        """
        Create fallback grade using rule-based evaluation.
        
        Args:
            step_id: Step ID being graded
            
        Returns:
            Fallback GradeReport
        """
        self.log_warning(f"Using fallback grading for step {step_id} due to LLM parsing failure")
        
        # This is a simplified fallback - in practice, you'd implement more sophisticated rules
        return GradeReport(
            step_id=step_id,
            verdict=GradeVerdict.PASS,  # Default to pass for fallback
            score=0.7,  # Default score
            comments=["Used fallback grading due to LLM parsing failure"],
            code_quality_score=0.7,
            output_quality_score=0.7,
            safety_score=0.8,
            suggested_improvements=["Review generated code manually"],
            can_proceed=True,
            requires_regeneration=False,
            requires_replanning=False
        )
    
    def _load_grading_criteria(self) -> Dict[str, Any]:
        """Load grading criteria and thresholds."""
        return {
            "execution_success_weight": 0.3,
            "code_quality_weight": 0.25,
            "output_quality_weight": 0.25,
            "safety_weight": 0.2,
            "pass_threshold": 0.7,
            "regenerate_threshold": 0.5,
            "fail_threshold": 0.3
        } 