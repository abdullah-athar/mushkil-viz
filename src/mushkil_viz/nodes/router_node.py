"""
Router node for determining workflow flow based on grading results.

This node is responsible for:
- Evaluating grade reports to determine next actions
- Routing workflow to appropriate next steps
- Managing iteration limits and workflow completion
- Only regenerating code when there are actual bugs
"""

from typing import Dict, Any, List
import logging

from ..schema import (
    WorkflowState, GradeVerdict, GradeReport
)


class RouterNode:
    """
    Node responsible for routing the workflow based on grading results.
    
    This node evaluates the grade report for the current step and
    determines the next action in the workflow. It only regenerates
    code when there are actual bugs, otherwise proceeds to run the code.
    """
    
    def __init__(self):
        """Initialize the router node."""
        self.logger = logging.getLogger(__name__)
    
    def route(self, state: WorkflowState) -> WorkflowState:
        """
        Determine the next step in the workflow based on current state.
        
        Args:
            state: Current workflow state with grade_report for current step
            
        Returns:
            Updated workflow state with next_step field
        """
        step_id = state.current_step
        self.logger.info(f"Routing workflow after step {step_id}")
        
        try:
            # Check if we have a grade report for the current step
            if step_id not in state.grade_reports:
                self.logger.error(f"No grade report found for step {step_id}")
                state.next_step = "error"
                return state
            
            grade_report = state.grade_reports[step_id]
            
            # Simplified routing logic
            if grade_report.requires_regeneration:
                # Only regenerate if there are actual bugs
                state.next_step = self._handle_regeneration_needed(state, grade_report)
            elif grade_report.requires_replanning:
                # Replan if the entire approach needs to change
                state.next_step = self._handle_replanning_needed(state, grade_report)
            elif grade_report.verdict == GradeVerdict.FAIL:
                # Handle complete failure
                state.next_step = self._handle_failure(state, grade_report)
            else:
                # Code is good enough to proceed (PASS or minor issues)
                state.next_step = self._handle_proceed(state, grade_report)
                
        except Exception as e:
            self.logger.error(f"Error in routing: {e}")
            state.next_step = "error"
        
        return state
    
    def _handle_proceed(self, state: WorkflowState, grade_report: GradeReport) -> str:
        """
        Handle cases where code is good enough to proceed.
        
        Args:
            state: Current workflow state
            grade_report: Grade report for current step
            
        Returns:
            Next node name
        """
        self.logger.info(f"Step {state.current_step} proceeding with score {grade_report.score:.2f}")
        
        # Check if there are more steps to process
        if self._has_more_steps(state):
            # Move to next step
            state.current_step = self._get_next_step(state)
            state.current_iteration = 0  # Reset iteration counter
            return "next_step"  # Generate code for next step
        else:
            # All steps completed, generate final report
            self.logger.info("All steps completed, generating final report")
            return "report"
    
    def _handle_regeneration_needed(self, state: WorkflowState, grade_report: GradeReport) -> str:
        """
        Handle cases where code needs to be regenerated due to bugs.
        
        Args:
            state: Current workflow state
            grade_report: Grade report for current step
            
        Returns:
            Next node name
        """
        self.logger.info(f"Step {state.current_step} needs regeneration due to bugs (score: {grade_report.score:.2f})")
        self.logger.info(f"Current iteration: {state.current_iteration}, Max iterations: {state.max_iterations}")
        
        # Check iteration limit
        if state.current_iteration >= state.max_iterations:
            self.logger.warning(f"Max iterations ({state.max_iterations}) reached for step {state.current_step}")
            # Mark step as failed and move to next
            state.failed_steps.append(state.current_step)
            if self._has_more_steps(state):
                state.current_step = self._get_next_step(state)
                state.current_iteration = 0
                self.logger.info(f"Moving to next step: {state.current_step}")
                return "next_step"
            else:
                self.logger.info("No more steps, generating report")
                return "report"
        
        # Increment iteration counter and regenerate code
        state.current_iteration += 1
        self.logger.info(f"Regenerating code for step {state.current_step} (iteration {state.current_iteration})")
        
        # Remove previous code bundle and execution result
        if state.current_step in state.code_bundles:
            del state.code_bundles[state.current_step]
        if state.current_step in state.execution_results:
            del state.execution_results[state.current_step]
        
        return "continue"  # Regenerate code
    
    def _handle_replanning_needed(self, state: WorkflowState, grade_report: GradeReport) -> str:
        """
        Handle cases where the entire approach needs to be replanned.
        
        Args:
            state: Current workflow state
            grade_report: Grade report for current step
            
        Returns:
            Next node name
        """
        self.logger.warning(f"Step {state.current_step} requires replanning (score: {grade_report.score:.2f})")
        
        # Check iteration limit
        if state.current_iteration >= state.max_iterations:
            self.logger.warning(f"Max iterations ({state.max_iterations}) reached for step {state.current_step}")
            # Mark step as failed and move to next
            state.failed_steps.append(state.current_step)
            if self._has_more_steps(state):
                state.current_step = self._get_next_step(state)
                state.current_iteration = 0
                return "next_step"
            else:
                return "report"
        
        # Increment iteration counter and replan
        state.current_iteration += 1
        self.logger.info(f"Replanning for step {state.current_step} (iteration {state.current_iteration})")
        
        # Clear all step-related data
        self._clear_step_data(state, state.current_step)
        
        return "replan"  # Replan the analysis
    
    def _handle_failure(self, state: WorkflowState, grade_report: GradeReport) -> str:
        """
        Handle complete failure cases.
        
        Args:
            state: Current workflow state
            grade_report: Grade report for current step
            
        Returns:
            Next node name
        """
        self.logger.error(f"Step {state.current_step} failed (score: {grade_report.score:.2f})")
        
        # Mark step as failed
        state.failed_steps.append(state.current_step)
        
        # Check if there are more steps to process
        if self._has_more_steps(state):
            # Move to next step
            state.current_step = self._get_next_step(state)
            state.current_iteration = 0
            return "next_step"
        else:
            # All steps processed, generate final report
            self.logger.info("All steps processed, generating final report")
            return "report"
    
    def _has_more_steps(self, state: WorkflowState) -> bool:
        """
        Check if there are more steps to process.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if there are more steps, False otherwise
        """
        if not state.analysis_plan:
            return False
        
        # Check if current step is the last step
        max_step_id = max(step.step_id for step in state.analysis_plan.steps)
        return state.current_step < max_step_id
    
    def _get_next_step(self, state: WorkflowState) -> int:
        """
        Get the next step ID to process.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next step ID
        """
        if not state.analysis_plan:
            raise ValueError("No analysis plan available")
        
        # Find the next step in order
        current_step_ids = [step.step_id for step in state.analysis_plan.steps]
        current_step_ids.sort()
        
        current_index = current_step_ids.index(state.current_step)
        if current_index + 1 < len(current_step_ids):
            return current_step_ids[current_index + 1]
        else:
            raise ValueError("No more steps available")
    
    def _clear_step_data(self, state: WorkflowState, step_id: int):
        """
        Clear all data related to a specific step.
        
        Args:
            state: Current workflow state
            step_id: Step ID to clear data for
        """
        # Remove code bundle
        if step_id in state.code_bundles:
            del state.code_bundles[step_id]
        
        # Remove execution result
        if step_id in state.execution_results:
            del state.execution_results[step_id]
        
        # Remove grade report
        if step_id in state.grade_reports:
            del state.grade_reports[step_id]
    
    def should_continue(self, state: WorkflowState) -> bool:
        """
        Check if the workflow should continue.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if workflow should continue, False otherwise
        """
        return state.next_step not in ["report", "end", "error"] 