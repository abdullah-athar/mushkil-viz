"""
Main workflow module for LLM-powered agentic analysis framework.

This module provides the main entry point and workflow orchestration
for the automated dataset analysis pipeline.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END

from .schema import (
    WorkflowState, DatasetFormat, ExecutionStatus, GradeVerdict
)
from .agents import (
    LoaderAgent, PlannerAgent, CoderAgent, GraderAgent, ReporterAgent
)
from .nodes import RuntimeNode, RouterNode


class AnalysisWorkflow:
    """
    Main workflow class for LLM-powered dataset analysis.
    
    This class orchestrates the entire analysis pipeline using LangGraph,
    managing the flow between different agents and nodes.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        artifacts_base_dir: str = "data/artifacts",
        max_iterations: int = 3,
        **kwargs
    ):
        """
        Initialize the analysis workflow.
        
        Args:
            model_name: LLM model name to use
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            api_key: API key for the LLM service
            artifacts_base_dir: Base directory for artifacts
            max_iterations: Maximum iterations per step
            **kwargs: Additional arguments passed to agents
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.artifacts_base_dir = artifacts_base_dir
        self.max_iterations = max_iterations
        
        # Initialize agents and nodes
        self.agents = self._initialize_agents(**kwargs)
        self.nodes = self._initialize_nodes()
        
        # Create workflow graph
        self.graph = self._create_workflow_graph()
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _initialize_agents(self, **kwargs) -> Dict[str, Any]:
        """Initialize all agents with common configuration."""
        agent_config = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
            **kwargs
        }
        
        return {
            "loader": LoaderAgent(**agent_config),
            "planner": PlannerAgent(**agent_config),
            "coder": CoderAgent(**agent_config),
            "grader": GraderAgent(**agent_config),
            "reporter": ReporterAgent(**agent_config)
        }
    
    def _initialize_nodes(self) -> Dict[str, Any]:
        """Initialize workflow nodes."""
        return {
            "runtime": RuntimeNode(artifacts_base_dir=self.artifacts_base_dir),
            "router": RouterNode()
        }
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("loader", self._loader_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("coder", self._coder_node)
        workflow.add_node("runtime", self._runtime_node)
        workflow.add_node("grader", self._grader_node)
        workflow.add_node("reporter", self._reporter_node)
        workflow.add_node("router", self._router_node)
        
        # Set entry point
        workflow.set_entry_point("loader")
        
        # Add edges
        workflow.add_edge("loader", "planner")
        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "runtime")
        workflow.add_edge("runtime", "grader")
        workflow.add_edge("grader", "router")
        workflow.add_edge("reporter", END)
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_condition,
            {
                "continue": "coder",
                "next_step": "coder", 
                "replan": "planner",
                "report": "reporter",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _loader_node(self, state: WorkflowState) -> WorkflowState:
        """Loader node wrapper."""
        return self.agents["loader"].process(state)
    
    def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """Planner node wrapper."""
        return self.agents["planner"].process(state)
    
    def _coder_node(self, state: WorkflowState) -> WorkflowState:
        """Coder node wrapper."""
        return self.agents["coder"].process(state)
    
    def _runtime_node(self, state: WorkflowState) -> WorkflowState:
        """Runtime node wrapper."""
        return self.nodes["runtime"].execute(state)
    
    def _grader_node(self, state: WorkflowState) -> WorkflowState:
        """Grader node wrapper."""
        return self.agents["grader"].process(state)
    
    def _reporter_node(self, state: WorkflowState) -> WorkflowState:
        """Reporter node wrapper."""
        return self.agents["reporter"].process(state)
    
    def _router_node(self, state: WorkflowState) -> WorkflowState:
        """Router node wrapper."""
        return self.nodes["router"].route(state)
    
    def _route_condition(self, state: WorkflowState) -> str:
        """Route condition based on router output."""
        return state.next_step
    
    def analyze_dataset(
        self,
        dataset_uri: str,
        dataset_format: DatasetFormat,
        workflow_id: Optional[str] = None
    ) -> WorkflowState:
        """
        Run the complete analysis workflow on a dataset.
        
        Args:
            dataset_uri: URI or path to the dataset
            dataset_format: Format of the dataset
            workflow_id: Optional workflow ID for tracking
            
        Returns:
            Final workflow state with all results
        """
        import uuid
        
        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())
        
        # Create initial state
        initial_state = WorkflowState(
            dataset_uri=dataset_uri,
            dataset_format=dataset_format,
            workflow_id=workflow_id,
            start_time=datetime.now().isoformat(),
            max_iterations=self.max_iterations
        )
        
        self.logger.info(f"Starting analysis workflow {workflow_id} for {dataset_uri}")
        
        try:
            # Run the workflow without checkpointer
            final_state_dict = self.graph.invoke(initial_state, config={"recursion_limit": 50})
            
            # Convert dictionary back to WorkflowState if needed
            if isinstance(final_state_dict, dict):
                final_state = WorkflowState(**final_state_dict)
            else:
                final_state = final_state_dict
            
            # Update final state
            final_state.end_time = datetime.now().isoformat()
            if final_state.start_time:
                start_time = datetime.fromisoformat(final_state.start_time)
                end_time = datetime.fromisoformat(final_state.end_time)
                final_state.total_execution_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Analysis workflow {workflow_id} completed successfully")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            # Update state with error information
            initial_state.end_time = datetime.now().isoformat()
            if initial_state.start_time:
                start_time = datetime.fromisoformat(initial_state.start_time)
                end_time = datetime.fromisoformat(initial_state.end_time)
                initial_state.total_execution_time = (end_time - start_time).total_seconds()
            raise
    
    def get_workflow_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Get a summary of the workflow execution.
        
        Args:
            state: Workflow state
            
        Returns:
            Summary dictionary
        """
        summary = {
            "workflow_id": state.workflow_id,
            "dataset_uri": state.dataset_uri,
            "dataset_format": state.dataset_format.value,
            "start_time": state.start_time,
            "end_time": state.end_time,
            "total_execution_time": state.total_execution_time,
            "current_step": state.current_step,
            "completed_steps": len(state.completed_steps),
            "failed_steps": len(state.failed_steps),
            "total_steps": len(state.analysis_plan.steps) if state.analysis_plan else 0,
            "has_final_report": state.final_report is not None
        }
        
        # Add execution statistics
        if state.execution_results:
            successful_executions = sum(
                1 for result in state.execution_results.values()
                if result.status == ExecutionStatus.SUCCESS
            )
            summary["successful_executions"] = successful_executions
            summary["failed_executions"] = len(state.execution_results) - successful_executions
        
        # Add grading statistics
        if state.grade_reports:
            pass_count = sum(
                1 for grade in state.grade_reports.values()
                if grade.verdict == GradeVerdict.PASS
            )
            summary["passed_grades"] = pass_count
            summary["total_grades"] = len(state.grade_reports)
        
        return summary
    
    def save_workflow_state(self, state: WorkflowState, output_dir: str) -> str:
        """
        Save workflow state and artifacts to disk.
        
        Args:
            state: Workflow state to save
            output_dir: Directory to save to
            
        Returns:
            Path to saved state file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save workflow state
        state_file = output_path / f"workflow_state_{state.workflow_id}.json"
        with open(state_file, "w") as f:
            json.dump(state.dict(), f, indent=2, default=str)
        
        # Save final report if available
        if state.final_report:
            report_file = output_path / f"analysis_report_{state.workflow_id}.md"
            with open(report_file, "w") as f:
                f.write(state.final_report.markdown_content)
        
        # Save summary
        summary_file = output_path / f"workflow_summary_{state.workflow_id}.json"
        with open(summary_file, "w") as f:
            json.dump(self.get_workflow_summary(state), f, indent=2)
        
        self.logger.info(f"Workflow state saved to {output_path}")
        return str(state_file)


def create_workflow(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    max_tokens: int = 4000,
    api_key: Optional[str] = None,
    artifacts_base_dir: str = "data/artifacts",
    max_iterations: int = 3,
    **kwargs
) -> AnalysisWorkflow:
    """
    Create a new analysis workflow instance.
    
    Args:
        model_name: LLM model name to use
        temperature: Sampling temperature for generation
        max_tokens: Maximum tokens to generate
        api_key: API key for the LLM service
        artifacts_base_dir: Base directory for artifacts
        max_iterations: Maximum iterations per step
        **kwargs: Additional arguments passed to workflow
        
    Returns:
        Configured AnalysisWorkflow instance
    """
    return AnalysisWorkflow(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        artifacts_base_dir=artifacts_base_dir,
        max_iterations=max_iterations,
        **kwargs
    )


def analyze_dataset_simple(
    dataset_uri: str,
    dataset_format: DatasetFormat,
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
    **kwargs
) -> WorkflowState:
    """
    Simple function to analyze a dataset with default settings.
    
    Args:
        dataset_uri: URI or path to the dataset
        dataset_format: Format of the dataset
        model_name: LLM model name to use
        api_key: API key for the LLM service
        **kwargs: Additional arguments passed to workflow
        
    Returns:
        Final workflow state with all results
    """
    workflow = create_workflow(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )
    
    return workflow.analyze_dataset(dataset_uri, dataset_format) 