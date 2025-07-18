"""
Runtime node for executing generated Python code in a sandbox.

This node is responsible for:
- Executing Python code safely in a sandbox environment
- Managing execution timeouts and memory limits
- Capturing stdout/stderr and artifacts
- Handling execution errors gracefully
"""

import os
import sys
import time
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import signal


from ..schema import (
    WorkflowState, ExecutionResult, ExecutionStatus, CodeBundle
)


class RuntimeNode:
    """
    Node responsible for executing Python code in a sandbox environment.
    
    This node takes a CodeBundle and executes the Python code safely,
    capturing all outputs, errors, and generated artifacts.
    """
    
    def __init__(self, artifacts_base_dir: str = "data/artifacts"):
        """
        Initialize the runtime node.
        
        Args:
            artifacts_base_dir: Base directory for storing artifacts
        """
        self.artifacts_base_dir = Path(artifacts_base_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure artifacts directory exists
        self.artifacts_base_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the code for the current analysis step.
        
        Args:
            state: Current workflow state with code_bundle for current step
            
        Returns:
            Updated workflow state with execution_result for current step
        """
        step_id = state.current_step
        self.logger.info(f"Executing code for step {step_id}")

        try:
            # Get code bundle for current step
            if step_id not in state.code_bundles:
                raise ValueError(f"No code bundle found for step {step_id}")
            
            code_bundle = state.code_bundles[step_id]
            
            # Execute the code
            execution_result = self._execute_code(code_bundle, state.data_spec.uri)
            
            # Update state
            state.execution_results[step_id] = execution_result
            
            if execution_result.status == ExecutionStatus.SUCCESS:
                state.completed_steps.append(step_id)
                self.logger.info(f"Step {step_id} executed successfully")
            else:
                state.failed_steps.append(step_id)
                self.logger.warning(f"Step {step_id} failed: {execution_result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute step {step_id}: {e}")
            # Create error execution result
            error_result = ExecutionResult(
                step_id=step_id,
                status=ExecutionStatus.FAILED,
                stdout="",
                stderr=str(e),
                execution_time_seconds=0.0,
                error_message=str(e),
                artifacts=[],
                return_value=None
            )
            state.execution_results[step_id] = error_result
            state.failed_steps.append(step_id)
        
        return state
    
    def _execute_code(self, code_bundle: CodeBundle, dataset_uri: str) -> ExecutionResult:
        """
        Execute Python code in a sandbox environment.
        
        Args:
            code_bundle: Code bundle to execute
            dataset_uri: URI of the dataset to analyze
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        
        # Create step-specific artifacts directory
        step_artifacts_dir = self.artifacts_base_dir / f"step_{code_bundle.step_id}"
        step_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            # Prepare the code for execution
            prepared_code = self._prepare_code_for_execution(code_bundle, dataset_uri, step_artifacts_dir)
            temp_file.write(prepared_code)
            temp_file_path = temp_file.name
            print(temp_file_path)
   
        
        try:
            # Execute the code
            result = self._run_code_in_sandbox(
                temp_file_path,
                step_artifacts_dir,
                code_bundle.execution_timeout_seconds,
                code_bundle.memory_limit_mb
            )
            
            execution_time = time.time() - start_time
            
            # Collect artifacts
            artifacts = self._collect_artifacts(step_artifacts_dir)
            
            # Create execution result
            execution_result = ExecutionResult(
                step_id=code_bundle.step_id,
                status=result['status'],
                stdout=result['stdout'],
                stderr=result['stderr'],
                execution_time_seconds=execution_time,
                memory_used_mb=result.get('memory_used_mb'),
                artifacts=artifacts,
                return_value=result.get('return_value'),
                error_message=result.get('error_message')
            )
            
            return execution_result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
    @staticmethod
    def _indent_code(code: str, num_tabs: int = 1) -> str:
        prefix = "    " * num_tabs
        return '\n'.join(prefix + line if line.strip() else line for line in code.splitlines())
    

    def _prepare_code_for_execution(self, code_bundle: CodeBundle, dataset_uri: str, artifacts_dir: Path) -> str:
        """
        Prepare code for execution by adding necessary setup and safety measures.
        
        Args:
            code_bundle: Code bundle to prepare
            dataset_uri: Dataset URI
            artifacts_dir: Directory for artifacts
            
        Returns:
            Prepared code string
        """
        safety_setup = f'''
import sys
import os
import signal
import resource
import tempfile
from pathlib import Path
import io
import contextlib
import json

# Set up resource limits
def set_resource_limits():
    try:
        memory_limit_bytes = {code_bundle.memory_limit_mb * 1024 * 1024}
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        cpu_time_limit = {code_bundle.execution_timeout_seconds}
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
        file_size_limit = 100 * 1024 * 1024  # 100MB
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))
    except Exception as e:
        print(f"Warning: Could not set resource limits: {{e}}", file=sys.stderr)

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({code_bundle.execution_timeout_seconds})
set_resource_limits()

ARTIFACTS_DIR = Path(r"{artifacts_dir}")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_URI = r"{dataset_uri}"

stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
'''

        code_body = f'''
with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
    try:
{self._indent_code(code_bundle.code, 2)}
        result = main() 
    except Exception as e:
        result = {{"status": "error", "error": str(e)}}
        raise

# Output result as JSON
print("\\n===EXECUTION_RESULT===")
print(json.dumps(result))
print("===END_EXECUTION_RESULT===")
'''

        return safety_setup + code_body

    
    def _run_code_in_sandbox(self, code_file: str, artifacts_dir: Path, timeout: int, memory_limit_mb: int) -> Dict[str, Any]:
        """
        Run code in a sandbox environment.
        
        Args:
            code_file: Path to the code file
            artifacts_dir: Directory for artifacts
            timeout: Execution timeout in seconds
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Dictionary with execution results
        """
        # Set up environment variables for sandbox
        env = os.environ.copy()
        env['PYTHONPATH'] = ':'.join([
            str(Path(__file__).parent.parent.parent.parent),  # Add project root to path
            env.get('PYTHONPATH', '')
        ])
        
        # Set up subprocess with restrictions
        try:
            # Start the process
            process = subprocess.Popen(
                [sys.executable, code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(artifacts_dir),
                preexec_fn=self._setup_sandbox_limits
            )
            
            # Monitor the process
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace')
            stderr_str = stderr.decode('utf-8', errors='replace')
            
            # Parse execution result
            return_value = self._parse_execution_result(stdout_str)
            
            # Determine status
            if process.returncode == 0:
                status = ExecutionStatus.SUCCESS
                error_message = None
            else:
                status = ExecutionStatus.FAILED
                error_message = stderr_str or f"Process exited with code {process.returncode}"
       
            return {
                'status': status,
                'stdout': stdout_str,
                'stderr': stderr_str,
                'return_value': return_value,
                'error_message': error_message
            }
            
        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            try:
                process.kill()
                process.wait()
            except:
                pass
            
            return {
                'status': ExecutionStatus.TIMEOUT,
                'stdout': '',
                'stderr': f'Execution timed out after {timeout} seconds',
                'return_value': None,
                'error_message': f'Execution timed out after {timeout} seconds'
            }
            
        except Exception as e:
            return {
                'status': ExecutionStatus.FAILED,
                'stdout': '',
                'stderr': str(e),
                'return_value': None,
                'error_message': str(e)
            }
    
    def _setup_sandbox_limits(self):
        """Set up sandbox limits for the subprocess."""
        try:
            import resource
            
            # Set memory limit
            memory_limit_bytes = 512 * 1024 * 1024  # 512MB default
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            
            # Set CPU time limit
            cpu_time_limit = 300  # 5 minutes default
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))
            
            # Set file size limit
            file_size_limit = 100 * 1024 * 1024  # 100MB
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_limit, file_size_limit))
            
        except Exception as e:
            self.logger.warning(f"Could not set sandbox limits: {e}")
    
    def _parse_execution_result(self, stdout: str) -> Optional[Dict[str, Any]]:
        """
        Parse execution result from stdout.
        
        Args:
            stdout: Standard output from execution
            
        Returns:
            Parsed result dictionary or None
        """
        try:
            # Look for execution result markers
            start_marker = "===EXECUTION_RESULT==="
            end_marker = "===END_EXECUTION_RESULT==="
            
            start_idx = stdout.find(start_marker)
            end_idx = stdout.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                result_str = stdout[start_idx + len(start_marker):end_idx].strip()
                return json.loads(result_str)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to parse execution result: {e}")
            return None
    
    def _collect_artifacts(self, artifacts_dir: Path) -> List[Dict[str, Any]]:
        """
        Collect artifacts from the artifacts directory.
        
        Args:
            artifacts_dir: Directory containing artifacts
            
        Returns:
            List of artifact dictionaries
        """
        artifacts = []
        
        if not artifacts_dir.exists():
            return artifacts
        
        for file_path in artifacts_dir.rglob('*'):
            if file_path.is_file():
                try:
                    artifact = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'size_bytes': file_path.stat().st_size,
                        'type': self._infer_artifact_type(file_path.name)
                    }
                    artifacts.append(artifact)
                except Exception as e:
                    self.logger.warning(f"Failed to collect artifact {file_path}: {e}")
        
        return artifacts
    
    def _infer_artifact_type(self, filename: str) -> str:
        """Infer artifact type from filename."""
        filename_lower = filename.lower()
        
        if any(ext in filename_lower for ext in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']):
            return 'figure'
        elif any(ext in filename_lower for ext in ['.csv', '.xlsx', '.json']):
            return 'table'
        elif any(ext in filename_lower for ext in ['.txt', '.md']):
            return 'text'
        else:
            return 'data' 