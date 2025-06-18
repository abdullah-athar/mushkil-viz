from pathlib import Path
from .agent import MultiAgentSystem, load_data, run_analysis


PACKAGE_ROOT = Path(__file__).parent


__version__ = "0.1.0"
__all__ = ["MultiAgentSystem", "load_data", "run_analysis", "PACKAGE_ROOT"]
