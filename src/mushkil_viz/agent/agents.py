"""
Legacy agents module - now uses the new modular system for backward compatibility.

This module maintains the original API while delegating to the new modular agent system.
The original MultiAgentSystem class is now imported from the package's __init__.py.
"""

# Import the new modular system
from . import MultiAgentSystem, load_data, run_analysis

# Keep the original imports for any direct usage

# Re-export everything for backward compatibility
__all__ = ["MultiAgentSystem", "load_data", "run_analysis"]
