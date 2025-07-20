#!/usr/bin/env python3
"""
Script to generate and display the MushkilViz workflow diagram.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mushkil_viz.main import create_workflow

def generate_workflow_diagram():
    """Generate and display the workflow diagram."""
    try:
        # Create a workflow instance
        workflow = create_workflow()
        
        # Get the graph
        graph = workflow.graph
        
        # Display the diagram
        from IPython.display import Image, display

        
        # Try to display the actual graph diagram
        try:
            # Generate the diagram
            diagram = graph.get_graph().draw_mermaid_png()
            # save bytes object to png file
            with open("workflow_diagram.png", "wb") as f:
                f.write(diagram)
            
            print("✅ Workflow diagram displayed successfully!")
        except Exception as e:
            print(f"⚠️  Could not generate visual diagram: {e}")
            print("📋 Using text representation instead.")
            
    except Exception as e:
        print(f"❌ Error generating workflow diagram: {e}")
        return None

if __name__ == "__main__":
    generate_workflow_diagram() 