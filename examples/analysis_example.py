#!/usr/bin/env python3
"""
Example usage of the LLM-powered agentic analysis framework.

This script demonstrates how to use the framework to analyze a dataset
with automated planning, code generation, execution, and reporting.
"""

import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mushkil_viz.main import analyze_dataset_simple, create_workflow
from mushkil_viz.schema import DatasetFormat
from dotenv import load_dotenv

load_dotenv()


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('analysis_example.log')
        ]
    )


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 10, n_samples).astype(int),
        'income': np.random.normal(50000, 15000, n_samples).astype(int),
        'purchase_amount': np.random.exponential(100, n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'is_premium': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    # Add some null values for data quality testing
    data['age'][np.random.choice(n_samples, 50, replace=False)] = None
    data['income'][np.random.choice(n_samples, 30, replace=False)] = None
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    dataset_path = "sample_customer_data.csv"
    df.to_csv(dataset_path, index=False)
    
    print(f"Created sample dataset: {dataset_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:\n{df.head()}")
    
    return dataset_path


def main():
    """Main example function."""
    print("=== LLM-Powered Agentic Analysis Framework Example ===\n")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set.")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        print("Continuing with example...\n")
    
    # Create sample dataset
    print("1. Creating sample dataset...")
    dataset_path = create_sample_dataset()
    print()
    
    # Example 1: Simple analysis with default settings
    print("2. Running simple analysis with default settings...")
    try:
        final_state = analyze_dataset_simple(
            dataset_uri=dataset_path,
            dataset_format=DatasetFormat.CSV,
            model_name="gemini-2.0-flash",
            api_key=api_key
        )
        
        print("✅ Analysis completed successfully!")
        print(f"Workflow ID: {final_state.workflow_id}")
        print(f"Total execution time: {final_state.total_execution_time:.2f} seconds")
        print(f"Completed steps: {len(final_state.completed_steps)}")
        print(f"Failed steps: {len(final_state.failed_steps)}")
        
        if final_state.final_report:
            print(f"Final report generated: {len(final_state.final_report.markdown_content)} characters")
            print(f"Key insights: {len(final_state.final_report.key_insights)}")
            print(f"Recommendations: {len(final_state.final_report.recommendations)}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"✗ Analysis failed: {e}")
    
    print()
    
    # Example 2: Custom workflow with specific settings
    print("3. Running custom workflow with specific settings...")
    try:
        workflow = create_workflow(
            model_name="gemini-2.0-flash",
            temperature=0.2,
            max_tokens=8000,
            api_key=api_key,
            artifacts_base_dir="./artifacts",
            max_iterations=2
        )
        
        final_state = workflow.analyze_dataset(
            dataset_uri=dataset_path,
            dataset_format=DatasetFormat.CSV,
            workflow_id="custom_example"
        )
        
        print("✅ Custom analysis completed successfully!")
        
        # Get workflow summary
        summary = workflow.get_workflow_summary(final_state)
        print("Workflow Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save workflow state
        output_dir = "./output"
        state_file = workflow.save_workflow_state(final_state, output_dir)
        print(f"Workflow state saved to: {state_file}")
        
    except Exception as e:
        logger.error(f"Custom analysis failed: {e}")
        print(f"✗ Custom analysis failed: {e}")
    
    print()
    
    # Example 3: Show how to create custom workflows
    print("4. Example of creating custom analysis workflows...")
    print("""
# Custom workflow example:
workflow = create_workflow(
    model_name="gemini-2.0-flash",
    temperature=0.1,
    max_tokens=4000,
    api_key=os.getenv("GEMINI_API_KEY"),
    artifacts_base_dir="/custom/artifacts",
    max_iterations=5
)

# Run analysis
final_state = workflow.analyze_dataset(
    dataset_uri="your_dataset.csv",
    dataset_format=DatasetFormat.CSV,
    workflow_id="my_analysis"
)

# Get results
if final_state.final_report:
    print(final_state.final_report.markdown_content)
    print("Key insights:", final_state.final_report.key_insights)
    print("Recommendations:", final_state.final_report.recommendations)
""")
    
    print("\n=== Example completed ===")
    print("Check the generated artifacts and logs for detailed results.")


if __name__ == "__main__":
    main() 