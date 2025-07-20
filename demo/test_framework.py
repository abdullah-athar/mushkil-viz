#!/usr/bin/env python3
"""
Simple test script for the LLM-powered agentic analysis framework.

This script tests the framework with the financial_data.csv example dataset.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


from mushkil_viz.main import analyze_dataset_simple, create_workflow
from mushkil_viz.schema import DatasetFormat


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_financial_data():
    """Test the framework with financial_data.csv."""
    print("=== Testing Framework with Financial Data ===\n")
    
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Test dataset path
    dataset_path = "examples/financial_data.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    try:
        print("\n🚀 Starting analysis...")
        
        # Run simple analysis
        final_state = analyze_dataset_simple(
            dataset_uri=dataset_path,
            dataset_format=DatasetFormat.CSV,
            model_name="gemini-2.0-flash",
            api_key=api_key
        )
        
        print("✅ Analysis completed successfully!")
        
        # Display results
        print(f"\n📊 Results Summary:")
        print(f"  Workflow ID: {final_state.workflow_id}")
        print(f"  Total execution time: {final_state.total_execution_time:.2f} seconds")
        print(f"  Completed steps: {len(final_state.completed_steps)}")
        print(f"  Failed steps: {len(final_state.failed_steps)}")
        
        if final_state.final_report:
            print(f"  Final report length: {len(final_state.final_report.markdown_content)} characters")
            print(f"  Key insights: {len(final_state.final_report.key_insights)}")
            print(f"  Recommendations: {len(final_state.final_report.recommendations)}")
            
            # Save the report
            report_file = "test_analysis_report.md"
            with open(report_file, "w") as f:
                f.write(final_state.final_report.markdown_content)
            print(f"  📄 Report saved to: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logging.error(f"Analysis failed: {e}", exc_info=True)
        return False


def test_netflix_data():
    """Test the framework with netflix_titles.csv."""
    print("\n=== Testing Framework with Netflix Data ===\n")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return False
    
    dataset_path = "examples/netflix_titles.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset found: {dataset_path}")
    
    try:
        print("\n🚀 Starting Netflix analysis...")
        
        # Create custom workflow for larger dataset
        workflow = create_workflow(
            model_name="gemini-2.0-flash",
            temperature=0.1,
            max_tokens=8000,
            api_key=api_key,
            max_iterations=2
        )
        
        final_state = workflow.analyze_dataset(
            dataset_uri=dataset_path,
            dataset_format=DatasetFormat.CSV,
            workflow_id="netflix_test"
        )
        
        print("✅ Netflix analysis completed!")
        
        # Get summary
        summary = workflow.get_workflow_summary(final_state)
        print(f"\n📊 Netflix Analysis Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Netflix analysis failed: {e}")
        logging.error(f"Netflix analysis failed: {e}", exc_info=True)
        return False


def main():
    """Main test function."""
    setup_logging()
    
    print("🧪 Testing LLM-Powered Agentic Analysis Framework")
    print("=" * 60)
    
    # Test 1: Financial data (smaller, faster)
    success1 = test_financial_data()
    
    # Test 2: Netflix data (larger, more complex)
    if success1:
        success2 = test_netflix_data()
    else:
        print("\n⏭️  Skipping Netflix test due to financial data failure")
        success2 = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 Test Summary:")
    print(f"  Financial Data Test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"  Netflix Data Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    return success2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 