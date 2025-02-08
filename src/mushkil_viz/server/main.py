from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ..core.engine import MushkilVizEngine
from .visualization_adapter import adapt_visualizations_for_frontend
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import json
from datetime import datetime, date
import traceback
import math
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="MushkilViz API", description="API for MushkilViz data analysis and visualization")

# Configure CORS with environment variables
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the engine
engine = MushkilVizEngine()

def convert_keys_to_str(obj):
    """Convert all dictionary keys to strings recursively"""
    if isinstance(obj, dict):
        return {str(key): convert_keys_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    return obj

def clean_numeric_value(value):
    """Clean numeric values for JSON serialization"""
    if isinstance(value, (int, float, np.number)):
        try:
            float_val = float(value)
            if math.isnan(float_val) or math.isinf(float_val):
                return None
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
        except (ValueError, TypeError, OverflowError):
            return None
    return value

def clean_data_for_json(obj):
    """Recursively clean data for JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_data_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_data_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return [clean_data_for_json(item) for item in obj.tolist()]
    elif isinstance(obj, pd.Series):
        return [clean_data_for_json(item) for item in obj.tolist()]
    else:
        return clean_numeric_value(obj)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    try:
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        raise TypeError(f"Type {type(obj)} not serializable")
    except Exception as e:
        print(f"Serialization error for object {type(obj)}: {str(e)}")
        raise

def validate_csv_file(file_content: bytes) -> bool:
    """Validate if the file is a proper CSV file"""
    try:
        # Try to read the first few lines to validate CSV format
        pd.read_csv(io.BytesIO(file_content), nrows=5)
        return True
    except Exception as e:
        print(f"CSV validation error: {str(e)}")
        return False

def get_visualization_metadata():
    """Get metadata about available visualizations and their configurations."""
    return {
        "spending_trends": {
            "title": "Spending Trends Over Time",
            "type": "time_series",
            "axes": {
                "x": {"title": "Date", "type": "date", "tickformat": "%b %Y"},
                "y": [
                    {"title": "Amount", "type": "numeric", "side": "left"},
                    {"title": "Transaction Count", "type": "numeric", "side": "right"}
                ]
            },
            "layout": {"span": 12}
        },
        "category_distribution": {
            "title": "Category Distribution",
            "type": "bar",
            "axes": {
                "x": {"title": "Category", "type": "category", "tickangle": -45},
                "y": {"title": "Amount ($)", "type": "numeric"}
            },
            "layout": {"span": 6}
        },
        "merchant_distribution": {
            "title": "Top Merchants",
            "type": "bar",
            "orientation": "horizontal",
            "axes": {
                "x": {"title": "Amount ($)", "type": "numeric"},
                "y": {"title": "Merchant", "type": "category"}
            },
            "layout": {"span": 6}
        },
        "cash_flow": {
            "title": "Cash Flow Analysis",
            "type": "bar",
            "axes": {
                "x": {"title": "Type", "type": "category"},
                "y": {"title": "Amount ($)", "type": "numeric"}
            },
            "showLegend": False,
            "layout": {"span": 12}
        }
    }

@app.get("/api/visualization-metadata")
async def get_visualization_config():
    """Get visualization configuration and metadata."""
    return JSONResponse(content=get_visualization_metadata())

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded CSV file and return visualizations
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported. Please upload a file with .csv extension."
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate CSV format
        if not validate_csv_file(content):
            raise HTTPException(
                status_code=400,
                detail="Invalid CSV file format. Please check if the file is properly formatted."
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file.write(content)
            temp_file.flush()
            
            print(f"Processing file: {file.filename}")
            
            try:
                # Process using existing engine
                results = engine.process_dataset(temp_file.name)
                
                # Add visualization metadata
                results["visualization_metadata"] = get_visualization_metadata()
                
                # Convert visualizations to frontend format
                if "visualizations" in results:
                    results["visualizations"] = adapt_visualizations_for_frontend(results["visualizations"])
                
                print(f"Results keys: {results.keys()}")
            except pd.errors.EmptyDataError:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded CSV file is empty."
                )
            except Exception as e:
                print(f"Processing error: {str(e)}")
                print("Full traceback:")
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing the file: {str(e)}"
                )
            finally:
                # Clean up temp file
                Path(temp_file.name).unlink()
            
            # Debug print the results before serialization
            print("Results before serialization:")
            for key, value in results.items():
                print(f"Key: {key}, Type: {type(value)}")
                if isinstance(value, dict):
                    print(f"Nested keys for {key}: {value.keys()}")
            
            # First convert all dictionary keys to strings
            results = convert_keys_to_str(results)
            
            # Clean numeric values
            results = clean_data_for_json(results)
            
            # Clean the results for JSON serialization using our custom serializer
            try:
                cleaned_results = json.loads(
                    json.dumps(results, default=json_serial)
                )
                return JSONResponse(content=cleaned_results)
            except Exception as e:
                print(f"Serialization error: {str(e)}")
                print("Full traceback:")
                traceback.print_exc()
                # Try to identify problematic keys
                problematic_keys = []
                for key, value in results.items():
                    try:
                        json.dumps({key: value}, default=json_serial)
                    except Exception as nested_e:
                        print(f"Problem with key '{key}': {str(nested_e)}")
                        problematic_keys.append(key)
                print(f"Problematic keys: {problematic_keys}")
                raise HTTPException(
                    status_code=500,
                    detail="Error serializing analysis results. This might be due to invalid data in the CSV file."
                )
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the file. Please try again or contact support if the problem persists."
        )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"} 