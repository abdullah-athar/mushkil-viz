import streamlit as st
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
from mushkil_viz.agent.utils import load_data, run_analysis
from mushkil_viz.streamlit.utils import validate_uploaded_file, load_sample_data, safe_read_csv, check_data_quality

# Basic page configuration
st.set_page_config(
    page_title="MushkilViz - AI Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .section-header {
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .analysis-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2980b9);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2980b9, #1f77b4);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize environment
load_dotenv()

# Main header
st.markdown("""
    <div class="main-header">
        <h1>📊 MushkilViz</h1>
        <p>AI-Powered Data Analysis & Visualization Platform</p>
    </div>
""", unsafe_allow_html=True)

# Check if API key is set
if (
    not os.getenv("GOOGLE_GEMINI_KEY")
    or os.getenv("GOOGLE_GEMINI_KEY") == "your_api_key_here"
):
    st.error("⚠️ Please set your GOOGLE_GEMINI_KEY in the .env file")
    st.info(
        "1. Create a .env file in the root directory\n2. Add: GOOGLE_GEMINI_KEY=your_actual_api_key"
    )
    st.stop()

# Main app tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗂️ Load Data", 
    "📊 Basic Analysis", 
    "💬 Chat with Data", 
    "🤖 Auto Analysis"
])

# Tab 1: Load Data
with tab1:
    # File upload section
    st.markdown('<div class="section-header"><h2>📁 Data Upload</h2></div>', unsafe_allow_html=True)
    
    # Sample data section
    sample_data = load_sample_data()
    if sample_data:
        with st.expander("📊 Use Sample Data", expanded=False):
            cols = st.columns(len(sample_data))
            for idx, (key, info) in enumerate(sample_data.items()):
                with cols[idx]:
                    if st.button(f"{info['name']}\n({info['size']})", key=f"sample_{key}"):
                        st.session_state["sample_file"] = info["path"]
                        st.rerun()

    with st.container():
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    # Handle data loading
    df = None
    data_source = "uploaded"

    # Check for sample data selection
    if "sample_file" in st.session_state and not uploaded_file:
        try:
            df = safe_read_csv(st.session_state["sample_file"])
            st.info(f"📊 Using sample data: {st.session_state['sample_file'].name}")
            data_source = "sample"
            
            if st.button("Clear Sample Data"):
                del st.session_state["sample_file"]
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            del st.session_state["sample_file"]

    # Handle uploaded file
    if uploaded_file:
        is_valid, error_msg = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            try:
                df = safe_read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")

# Tab 2: Basic Analysis
with tab2:
    if df is None:
        st.warning("⚠️ Please load data in the 'Load Data' tab first.")
    else:
        # Generate summary
        try:
            if data_source == "uploaded":
                uploaded_file.seek(0)
                _, summary = load_data(uploaded_file)
            else:
                summary = df.describe()
        except:
            summary = df.describe()

        st.markdown('<div class="section-header"><h2>📄 Data Overview</h2></div>', unsafe_allow_html=True)
        
        subtab1, subtab2, subtab3 = st.tabs(["📄 Preview", "📊 Statistics", "🔍 Quality"])
        
        with subtab1:
            st.dataframe(df.head(10))
            
        with subtab2:
            st.dataframe(summary)
            
        with subtab3:
            quality_info = check_data_quality(df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", quality_info["rows"])
                st.metric("Columns", quality_info["columns"])
            with col2:
                st.metric("Missing Values", quality_info["missing_values"])
                st.metric("Duplicates", quality_info["duplicates"])
            with col3:
                st.metric("Memory Usage", quality_info["memory_usage"])
                
            if df.isnull().sum().sum() > 0:
                st.markdown("**Missing Values by Column:**")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                st.bar_chart(missing_data)

# Tab 3: Chat with Data
with tab3:
    if df is None:
        st.warning("⚠️ Please load data in the 'Load Data' tab first.")
    else:
        # Analysis section
        st.markdown('<div class="section-header"><h2>🤖 Multi-Agent Analysis</h2></div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
            
            # Quick analysis buttons
            st.markdown("**🚀 Quick Analysis:**")
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            
            with quick_col1:
                if st.button("📊 Data Summary"):
                    st.session_state["quick_prompt"] = "Provide a comprehensive summary of this dataset including key insights"
                if st.button("🔗 Correlations"):
                    st.session_state["quick_prompt"] = "Show correlation matrix and identify strong relationships between variables"
                    
            with quick_col2:
                if st.button("📈 Distributions"):
                    st.session_state["quick_prompt"] = "Analyze the distribution of all numeric variables and create visualizations"
                if st.button("🎯 Outliers"):
                    st.session_state["quick_prompt"] = "Detect and analyze outliers in the data"
                    
            with quick_col3:
                if st.button("📋 Data Quality"):
                    st.session_state["quick_prompt"] = "Assess data quality including missing values, duplicates, and data types"
                if st.button("🔍 Feature Analysis"):
                    st.session_state["quick_prompt"] = "Analyze individual features and their characteristics"
            
            st.markdown("---")
            
            # Custom analysis input
            col1, col2 = st.columns([3, 1])
            with col1:
                prompt = st.text_area(
                    "Ask your question:",
                    value=st.session_state.get("quick_prompt", ""),
                    placeholder="e.g., 'Show correlation matrix', 'Plot distribution of sales', 'Analyze missing values'",
                )
                # Clear quick prompt after use
                if "quick_prompt" in st.session_state:
                    del st.session_state["quick_prompt"]
            with col2:
                st.write("")  # Spacing
                if st.button("💡 Get Sample Prompts"):
                    with st.spinner("Generating relevant prompts..."):
                        try:
                            sample_result = run_analysis(df, "get sample prompts")
                            if not sample_result["error"]:
                                st.info("**Suggested Analysis Ideas:**")
                                st.markdown(sample_result["output"])
                        except Exception as e:
                            st.error(f"Error getting prompts: {str(e)}")

            if st.button("🚀 Run Analysis") and prompt:
                with st.spinner("🔄 Agents working on your request..."):
                    try:
                        result = run_analysis(df, prompt)

                        # Debug information
                        with st.expander("🔍 Debug Info", expanded=False):
                            st.json(
                                {
                                    "result_keys": list(result.keys()),
                                    "has_output": bool(result.get("output")),
                                    "has_chart": bool(result.get("chart")),
                                    "has_table": bool(result.get("table")),
                                    "error": result.get("error", False),
                                    "output_length": len(str(result.get("output", ""))),
                                }
                            )

                        if result["error"]:
                            st.error(f"❌ Error: {result['output']}")
                        else:
                            st.success("✅ Analysis Complete!")

                            # Always show the output
                            if result.get("output"):
                                st.markdown("### 📝 Analysis Result:")
                                st.markdown(result["output"])
                            else:
                                st.warning("⚠️ No output text received from agents")

                            # Display interactive visualization if generated
                            if result.get("chart"):
                                st.subheader("📈 Interactive Visualization")
                                try:
                                    # Parse the Plotly JSON and display
                                    fig_dict = json.loads(result["chart"])
                                    fig = go.Figure(fig_dict)
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.success("✅ Chart displayed successfully!")
                                except Exception as e:
                                    st.error(f"❌ Error displaying chart: {str(e)}")
                                    with st.expander("Chart data preview"):
                                        st.text(
                                            str(result["chart"])[:500] + "..."
                                            if len(str(result["chart"])) > 500
                                            else str(result["chart"])
                                        )

                            # Display enhanced table results if generated
                            if result.get("table"):
                                st.subheader(f"📊 {result['table']['title']}")
                                try:
                                    # Parse the dataframe JSON and display
                                    import pandas as pd

                                    df_result = pd.read_json(result["table"]["data"])
                                    st.dataframe(df_result, use_container_width=True)
                                    st.success("✅ Table displayed successfully!")
                                except Exception as e:
                                    st.error(f"❌ Error displaying table: {str(e)}")
                                    with st.expander("Table data preview"):
                                        st.text(
                                            str(result["table"]["data"])[:500] + "..."
                                            if len(str(result["table"]["data"])) > 500
                                            else str(result["table"]["data"])
                                        )

                    except Exception as e:
                        st.error(f"💥 Critical error: {str(e)}")
                        import traceback

                        st.text("Full traceback:")
                        st.text(traceback.format_exc())
            
            st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Auto Analysis
with tab4:
    if df is None:
        st.warning("⚠️ Please load data in the 'Load Data' tab first.")
    else:
        st.info("🚧 **Coming Soon!**")
        st.markdown("This feature will include automated analysis and insights.")

# Minimal sidebar - pinned items only
with st.sidebar:
    # Top section
    st.markdown("### 🚀 Quick Start")
    st.markdown("1. Upload CSV or use sample data\n2. Ask questions\n3. Get AI insights")
    
    # Bottom section (spacer + status)
    st.markdown("<br>" * 10, unsafe_allow_html=True)
    
    st.markdown("### Status")
    if os.getenv("GOOGLE_GEMINI_KEY") and os.getenv("GOOGLE_GEMINI_KEY") != "your_api_key_here":
        st.success("🔑 API Ready")
    else:
        st.error("🔑 API Missing")
