import streamlit as st
import os
from dotenv import load_dotenv

# Import modular components
from mushkil_viz.streamlit.ui_components import load_custom_css, render_header, render_sidebar
from mushkil_viz.streamlit.tab_load_data import render_load_data_tab
from mushkil_viz.streamlit.tab_basic_analysis import render_basic_analysis_tab
from mushkil_viz.streamlit.tab_chat_data import render_chat_data_tab
from mushkil_viz.streamlit.tab_auto_analysis import render_auto_analysis_tab

# Basic page configuration
st.set_page_config(
    page_title="MushkilViz - AI Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Load custom styling
load_custom_css()

# Initialize environment
load_dotenv()

# Render header
render_header()

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
    df, data_source, uploaded_file = render_load_data_tab()

# Tab 2: Basic Analysis
with tab2:
    render_basic_analysis_tab(df, data_source, uploaded_file)

# Tab 3: Chat with Data
with tab3:
    render_chat_data_tab(df)

# Tab 4: Auto Analysis
with tab4:
    render_auto_analysis_tab(df)

# Render sidebar
render_sidebar()
