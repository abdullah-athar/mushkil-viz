import streamlit as st


def load_custom_css():
    """Load custom CSS styling for the app."""
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


def render_header():
    """Render the main application header."""
    st.markdown("""
        <div class="main-header">
            <h1>📊 MushkilViz</h1>
            <p>AI-Powered Data Analysis & Visualization Platform</p>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the minimal sidebar."""
    with st.sidebar:
        # Top section
        st.markdown("### 🚀 Quick Start")
        st.markdown("1. Upload CSV or use sample data\n2. Ask questions\n3. Get AI insights")
        
        # Bottom section (spacer + status)
        st.markdown("<br>" * 10, unsafe_allow_html=True)
        
        st.markdown("### Status")
        import os
        if os.getenv("GOOGLE_GEMINI_KEY") and os.getenv("GOOGLE_GEMINI_KEY") != "your_api_key_here":
            st.success("🔑 API Ready")
        else:
            st.error("🔑 API Missing") 