import streamlit as st
from mushkil_viz.agent.utils import load_data
from mushkil_viz.streamlit.utils import check_data_quality


def render_basic_analysis_tab(df, data_source, uploaded_file):
    """Render the Basic Analysis tab content."""
    if df is None:
        st.warning("⚠️ Please load data in the 'Load Data' tab first.")
        return
    
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