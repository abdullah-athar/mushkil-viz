import streamlit as st
from mushkil_viz.streamlit.utils import validate_uploaded_file, load_sample_data, safe_read_csv


def render_load_data_tab():
    """Render the Load Data tab content."""
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

    return df, data_source, uploaded_file 