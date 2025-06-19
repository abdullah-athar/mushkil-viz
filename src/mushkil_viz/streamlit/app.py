import streamlit as st
import os
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
from mushkil_viz.agent.utils import load_data, run_analysis

# Basic page configuration
st.set_page_config(
    page_title="MushkilViz - AI Data Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize environment
load_dotenv()

st.title("📊 Multi-Agent CSV Analysis with Gemini")

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

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load data and show preview
        df, summary = load_data(uploaded_file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head(3))

        st.subheader("📊 Basic Statistics")
        st.dataframe(summary)

        # Analysis prompt
        st.subheader("🤖 Multi-Agent Analysis")

        # Sample prompts button
        col1, col2 = st.columns([3, 1])
        with col1:
            prompt = st.text_area(
                "Ask your question:",
                placeholder="e.g., 'Show correlation matrix', 'Plot distribution of sales', 'Analyze missing values'",
            )
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
                                st.text("Chart JSON preview:")
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
                                st.text("Table JSON preview:")
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

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

# Empty sidebar (as requested)
with st.sidebar:
    st.empty()
