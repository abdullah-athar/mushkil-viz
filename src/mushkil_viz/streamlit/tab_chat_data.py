import streamlit as st
import json
import plotly.graph_objects as go
from mushkil_viz.agent.utils import run_analysis


def render_chat_data_tab(df):
    """Render the Chat with Data tab content."""
    if df is None:
        st.warning("⚠️ Please load data in the 'Load Data' tab first.")
        return
    
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