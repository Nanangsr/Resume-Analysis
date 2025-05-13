import os
import streamlit as st
from dotenv import load_dotenv
from app.ui import render_ui
from app.controller import process_use_case
import pandas as pd
import plotly.express as px
import json
from utils.name_extractor import extract_name_from_resume

load_dotenv()

def main():
    st.set_page_config(page_title="RAG Resume Analyzer", layout="wide")
    st.title("📄 AI Resume Analyzer with Groq")
    
    use_case, inputs, question = render_ui()
    
    if use_case and inputs:
        with st.spinner("Processing..."):
            try:
                results = process_use_case(use_case, inputs, question)
                
                st.subheader("Results")
                
                if use_case == "Compare with Scoring" and isinstance(results, dict):
                    # Validate results structure
                    if "ranking" not in results or not isinstance(results["ranking"], list):
                        st.error("Invalid results format from scoring system")
                        return
                    
                    # Get candidate names from session state or extract them
                    candidate_names = st.session_state.get("candidate_names", [])
                    if len(candidate_names) != len(results["ranking"]):
                        # Regenerate names if mismatch
                        candidate_names = []
                        for i, candidate in enumerate(results["ranking"]):
                            if isinstance(candidate, dict) and "text" in candidate:
                                try:
                                    name = extract_name_from_resume(candidate["text"])
                                    candidate_names.append(name or f"Candidate {i+1}")
                                except Exception:
                                    candidate_names.append(f"Candidate {i+1}")
                            else:
                                candidate_names.append(f"Candidate {i+1}")
                        st.session_state.candidate_names = candidate_names
                        
                    # Display ranking table
                    st.markdown("### 🏆 Candidate Ranking")
                    
                    # Prepare data for display
                    display_data = []
                    for idx, candidate in enumerate(results["ranking"]):
                        if not isinstance(candidate, dict):
                            continue
                            
                        row = {
                            "Rank": candidate.get("rank", idx+1),
                            "Candidate": candidate_names[idx],
                            "Total Score": f"{candidate.get('total_score', 0):.1f}/{results.get('max_score', 100)}",
                            "Percentage": f"{candidate.get('percentage', 0):.1f}%"
                        }
                        
                        # Add individual scores
                        if "scores" in candidate and isinstance(candidate["scores"], dict):
                            for criterion, score in candidate["scores"].items():
                                row[criterion] = f"{score:.1f}"
                        
                        display_data.append(row)
                    
                    df = pd.DataFrame(display_data)
                    df = df.set_index("Rank")
                    
                    # Display table
                    st.dataframe(
                        df.style.format(precision=1),
                        use_container_width=True
                    )

                    # Visualizations for top candidates
                    top_n = min(3, len(results["ranking"]))
                    
                    # Radar chart
                    st.markdown("### 📊 Top Candidates Comparison")
                    plot_data = []
                    for idx in range(top_n):
                        candidate = results["ranking"][idx]
                        if not isinstance(candidate, dict) or "scores" not in candidate:
                            continue
                            
                        for criterion, score in candidate["scores"].items():
                            plot_data.append({
                                "Candidate": candidate_names[idx],
                                "Criterion": criterion,
                                "Score": score,
                                "Max": results.get("criteria", {}).get(criterion, 10)
                            })
                    
                    if plot_data:
                        fig = px.line_polar(
                            pd.DataFrame(plot_data),
                            r="Score",
                            theta="Criterion",
                            color="Candidate",
                            line_close=True,
                            template="plotly_white",
                            height=500,
                            range_r=[0, 10]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bar chart
                    st.markdown("### 📊 Detailed Criteria Comparison")
                    if plot_data:  # Reuse data from radar chart
                        fig_bar = px.bar(
                            pd.DataFrame(plot_data),
                            x="Criterion",
                            y="Score",
                            color="Candidate",
                            barmode="group",
                            title="Score Comparison by Criteria",
                            hover_data=["Max"],
                            height=500
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # Export functionality
                    @st.cache_data
                    def prepare_export_data():
                        csv_data = []
                        json_data = results.copy()
                        
                        for idx, candidate in enumerate(results["ranking"]):
                            if not isinstance(candidate, dict):
                                continue
                                
                            csv_row = {
                                "Rank": candidate.get("rank", idx+1),
                                "Candidate": candidate_names[idx],
                                "Total Score": candidate.get("total_score", 0),
                                "Percentage": candidate.get("percentage", 0)
                            }
                            
                            if "scores" in candidate:
                                csv_row.update(candidate["scores"])
                            
                            csv_data.append(csv_row)
                            json_data["ranking"][idx]["name"] = candidate_names[idx]
                        
                        return csv_data, json_data

                    st.markdown("### 📤 Export Results")
                    csv_data, json_data = prepare_export_data()

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🖨️ Export as CSV"):
                            df_export = pd.DataFrame(csv_data)
                            st.download_button(
                                label="💾 Download CSV",
                                data=df_export.to_csv(index=False).encode('utf-8'),
                                file_name="candidate_scores.csv",
                                mime="text/csv"
                            )

                    with col2:
                        if st.button("🖨️ Export as JSON"):
                            st.download_button(
                                label="💾 Download JSON",
                                data=json.dumps(json_data, indent=2).encode('utf-8'),
                                file_name="candidate_scores.json",
                                mime="application/json"
                            )

                    # Display analysis
                    if "analysis" in results:
                        st.markdown("### 📝 Detailed Analysis")
                        st.write(results["analysis"])
                    
                else:
                    st.write(results)
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
