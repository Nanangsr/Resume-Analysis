import streamlit as st
from typing import Tuple, Union, Dict, List, Optional
from utils.resume_parser import parse_resume, parse_uploaded_folder
from utils.jd_parser import parse_jd
from utils.name_extractor import NameExtractor
from utils.resume_standardizer import ResumeStandardizer
import pandas as pd
import plotly.express as px
import logging
import traceback
import json
import re

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Inisialisasi NameExtractor
name_extractor = NameExtractor()

# Daftar domain yang didukung
SUPPORTED_DOMAINS = ["General", "IT", "HR", "Finance", "Marketing", "Sales", "Operations"]

# Inisialisasi session state lainnya
if 'upload_errors' not in st.session_state:
    st.session_state.upload_errors = []

if 'last_jd_text' not in st.session_state:
    st.session_state.last_jd_text = None

if 'last_scoring_results' not in st.session_state:
    st.session_state.last_scoring_results = None

if 'last_narrative_analysis' not in st.session_state:
    st.session_state.last_narrative_analysis = None

if 'show_scoring_results' not in st.session_state:
    st.session_state.show_scoring_results = False

if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "General"

if 'processed_flag' not in st.session_state:
    st.session_state.processed_flag = False  # Flag untuk mencegah pemrosesan ganda

def display_scoring_results(results: Dict):
    """Display scoring results with tabs for different analyses"""
    logger.info(f"Displaying scoring results: {type(results)}")
    
    if not results or "ranking" not in results:
        st.error("❌ Hasil scoring tidak valid atau kosong")
        logger.error(f"Invalid scoring results: {results}")
        return
    
    st.session_state.show_scoring_results = True
    
    try:
        st.markdown("---")
        st.markdown("## 🎯 Hasil Analisis Kandidat")
        st.markdown(f"**Jumlah Kandidat Dianalisis:** {len(results['ranking'])}")
        st.markdown(f"**Domain:** {st.session_state.selected_domain.title()}")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Ranking", 
            "📊 Visualisasi", 
            "📝 Analisis Naratif", 
            "📤 Export"
        ])
        
        with tab1:
            st.subheader("🏆 Hasil Ranking Kandidat")
            ranking_data = []
            for candidate in results["ranking"]:
                row = {
                    "Ranking": candidate["rank"],
                    "Nama": candidate.get("name", f"Kandidat {candidate['candidate_id']}"),
                    "AI Score": f"{candidate['ai_score']:.1f}",
                    "Total Score": f"{candidate['total_score']:.1f}",
                    "Level": candidate["level"].title()
                }
                for criterion, score in candidate["scores"].items():
                    row[criterion] = f"{score:.1f}"
                ranking_data.append(row)
            
            df = pd.DataFrame(ranking_data)
            fixed_columns = ["Ranking", "Nama", "AI Score", "Total Score", "Level"]
            criteria_columns = [c for c in df.columns if c not in fixed_columns]
            df = df[fixed_columns + criteria_columns]
            
            st.dataframe(
                df, 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ranking": st.column_config.NumberColumn("Ranking", format="%d"),
                    "AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100),
                    "Total Score": st.column_config.NumberColumn("Total Score", format="%.1f")
                }
            )
            
        with tab2:
            st.subheader("📊 Visualisasi Performa Kandidat")
            col1, col2 = st.columns(2)
            
            with col1:
                top_candidates = results["ranking"][:3]
                if len(top_candidates) >= 2:
                    radar_data = []
                    for candidate in top_candidates:
                        for criterion, score in candidate["scores"].items():
                            radar_data.append({
                                "Kandidat": candidate.get("name", f"Kandidat {candidate['candidate_id']}"),
                                "Kriteria": criterion,
                                "Skor": score
                            })
                    
                    radar_df = pd.DataFrame(radar_data)
                    fig = px.line_polar(
                        radar_df, 
                        r="Skor", 
                        theta="Kriteria", 
                        color="Kandidat",
                        line_close=True,
                        template="plotly_white",
                        title="Perbandingan Top 3 Kandidat"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True, key=f"radar_chart_{id(results)}")
            
            with col2:
                bar_data = []
                for candidate in results["ranking"]:
                    bar_data.append({
                        "Kandidat": candidate.get("name", f"Kandidat {candidate['candidate_id']}"),
                        "AI Score": candidate["ai_score"],
                        "Rank": candidate["rank"]
                    })
                
                bar_df = pd.DataFrame(bar_data)
                fig = px.bar(
                    bar_df,
                    x="Kandidat",
                    y="AI Score",
                    color="AI Score",
                    color_continuous_scale="RdYlBu_r",
                    title="AI Score per Kandidat",
                    text="AI Score"
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{id(results)}")
        
        with tab3:
            st.subheader("📝 Analisis Naratif Mendalam")
            narrative = None
            sources_checked = []
            
            if results.get("narrative_analysis") and str(results["narrative_analysis"]).strip():
                narrative = str(results["narrative_analysis"]).strip()
                sources_checked.append("results parameter")
                logger.info("Using narrative from results parameter")
            
            elif st.session_state.get("last_narrative_analysis") and str(st.session_state["last_narrative_analysis"]).strip():
                narrative = str(st.session_state["last_narrative_analysis"]).strip()
                sources_checked.append("session state")
                logger.info("Using narrative from session state")
            
            logger.info(f"Narrative sources checked: {sources_checked}")
            logger.info(f"Narrative content available: {bool(narrative)}")
            
            if narrative:
                cleaned_narrative = re.sub(r"<think>.*?</think>", "", narrative, flags=re.DOTALL | re.IGNORECASE)
                cleaned_narrative = re.sub(r"<think>.*?(?=###|\n\n|$)", "", cleaned_narrative, flags=re.DOTALL | re.IGNORECASE)
                cleaned_narrative = re.sub(r"^.*?(?=###|Ringkasan|Analisis)", "", cleaned_narrative, flags=re.DOTALL | re.IGNORECASE)
                cleaned_narrative = cleaned_narrative.strip()
                
                logger.info(f"Cleaned narrative length: {len(cleaned_narrative)}")
                
                if cleaned_narrative and len(cleaned_narrative) > 50:
                    if cleaned_narrative.startswith("⚠️"):
                        st.error(f"**Error dalam Analisis:** {cleaned_narrative[2:]}")
                        if st.button("🔄 Generate Ulang Analisis", key=f"regenerate_from_error_{id(results)}"):
                            with st.spinner("Membuat analisis baru..."):
                                try:
                                    from core.rag_chain import ResumeRagChain
                                    rag_chain = ResumeRagChain(domain=st.session_state.selected_domain)
                                    new_analysis = rag_chain.generate_llm_narrative_analysis(
                                        results,
                                        st.session_state.get("last_jd_text")
                                    )
                                    if new_analysis and not str(new_analysis).startswith("⚠️"):
                                        st.session_state["last_narrative_analysis"] = new_analysis
                                        results["narrative_analysis"] = new_analysis
                                        st.success("✅ Analisis berhasil dibuat ulang!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Gagal membuat analisis baru")
                                except Exception as e:
                                    st.error(f"❌ Error saat generate ulang: {str(e)}")
                                    logger.error(f"Regeneration error: {str(e)}")
                    elif any(section.lower() in cleaned_narrative.lower() for section in [
                        "ringkasan eksekutif", "analisis komparatif", 
                        "rekomendasi", "eksekutif", "komparatif", 
                        "### 1", "### 2", "### 3"
                    ]):
                        with st.container():
                            st.markdown("### 🎯 Analisis Lengkap")
                            st.markdown(cleaned_narrative)
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                if st.button("🔄 Regenerate", key=f"regenerate_narrative_{id(results)}"):
                                    with st.spinner("Membuat analisis baru..."):
                                        try:
                                            from core.rag_chain import ResumeRagChain
                                            rag_chain = ResumeRagChain(domain=st.session_state.selected_domain)
                                            new_analysis = rag_chain.generate_llm_narrative_analysis(
                                                results,
                                                st.session_state.get("last_jd_text")
                                            )
                                            if new_analysis:
                                                st.session_state["last_narrative_analysis"] = new_analysis
                                                results["narrative_analysis"] = new_analysis
                                                st.success("✅ Analisis berhasil dibuat ulang!")
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                            with col2:
                                if st.button("📋 Copy Text", key=f"copy_narrative_{id(results)}"):
                                    st.info("💡 Gunakan Ctrl+A untuk select all, lalu Ctrl+C untuk copy")
                    else:
                        st.warning("⚠️ Format analisis tidak standar. Menampilkan hasil mentah:")
                        with st.expander("📄 Lihat Analisis", expanded=True):
                            st.markdown(cleaned_narrative)
                        if st.button("🔄 Generate Analisis Standar", key=f"regenerate_standard_{id(results)}"):
                            with st.spinner("Membuat analisis standar..."):
                                try:
                                    from core.rag_chain import ResumeRagChain
                                    rag_chain = ResumeRagChain(domain=st.session_state.selected_domain)
                                    new_analysis = rag_chain.generate_llm_narrative_analysis(
                                        results,
                                        st.session_state.get("last_jd_text")
                                    )
                                    if new_analysis:
                                        st.session_state["last_narrative_analysis"] = new_analysis
                                        results["narrative_analysis"] = new_analysis
                                        st.success("✅ Analisis standar berhasil dibuat!")
                                        st.rerun()
                                except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("📝 Konten analisis terlalu pendek atau kosong")
                    logger.info(f"Short or empty narrative: '{cleaned_narrative[:100]}...'")
            else:
                st.warning("📝 Analisis naratif belum tersedia atau kosong")
                with st.expander("🔍 Debug Information", expanded=False):
                    debug_info = {
                        "Results has narrative_analysis": bool(results.get("narrative_analysis")),
                        "Results narrative length": len(str(results.get("narrative_analysis", ""))) if results.get("narrative_analysis") else 0,
                        "Session state has narrative": bool(st.session_state.get("last_narrative_analysis")),
                        "Session narrative length": len(str(st.session_state.get("last_narrative_analysis", ""))) if st.session_state.get("last_narrative_analysis") else 0,
                        "Sources checked": sources_checked,
                        "Results keys": list(results.keys()) if results else []
                    }
                    st.json(debug_info)
                if st.button("🚀 Generate Analisis Naratif", key=f"force_generate_{id(results)}", type="primary"):
                    with st.spinner("Membuat analisis naratif..."):
                        try:
                            from core.rag_chain import ResumeRagChain
                            rag_chain = ResumeRagChain(domain=st.session_state.selected_domain)
                            analysis_results = results if results else st.session_state.get("last_scoring_results", {})
                            if analysis_results and "ranking" in analysis_results:
                                new_analysis = rag_chain.generate_llm_narrative_analysis(
                                    analysis_results,
                                    st.session_state.get("last_jd_text")
                                )
                                if new_analysis and str(new_analysis).strip():
                                    st.session_state["last_narrative_analysis"] = new_analysis
                                    results["narrative_analysis"] = new_analysis
                                    st.success("✅ Analisis naratif berhasil dibuat!")
                                    logger.info(f"Force generated narrative length: {len(str(new_analysis))}")
                                    st.rerun()
                                else:
                                    st.error("❌ Gagal membuat analisis - hasil kosong")
                                    logger.error("Force generation returned empty result")
                            else:
                                st.error("❌ Data scoring tidak valid untuk analisis")
                                logger.error("Invalid scoring data for analysis")
                        except Exception as e:
                            error_msg = f"Error saat membuat analisis: {str(e)}"
                            st.error(f"❌ {error_msg}")
                            logger.error(f"Force generation error: {error_msg}")
        
        with tab4:
            st.subheader("📤 Export Hasil Analisis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export CSV", key=f"export_csv_{id(results)}"):
                    try:
                        csv_data = []
                        for candidate in results["ranking"]:
                            row = {
                                "Ranking": candidate["rank"],
                                "Nama": candidate.get("name", f"Kandidat {candidate['candidate_id']}"),
                                "AI_Score": candidate["ai_score"],
                                "Total_Score": candidate["total_score"],
                                "Level": candidate["level"],
                                "Domain": st.session_state.selected_domain
                            }
                            for criterion, score in candidate["scores"].items():
                                row[criterion.replace(" ", "_")] = score
                            csv_data.append(row)
                        
                        df_export = pd.DataFrame(csv_data)
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv,
                            file_name=f"candidate_scoring_{st.session_state.selected_domain}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error creating CSV: {str(e)}")
            
            with col2:
                if st.button("📋 Export JSON", key=f"export_json_{id(results)}"):
                    try:
                        json_data = {
                            "timestamp": pd.Timestamp.now().isoformat(),
                            "domain": st.session_state.selected_domain,
                            "total_candidates": len(results["ranking"]),
                            "ranking": results["ranking"],
                            "criteria": results.get("criteria", {}),
                            "narrative_analysis": results.get("narrative_analysis", "")
                        }
                        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_str,
                            file_name=f"candidate_analysis_{st.session_state.selected_domain}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Error creating JSON: {str(e)}")
            
            with col3:
                if st.button("📄 Export Report", key=f"export_report_{id(results)}"):
                    try:
                        report_content = f"""
# Laporan Analisis Kandidat
Tanggal: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Domain: {st.session_state.selected_domain.title()}
Jumlah Kandidat: {len(results["ranking"])}

## Ranking Kandidat
"""
                        for candidate in results["ranking"]:
                            report_content += f"""
### {candidate['rank']}. {candidate.get('name', f"Kandidat {candidate['candidate_id']}")}
- AI Score: {candidate['ai_score']:.1f}/100
- Total Score: {candidate['total_score']:.1f}
- Level: {candidate['level'].title()}
"""
                        narrative_for_export = results.get("narrative_analysis") or st.session_state.get("last_narrative_analysis")
                        if narrative_for_export:
                            report_content += f"\n## Analisis Naratif\n{narrative_for_export}"
                        st.download_button(
                            label="⬇️ Download Report",
                            data=report_content,
                            file_name=f"candidate_report_{st.session_state.selected_domain}.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error creating report: {str(e)}")
        
        if st.checkbox("🔧 Tampilkan Data Mentah", key=f"show_raw_data_{id(results)}"):
            with st.expander("📊 Raw Data Results", expanded=False):
                st.json(results)
        
        logger.info("Scoring results displayed successfully")
        
    except Exception as e:
        error_msg = f"Error displaying scoring results: {str(e)}"
        st.error(f"❌ {error_msg}")
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

def render_ui() -> Tuple[str, Union[Dict, List, None], Optional[str]]:
    """Render Streamlit UI and collect inputs"""
    use_case = "Candidate Search by Job Description"
    inputs = None
    question = None
    
    st.sidebar.header("⚙️ Konfigurasi")
    st.session_state.selected_domain = st.sidebar.selectbox(
        "Pilih Domain",
        SUPPORTED_DOMAINS,
        index=SUPPORTED_DOMAINS.index(st.session_state.get('selected_domain', 'General')),
        help="Pilih domain untuk menyesuaikan kriteria penilaian dan analisis."
    )
    
    st.sidebar.header("📋 Use Cases")
    use_case = st.sidebar.radio(
        "Pilih Use Case",
        ["Candidate Search by Job Description", 
         "Candidate Profiling / Resume QA",
         "Compare Multiple Candidates",
         "Compare with Scoring"]
    )
    
    if use_case == "Candidate Search by Job Description":
        st.header("🔍 Candidate Search by Job Description")
        jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
        if jd_file:
            try:
                jd_text, error = parse_jd(jd_file)
                if error:
                    st.error(f"Gagal memproses JD: {error}")
                    logger.error(f"JD parsing error: {error}")
                else:
                    inputs = {"jd_text": jd_text}
                    st.session_state.last_jd_text = jd_text
            except Exception as e:
                error_msg = f"Gagal memparsing JD: {str(e)}"
                st.error(error_msg)
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    
    elif use_case == "Candidate Profiling / Resume QA":
        st.header("📝 Candidate Profiling / Resume QA")
        col1, col2 = st.columns([3, 1])
        with col1:
            resume_file = st.file_uploader(
                "Upload Resume (PDF/DOCX)", 
                type=["pdf", "docx"],
                key="profiling_uploader"
            )
        with col2:
            if st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"] and st.button("Clear Resume"):
                st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"] = None
                st.rerun()
                
        if resume_file:
            try:
                resume_text, error = parse_resume(resume_file, resume_file.name)
                if error:
                    st.error(f"Error processing resume: {error}")
                    logger.error(f"Resume parsing error: {error}")
                else:
                    st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"] = (resume_text, resume_file.name)
                    from core.retriever import add_resume_to_vector_store
                    add_resume_to_vector_store(resume_text, resume_file.name)
            except Exception as e:
                st.error(f"Failed to parse resume: {str(e)}")
                logger.error(f"Resume parsing exception: {str(e)}")
            
        if st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"]:
            inputs = {
                "resume_text": st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"][0],
                "filename": st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"][1],
                "domain": st.session_state.selected_domain
            }
            st.subheader("Ajukan Pertanyaan tentang Resume")
            question = st.text_input("Masukkan pertanyaan Anda tentang resume")
            
    elif use_case == "Compare Multiple Candidates":
        st.header("📊 Compare Multiple Candidates")
        if st.session_state.last_jd_text:
            st.info("Membandingkan kandidat terhadap deskripsi pekerjaan terakhir yang diunggah")
            with st.expander("Lihat Deskripsi Pekerjaan"):
                st.write(st.session_state.last_jd_text)
        
        upload_option = st.radio(
            "Opsi Upload:", 
            ["Multiple Files", "Folder (ZIP)"],
            key="compare_upload_option"
        )
        
        if upload_option == "Multiple Files":
            resume_files = st.file_uploader(
                "Upload Resume (PDF/DOCX)", 
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="compare_multiple_files"
            )
            if resume_files:
                valid_resumes = []
                error_messages = []
                for file in resume_files:
                    try:
                        text, error = parse_resume(file, file.name)
                        if text:
                            valid_resumes.append((text, file.name))
                            from core.retriever import add_resume_to_vector_store
                            add_resume_to_vector_store(text, file.name)
                        if error:
                            error_messages.append(error)
                            logger.error(f"Resume parsing error for {file.name}: {error}")
                    except Exception as e:
                        error_messages.append(f"Failed to parse {file.name}: {str(e)}")
                        logger.error(f"Resume parsing exception for {file.name}: {str(e)}")
                
                st.session_state.uploaded_resumes["Compare Multiple Candidates"] = valid_resumes
                st.session_state.upload_errors = error_messages
        else:
            uploaded_folder = st.file_uploader(
                "Upload Folder (sebagai ZIP)", 
                type=["zip"],
                accept_multiple_files=False,
                key="compare_folder_upload"
            )
            if uploaded_folder:
                try:
                    valid_resumes, error_messages = parse_uploaded_folder(uploaded_folder)
                    for text in valid_resumes:
                        from core.retriever import add_resume_to_vector_store
                        add_resume_to_vector_store(text, "folder_uploaded_resume")
                    st.session_state.uploaded_resumes["Compare Multiple Candidates"] = [(text, "folder_uploaded_resume") for text in valid_resumes]
                    st.session_state.upload_errors = error_messages
                    for error in error_messages:
                        logger.error(f"Folder parsing error: {error}")
                except Exception as e:
                    st.session_state.upload_errors = [f"Failed to parse folder: {str(e)}"]
                    logger.error(f"Folder parsing exception: {str(e)}")
        
        current_resumes = st.session_state.uploaded_resumes["Compare Multiple Candidates"]
        if current_resumes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.success(f"Berhasil mengunggah {len(current_resumes)} resume")
            with col2:
                if st.button("Hapus Semua Resume", key="clear_compare_resume"):
                    st.session_state.uploaded_resumes["Compare Multiple Candidates"] = []
                    st.session_state.upload_errors = []
                    st.rerun()
            
        if st.session_state.upload_errors:
            st.warning("Beberapa file tidak dapat diproses:")
            for error in st.session_state.upload_errors:
                st.error(error)
        
        if current_resumes and len(current_resumes) >= 2:
            inputs = {
                "resume_data": current_resumes,
                "domain": st.session_state.selected_domain
            }
            
    elif use_case == "Compare with Scoring":
        st.header("📈 Compare with Scoring")
        
        # Tampilkan hasil sebelumnya jika ada
        if st.session_state.get('show_scoring_results') and st.session_state.get('last_scoring_results'):
            st.subheader("Hasil Perbandingan Skor Terakhir")
            display_scoring_results(st.session_state.last_scoring_results)
            st.markdown("---")
        
        if st.session_state.last_jd_text:
            st.info("Penilaian kandidat berdasarkan deskripsi pekerjaan terakhir yang diunggah")
            with st.expander("Lihat Deskripsi Pekerjaan"):
                st.write(st.session_state.last_jd_text)
        else:
            st.warning("Belum ada deskripsi pekerjaan yang diunggah. Silakan unggah JD melalui use case 'Candidate Search by Job Description' terlebih dahulu.")
        
        standardizer = ResumeStandardizer(domain=st.session_state.selected_domain)
        
        with st.expander("⚙️ Konfigurasi Kriteria Penilaian", expanded=True):
            st.write("Konfigurasi bobot untuk setiap kriteria penilaian (1-10):")
            domain_criteria = standardizer.get_domain_specific_criteria()
            criteria = {}
            cols = st.columns(3)
            for i, criterion in enumerate(domain_criteria.keys()):
                with cols[i % 3]:
                    criteria[criterion] = st.slider(
                        criterion,
                        1, 10,
                        domain_criteria[criterion],
                        key=f"score_criteria_{criterion}_{st.session_state.selected_domain}"
                    )
        
        upload_option = st.radio(
            "Opsi Upload:", 
            ["Multiple Files", "Folder (ZIP)"],
            key="score_upload_option"
        )
        
        current_resumes = st.session_state.uploaded_resumes.get("Compare with Scoring", [])
        
        if upload_option == "Multiple Files":
            resume_files = st.file_uploader(
                "Upload Resume (PDF/DOCX)", 
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="score_multiple_files"
            )
            if resume_files:
                valid_resumes = []
                error_messages = []
                for file in resume_files:
                    try:
                        text, error = parse_resume(file, file.name)
                        if text:
                            valid_resumes.append((text, file.name))
                            from core.retriever import add_resume_to_vector_store
                            add_resume_to_vector_store(text, file.name)
                        if error:
                            error_messages.append(error)
                            logger.error(f"Resume parsing error for {file.name}: {error}")
                    except Exception as e:
                        error_messages.append(f"Failed to parse {file.name}: {str(e)}")
                        logger.error(f"Resume parsing exception for {file.name}: {str(e)}")
                
                st.session_state.uploaded_resumes["Compare with Scoring"] = valid_resumes
                st.session_state.upload_errors = error_messages
                current_resumes = valid_resumes  # Update current_resumes immediately
        
        else:
            uploaded_folder = st.file_uploader(
                "Upload Folder (sebagai ZIP)", 
                type=["zip"],
                accept_multiple_files=False,
                key="score_folder_upload"
            )
            if uploaded_folder:
                try:
                    valid_resumes, error_messages = parse_uploaded_folder(uploaded_folder)
                    for text in valid_resumes:
                        from core.retriever import add_resume_to_vector_store
                        add_resume_to_vector_store(text, "folder_uploaded_resume")
                    st.session_state.uploaded_resumes["Compare with Scoring"] = [(text, "folder_uploaded_resume") for text in valid_resumes]
                    st.session_state.upload_errors = error_messages
                    current_resumes = [(text, "folder_uploaded_resume") for text in valid_resumes]  # Update current_resumes immediately
                    for error in error_messages:
                        logger.error(f"Folder parsing error: {error}")
                except Exception as e:
                    st.session_state.upload_errors = [f"Failed to parse folder: {str(e)}"]
                    logger.error(f"Folder parsing exception: {str(e)}")
        
        if st.session_state.upload_errors:
            st.warning("Beberapa file tidak dapat diproses:")
            for error in st.session_state.upload_errors:
                st.error(error)
        
        if current_resumes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.success(f"Berhasil mengunggah {len(current_resumes)} resume")
            with col2:
                if st.button("Hapus Semua Resume", key="clear_score_resume"):
                    st.session_state.uploaded_resumes["Compare with Scoring"] = []
                    st.session_state.upload_errors = []
                    st.session_state.show_scoring_results = False
                    st.session_state.last_scoring_results = None
                    st.session_state.processed_flag = False
                    st.rerun()
        
        if current_resumes and len(current_resumes) >= 2:
            candidate_names = []
            for i, (resume_text, filename) in enumerate(current_resumes):
                try:
                    name = name_extractor.extract_name_from_resume(resume_text, filename)
                    candidate_names.append(name or f"Kandidat {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting name for resume {i+1}: {str(e)}")
                    candidate_names.append(f"Kandidat {i+1}")
            
            st.session_state.candidate_names = candidate_names
            
            inputs = {
                "resume_data": current_resumes,
                "criteria": criteria,
                "domain": st.session_state.selected_domain
            }
            
            # Proses langsung setelah resume diunggah, hanya jika belum diproses
            if not st.session_state.processed_flag:
                st.session_state.processed_flag = True  # Set flag untuk mencegah pemrosesan ganda
                return use_case, inputs, question  # Kembalikan inputs untuk diproses di main.py
        else:
            st.session_state.processed_flag = False  # Reset flag jika tidak ada resume
        
    return use_case, inputs, question
