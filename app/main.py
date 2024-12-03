import tempfile
import asyncio
import json
import os
import pprint as pp
import streamlit as st
from pathlib import Path
from typing import Dict, Union
from dotenv import load_dotenv


from components.llm_interface import LlmInterface
from components.resume_processor import ResumeProcessor

load_dotenv()

class ResumeOptimizerApp:
    """Main streamlit application for resume optimization"""

    def __init__(self):
        """Initialize application components"""
        self.setup_streamlit()
        self.initialize_components()
    
    def setup_streamlit(self):
        """Configure Initial Streamlit page settings and session state"""
        # Configure page
        st.set_page_config(
            page_title = "Wera AI Resume Optimizer",
            page_icon="📝",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Initialize session state variables
        if 'processing_state' not in st.session_state:
            st.session_state.processing_state = None
        if 'result' not in st.session_state:
            st.session_state.result = None
        if 'original_content' not in st.session_state:
            st.session_state.original_content = None



    def initialize_components(self):
        groq_key = os.environ.get("GROQ_API_KEY")
        project_root = Path(__file__).parent.parent
        prompt_file = project_root/ "prompts" / "parse_resume_prompt.txt"
        if not groq_key:
            raise EnvironmentError("GROQ_API_KEY not set in enviroment variables")
        self.llm = LlmInterface(api_key=groq_key)
        self.processor = ResumeProcessor(self.llm)

    def render_sidebar(self) -> Dict:
        """
        Render sidebar with configuration options

        Returns:
            Dictionary containing user configuration
        """
        try:
            with st.sidebar:
              
                model_options = {"mixtral": "mixtral", "llama3.1": "llama3.2"}
                
                selected_model = st.selectbox(
                    "Select LLM Model",
                    options = list(model_options.keys())
                )
                return {
                    "model": model_options[selected_model]
                }

        except Exception as e:
            st.error(f"Error rendering sidebar: {str(e)}")
            raise

    def render_file_upload(self):
        """Render file upload section"""
        st.header("Upload Documents")

        col1, col2 = st.columns(2)
        with col1:
            resume_file = st.file_uploader(
                "Upload Your Resume",
                type = ["pdf","docx","txt"],
                help="Supported formats: PDF, DOCX, TXT"
            )
            if resume_file:
                st.success("Resume uploaded successfully!")

        with col2:
            job_desc_file = st.file_uploader(
                "Upload Job Description",
                type = ["pdf","docx","txt"],
                help="Supported formats: PDF, DOCX, TXT"
            )
            if job_desc_file:
                st.success("Job Description uploaded successfully!")

        return resume_file, job_desc_file



       
    async def process_files(self, resume_file, jd_file, save_json:bool=True) -> Dict:
        """
        Process uploaded files

        Args:
            resume_file: Uploaded resume file
            TODO:
                job_desc_file: Uploaded job description file
                config: Processing configurations
        Returns:
            Dictionary contaning processing results
        """
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(resume_file.name).suffix, 
                prefix=f"{Path(resume_file.name).stem}_"
            ) as temp_resume:
                temp_resume.write(resume_file.getbuffer())
                resume_path = temp_resume.name

            resume_json = await self.processor._parse_resume(resume_path, save_json)
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(jd_file.name).suffix, 
                prefix=f"{Path(jd_file.name).stem}_"
            ) as temp_jd:
                temp_jd.write(jd_file.getbuffer())
                jd_path = temp_jd.name

            jd_json = await self.processor._parse_jd(jd_path, save_json)
            if not self.processor.resume_json_path and self.processor.jd_json_path:
                return {}
            
            return await self.processor._optimize_resume(
                self.processor.resume_json_path,
                self.processor.jd_json_path,
                "balanced",
                save_json
            )
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            raise
    
    def render_results(self, results:Dict):
        """Render processing results with visualizations"""
        # Disiplay Content Comparison
        st.header("Resume Optimizaton Results")
        self._render_content_comparison(results["original_content"], results["modified_content"])

    def _render_ats_score(self):
        pass

    def _render_content_comparison(
        self, 
        original: Dict[str, str], 
        modified: Dict[str, dict], 
    ):
        """Render side-by-side commparison of original and modified content"""
        st.subheader("Content Comparison")
        
        for section in modified.keys():
            with st.expander(f"📝 {section.title()}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Content**")
                    st.text_area(
                        "Original",
                        value= original[section],
                        height = 200,
                        key = f"original_{section}",
                        disabled=True
                    )
                with col2:
                    st.markdown("**Modified Content**")
                    st.text_area(
                        "Modified",
                        value=modified[section]["updated_content"],
                        height = 200,
                        key = f"modified_{section}_1",
                        disabled=True
                    )

                st.info(f"Comments: {modified[section]['improvements']}")

        

    def _render_detailed_analysis(self):
        pass

    def _render_suggestions(self):
        pass

    def _render_download_options(self):
        pass

    def _generate_analysis_report(self):
        pass

    async def main(self):
        st.title("Wera AI Resume Optimizer")

        # Get configuration from sidebar
        config = self.render_sidebar()

        # Handle file uploads
        resume_file, job_desc_file = self.render_file_upload()

        if resume_file:
            if st.button("🚀 Optimize Resume"):
                with st.spinner("Processing..."):
                    # Process files. Uses dummy/mock data for now
                    result =  await self.process_files(resume_file, job_desc_file)
                    if result:
                        # Store result in session state
                        st.session_state.processing_state = "completed"
                        st.session_state.result = result
                        
                        # Display result
                        self.render_results(result)
                    else:
                        st.error("Error processing provided files. Result was empty. Please try again.")
        # Display results if already proceed
        elif st.session_state.processing_state == "completed":
            self.render_results(st.session_state.result)
        

if __name__ == "__main__":
    try:
        app = ResumeOptimizerApp()
        asyncio.run(app.main())
    except KeyboardInterrupt:
        st.warning("\n⚠️  Verification interrupted by user")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        raise

    
