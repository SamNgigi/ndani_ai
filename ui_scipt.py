import streamlit as st
import numpy as np
import pandas as pd

def main():
    # Configure the page
    st.set_page_config(
        page_title="Resume Optimizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state variables
    if 'processing_state' not in st.session_state:
        st.session_state.processing_state = None
    if 'result' not in st.session_state:
        st.session_state.result = None

    st.title("Resume Optimizer")

    # Get configuration from sidebar
    config = render_sidebar()

    # Handle file uploads
    resume_file, job_desc_file = render_file_upload()

    # Process files when both are uploaded
    if resume_file and job_desc_file:
        if st.button("🚀 Optimize Resume"):
            with st.spinner("Processing..."):
                # Here, you would process the files and obtain the result
                # For now, we use dummy data
                result = process_files_dummy(resume_file, job_desc_file, config)
                if result:
                    # Store results in session state
                    st.session_state.processing_state = "completed"
                    st.session_state.result = result

                    # Display results
                    render_results(result)
                else:
                    st.error("Error processing files. Please try again.")

    # Display results if already processed
    elif st.session_state.processing_state == "completed":
        render_results(st.session_state.result)


def render_sidebar():
    """
    Render sidebar with configuration options
    Returns:
        Dictionary containing user configuration
    """
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        st.subheader("Model Selection")
        model_options = {
            "Default (Llama2)": "llama2",
            "Custom Dolphin": "dolphin",
            "Mistral": "mistral"
        }
        selected_model = st.selectbox(
            "Select LLM Model",
            options=list(model_options.keys())
        )

        # Temperature/Creativity setting
        st.subheader("Optimization Style")
        temp_setting = st.select_slider(
            "Select optimization approach",
            options=["Precise", "Balanced", "Creative"],
            value="Balanced",
            help="Controls how creative the AI can be with modifications:\n" +
                 "- Precise: Minimal changes, focusing on relevant content\n" +
                 "- Balanced: Moderate modifications for better matching\n" +
                 "- Creative: More substantial rewording while maintaining truth"
        )

        # Section selection
        st.subheader("Sections to Modify")
        sections = {
            "Professional Summary": "summary",
            "Work Experience": "experience",
            "Skills": "skills",
            "Education": "education",
            "Additional Information": "other"
        }
        selected_sections = []
        for display_name, section_id in sections.items():
            if st.checkbox(display_name, value=True):
                selected_sections.append(section_id)

        # Industry selection
        st.subheader("Target Industry")
        industry = st.selectbox(
            "Select target industry",
            options=["Technology", "Finance", "Agriculture", "General"],
            help="Helps optimize content for industry-specific expectations"
        )

        return {
            "model": model_options[selected_model],
            "temperature": temp_setting.lower(),
            "sections": selected_sections,
            "industry": industry.lower()
        }


def render_file_upload():
    """Render file upload section"""
    st.header("Upload Documents")

    col1, col2 = st.columns(2)

    with col1:
        resume_file = st.file_uploader(
            "Upload Your Resume",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )

        if resume_file:
            st.success("Resume uploaded successfully!")

    with col2:
        job_desc_file = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )

        if job_desc_file:
            st.success("Job description uploaded successfully!")

    return resume_file, job_desc_file


def process_files_dummy(resume_file, job_desc_file, config):
    """
    Dummy function to simulate file processing
    """
    # Simulate processing time
    import time
    time.sleep(2)

    # Create dummy result
    result = {
        'original_content': {
            'summary': 'Experienced software engineer with a passion for developing innovative programs.',
            'experience': 'Worked at XYZ Corp for 5 years as a backend developer.',
            'skills': 'Python, Java, SQL',
            'education': 'B.Sc. in Computer Science from ABC University',
            'other': 'Volunteer at local community center.'
        },
        'modified_content': {
            'summary': 'Skilled software engineer specializing in backend development, eager to contribute to cutting-edge projects.',
            'experience': 'Backend developer at XYZ Corp with 5 years of experience in optimizing database performance.',
            'skills': 'Python, Java, SQL, NoSQL, Cloud Computing',
            'education': 'B.Sc. in Computer Science from ABC University, with honors',
            'other': 'Community center volunteer focused on teaching coding to youth.'
        },
        'modifications': {
            'summary': {'percentage': 0.3, 'changes': [{'original': 'Experienced software engineer', 'new': 'Skilled software engineer specializing in backend development'}]},
            'experience': {'percentage': 0.2, 'changes': [{'original': 'Worked at XYZ Corp for 5 years as a backend developer.', 'new': 'Backend developer at XYZ Corp with 5 years of experience in optimizing database performance.'}]},
            'skills': {'percentage': 0.4, 'changes': [{'original': 'Python, Java, SQL', 'new': 'Python, Java, SQL, NoSQL, Cloud Computing'}]},
            'education': {'percentage': 0.1, 'changes': [{'original': 'B.Sc. in Computer Science from ABC University', 'new': 'B.Sc. in Computer Science from ABC University, with honors'}]},
            'other': {'percentage': 0.2, 'changes': [{'original': 'Volunteer at local community center.', 'new': 'Community center volunteer focused on teaching coding to youth.'}]}
        },
        'ats_analysis': {
            'overall_score': 85.0,
            'detailed_scores': {
                'relevance': {'score': 90.0, 'details': ['Your experience matches the job requirements.'], 'suggestions': []},
                'keywords': {'score': 80.0, 'details': ['Good use of industry-specific keywords.'], 'suggestions': ['Consider adding more cloud computing terms.']},
                'formatting': {'score': 85.0, 'details': ['Formatting is ATS-friendly.'], 'suggestions': []},
                'skills_match': {'score': 80.0, 'details': ['Most required skills are listed.'], 'suggestions': ['Highlight leadership skills.']}
            },
            'suggestions': [
                'Add more cloud computing terms to your skills section.',
                'Highlight any leadership or team management experience.'
            ]
        }
    }
    return result


def render_results(result):
    """
    Render processing results with visualizations
    """
    # Display ATS Score
    _render_ats_score(result['ats_analysis'])

    # Display Content Comparison
    st.header("Resume Optimization Results")
    _render_content_comparison(
        result['original_content'],
        result['modified_content'],
        result['modifications']
    )

    # Display Detailed Analysis (Removed outer expander)
    st.header("📊 Detailed Analysis")
    _render_detailed_analysis(result['ats_analysis'])

    # Display Suggestions (Removed outer expander)
    st.header("💡 Improvement Suggestions")
    _render_suggestions(result['ats_analysis'])

    # Download Options
    _render_download_options(result)


def _render_ats_score(ats_analysis):
    """Render ATS compatibility score with gauge chart"""
    st.header("ATS Compatibility Score")

    score = ats_analysis['overall_score']

    # Create three columns for score display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        # Display numeric score
        st.metric(
            "Score",
            f"{score:.1f}%",
            delta=None,
            help="Overall ATS compatibility score"
        )

    with col2:
        # Create gauge chart using Plotly
        import plotly.graph_objects as go

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': 'red'},
                    {'range': [60, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'green'}
                ],
            }
        ))
        fig.update_layout(width=300, height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig)

    with col3:
        # Display rating text
        rating = (
            "Excellent" if score >= 80
            else "Good" if score >= 60
            else "Needs Improvement"
        )
        st.info(f"Rating: {rating}")


def _render_content_comparison(original, modified, modifications):
    """Render side-by-side comparison of original and modified content"""
    st.subheader("Content Comparison")

    for section in original.keys():
        if section in modified:
            with st.expander(f"📄 {section.title()}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Content**")
                    st.text_area(
                        "Original",
                        value=original[section],
                        height=200,
                        key=f"original_{section}",
                        disabled=True
                    )

                with col2:
                    st.markdown("**Optimized Content**")
                    st.text_area(
                        "Modified",
                        value=modified[section],
                        height=200,
                        key=f"modified_{section}",
                        disabled=True
                    )

                # Display modification details
                if section in modifications:
                    mod_info = modifications[section]
                    st.info(
                        f"Modification Percentage: "
                        f"{mod_info['percentage']*100:.1f}%"
                    )

                    if mod_info['changes']:
                        # Instead of nesting an expander, use a button to show/hide changes
                        show_changes = st.checkbox(f"Show Changes in {section.title()}", key=f"show_changes_{section}")
                        if show_changes:
                            for change in mod_info['changes']:
                                st.write(
                                    f"- Changed: '{change['original']}' "
                                    f"→ '{change['new']}'"
                                )


def _render_detailed_analysis(ats_analysis):
    """Render detailed ATS analysis results"""
    # No change in function structure, inner expanders are now at the top level
    st.subheader("Detailed Analysis")

    # Create radar chart for score components
    scores = ats_analysis['detailed_scores']

    # Prepare data for radar chart
    categories = list(scores.keys())
    values = [scores[cat]['score'] for cat in categories]

    # Create radar chart using Plotly
    import plotly.express as px

    df = pd.DataFrame({
        'Category': categories,
        'Score': values
    })
    fig = px.line_polar(df, r='Score', theta='Category', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(width=400, height=400, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig)

    # Display detailed breakdowns
    for category, details in scores.items():
        with st.expander(f"📊 {category.title()} Details"):
            st.metric(
                "Score",
                f"{details['score']:.1f}%"
            )

            if details['details']:
                st.markdown("**Analysis Details:**")
                for detail in details['details']:
                    st.write(f"- {detail}")


def _render_suggestions(ats_analysis):
    """Render improvement suggestions"""
    # No change in function structure, inner expanders are now at the top level
    st.subheader("Improvement Suggestions")

    suggestions = ats_analysis.get('suggestions', [])
    scores = ats_analysis['detailed_scores']

    # Group suggestions by category
    categorized_suggestions = {}
    for category, details in scores.items():
        if details['suggestions']:
            categorized_suggestions[category] = details['suggestions']

    # Display suggestions with priority indicators
    for category, category_suggestions in categorized_suggestions.items():
        score = scores[category]['score']
        priority = (
            "🔴 High Priority" if score < 60
            else "🟡 Medium Priority" if score < 80
            else "🟢 Low Priority"
        )

        with st.expander(f"{priority} - {category.title()}"):
            for suggestion in category_suggestions:
                st.write(f"- {suggestion}")

    # Also display general suggestions
    if suggestions:
        with st.expander("General Suggestions"):
            for suggestion in suggestions:
                st.write(f"- {suggestion}")


def _render_download_options(result):
    """Render download options for results"""
    st.header("Download Options")

    col1, col2 = st.columns(2)

    with col1:
        # Download optimized resume
        optimized_content = "\n\n".join(result['modified_content'].values())
        st.download_button(
            label="📥 Download Optimized Resume",
            data=optimized_content,
            file_name="optimized_resume.txt",
            mime="text/plain"
        )

    with col2:
        # Download full analysis report
        report = _generate_analysis_report(result)
        st.download_button(
            label="📊 Download Analysis Report",
            data=report,
            file_name="ats_analysis_report.txt",
            mime="text/plain"
        )


def _generate_analysis_report(result):
    """Generate detailed analysis report"""
    report = []

    # Add header
    report.append("ATS OPTIMIZATION ANALYSIS REPORT")
    report.append("=" * 30 + "\n")

    # Add overall score
    report.append(f"Overall ATS Compatibility Score: {result['ats_analysis']['overall_score']:.1f}%\n")

    # Add detailed scores
    report.append("DETAILED SCORES")
    report.append("-" * 20)
    for category, details in result['ats_analysis']['detailed_scores'].items():
        report.append(f"\n{category.title()}")
        report.append(f"Score: {details['score']:.1f}%")

        if details['details']:
            report.append("Details:")
            for detail in details['details']:
                report.append(f"- {detail}")

        if details['suggestions']:
            report.append("Suggestions:")
            for suggestion in details['suggestions']:
                report.append(f"- {suggestion}")

    return "\n".join(report)


if __name__ == "__main__":
    main()

