# Wera Ai Resume Optimizer (W.I.P)

This is an MVP for an intelligent resume optimization AI agent powered by advanced open source language models that helps job seekers tailor their resumes to specific job descriptions and generate appropriate cover letters.

Optimization and additional features coming soon

## 🌟 Features

- **Resume Analysis**: Automatically extracts and structures information from PDF resumes
- **Job Description Parsing**: Analyzes job descriptions to identify key requirements and qualifications
- **Smart Optimization**: Uses AI to tailor resume content to match job requirements
- **Interactive UI**: Clean, user-friendly interface built with Streamlit
- **Real-time Processing**: Instant feedback and optimization suggestions
- **Content Comparison**: Side-by-side view of original and optimized content
- **Multiple Model Support**: Compatible with various LLM models (Mixtral, LLaMA)

**NOTE**: Currently total inference may take up to 2 minutes because of the multi-step inferencing required. Also one of the models was decommissioned, updating model soon

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.x
- **LLM Integration**: Groq Cloud API
- **PDF Processing**: PyPDF
- **Async Support**: asyncio
- **Error Handling**: tenacity, backoff
- **Environment Management**: python-dotenv

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Internet connection for API access

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ndani_ai.git
cd ndani_ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add:
```
GROQ_API_KEY=your_api_key_here
```

## 🚀 Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Navigate to the provided local URL (typically `http://localhost:8501`)

3. Upload your resume and job description (supported formats: PDF, DOCX, TXT)

4. Select your preferred LLM model and optimization settings

5. Click "Optimize Resume" to process your documents

## 📁 Core Project Structure

```
ndani-resume-optimizer/
├── app/
│   ├── __init__.py                # Package initializer
│   ├── main.py                    # Main Streamlit application entry point
│   └── components/                # Application components
│       ├── __init__.py
│       ├── llm_interface.py       # LLM interaction handling
│       ├── resume_processor.py    # Resume optimization logic
├── prompts/                # LLM prompt templates
├── output/                 # Generated JSON outputs
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 🔍 Key Components

### LLM Interface
- Handles communication with Groq Cloud API
- Supports multiple models with different configurations
- Implements retry logic and error handling
- Configurable temperature settings for different optimization styles

### Resume Processor
- Parses PDF documents into structured format
- Extracts relevant sections from resumes and job descriptions
- Implements optimization logic using LLM
- Manages file I/O and JSON processing

### Streamlit Interface
- Provides intuitive file upload functionality
- Displays real-time processing status
- Shows side-by-side comparisons of original and optimized content
- Offers model selection and configuration options

## ⚠️ Error Handling

The application implements robust error handling through:
- Exponential backoff for API calls
- Multiple retry attempts for failed operations
- Comprehensive logging for debugging
- User-friendly error messages in the UI

## 🔐 Security

- API keys are managed through environment variables
- Temporary file handling for uploaded documents
- Input validation and sanitization
- Secure file processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Groq Cloud](https://groq.com/)
- PDF processing with [PyPDF](https://pypdf.readthedocs.io/)

## 📞 Contact

For questions and support, please open an issue in the GitHub repository.
