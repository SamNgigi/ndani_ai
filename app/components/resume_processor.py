import os
import json
import asyncio
import logging
import pypdf
import pprint as pp
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from difflib import SequenceMatcher
from datetime import datetime
from dataclasses import dataclass

from components.llm_interface import LlmInterface
from verify_ollama import get_error_details

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SectionModification:
    """Tracks modifications made to a resume section"""
    original_content: str
    modified_content: str
    modification_percentage: float
    changes: List[Dict[str, str]]

class ResumeProcessor:
    """Handles resume processing and optimization"""
    def __init__(self, llm: LlmInterface):
        """
        Initialize the resume processor
        """
        self.llm = llm
        self.project_root = Path(__file__).parent.parent.parent
        self.supported_formats = {
            '.pdf': self._read_pdf
        }


    async def _parse_resume(self, file_path:Union[str, Path], save_json:bool=False) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"❌ Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported file format: {file_ext}")

        try:
            # Read pdf
            pdf_text = self.supported_formats[file_ext](file_path)
            # Load prompt
            system_prompt = self._load_prompt('sys_valid_json')
            user_prompt = self._load_prompt('parse_resume')
            # Format prompt with pdf_text
            parse_resume_prompt = user_prompt.format(pdf_text = pdf_text)
            resume_json = await self.llm.generate(parse_resume_prompt, system_prompt)
            if not resume_json:
                logger.error(f"❌ Resume Processor: `result` was not returned")
                return {}
            if save_json:
                self._write_json(resume_json, file_path.stem, self.llm.current_model)
            return resume_json
        except Exception as e:
            logger.error(f"❌ ResumeProcessor:: Error parsing resume: {str(e)}")
            raise

    def _read_pdf(self, file_path: Path)->str:
        try:
            text = []
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                for page in pdf.pages:
                    extracted_text = page.extract_text()
                    if extracted_text: text.append(extracted_text)
            if not text:
                logger.error("❌ No text extracted from the PDF")
                raise ValueError("No text extracted from the PDF")

            return '\n'.join(text)
        except Exception as e:
            logger.error(f"❌ Error reading PDF: {str(e)}")
            raise

    def _load_prompt(self, prompt_name:str)->str:
        """Load prompt template from predifiend prompts"""
        prompt_file = self.project_root / "prompts" / f"{prompt_name}.txt" 
        if not prompt_file.exists():
            logger.error(f"❌ {prompt_name} file not found at {prompt_file}")
            raise FileNotFoundError(f"{prompt_name} file not found at {prompt_file}")
        try:
            with open(prompt_file, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        except Exception as e:
            logger.error(f"❌ Error reading prompt file: {str(e)}")
            raise

    def _write_json(self, result:dict, resume_name:str, model_name:str):
        output_dir = self.project_root / "output"
        output_dir.mkdir(exist_ok = True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{resume_name.lower()}_{model_name}_{timestamp}.json"

        try:
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Error saving JSON file: {str(e)}")



