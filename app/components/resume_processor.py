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
        self.resume_json_path:Path = Path("") 
        self.jd_json_path:Path = Path("") 
        self.optimized_resume_json_path:Path = Path("") 
        self.supported_formats = {
            '.pdf': self._read_pdf
        }


    async def _apply(self):
        pass

    async def _process_resume(self):
        pass

    async def _optimize_resume(self, resume_json:Union[str, Path], jd_json:Union[str, Path], temperature:str, save_json:bool=True) -> dict:
        try:
            resume_data = self._read_json(resume_json)
            jd_data = self._read_json(jd_json)
            resume_required_sections = {k:v for k,v in resume_data.items() if k in ["summary", "skill_or_tech_stack", "experience"]}
            jd_required_sections = {k:v for k,v in jd_data.items() if k in ["job_description", "qualifications"]}
            sys_prompt = self._load_prompt('sys_resume_optimizer')
            user_prompt = self._load_prompt('claude_optimize_resume')
            optimize_resume_prompt = user_prompt.format(resume_json=resume_required_sections, jd_json=jd_required_sections)
            # self.llm.set_model('deepseek')
            # self.llm.set_model('llama-versatile')
            self.llm.set_temperature(temperature)
            optimized_resume_data = await self.llm.generate(optimize_resume_prompt, sys_prompt)
            if not optimized_resume_data:
                logger.error("ResumeProcessor: `optimized_resume_data` is empty")
                return {}
            if save_json:
                self.optimized_resume_json_path = self._write_json(optimized_resume_data, "opt_resume",  self.llm.current_model) 
            return {
            "original_content": resume_data,
            "modified_content": optimized_resume_data
        }
        except Exception as e:
            logger.error(f"❌ ResumeProcessor: _optimize_resume failed: {str(e)}")
            raise


    async def generate_cover_letter(self):
        pass
    
    async def scrape_jobs(self):
        pass

    async def _parse_resume(self, file_path:Union[str, Path], save_json:bool) -> dict:
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
            _text = self.supported_formats[file_ext](file_path)
            # Load prompt
            system_prompt = self._load_prompt('sys_valid_json')
            user_prompt = self._load_prompt('parse_resume')
            # Format prompt with _text
            parse_resume_prompt = user_prompt.format(_text = _text)
            resume_json = await self.llm.generate(parse_resume_prompt, system_prompt)
            if not resume_json:
                logger.error(f"❌ ResumeProcessor: `resume_json` is empty")
                return {}
            logger.info("✅ ResumeProcessor: `resume_json` returned successfully")
            if save_json:
                self.resume_json_path = self._write_json(resume_json, file_path.stem, self.llm.current_model)
            return resume_json
        except Exception as e:
            logger.error(f"❌ ResumeProcessor:: Error parsing resume: {str(e)}")
            raise

    async def _parse_jd(self, file_path:Union[str, Path], save_json:bool=True):
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            logger.error(f"❌ Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported file format: {file_ext}")

        try:
            _text = self.supported_formats[file_ext](file_path)
            system_prompt = self._load_prompt('sys_valid_json')
            user_prompt = self._load_prompt('parse_jd')
            parse_jd_prompt = user_prompt.format(_text=_text)
            jd_json = await self.llm.generate(parse_jd_prompt, system_prompt)
            if not jd_json:
                logger.error("❌ Resume Processor: `jd_json` is empty")
                return {}
            logger.info("✅ ResumeProcessor:`jd_json` returned successfully")
            if save_json:
                self.jd_json_path = self._write_json(jd_json, file_path.stem, self.llm.current_model)
        except Exception as e:
            logger.error(f"❌ Resume Processor:: Error parsing job description: {str(e)}")
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

    def _read_json(self, file_path:Union[str, Path]) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            raise FileNotFoundError(f"❌ File not found: {file_path}")
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            logger.error(f"❌ No data found from {file_path.name}")
            ValueError(f"❌ No data found from {file_path.name}")
        logger.info("✅ Json Data found and read successfully")
        return data


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

    def _write_json(self, result:dict, resume_name:str, model_name:str) -> Path:
        output_dir = self.project_root / "output"
        output_dir.mkdir(exist_ok = True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{resume_name.lower()}_{model_name}_{timestamp}.json"

        try:
            with output_path.open('w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Error saving JSON file: {str(e)}")
            raise



