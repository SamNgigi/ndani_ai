import logging
import json
import asyncio
from typing import Dict, List, Optional, Union, Tuple
from difflib import SequenceMatcher
from datetime import datetime

from verify_ollama import get_error_details
from components.document_processor import DocumentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeProcessor:
    """Handles resume processing and optimization"""
    def __init__(self, resume_raw:dict):
        """
        Initialize the resume processor
        """
        self.parser = DocumentParser()
        self.core_resume_content = {
                key: value for key, value in resume_raw if key in ["summary", "skill_or_tech_stack", "experience"]
            }

        self.prompts = {
            'analyze_job': self._load_prompts('analyze_job'),
            'modify_section': self._load_prompts('modify_section'),
            'generate_cover_letter': self._load_prompts('generate_cover_letter'),
        }


    def _load_prompts(self, prompt_name:str)->str:
        """Load prompt template from predifiend prompts"""
        prompts = {
            'analyze_job': '', # TODO: Analyze job prompt
            'modify_section': '', # TODO: Modify section prompt
            'generate_cover_letter': '', # TODO: Generate section_prompt
        }

        return prompts.get(prompt_name, "")
        
    async def _analyze_job_description(self):
        # TODO: 
        pass
    async def process_resume(self):

        """
        Process and optimize resume based on Job Description

        """
        try:
            return {
                "original_content": self.core_resume_content
            }
        except Exception as e:
            logger.error(f"❌ Error in resume processing: {get_error_details(e)}")
            raise
