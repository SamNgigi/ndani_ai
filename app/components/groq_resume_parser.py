import os
import json
import asyncio
import logging
import pypdf
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

from groq import AsyncGroq
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class ResumeParser:
    """
    Uses Groq Cloud LLM inference to parse different resume document formats
    into a JSON format of the different simplified sections
    """

    def __init__(self, api_key:str, parsing_prompt: Union[str, Path]):
        self.parsing_prompt: Path = Path(parsing_prompt)
        self.client = AsyncGroq(api_key = api_key)
        self.supported_formats = {
            '.pdf': self._read_pdf,
        }
    
    def _read_pdf(self, file_path:Path) -> str:
        """
        Read PDF documents
 
        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text from the PDF
        """
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

    def _load_prompt(self) -> str:
        if not self.parsing_prompt.exists():
            logger.error(f"❌ Parsing prompt file not found at {self.parsing_prompt}")
            raise FileNotFoundError(f"Parsing prompt file not found at {self.parsing_prompt}")
        try:
            with open(self.parsing_prompt, 'r', encoding='utf-8') as prompt:
                return prompt.read()
        except Exception as e:
            logger.error(f"❌ Error reading prompt file: {str(e)}")
            raise

    async def groq_parse(self, file_path:Union[str, Path]) -> dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")

        _prompt = self._load_prompt()
        pdf_text = self.supported_formats[file_ext](file_path)
        parsing_prompt = _prompt.format(pdf_text = pdf_text)

        mxTokens = 11000
        try:
            completion = await self.client.chat.completions.create(
                model = "mixtral-8x7b-32768",
                messages = [
                    {
                        "role":"system",
                        "content":"Respond with valid JSON only. Do not include other text or markdown formatting"
                    },
                    {
                        "role": "user",
                        "content": parsing_prompt 
                    },
                ],
                temperature = 0,
                max_tokens = mxTokens,
                top_p = 1,
                seed = 100,
                stream = False,
                response_format = {"type":"json_object"},
                stop = None

            )

        except Exception as e:
            logger.error(f"❌ Error during API call: {str(e)}")
            raise
        if completion.choices and completion.choices[0].message.content:
            try:
                result = json.loads(completion.choices[0].message.content)
                logger.info(f"✅ Successfully return resume json data")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error decoding JSON response: {str(e)}")
                raise
            return result
        else:
            logger.error("❌ Did not receive a valid response from the API.")
            raise ValueError("❌ Did not receive a valid response from the API.")
            
    def _write_json(self, result: dict, resume_name:str, model_name:str, output_dir: Optional[Union[str, Path]]):
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "output"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{resume_name.lower()}_{model_name}.json"
        
        try:
            with output_path.open('w', encoding = 'utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Error saving JSON file: {str(e)}")
            raise

