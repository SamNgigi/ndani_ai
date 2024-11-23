import os
from datetime import datetime
import json
import asyncio
import pypdf
import logging
import pprint as pp
from pathlib import Path
from typing import Union
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_key = os.environ.get("GROQ_API_KEY")

def _read_pdf(file_path:Union[str, Path]) -> str:
    try:
        text = []
        with open(file_path, 'rb') as file:
            pdf  = pypdf.PdfReader(file)
            for pages in pdf.pages:
                text.append(pages.extract_text())
            return '\n'.join(text)
    except Exception as e:
        logger.error(f"❌ Error parsing pdf: {str(e)}")
        raise


pdf_text = _read_pdf('data/sn_resume.pdf')

# pp.pprint(pdf_text)


client = AsyncGroq(api_key = groq_key)

async def parse_resume(pdf_text:str):
    # Read the parse_resume_prompt
    with open('prompts/parse_resume_prompt.txt', 'r', encoding='utf-8') as prompt:
        _prompt = prompt.read()
    parsing_prompt = _prompt.format(pdf_text = pdf_text)

    mxTokens = 11000
    completion = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": "Respond with valid JSON only. Do not include any other text or markdown formatting."
            },
            {
                "role": "user",
                "content": parsing_prompt
            }
        ],
        temperature=0,
        max_tokens=mxTokens,
        top_p=1,
        seed=100,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )

    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "output" / f"parsed_resume_{timestamp}_{mxTokens}tokens.json"
    output_path.parent.mkdir(exist_ok=True)
    if completion.choices[0].message.content:
        result = json.loads(completion.choices[0].message.content)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Results saved to {output_path}")
    else:
        logger.error("❌ Did not get response")
    

async def parse_job_description(): 
    _text = _read_pdf("./data/salesforce_jd.pdf")

    with open("./prompts/parse_jd_prompt.txt", 'r', encoding='utf-8') as prompt:
        _prompt  = prompt.read()
    parsing_prompt = _prompt.format(_text=_text)
    mxTokens = 11000
    completions = await client.chat.completions.create(
        model = "mixtral-8x7b-32768",
        messages = [
            {
                "role":"system",
                "content":"Respond with valid JSON only. Do not include any other text or markdow formatting"
            },
            {
                "role": "user",
                "content": parsing_prompt
            }
        ],
        temperature=0,
        max_tokens=mxTokens,
        top_p=1,
        seed=100,
        stream=False,
        response_format={"type":"json_object"},
        stop=None
    )
    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "output" / f"jd_parsed_{timestamp}_{mxTokens}tokens.json"
    output_path.parent.mkdir(exist_ok=True)
    if completions.choices[0].message.content:
        result = json.loads(completions.choices[0].message.content)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Results save to {output_path}")
    else:
        logger.error(f"❌ Did not get response")



# asyncio.run(parse_resume(pdf_text))
asyncio.run(parse_job_description())
