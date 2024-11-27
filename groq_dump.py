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



async def process_resume(resume:str, jd:str, user_prompt:str, sys_prompt):
    project_root = Path(__file__).parent
    output_dir = project_root / "output"
    resume_path = output_dir/f"{resume}"
    jd_path = output_dir/f"{jd}"
    prompts_dir = project_root / "prompts"
    user_prompt_path = prompts_dir / f"{user_prompt}"
    sys_prompt_path = prompts_dir / f"{sys_prompt}"
    with resume_path.open('r', encoding="utf-8") as resume_json:
        resume_data = json.load(resume_json)
    required_resume_sections = ["summary", "skill_or_tech_stack", "experience"]
    resume_sections = {k:v for k,v  in resume_data.items() if k in required_resume_sections}

    with jd_path.open('r', encoding="utf-8") as jd_json:
        jd_data = json.load(jd_json)
    required_jd_sections = ["job_title", "job_description", "qualifications"]
    jd_sections = {k:v for k,v  in jd_data.items() if k in required_jd_sections}


    with user_prompt_path.open('r', encoding='utf-8') as p:
        user_prompt = p.read()
    with sys_prompt_path.open('r', encoding='utf-8') as p:
        sys_prompt = p.read()


    optimize_resume_prompt = user_prompt.format(resume_json=resume_sections, job_json=jd_sections)

    await _make_request(optimize_resume_prompt, sys_prompt, "optimized_resume")

async def _make_request(user_prompt:str, system_prompt:str, outfile_name:str):
    mxTokens = 11000
    completions = await client.chat.completions.create(
        model = "mixtral-8x7b-32768",
        messages = [
            {
                "role":"system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.5,
        max_tokens=mxTokens,
        top_p=1,
        seed=100,
        stream=False,
        response_format={"type":"json_object"},
        stop=None
    )
    project_root = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = project_root / "output" / f"{outfile_name}_{timestamp}_{mxTokens}tokens.json"
    output_path.parent.mkdir(exist_ok=True)
    if completions.choices[0].message.content:
        result = json.loads(completions.choices[0].message.content)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Results save to {output_path}")
    else:
        logger.error(f"❌ Did not get response")

jd_file = "salesforce_jd_hseft4sm_mixtral_20241125_140935.json"
resume_file = "sn_resume_46uiad2r_mixtral_20241125_140904.json"
user_prompt_file = "claude_optimize_resume.txt"
sys_prompt_file = "sys_resume_optimizer.txt"
# asyncio.run(parse_resume(pdf_text))
asyncio.run(process_resume(resume_file, jd_file, user_prompt_file, sys_prompt_file))
