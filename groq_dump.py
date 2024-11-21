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

def _parse_pdf(file_path:Union[str, Path]) -> str:
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


pdf_text = _parse_pdf('data/sn_resume.pdf')

# pp.pprint(pdf_text)


client = AsyncGroq(api_key = groq_key)

async def parse_resume(pdf_text:str):
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
                "content": """Extract the following CV into a JSON dictionary with the exact structure provided below. Each key must contain the full text of the corresponding section from the CV.

**Expected JSON Structure**:

{{
  "contact_details": "Full text of the contact details section",
  "bio_summary": "Full text of the bio summary section",
  "skill_or_tech_stack": "Full text of the skills or tech stack section",
  "experience": "Full text of the experience section",
  "education": "Full text of the education section",
  "licences_certifications": "Full text of the licences and certifications section",
  "interest_or_milestones": "Full text of the interests or milestones section",
  "references": "Full text of the references section"
}}

**CV**:

{pdf_text}

**Instructions**:

- **Do not include any hierarchical or nested structures**.
- **Do not break down sections into subfields**.
- **Do not provide additional formatting or summaries**.
- **Each key must contain the full, unaltered text of the corresponding section from the CV**.
- **Return only the JSON dictionary with the extracted information, matching the specified structure exactly**.

Return only the JSON dictionary with the extracted information.""".format(pdf_text=pdf_text)
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
    


asyncio.run(parse_resume(pdf_text))
