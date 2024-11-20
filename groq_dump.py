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


pdf_text = _parse_pdf('data/jn_resume.pdf')

# pp.pprint(pdf_text)


client = AsyncGroq(api_key = groq_key)

async def parse_resume(pdf_text:str):
    mxTokens = 10000
    completion = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {
                "role": "system",
                "content": "Respond with valid JSON only. Do not include any other text or markdown formatting."
            },
            {
                "role": "user",
                "content": """Given the following CV text, extract and organize the content into a dictionary where the keys are the section headings and values are the corresponding content. Maintain the hierarchical structure and return only valid JSON. Format the section as follows:
                    
                    ###START OF FORMAT
{{
  "contact_details":{{
    "name":"",
    "email": "",
    "linkedIn":"",
    "github":""
  }},
  "summary": "",

  "skill_or_tech_stack": [
    {{
    "competency":"",
    "tools": ""
    }},
    {{
    "competency": "",
    "tools":""
    }}
  ],
  "experience": [
    {{
      "role": "title and company",
      "period": "dates",
      "Description": "",
      "Tasks": {{
        "": "",
        "": "",
        "": "",
        "": "",
        "": ""
      }}
    }}
  ],
  "education": [
    {{ 
      "school": "" , 
      "course": "",
      "dates": ""
    }}
  ],
  "licences_certifications": [
    "cert 1",
    "cert 2"
  ],
  "personal_interest": {{
    "projects": [
        {{
            "name":"",
            "description":""
        }},
        {{
            "name":"",
            "description":""
        }}
    ],
    "interests": ["interest 1", "interest 2"],
    "milestones": [
        "milestone 1",
        "milestone 2"
    ]
  }}
}}
                    ###END OF FORMAT

                    ### START OF RESUME
                    {pdf_text}
                    ### END OF RESUME
                    Return only a valid JSON dictionary with the extracted information.""".format(pdf_text=pdf_text)
            }
        ],
        temperature=0.1,
        max_tokens=mxTokens,
        top_p=1,
        # seed=3,
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
