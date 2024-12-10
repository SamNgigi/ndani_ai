from typing import Union
from pathlib import Path
from datetime import datetime
import json
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_json(data:dict, file_name:str, **kwargs)->Path:
    output_dir = Path(__file__).parent.parent/ "output"
    output_dir.mkdir(exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{file_name}_{timestamp}.json"
    try:
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"❌ Error saving JSON file: {str(e)}")
        raise


def read_json(file_path:Union[str, Path]) -> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"❌ File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with file_path.open('r', encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            logger.error(f"❌ No data found from {file_path.name}")
            raise ValueError(f"No data found from {file_path.name}")
        return data
    except Exception as e:
        logger.error(f"Error reading json: {str(e)}")
        raise
