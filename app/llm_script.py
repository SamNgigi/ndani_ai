import asyncio
import httpx
import logging
import pprint as pp

from verify_ollama import load_config, OllamaVerifier, _get_available_gguf_models

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

async def get_llm_response():

    try:
        client = httpx.AsyncClient(timeout = 30)
        config = load_config()
        models = _get_available_gguf_models(config)
        modelfile_path = config.modelfiles_dir / "Modelfile.codellama"
        modelfile_content = modelfile_path.read_text()
        response = await client.post(
            f"{config.base_url}/api/create",
            json = {
                "model": 'codellama',
                "modelfile": modelfile_content,

            }
        )
        response.raise_for_status()
        response = await client.post(
            f"{config.base_url}/api/generate",
            json = {
                "model": 'codellama',
                "prompt": 'Hello world program in modern C++?',
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        pp.pprint(result)
    except Exception as e:
        logger.info(f"❌ Error in get llm_response: {str(e)}")


if __name__ == "__main__":
    asyncio.run(get_llm_response())
