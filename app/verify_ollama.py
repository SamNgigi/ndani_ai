import asyncio
import sys
import httpx
import logging
from typing import List,Tuple, Dict, Optional
import json
from pathlib import Path

# logging config
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaVerifier:
    """Verify Ollama installation and model availability"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        models_dir: Optional[str] = None,
        modelfiles_dir: Optional[str] = None,
    ):
        """Initialize verifier with Ollama API URL and directory paths"""
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)

        # Set up directory paths
        self.models_dir = Path(models_dir or Path.home() / "Documents" / "models" )
        self.modelfiles_dir = Path(modelfiles_dir or Path(__file__).parent / "modelfiles")

        # Ensure modelfiles directory exists
        self.modelfiles_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def _create_modelfile_content(self, model_path: Path) -> str:
        """Create content for a ModelFile with specific configurations"""
        return f"""FROM {model_path}
# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.7
PARAMETER top_k 50
PARAMETER num_ctx 4096
PARAMETER stop "</s>"
PARAMETER stop "User:"
PARAMETER stop "Assistant:"

# System prompt for resume optimization
SYSTEM \"""
You are an expert AI assistant specializing in resume optimization and ATS (Applicant Tracking System) analysis.
Your role is to help improve resumes for better job application success rates while maintaining authenticity
\"""

# Template for consistent response format
TEMPLATE \"""
{{if .System}}{{.System}}{{end}}

{{if .Prompt}}{{.Prompt}}{{end}}

{{if .Response}}{{.Response}}{{end}}
\"""
"""
    
    async def _save_modelfile(self, model_name:str, content:str)->Path:
        """Save a Modelfile to the modelfile directory"""
        modelfile_path = self.modelfiles_dir / f"Modelfile.{model_name}"
        modelfile_path.write_text(content)
        logger.info(f"📝 Created Modelfile at: {modelfile_path}")
        return modelfile_path
    
    async def _get_available_gguf_models(self) -> List[Tuple[str, Path]]:
        """Get all GGUF models from the models directory"""
        if not self.models_dir.exists():
            logger.error(f"❌ Models directory not found: {self.models_dir}")
            return []
        gguf_files = list(self.models_dir.glob("*.gguf"))
        models = []
        for path in gguf_files:
            # Creating a simplified model name from the filename
            model_name = path.stem.split('-')[0].lower() # Take first part before hyphen
            models.append((model_name, path))

        if models:
            logger.info(f"📦 Found {len(models)} GGUF model(s) in {self.models_dir}")
        else:
            logger.warning(f" No GGUF models found in {self.models_dir}")

        return models

    async def verify_ollama_connection(self) -> bool:
        """Verify basic connection to Ollama"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            logger.info("✅ Successfully connected to Ollama")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {str(e)}")
            return False

    async def verify_model(self, model_name: str) -> bool:
        """Verify specific model functionality"""
        try:
            test_prompt = "Respond with 'OK' if you can process this message"
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json = {
                    "model":model_name,
                    "prompt":test_prompt,
                    "stream":False
                }
            )
            response.raise_for_status()

            result = response.json()
            if 'response' in result:
                logger.info(f"✅ Successfully tested model: {model_name}")
                return True

            logger.warning(f"⚠️  Model {model_name} response format unexpected")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to verify model {model_name}: {str(e)}")
            return False

    async def verify_custom_model(self, model_name: str, model_path: Path) -> bool:
        """Verify custom model availability and functionality"""
        try:
            # Generate and save modelfile
            modelfile_content = self._create_modelfile_content(model_path)
            modelfile_path = await self._save_modelfile(model_name, modelfile_content)

            # Create/Update model in Ollama
            logger.info(f"🏗️ Creating/Updating model: {model_name} :: modelfile_path: {modelfile_path}")
            response = await self.client.post(
                f"{self.base_url}/api/create",
                json = {
                    "name": model_name,
                    "modelfile": modelfile_content
                }
            )
            response.raise_for_status
            return await self.verify_model(model_name)
        except Exception as e:
            logger.error(f"❌ Failed to verify custom model {model_name}: {str(e)}")
            return False
    
    async def test_model_performance(self, model_name:str) -> Dict:
        """Test model performance with resume-related prompts"""
        test_prompts = [
            "Summarize this experience: Developed web applications using Python and React",
            "List key skills from: Expert in cloud architecture, CI/CD, and agile methodologies",
            "Improve this bullet point: Managed team of developers"
        ]

        results = {
            "successful_test": 0,
            "total_tests": len(test_prompts),
            "response_times": []
        }

        for prompt in test_prompts:
            try:
                start_time = asyncio.get_event_loop().time()
                response = await self.client.post(
                    f"{self.base_url}/api/generate",
                    json = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream":False
                    }
                )
                response.raise_for_status()
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time

                result = response.json()
                
                if 'response' in result and result['response'].strip():
                    results['successful_test'] += 1
                    results['response_times'].append(response_time)

            except Exception as e:
                logger.error(f"❌ Test prompt failed: {str(e)}")

        if results['response_times']:
            results['average_response_times'] = sum(results['response_times'])/len(results['response_times'])
        else:
            results['average_response_times'] = 0
        
        return results


async def main():
    """Main verification flow"""
    logger.info("🔍 Starting Ollama verification")

    async with OllamaVerifier() as verifier:
        # Check basic connection
        if not await verifier.verify_ollama_connection():
            logger.error("❌ Cannot proceed: Ollama connection failed")
            return 1

        # Get and verify all GGUF models
        models = await verifier._get_available_gguf_models()
        if not models:
            logger.error("❌ No models found to verify")
            return 1
    
        # Track verification results
        results = []

        # Verify each model
        for model_name, model_path in models:
            logger.info(f"\n🔍 Verifying model: {model_name}")

            # Verifying and testing model
            if await verifier.verify_custom_model(model_name, model_path):
                performance = await verifier.test_model_performance(model_name)
                results.append({
                    "model": model_name,
                    "status": "✅ Passed",
                    "performance": performance
                })
            else:
                results.append({
                    "model": model_name,
                    "status": "❌ Failed",
                    "performance":None
                })

        logger.info("\n📊 Verification Summary")
        for result in results:
            logger.info(f"Model: {result['model']}")
            logger.info(f"Status: {result['status']}")
            if result['performance']:
                perf = result['performance']
                logger.info(
                    f"Performance:\n"
                    f"  - Success rate: {perf['successful_test']}/{perf['total_tests']}\n"
                    f"  - Average response time: {perf['average_response_times']:.2f}s"
                )

        return 0 if any(r['status'].startswith('✅') for r in results) else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)
