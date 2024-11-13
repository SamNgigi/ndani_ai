import asyncio
import json
import httpx
import structlog
import shutil
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, field_validator 
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List,Tuple, Dict, Optional, Callable, Awaitable
from prometheus_client import Counter, Histogram, start_http_server
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

# Configure structured logging
structlog.configure(
    processors = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

def get_error_details(e:Exception) -> Dict:
    return  {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'error_args': getattr(e, 'args', None),
        'response_status': getattr(getattr(e, 'response', None), 'status_code', None),
        'response_text': getattr(getattr(e, 'response', None), 'text', None)
    }

class ModelParameters(BaseModel):
    """Model specific parameter config"""
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 50
    ctx_size: int = 4096

    @field_validator('temperature', 'top_p')
    def validate_float_params(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Parameter must be between 0 and 1")
        return v

class OllamaConfig(BaseSettings):
    """Configuration settings for Ollama verification"""
    # Server Configuration
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    max_retries: int = 3

    # Directory Paths
    models_dir: Path = Path.home() / "Documents" / "models" 
    modelfiles_dir: Path = Path(__file__).parent / "modelfiles"

    # Metric Configuration
    metric_port: int = 9090
    metrics_enabled: bool = True

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"

    # Model params
    model_parameters: ModelParameters = ModelParameters()

    # Health Checks config
    health_check_interval: int = 60 # Seconds
    min_disk_space_pct: int = 10 # minimum free disk space percentage

    class Config:
        env_perfix = "OLLAMA_"
        env_nested_delimiter = "__"

        @classmethod
        def customize_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )
    
    @field_validator('models_dir', 'modelfiles_dir')
    def validate_directory(cls, v):
        return Path(os.path.expandvars(os.path.expanduser(str(v))))

def load_config() -> OllamaConfig: 
    """Load configuration from .env file and environment variables"""
    try:
        config = OllamaConfig()

        # Ensuring directories exist
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.modelfiles_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging level
        logger.info(
            "configuration_loaded",
            base_url=config.base_url,
            models_dir=str(config.models_dir),
            modelfiles_dir=str(config.modelfiles_dir)
        )
        return config
    except Exception as e:
        logger.error("load_config_failed", error=str(e), exc_info=True)
        raise

class MetricsCollector:
    """Prometheus metrics collection"""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
    
        self.verification_attempts = Counter(
            'model_verification_attempts_total',
            'Number of model verification attempts',
            ['model_name', 'status']
        )
        self.verification_duration = Histogram(
            'model_verification_duration_seconds',
            'Time spend verifying models',
            ['model_name']
        )
        self.verification_time = Histogram(
            'model_response_time_seconds',
            'Model response times for test prompts',
            ['model_name']
        )

    def record_attempts(self, model_name:str, status:str):
        """"Record verification attempts"""
        if self.enabled:
            self.verification_attempts.labels(
                model_name = model_name,
                status=status
            ).inc()

class HealthCheck:
    """System health monitoring"""
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.last_check: Optional[datetime] = None
        self.last_status: Dict[str, bool] = {}

    async def check_disk_space(self) -> bool:
        """Check if there's sufficient disk space"""
        try:
            total, _, free = shutil.disk_usage(self.config.models_dir) # total, used, free
            free_pct = (free/total) * 100
            return free_pct >= self.config.min_disk_space_pct
        except Exception as e:
            logger.error("disk_space_check_failed", error=get_error_details(e))
            return False
    
    async def check_ollama_server(self) -> bool:
        """Check if Ollama server is responding"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.config.base_url}/api/tags",
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception as e:
            logger.error("check_ollama_server_failed", error=get_error_details(e))
            return False

    async def should_check_health(self) -> bool:
        """Determine if health check is needed based on interval"""
        if not self.last_check:
            return True
        elapsed = (datetime.now() - self.last_check).total_seconds()
        return elapsed >= self.config.health_check_interval

    async def check_system_health(self) -> Dict[str, bool]:
        """Perform comprehensive system health check."""
        if not await self.should_check_health():
            return self.last_status
        
        status = {
            "ollama_server": await self.check_ollama_server(),
            "models_directory": self.config.models_dir.exists(),
            "modelfiles_directory": self.config.modelfiles_dir.exists(),
            "disk_space": await self.check_disk_space()
        }
        return status

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
            error_details = get_error_details(e)
            logger.error(f"❌ Failed to verify model {model_name}: {error_details}")
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
            error_details = get_error_details(e)
            logger.error(f"❌ Failed to verify custom model {model_name}: {error_details}")
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
                error_details = get_error_details(e)
                logger.error(f"❌ Test prompt failed: {error_details}")

        if results['response_times']:
            results['average_response_times'] = sum(results['response_times'])/len(results['response_times'])
        else:
            results['average_response_times'] = 0
        
        return results


async def main():
    """Main verification flow"""
    logger.info("🔍 Starting Ollama verification")

    try:
        config = load_config()
        if config.metrics_enabled:
            start_http_server(config.metric_port)
            logger.info("metrics_server_started", port=config.metric_port)
    except Exception as e:
        logger.error("main_executation_failed", error=get_error_details(e), exc_info=True)

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Verification interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        exit(1)
