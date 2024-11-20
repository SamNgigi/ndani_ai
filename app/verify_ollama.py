import asyncio
import time
import json
import httpx
import logging
import shutil
import pprint as pp
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, field_validator 
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List,Tuple, Dict, Optional, Callable, Awaitable
from prometheus_client import Counter, Histogram, start_http_server
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

# logging config
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

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

class ModelMetadata(BaseModel):
    """Model metadata information"""
    name: str
    path: Path
    size: int
    created_at: datetime
    last_verified: Optional[datetime] = None
    performance_metrics: Dict[str, float] = {}

class OllamaConfig(BaseSettings):
    """Configuration settings for Ollama verification"""
    # Server Configuration
    base_url: str = "http://localhost:11434"
    timeout: float = 100.0
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
    llm_params: ModelParameters = ModelParameters()

    # Health Checks config
    health_check_interval: int = 60 # Seconds
    min_disk_space_pct: int = 10 # minimum free disk space percentage

    model_config = SettingsConfigDict(
        env_prefix = "OLLAMA_",
        env_nested_delimiter = "__"
    )
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
            f"✅ Configuration loaded successfully\n"
            f"  - base_url: {config.base_url}\n"
            f"  - models_dir: {str(config.models_dir)}\n"
            f"  - modelfiles_dir: {str(config.modelfiles_dir)}"
        )
        return config
    except Exception as e:
        logger.error(f"❌ Loading configuration failed. Error : {get_error_details(e)}")
        raise

def _get_available_gguf_models(config: OllamaConfig) -> List[Tuple[str, Path]]:
    """Get all GGUF models from the models directory"""
    if not config.models_dir.exists():
        logger.error(f"❌ Models directory not found: {config.models_dir}")
        return []
    gguf_files = list(config.models_dir.glob("*.gguf"))
    models = []
    for path in gguf_files:
        # Creating a simplified model name from the filename
        model_name = path.stem.split('-')[0].lower() # Take first part before hyphen
        models.append((model_name, path))

    if models:
        logger.info(f"📦 Found {len(models)} GGUF model(s) in {config.models_dir}")
    else:
        logger.warning(f" No GGUF models found in {config.models_dir}")

    return models

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
            logger.error(f"❌ Disk space check failed. Error: {str(get_error_details(e))}")
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
            logger.error(f"❌ Check ollama server failed. Error:{str(get_error_details(e))}")
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
        self.last_check = datetime.now()
        self.last_status = status
        return status

class OllamaVerifier:
    """Verify Ollama installation and model availability"""

    def __init__(self, config:OllamaConfig):
        """Initialize verifier with Ollama API URL and directory paths"""
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.metrics = MetricsCollector()
        self.health = HealthCheck(config)
        self.verification_callbacks: List[Callable[[str, bool], Awaitable[None]]] = []

        # Ensure modelfiles directory exists
        self.config.modelfiles_dir.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    @lru_cache(maxsize=32)
    async def get_model_metadata(self, model_path: Path) -> ModelMetadata:
        """"Get cached model metadata"""
        stats = model_path.stat()
        return ModelMetadata(
            name = model_path.stem.split('-')[0].lower(),
            path = model_path,
            size = stats.st_size,
            created_at = datetime.fromtimestamp(stats.st_birthtime)
        )

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
TEMPLATE \"""{{ if .System }}<|start_header|>system<|end_header|>
{{ .System }}<|end_of_text|>{{ end }}
{{ if .Prompt }}<|start_header|>user<|end_header|>
{{ .Prompt }}<|end_of_text|>{{ end }}
<|start_header|>assistant<|end_header|>
{{ .Response }}<|end_of_text|>
\"""
"""
    
    async def _save_modelfile(self, model_name:str, content:str)->Path:
        """Save a Modelfile to the modelfile directory"""
        modelfile_path = self.config.modelfiles_dir / f"Modelfile.{model_name}"
        modelfile_path.write_text(content)
        logger.info(f"📝 Created Modelfile at: {modelfile_path}")
        return modelfile_path
    


    async def verify_ollama_connection(self) -> bool:
        """Verify basic connection to Ollama"""
        try:
            response = await self.client.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            logger.info("✅ Successfully connected to Ollama")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {str(e)}")
            return False


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def verify_model(self, model_name: str) -> bool:
        """Verify specific model functionality with retries"""
        try:
            with self.metrics.verification_duration.labels(model_name).time():
                response = await self.client.post(
                    f"{self.config.base_url}/api/generate",
                    json = {
                        "model":model_name,
                        "prompt":"System check: Respond with 'OK' if functioning",
                        "stream":False
                    }
                )
                response.raise_for_status()

                result = response.json()
                success = bool('response' in result and result['response'].strip())
                pp.pprint(result)
                self.metrics.verification_attempts.labels(
                    model_name = model_name,
                    status = "success" if success else "failure"
                ).inc()
                if success:
                    logger.info(f"✅ Successfully tested model: {model_name}")
                    return success

                logger.warning(f"⚠️  Model {model_name} response format unexpected")
                return success

        except Exception as e:
            error_details = get_error_details(e)
            logger.error(f"❌ Failed to verify model {model_name}: {error_details}")
            self.metrics.verification_attempts.labels(
                model_name = model_name,
                status = "error"
            ).inc()
            raise

    async def _trigger_callbacks(self, model_name: str, success: bool):
        """Trigger verification completion callbacks"""
        for callback in self.verification_callbacks:
            try:
                await callback(model_name, success)
            except Exception as e:
                logger.error(
                    "❌ Callback failed\n"
                    f" - model: {model_name}\n"
                    f" - error: {str(get_error_details(e))}"
                )

    async def verify_custom_model(self, model_name: str, model_path: Path) -> bool:
        """Verify custom model availability and functionality"""
        try:
            # Get/update metadata
            metadata = await self.get_model_metadata(model_path)
            modelfile_path = self.config.modelfiles_dir / f"Modelfile.{model_name}"

            if modelfile_path.exists() and modelfile_path.stat().st_size > 0:
                modelfile_content = modelfile_path.read_text()
            else:
            # Generate and save modelfile
                modelfile_content = self._create_modelfile_content(model_path)
                modelfile_path = await self._save_modelfile(model_name, modelfile_content)

            # Create/Update model in Ollama
            logger.info(f"🏗️ Creating/Updating model: {model_name} :: modelfile_path: {modelfile_path}")
            response = await self.client.post(
                f"{self.config.base_url}/api/create",
                json = {
                    "name": model_name,
                    "modelfile": modelfile_content
                }
            )
            response.raise_for_status()
            
            # Verify the model
            success = await self.verify_model(model_name)

            # Update metadata
            metadata.last_verified = datetime.now()

            # Trigger callbacks
            await self._trigger_callbacks(model_name, success)

            return success
        except Exception as e:
            error_details = get_error_details(e)
            logger.error(
                "❌ Failed to verify custom model\n"
                f"  - model_name: {model_name}\n" 
                f"  - error: {error_details}"
            )
            return False

    def on_verification_complete(self, callback: Callable[[str, bool], Awaitable[None]]):
        """Register a verification completion callback"""
        self.verification_callbacks.append(callback)

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
                start_time = time.perf_counter()
                response = await self.client.post(
                    f"{self.config.base_url}/api/generate",
                    json = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream":False
                    }
                )
                response.raise_for_status()
                end_time = time.perf_counter()
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

    async def verify_and_test_model(self, model_name:str, model_path:Path) -> Dict:
        """Verify and test a single model, returning the result"""
        result = {
            "model": model_name,
            "status": "",
            "performance": None 
        }

        try:
            # Verifying and testing model
            logger.info(f"\n🔍 Verifying model: {model_name}")
            if await self.verify_custom_model(model_name, model_path):
                performance = await self.test_model_performance(model_name)
                result["status"] = "✅ Passed"
                result["performance"] = performance
            else:
                result["status"] = "❌ Failed"

        except Exception as e:
            logger.error(f"❌ Model {model_name} verification failed with exception: {get_error_details(e)}")
            result["status"] = "❌ Failed"

        return result

    async def verify_all_models(self) -> List[Dict]:
        """Verify all models in parallel"""
        try:
            # Get all GGUF models
            models = _get_available_gguf_models(self.config)
            if not models: 
                logger.error("❌ No models found to verify")
                return []

            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self.verify_and_test_model(name, path))
                    for name, path in models
                ]
            results = [
                task.result() for task in tasks
            ]
            return results
        except Exception as e:
            logger.error(f"❌ Bulk verification failed. Error: {str(get_error_details(e))}")
            return []
    



async def main():
    """Main verification flow"""
    logger.info("🔍 Starting Ollama verification")

    try:
        config = load_config()
        if config.metrics_enabled:
            start_http_server(config.metric_port)
            logger.info(f"💻 Metric server started on port: {config.metric_port}")

        async with OllamaVerifier(config) as verifier:
            # Check basic connection
            if not await verifier.verify_ollama_connection():
                logger.error("❌ Cannot proceed: Ollama connection failed")
                return 1

            # Check system health
            health_status = await verifier.health.check_system_health()
            if not all(health_status.values()):
                logger.error(f"❌ Health check failed. Status={health_status}")
                return 1
            
            results = await verifier.verify_all_models()
            
            if not results:
                logger.error("❌ No models were verified")
                return 1
            # Log summary
            logger.info("📊 Verification summary\n")
            for result in results:
                logger.info(f"Model: {result['model']}")
                logger.info(f"Status: {result['status']}")
                if result['performance']:
                    perf = result['performance']
                    logger.info(
                        f"Performance:\n"
                            f"  - Success rate: {perf['successful_test']/perf['total_tests']}\n"
                            f"  - Average response time: {perf['average_response_times']:.2f}s"
                    )

        success_counts = sum(1 for r in results if r['status'].startswith('✅'))
        return 0 if success_counts > 0 else 1

    except Exception as e:
        logger.error(f"❌ Main execution failed. Error: {str(get_error_details(e))}")
        return 1

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

