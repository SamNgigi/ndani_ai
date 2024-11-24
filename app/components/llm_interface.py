import os
import json
import asyncio
import logging
import backoff # For exponential backoff in retries 
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Optional, Dict, Union, List
from dataclasses import dataclass
from groq import AsyncGroq
from groq.types.chat.chat_completion import ChatCompletion
from groq.types.chat.completion_create_params import ResponseFormat


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """Configuration for Groq Cloud"""
    model_name:str
    max_tokens:int
    temperature:Dict[str, float] = {"strict":0.0, "precise":0.3, "balanced": 0.5, "creative":0.7}
    top_p:int=1
    seed:Optional[int]=100
    stream:bool=False
    response_format:ResponseFormat = {'type':'json_object'} # text or json_object
    stop:Union[str, None] = None
    message: Optional[str] = None

    def __post_init__(self):
        if self.message is None:
            self.message = f"ℹ️  Running Model: {self.model_name}, max_tokens: {self.max_tokens}"
        

class LlmInterface:
    """
    Interface for interacting with LLM Apis (Groq Cloud)
    for prototyping.
    # TODO: Generalize to Ollama or Llama.cpp
    """

    def __init__(self, api_key:str):
        """
        Initialize the interace

        Args:
            api_key: groq cloud api key
"""
        self.client = AsyncGroq(api_key=api_key)
        self.models = {
            'mixtral': GroqConfig(
                model_name='mixtral-8x7-32768',
                max_tokens = 11000
            ),
            'llama-versatile': GroqConfig(
                model_name='llama-3.1-70b-versatile',
                max_tokens = 8000
            ),
            'llama-instant': GroqConfig(
                model_name='llama-3.1-8b-instant',
                max_tokens = 8000
            )
        }
        self.current_model = 'mixtral' # Default
        self.current_temperature = 'strict'

    
    def set_model(self, model_name:str) -> None:
        """
        Set the active model

        Args:
            model_name: Name of the model to use

        Raises:
            ValueError: if model is not supported
        """

        if model_name not in self.models:
            raise ValueError(f"Unsupported model: {model_name}. Avaliable models: {list(self.models.keys())}")
        self.current_model = model_name
        logger.info(f"ℹ️  Switched to model: {model_name}")

    @backoff.on_exception(
        backoff.expo,
        (asyncio.TimeoutError),
        max_tries=3
    )
    async def _make_request(self, payload:List) -> ChatCompletion:
        """
        Make request to Groq Cloud with retry

        Args:
            payload: message (Groq request payload)
        """
        try:
            config = self.models[self.current_model]

            completion = await self.client.chat.completions.create(
                model = config.model_name,
                messages = payload,
                temperature = config.temperature[self.current_temperature],
                max_tokens = config.max_tokens,
                top_p = config.top_p,
                seed = config.seed,
                stream = False,
                response_format = config.response_format,
                stop = config.stop,

            )

            return completion
        except Exception as e:
            logger.error(f"❌ Error during API call: {str(e)}")
            raise

    async def generate(
            self,
            user_prompt: str ,
            system_prompt: Optional[str]=None,
    ) -> dict:
        """
        Generate response from the model
        """
        payload = [
            {
                "role":"system",
                "content": system_prompt,
            },
            {
                "role":"user",
                "content": user_prompt,
            }
        ]

        completion = await self._make_request(payload)
        if completion.choices[0] and completion.choices[0].message.content:
            try:
                result = json.loads(completion.choices[0].message.content)
                logger.info("✅ Successfully returned generated JSON response")
            except json.JSONDecodeError as e:
                logger.error(f"❌ Error decoding JSON response: {str(e)}")
                raise
            return result
        else:
            logger.error("❌ Did not receive a valid response from API call")
            raise ValueError("Did not recieve a valid response from the API")

    async def generate_with_retry(
        self,
        user_prompt: str ,
        system_prompt: Optional[str]=None,
        max_attempts:int=3
    ) -> dict:
        @retry(
            stop = stop_after_attempt(max_attempts),
            wait = wait_exponential(multiplier=1, min=4, max=10)
        )
        async def _generate_with_retry():
            response = await self.generate(user_prompt, system_prompt)
            if not response:
                logger.error("❌ Error in generating a response...retrying")
            return response
        try:
            return await _generate_with_retry()
        except Exception as e:
            logger.error(f"❌ Failed after {max_attempts} attempts: {str(e)}")
            raise




