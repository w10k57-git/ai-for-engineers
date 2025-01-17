"""
Basic OpenAI Service
"""
import os
from typing import Dict, List
from openai import OpenAI
import instructor
from pydantic import BaseModel

class OpenAIService:
    """
    OpenAI Service
    """
    def __init__(self, provider: str = "ollama"):
        self.client = self._get_client(provider)

    def _get_client(self, provider: str) -> OpenAI:
        if provider == "ollama":
            return OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
        if provider == "groq":
            return OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY")
            )
        raise ValueError(f"Invalid provider: {provider}")

    def create_chat_completion(
        self,
        messages: list[dict],
        model: str,
        **kwargs
    ) -> dict:
        """
        Create a chat completion from the messages.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs)

    def create_structured_output(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response_model: BaseModel,
        **kwargs) -> BaseModel:
        """
        Create structured output from the chat completion.
        """
        client = instructor.from_openai(self.client)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            max_retries=3,
            **kwargs
        )

        return response
