"""
Configurable LLM client using OpenAI-compatible API.
Supports NVIDIA NIM, OpenAI, and any OpenAI-compatible endpoint.
"""

import logging

from openai import OpenAI

from backend.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Configurable LLM client for grounded answer generation.
    Uses temperature=0 for deterministic, non-hallucinatory output.
    """

    def __init__(self):
        if not settings.has_nvidia_key:
            logger.warning("No NVIDIA API key — LLM generation will be unavailable")
            self.client = None
            self.available = False
            return

        self.client = OpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
        self.model = settings.nvidia_llm_model
        self.available = True
        logger.info(f"LLM client initialized: {self.model}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8096,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instructions.
            user_prompt: User query with context.
            max_tokens: Maximum response length.
            temperature: 0.0 for deterministic output.

        Returns:
            Generated text response.
        """
        if not self.available or self.client is None:
            return (
                "⚠️ LLM is not configured. Set NVIDIA_API_KEY in your .env file "
                "to enable AI-powered answers."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"⚠️ Generation error: {str(e)}"
