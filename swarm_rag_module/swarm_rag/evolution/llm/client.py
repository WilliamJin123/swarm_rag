"""
LLM Client - Single source of truth for all LLM API calls.

All LLM calls in the evolution system go through this client:
- StrategicOracle.get_directive()
- TacticalAdvisor.get_prescription()
- CreativeSynthesizer.synthesize()

Features:
- Retry with exponential backoff
- JSON response mode
- Configurable temperature
- Logging for debugging
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMCallResult:
    """Result from an LLM call."""
    content: str
    parsed: Optional[Dict[str, Any]]
    success: bool
    error: Optional[str] = None
    attempts: int = 1


class LLMClient:
    """
    Unified LLM client for the evolution system.

    Wraps keycycle's MultiClientWrapper and provides a consistent
    interface for all LLM calls with retry logic and JSON parsing.

    Usage:
        from keycycle import MultiClientWrapper, ProviderEnvConfig
        wrapper = MultiClientWrapper.from_env(
            providers={"cerebras": ProviderEnvConfig(default_model="zai-glm-4.7")},
            env_file=".env"
        )
        client = LLMClient(wrapper, model="zai-glm-4.7", provider="cerebras")

        result = client.call(
            system_prompt="You are an expert...",
            user_prompt="Analyze this genome...",
        )
        if result.success:
            data = result.parsed
    """

    def __init__(
        self,
        wrapper: Any,  # keycycle.MultiClientWrapper
        model: str,
        provider: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM client.

        Args:
            wrapper: keycycle MultiClientWrapper instance
            model: Model ID (e.g., "zai-glm-4.7", "gpt-4o-mini")
            provider: Provider name (e.g., "cerebras", "openai", "groq")
            max_retries: Max retry attempts
            retry_delay: Base delay for exponential backoff
        """
        self.wrapper = wrapper
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float = 0.3,
    ) -> LLMCallResult:
        """
        Make an LLM API call with retry logic.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            json_mode: Whether to request JSON response format
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMCallResult with content, parsed JSON (if json_mode), and status
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = self._make_call(
                    system_prompt, user_prompt, json_mode, temperature
                )
                result.attempts = attempt + 1
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    time.sleep(sleep_time)

        return LLMCallResult(
            content="",
            parsed=None,
            success=False,
            error=f"Max retries exceeded: {last_error}",
            attempts=self.max_retries,
        )

    def _make_call(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool,
        temperature: float,
    ) -> LLMCallResult:
        """Make single LLM call (no retry)."""
        from keycycle import RotatingOpenAIClient

        manager = self.wrapper.get_manager(self.provider)
        openai_client = RotatingOpenAIClient(
            manager=manager,
            limit_resolver=lambda m, k: self.wrapper._resolve_limits(self.provider, m, k),
            default_model=self.model,
            provider=self.provider,
        )

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = openai_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        # Parse JSON if in json_mode
        parsed = None
        if json_mode and content:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                return LLMCallResult(
                    content=content,
                    parsed=None,
                    success=False,
                    error=f"JSON parse error: {e}",
                )

        return LLMCallResult(
            content=content,
            parsed=parsed,
            success=True,
        )

    @classmethod
    def from_config(
        cls,
        provider: str = "cerebras",
        model: str = "zai-glm-4.7",
        env_path: str = ".env",
        **kwargs,
    ) -> "LLMClient":
        """
        Create an LLMClient from configuration.

        Args:
            provider: Provider name (cerebras, openai, groq, openrouter, gemini, cohere)
            model: Model ID
            env_path: Path to .env file with API keys
            **kwargs: Additional args (max_retries, retry_delay)

        Returns:
            Configured LLMClient instance
        """
        try:
            from keycycle import MultiClientWrapper, ProviderEnvConfig
        except ImportError:
            raise ImportError(
                "keycycle is required for LLMClient. "
                "Install with: pip install keycycle"
            )

        from dotenv import load_dotenv
        load_dotenv(env_path)

        wrapper = MultiClientWrapper.from_env(
            providers={
                provider: ProviderEnvConfig(
                    default_model=model,
                )
            },
            env_file=env_path,
        )

        logger.info(f"Created LLMClient: {provider}/{model}")
        return cls(wrapper, model, provider=provider, **kwargs)


# Supported providers (matches keycycle PROVIDER_BASE_URLS)
SUPPORTED_PROVIDERS = [
    "openai",
    "openrouter",
    "gemini",
    "cerebras",
    "groq",
    "cohere",
]
