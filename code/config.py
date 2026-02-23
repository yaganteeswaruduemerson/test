
# config.py

import os
from dotenv import load_dotenv
from typing import Any, Dict, Optional
import logging

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Configuration management for the Email Payload Validation Agent.
    Handles environment variable loading, API key management, LLM config,
    domain-specific settings, validation, error handling, and defaults.
    """

    # Default values
    DEFAULT_LLM_PROVIDER = "openai"
    DEFAULT_LLM_MODEL = "gpt-4o"
    DEFAULT_LLM_TEMPERATURE = 0.7
    DEFAULT_LLM_MAX_TOKENS = 2000
    DEFAULT_SYSTEM_PROMPT = "You are a professional general agent."
    DEFAULT_USER_PROMPT_TEMPLATE = "How can I help you today?"
    DEFAULT_FEW_SHOT_EXAMPLES = []
    DEFAULT_DOMAIN = "general"
    DEFAULT_PERSONALITY = "professional"

    def __init__(self):
        # Load environment variables
        load_dotenv()

        # API Key Management
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key or not self.openai_api_key.strip():
            raise ConfigError("Missing required environment variable: OPENAI_API_KEY")

        # LLM Configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", self.DEFAULT_LLM_PROVIDER)
        self.llm_model = os.getenv("LLM_MODEL", self.DEFAULT_LLM_MODEL)
        self.llm_temperature = self._get_float_env("LLM_TEMPERATURE", self.DEFAULT_LLM_TEMPERATURE)
        self.llm_max_tokens = self._get_int_env("LLM_MAX_TOKENS", self.DEFAULT_LLM_MAX_TOKENS)
        self.system_prompt = os.getenv("SYSTEM_PROMPT", self.DEFAULT_SYSTEM_PROMPT)
        self.user_prompt_template = os.getenv("USER_PROMPT_TEMPLATE", self.DEFAULT_USER_PROMPT_TEMPLATE)
        self.few_shot_examples = self._get_few_shot_examples_env("FEW_SHOT_EXAMPLES", self.DEFAULT_FEW_SHOT_EXAMPLES)

        # Domain-specific settings
        self.domain = os.getenv("AGENT_DOMAIN", self.DEFAULT_DOMAIN)
        self.personality = os.getenv("AGENT_PERSONALITY", self.DEFAULT_PERSONALITY)

        # API Requirements
        self.api_requirements = [
            {
                "name": "OpenAI API",
                "type": "external",
                "purpose": "LLM inference",
                "authentication": "API Key",
                "rate_limits": "Per-minute limits"
            }
        ]

    def _get_int_env(self, var: str, default: int) -> int:
        val = os.getenv(var)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            logging.warning(f"Invalid int for {var}, using default {default}")
            return default

    def _get_float_env(self, var: str, default: float) -> float:
        val = os.getenv(var)
        if val is None:
            return default
        try:
            return float(val)
        except ValueError:
            logging.warning(f"Invalid float for {var}, using default {default}")
            return default

    def _get_few_shot_examples_env(self, var: str, default: list) -> list:
        val = os.getenv(var)
        if not val:
            return default
        try:
            import json
            return json.loads(val)
        except Exception:
            logging.warning(f"Invalid JSON for {var}, using default empty list")
            return default

    def get_llm_config(self) -> Dict[str, Any]:
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "few_shot_examples": self.few_shot_examples
        }

    def get_api_key(self) -> str:
        if not self.openai_api_key:
            raise ConfigError("OpenAI API key is not set.")
        return self.openai_api_key

    def get_domain_settings(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "personality": self.personality
        }

    def get_api_requirements(self) -> list:
        return self.api_requirements

    @classmethod
    def load(cls) -> "Config":
        try:
            return cls()
        except ConfigError as e:
            logging.critical(f"Configuration error: {e}")
            raise
        except Exception as e:
            logging.critical(f"Unexpected error during config loading: {e}")
            raise

# Example usage:
# try:
#     config = Config.load()
# except ConfigError as e:
#     print(f"Configuration failed: {e}")
#     exit(1)
