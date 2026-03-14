import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from mistralai.client import Mistral


def _ensure_src_on_path() -> Path:
    src_path = Path(__file__).resolve().parents[3]
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src_path


_SRC_PATH = _ensure_src_on_path()

from byeias.backend.config_loader import get_backend_config, get_logger  # noqa: E402

BACKEND_CONFIG = get_backend_config()
logger = get_logger("byeias.llm_communicator", BACKEND_CONFIG)
LLM_CONFIG = BACKEND_CONFIG.llm


class LLMCommunicator:
    """LLM wrapper for bias explanation generation."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name or LLM_CONFIG.model_name
        self.max_tokens = (
            max_tokens if max_tokens is not None else LLM_CONFIG.max_tokens
        )
        self.temperature = (
            temperature if temperature is not None else LLM_CONFIG.temperature
        )
        self.system_prompt = system_prompt or self._load_system_prompt()

        resolved_api_key = api_key or os.getenv("MISTRAL_API_KEY") or LLM_CONFIG.api_key
        if not resolved_api_key:
            raise ValueError(
                "Missing API key. Set MISTRAL_API_KEY or backend.llm.api_key in config.yaml."
            )

        self.client = Mistral(api_key=resolved_api_key)
        logger.info(
            "LLM communicator initialized | model=%s max_tokens=%d temperature=%.2f",
            self.model_name,
            self.max_tokens,
            self.temperature,
        )

    def _load_system_prompt(self) -> str:
        prompt_path = LLM_CONFIG.system_prompt_path
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"System prompt file is empty: {prompt_path}")
        return prompt_text

    @staticmethod
    def _build_user_prompt(
        context_before: str, flagged_sentence: str, context_after: str
    ) -> str:
        return (
            f'Context before: "{context_before}"\n'
            f'Highlighted sentence: "{flagged_sentence}"\n'
            f'Context after: "{context_after}"'
        )

    def explain_bias(
        self,
        context_before: str,
        flagged_sentence: str,
        context_after: str,
    ) -> Dict[str, str]:
        user_prompt = self._build_user_prompt(
            context_before, flagged_sentence, context_after
        )
        logger.info("Requesting LLM explanation")

        response = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        return {
            "bias_type": str(result_json.get("bias_type", "")),
            "explanation": str(result_json.get("explanation", "")),
            "rewrite_suggestion": str(result_json.get("rewrite_suggestion", "")),
        }
