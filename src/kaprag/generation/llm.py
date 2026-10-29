"""LLMClient: one interface, three backends (Ollama default, Anthropic, OpenAI).

The pipeline only calls `client.generate(system, user)`; switching backend in
config.yaml changes nothing else. Paid backends read keys from .env.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import httpx
from dotenv import load_dotenv

from kaprag.config import LLMConfig

log = logging.getLogger(__name__)
load_dotenv()  # makes ANTHROPIC_API_KEY / OPENAI_API_KEY from .env visible


class LLMUnavailableError(RuntimeError):
    """Raised with a human-readable fix (install command, missing key, ...)."""


class LLMClient(ABC):
    name: str = "base"

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def check(self) -> None:
        """Raise LLMUnavailableError if the backend cannot be used."""

    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class OllamaClient(LLMClient):
    """Local, free. Talks to the Ollama HTTP server (default port 11434)."""

    name = "ollama"
    INSTALL_HINT = ("Ollama is not running at {url}.\n"
                    "Install:  curl -fsSL https://ollama.com/install.sh | sh\n"
                    "Start:    ollama serve   (in another terminal)\n"
                    "Model:    ollama pull {model}")

    @property
    def model(self) -> str:
        return self.cfg.ollama_model

    def check(self) -> None:
        try:
            tags = httpx.get(f"{self.cfg.ollama_url}/api/tags", timeout=5).json()
        except Exception as exc:
            raise LLMUnavailableError(self.INSTALL_HINT.format(url=self.cfg.ollama_url, model=self.model)) from exc
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMUnavailableError(f"Ollama is running but model '{self.model}' is missing. "
                                      f"Run: ollama pull {self.model}")

    def generate(self, system: str, user: str) -> str:
        resp = httpx.post(
            f"{self.cfg.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                # num_ctx: Ollama's default context is only 2048 tokens, too small
                # for 6 chunks of ~400 tokens + instructions.
                "options": {"temperature": self.cfg.temperature, "num_ctx": 8192,
                            "num_predict": self.cfg.max_tokens},
            },
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class AnthropicClient(LLMClient):
    """Optional paid backend (Claude). Needs ANTHROPIC_API_KEY in .env."""

    name = "anthropic"

    @property
    def model(self) -> str:
        return self.cfg.anthropic_model

    def check(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise LLMUnavailableError("Install the optional extra: uv sync --extra api") from exc
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMUnavailableError("Set ANTHROPIC_API_KEY in .env to use the anthropic backend.")

    def generate(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        # Current Claude models take no `temperature`; `effort` controls depth instead.
        # Server-side fallbacks re-run the request on another model if it is declined.
        resp = client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise LLMUnavailableError("The model declined this request.")
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIClient(LLMClient):
    """Optional paid backend. Needs OPENAI_API_KEY in .env."""

    name = "openai"

    @property
    def model(self) -> str:
        return self.cfg.openai_model

    def check(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise LLMUnavailableError("Install the optional extra: uv sync --extra api") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMUnavailableError("Set OPENAI_API_KEY in .env to use the openai backend.")

    def generate(self, system: str, user: str) -> str:
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=self.model, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


def make_llm(cfg: LLMConfig, backend: str | None = None) -> LLMClient:
    """Factory: pick the backend named in config (or the override)."""
    backend = backend or cfg.backend
    classes = {"ollama": OllamaClient, "anthropic": AnthropicClient, "openai": OpenAIClient}
    return classes[backend](cfg)
