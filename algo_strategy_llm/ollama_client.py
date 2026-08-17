"""Thin HTTP wrapper for the local Ollama server (same pattern as model_debate.py)."""

import requests

from . import config


def ask_model(prompt: str, model: str = config.MODEL, format_json: bool = True) -> str:
    """Send a prompt to Ollama and return the raw response text.

    format_json=True asks Ollama to constrain the output to valid JSON.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if format_json:
        payload["format"] = "json"

    try:
        response = requests.post(config.OLLAMA_URL, json=payload, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at "
            f"{config.OLLAMA_URL}. Is the Ollama server running?"
        )
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise RuntimeError(
                f"Model '{model}' not found. Pull it first: ollama pull {model}"
            )
        raise RuntimeError(f"Ollama request failed: {e}")

    return response.json().get("response", "").strip()
