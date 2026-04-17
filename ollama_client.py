"""
ollama_client.py
Manages all communication with Ollama (llama3 at http://127.0.0.1:11434).
Features: retry with exponential backoff, request-level caching, graceful fallback.
"""

import hashlib
import json
import os
import time
from typing import Optional

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3"
CACHE_PATH = "llm_cache.json"
MAX_RETRIES = 3
TIMEOUT = 30  # seconds per attempt


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass  # Non-fatal; cache is best-effort


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def query_llm(prompt: str, use_cache: bool = True) -> Optional[str]:
    """
    Send a prompt to Ollama llama3 and return the response text.
    Returns None if all retries fail (caller should use rule-based fallback).
    """
    cache = _load_cache()
    key = _cache_key(prompt)

    if use_cache and key in cache:
        return cache[key]

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            if text:
                if use_cache:
                    cache[key] = text
                    _save_cache(cache)
                return text
        except requests.exceptions.ConnectionError as e:
            last_error = e
            break  # Ollama not running — no point retrying
        except requests.exceptions.Timeout as e:
            last_error = e
        except requests.exceptions.HTTPError as e:
            last_error = e
            break  # HTTP errors are deterministic
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            last_error = e
            break

        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)  # 2s, 4s

    return None  # Signal to caller: use fallback


def is_ollama_available() -> bool:
    """Quick health check — does not count toward query retries."""
    try:
        r = requests.get("http://127.0.0.1:11434", timeout=3)
        return r.status_code < 500
    except requests.exceptions.RequestException:
        return False

