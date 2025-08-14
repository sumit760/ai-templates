#!/usr/bin/env python3
import os, json, requests, argparse, textwrap, re, sys, math, yaml
from typing import List, Dict, Optional

# --------------------------
# Config & Helpers
# --------------------------

PR_SUMMARY_MARKER = "<!-- pr-summarizer-bot -->"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
SUMMARY_TONE = os.getenv("SUMMARY_TONE", "concise")  # concise|detailed
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "220000"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # openai|ollama (sample)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

EXCLUDE_GLOBS_ENV = os.getenv("EXCLUDE_GLOBS", "").strip()
EXCLUDE_GLOBS = [g.strip() for g in EXCLUDE_GLOBS_ENV.splitlines() if g.strip()]

def glob_to_regex(glob: str) -> str:
    # very lightweight glob→regex (**, *, .) — good enough for our excludes
    rx = re.escape(glob)
    rx = rx.replace(r"\*\*", ".*")
    rx = rx.replace(r"\*", "[^/]*")
    rx = rx.replace(r"\?", ".")
    return f"^{rx}$"

EXCLUDE_PATTERNS = [re.compile(glob_to_regex(g)) for g in EXCLUDE_GLOBS]

def is_excluded(path: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat.search(path):
            return True
    return False

def chunk_text(s: str, max_chars: int = 24000) -> List[str]:
    s = s.strip()
    if len(s) <= max_chars:
        return [s]
    chunks, i = [], 0
    while i < len(s):
        j = min(i + max_chars, len(s))
        # try to split at patch/file boundary for nicer chunks
        k = s.rfind("\ndiff --git ", i, j)
        if k == -1 or j == len(s):
            k = j
        chunks.append(s[i:k])
        i = k
    return chunks

def read_yaml_if_exists(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

# --------------------------
# GitHub API
# --------------------------

def gh_headers(token: str, accept: Optional[str] = None) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    if accept:
        h["Accept"] = accept
    return h

def gh_get(url: str, token: str, accept: Optional[str] = None):
    r = requests.get(url, headers=gh_headers(token, accept))
    r.raise_for_status()
    return r

def gh_post(url: str, token: str, json_body: dict):
    r = requests.post(url, headers=gh_headers(token, "application/json"), json=json_body)
    r.raise_for_status()
    return r

def gh_patch(url: str, token: str, json_body: dict):
    r = requests.patch(url, headers=gh_headers(token, "application/json"), json=json_body)
    r.raise_for_status()
    return r

# --------------------------
# LLM Providers
# --------------------------

def llm_summarize_openai(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 700) -> str:
    # Requires: pip install openai
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY or None)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "You are a precise, senior software engineer who writes terse, actionable PR summaries."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

def llm_summarize_ollama(prompt: str, model: str = OLLAMA_MODEL, max_tokens: int = 700) -> str:
    # Local fallback if you prefer self-hosting
    r = requests.post(
        f"{OLLAMA_ENDPOINT}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
        timeout=300
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

def llm_summarize(prompt: str, max_tokens=700) -> str:
    if LLM_PROVIDER == "ollama":
        return llm_summarize_ollama(prompt, max_tokens=max_tokens)
    return llm_summarize_openai(prompt, max_tokens=max_tokens)

# --------------------------
# Summarization logic
# --------------------------

SECTION_PROMPT = """Summarize the following pull request changes for reviewers.

Write {tone} and DO NOT paste long code. Produce Markdown with these sections:

### What changed
- 4–8 bullets with *high-level* intent (features, fixes, refactors)

### Risk & breaking considerations
- Note migrations, config/env changes, backward incompatibilities, security/privacy implications

### Test & verification hints
- How to verify manually; key unit/integration tests to add; edge cases to check

### Files of interest
- 5–12 most impactful files with one short note each

Context:
{context}

Diff / patches (chunk):
