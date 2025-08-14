#!/usr/bin/env python3
import os, json, requests, argparse, textwrap, re, sys, math, yaml
from typing import List, Dict, Optional

# --------------------------
# Config & Helpers
# --------------------------

PR_SUMMARY_MARKER = "<!-- pr-summarizer-bot -->"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-5-mini")
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

""".strip()

REDUCE_PROMPT = """Combine partial PR summaries into a single, non-repetitive final summary.

Rules:
- Keep it crisp and reviewer-friendly
- Merge similar bullets, remove duplicates
- Keep the same section headings
- Ensure it stands alone without missing context

Partials:
{partials}
""".strip()

def summarize_map_reduce(context: str, diff_text: str) -> str:
    chunks = chunk_text(diff_text, max_chars=24000)
    partials = []
    for idx, ch in enumerate(chunks, 1):
        p = SECTION_PROMPT.format(tone=SUMMARY_TONE, context=context, chunk=ch)
        partial = llm_summarize(p, max_tokens=650)
        partials.append(f"--- Partial {idx} ---\n{partial}\n")
    if len(partials) == 1:
        return partials[0].split("--- Partial 1 ---\n",1)[-1].strip()
    combined = llm_summarize(REDUCE_PROMPT.format(partials="\n".join(partials)), max_tokens=700)
    return combined

# --------------------------
# Build PR context
# --------------------------

def build_context(pr: dict, commits: List[dict], files: List[dict]) -> str:
    title = pr.get("title","").strip()
    body = (pr.get("body") or "").strip()
    author = pr.get("user",{}).get("login","")
    labels = [l.get("name","") for l in pr.get("labels",[])]
    commit_msgs = [c.get("commit",{}).get("message","").strip() for c in commits]
    changed = []
    for f in files:
        filename = f.get("filename","")
        if is_excluded(filename):
            continue
        status = f.get("status","")
        additions = f.get("additions",0)
        deletions = f.get("deletions",0)
        changed.append(f"- {filename} ({status}, +{additions}/-{deletions})")
    ctx = textwrap.dedent(f"""
    PR: {title}
    Author: @{author}
    Labels: {", ".join(labels) if labels else "none"}

    PR description:
    {body if body else "(no description)"}

    Recent commits (head → base):
    {os.linesep.join(f"* {m}" for m in commit_msgs[:15])}

    Changed files:
    {os.linesep.join(changed[:60])}
    """).strip()
    return ctx

# --------------------------
# GH comment (sticky)
# --------------------------

def upsert_comment(owner: str, repo: str, pr_number: int, token: str, body_md: str):
    # Find existing bot comment
    comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments?per_page=100"
    resp = gh_get(comments_url, token)
    existing = None
    for c in resp.json():
        if c.get("body","").startswith(PR_SUMMARY_MARKER):
            existing = c
            break

    payload = {"body": f"{PR_SUMMARY_MARKER}\n{body_md}"}
    if existing:
        cid = existing["id"]
        gh_patch(f"https://api.github.com/repos/{owner}/{repo}/issues/comments/{cid}", token, payload)
    else:
        gh_post(comments_url, token, payload)

# --------------------------
# Main: GitHub mode or CLI mode
# --------------------------

def fetch_pr_data_from_env():
    token = os.getenv("GITHUB_TOKEN")
    repo_full = os.getenv("GITHUB_REPOSITORY")  # owner/repo
    event_path = os.getenv("GITHUB_EVENT_PATH")
    assert token and repo_full and event_path, "Missing GITHUB_* envs"

    with open(event_path, "r", encoding="utf-8") as f:
        event = json.load(f)
    pr_number = event.get("pull_request",{}).get("number")
    assert pr_number, "Not a pull_request event"
    owner, repo = repo_full.split("/", 1)

    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    pr = gh_get(pr_url, token).json()
    commits = gh_get(pr_url + "/commits?per_page=250", token).json()
    files = gh_get(pr_url + "/files?per_page=300", token).json()

    # Diff (unified)
    diff_resp = gh_get(pr_url, token, accept="application/vnd.github.v3.diff")
    diff_text = diff_resp.text

    # Filter out excluded files from diff, if any patterns were provided
    if EXCLUDE_PATTERNS:
        filtered_lines = []
        include = True
        for line in diff_text.splitlines(True):
            if line.startswith("diff --git "):
                # extract path after b/
                m = re.search(r" a/(.*?) b/(.*?)\n", line)
                path = m.group(2) if m else ""
                include = not is_excluded(path)
            if include:
                filtered_lines.append(line)
        diff_text = "".join(filtered_lines)

    # Budget input size
    if len(diff_text) > MAX_INPUT_CHARS:
        diff_text = diff_text[:MAX_INPUT_CHARS] + "\n... [truncated]\n"

    return owner, repo, pr_number, token, pr, commits, files, diff_text

def build_markdown_summary(final_summary: str) -> str:
    return f"""
## 🤖 PR Summary (auto-generated)
> Model: `{DEFAULT_MODEL}` | Tone: `{SUMMARY_TONE}`

{final_summary}

---
_This is an automated summary to assist reviewers. Please verify details and tests._
""".strip()

def run_github_mode():
    owner, repo, pr_number, token, pr, commits, files, diff_text = fetch_pr_data_from_env()
    context = build_context(pr, commits, files)
    final_summary = summarize_map_reduce(context, diff_text)
    md = build_markdown_summary(final_summary)
    upsert_comment(owner, repo, pr_number, token, md)

def run_cli_mode(args):
    # Reusable outside GitHub: pipe a diff or pass a file path.
    if args.diff_file and os.path.exists(args.diff_file):
        with open(args.diff_file, "r", encoding="utf-8", errors="ignore") as f:
            diff_text = f.read()
    else:
        diff_text = sys.stdin.read()

    if len(diff_text) > MAX_INPUT_CHARS:
        diff_text = diff_text[:MAX_INPUT_CHARS] + "\n... [truncated]\n"

    context = textwrap.dedent(f"""
    Source: CLI
    Note: This summary was generated from a raw unified diff provided to the tool.
    """).strip()
    final_summary = summarize_map_reduce(context, diff_text)
    print(build_markdown_summary(final_summary))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize PR diffs (GitHub Action or CLI).")
    parser.add_argument("--diff-file", help="Path to a unified diff (if using CLI mode).")
    args = parser.parse_args()

    try:
        if os.getenv("GITHUB_EVENT_PATH"):
            run_github_mode()
        else:
            run_cli_mode(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

