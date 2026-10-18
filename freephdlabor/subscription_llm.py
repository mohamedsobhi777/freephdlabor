"""
Subscription-backed LLM calls via the Claude Code and Codex CLIs.

Instead of calling the Anthropic/OpenAI APIs with an API key, these backends
shell out to `claude -p` (Claude Pro/Max login) or `codex exec` (ChatGPT login),
so usage is billed against your subscription rather than API credits.

Model names:
    claude-code              -> Claude Code's default model
    claude-code/<model>      -> e.g. claude-code/opus, claude-code/claude-opus-5-5
    codex                    -> Codex's default model
    codex/<model>            -> e.g. codex/gpt-5.6

This module only depends on the standard library so the AI-Scientist subprocess
(external_tools/run_experiment_tool) can import it too.
"""

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

SUBSCRIPTION_BACKENDS = ("claude-code", "codex")

# Env vars that would make the CLIs bill an API key instead of the subscription login
_API_KEY_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CLAUDECODE",
)

_USAGE_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "rate_limit",
    "limit reached",
    "hit your limit",
    "too many requests",
    "429",
    "overloaded",
)


class SubscriptionLLMError(RuntimeError):
    pass


@dataclass
class CompletionResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    structured: Optional[Any] = None
    raw: dict = field(default_factory=dict)


def is_subscription_model(model: Optional[str]) -> bool:
    if not isinstance(model, str):
        return False
    return any(model == b or model.startswith(b + "/") for b in SUBSCRIPTION_BACKENDS)


def parse_subscription_model(model: str) -> tuple[str, Optional[str]]:
    """Split 'codex/gpt-5.6' into ('codex', 'gpt-5.6'); bare backend names give (backend, None)."""
    backend, _, name = model.partition("/")
    if backend not in SUBSCRIPTION_BACKENDS:
        raise ValueError(f"Not a subscription model: {model}")
    return backend, name or None


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------

def _content_to_parts(content: Any) -> list[dict]:
    """Normalise OpenAI/Anthropic/smolagents content into [{'type':'text'|'image', ...}]."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, dict):
        content = [content]
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif kind == "image_url":
            url = item["image_url"]["url"] if isinstance(item["image_url"], dict) else item["image_url"]
            parts.append(_image_part_from_url(url))
        elif kind == "image" and isinstance(item.get("source"), dict):
            src = item["source"]
            parts.append({"type": "image", "media_type": _sniff_media_type(src["data"], src.get("media_type", "image/png")), "data": src["data"]})
        elif kind == "image" and "image" in item:
            parts.append(_image_part_from_pil(item["image"]))
        else:
            parts.append({"type": "text", "text": json.dumps(item, default=str)})
    return parts


_IMAGE_SIGNATURES = (
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
    (b"RIFF", "image/webp"),
)


def _sniff_media_type(data_b64: str, declared: str) -> str:
    """Callers often label PNGs as image/jpeg; Claude rejects mismatches, so trust the bytes."""
    head = base64.b64decode(data_b64[:16] + "=" * (-len(data_b64[:16]) % 4))
    for signature, media_type in _IMAGE_SIGNATURES:
        if head.startswith(signature):
            return media_type
    return declared


def _image_part_from_url(url: str) -> dict:
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        media_type = header[5:].split(";")[0] or "image/png"
        return {"type": "image", "media_type": _sniff_media_type(data, media_type), "data": data}
    if os.path.exists(url):
        with open(url, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return {"type": "image", "media_type": _sniff_media_type(data, "image/png"), "data": data}
    return {"type": "text", "text": f"[image: {url}]"}


def _image_part_from_pil(image: Any) -> dict:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}


def _role_of(message: Any) -> str:
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "user")
    return getattr(role, "value", role) or "user"


def _content_of(message: Any) -> Any:
    return message.get("content") if isinstance(message, dict) else getattr(message, "content", None)


def _build_prompt(messages: list, system: Optional[str]) -> tuple[str, list[dict]]:
    """
    Flatten a chat history into (system_prompt, parts) for a single-turn CLI call.

    The CLIs only accept one user turn, so multi-turn histories are rendered as a
    transcript and the model is asked to write the next assistant message.
    """
    system_chunks = [system] if system else []
    turns = []
    for message in messages:
        role = _role_of(message)
        parts = _content_to_parts(_content_of(message))
        if role == "system":
            system_chunks.extend(p["text"] for p in parts if p["type"] == "text")
        else:
            turns.append(("assistant" if role == "assistant" else "user", parts))

    if len(turns) == 1 and turns[0][0] == "user":
        return "\n\n".join(system_chunks), turns[0][1]

    parts: list[dict] = [{
        "type": "text",
        "text": "Below is the conversation so far. Write ONLY the next assistant message, "
                "continuing directly from it. Do not repeat earlier messages or role headers.\n",
    }]
    for role, turn_parts in turns:
        parts.append({"type": "text", "text": f"\n<<<{role.upper()}>>>\n"})
        parts.extend(turn_parts)
    parts.append({"type": "text", "text": "\n<<<ASSISTANT>>>\n"})
    return "\n\n".join(system_chunks), parts


def _clean_env() -> dict:
    env = os.environ.copy()
    for var in _API_KEY_ENV_VARS:
        env.pop(var, None)
    return env


def _is_usage_limit(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _USAGE_LIMIT_MARKERS)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _run_claude(model, system, parts, json_schema, effort, timeout) -> CompletionResult:
    exe = shutil.which(os.environ.get("CLAUDE_CODE_CLI", "claude"))
    if not exe:
        raise SubscriptionLLMError("Claude Code CLI not found; install it and run `claude` once to log in.")

    content = []
    for p in parts:
        if p["type"] == "text":
            content.append({"type": "text", "text": p["text"]})
        else:
            content.append({"type": "image", "source": {"type": "base64", "media_type": p["media_type"], "data": p["data"]}})
    stdin = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"

    with tempfile.TemporaryDirectory(prefix="freephdlabor_claude_") as tmp:
        cmd = [
            exe, "-p",
            "--tools", "",
            "--safe-mode",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
        ]
        if system:
            system_file = os.path.join(tmp, "system.txt")
            with open(system_file, "w") as f:
                f.write(system)
            cmd += ["--system-prompt-file", system_file]
        if model:
            cmd += ["--model", model]
        if effort:
            cmd += ["--effort", effort]
        if json_schema:
            cmd += ["--json-schema", json.dumps(json_schema)]

        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                              cwd=tmp, env=_clean_env(), timeout=timeout)

    result = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    if result is None or result.get("is_error") or proc.returncode != 0:
        detail = (result or {}).get("result") or proc.stderr.strip() or proc.stdout.strip()[-2000:]
        raise SubscriptionLLMError(f"claude CLI failed (rc={proc.returncode}): {detail}")

    usage = result.get("usage", {})
    return CompletionResult(
        text=result.get("result") or "",
        input_tokens=usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        structured=result.get("structured_output"),
        raw=result,
    )


def _run_codex(model, system, parts, json_schema, effort, timeout) -> CompletionResult:
    exe = shutil.which(os.environ.get("CODEX_CLI", "codex"))
    if not exe:
        raise SubscriptionLLMError("Codex CLI not found; install it and run `codex login`.")

    with tempfile.TemporaryDirectory(prefix="freephdlabor_codex_") as tmp:
        workdir = os.path.join(tmp, "work")
        os.makedirs(workdir)
        out_file = os.path.join(tmp, "last_message.txt")

        text_chunks = []
        if system:
            text_chunks.append(f"<system_instructions>\n{system}\n</system_instructions>\n")
        text_chunks.append(
            "Answer directly from your own knowledge and the content below. "
            "Do not run shell commands, read files, or edit anything.\n"
        )
        cmd = [
            exe, "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox", "read-only",
            "--cd", workdir,
            "--json",
            "--output-last-message", out_file,
        ]
        image_count = 0
        for p in parts:
            if p["type"] == "text":
                text_chunks.append(p["text"])
            else:
                image_count += 1
                ext = p["media_type"].split("/")[-1].replace("jpeg", "jpg")
                image_path = os.path.join(tmp, f"image_{image_count}.{ext}")
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(p["data"]))
                cmd += ["--image", image_path]
                text_chunks.append(f"[attached image {image_count}]")
        if model:
            cmd += ["--model", model]
        if effort:
            cmd += ["-c", f'model_reasoning_effort="{effort}"']
        if json_schema:
            schema_file = os.path.join(tmp, "schema.json")
            with open(schema_file, "w") as f:
                json.dump(_strict_schema(json_schema), f)
            cmd += ["--output-schema", schema_file]
        cmd.append("-")

        proc = subprocess.run(cmd, input="".join(text_chunks), capture_output=True, text=True,
                              cwd=workdir, env=_clean_env(), timeout=timeout)

        text = ""
        if os.path.exists(out_file):
            with open(out_file) as f:
                text = f.read()

    usage, error = {}, None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage", {})
        elif event.get("type") in ("error", "turn.failed"):
            error = event.get("message") or json.dumps(event.get("error", event))
    if proc.returncode != 0 or error or not text:
        detail = error or proc.stderr.strip()[-2000:] or "empty response"
        raise SubscriptionLLMError(f"codex CLI failed (rc={proc.returncode}): {detail}")

    structured = None
    if json_schema:
        try:
            structured = json.loads(text)
        except json.JSONDecodeError:
            pass
    return CompletionResult(
        text=text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        structured=structured,
        raw=usage,
    )


def _strict_schema(schema: Any) -> Any:
    """Codex structured output requires additionalProperties=false and all properties required."""
    if isinstance(schema, list):
        return [_strict_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    schema = {k: _strict_schema(v) for k, v in schema.items()}
    if schema.get("type") == "object" and "properties" in schema:
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
    return schema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def complete(
    model: str,
    messages: list,
    system: Optional[str] = None,
    json_schema: Optional[dict] = None,
    stop_sequences: Optional[list[str]] = None,
    effort: Optional[str] = None,
) -> CompletionResult:
    """
    Run one completion through the subscription CLI named by `model`.

    `messages` may be OpenAI/Anthropic-style dicts or smolagents ChatMessages;
    image parts are forwarded to the CLI. Stop sequences are applied by truncating
    the output, since neither CLI supports them natively.

    Retries transient failures; when the subscription's usage limit is hit it keeps
    waiting (up to SUBSCRIPTION_LIMIT_MAX_WAIT seconds, default 6h) for the window to reset.
    """
    backend, cli_model = parse_subscription_model(model)
    system_prompt, parts = _build_prompt(messages, system)
    if effort is None:
        effort = os.environ.get("SUBSCRIPTION_EFFORT") or None
    # Claude Code accepts low|medium|high|xhigh|max; Codex accepts minimal|low|medium|high
    if backend == "claude-code" and effort == "minimal":
        effort = "low"
    elif backend == "codex" and effort in ("xhigh", "max"):
        effort = "high"
    timeout = int(os.environ.get("SUBSCRIPTION_CALL_TIMEOUT", "1800"))
    max_retries = int(os.environ.get("SUBSCRIPTION_MAX_RETRIES", "4"))
    limit_wait = int(os.environ.get("SUBSCRIPTION_LIMIT_POLL", "600"))
    limit_max_wait = int(os.environ.get("SUBSCRIPTION_LIMIT_MAX_WAIT", str(6 * 3600)))
    runner = _run_claude if backend == "claude-code" else _run_codex

    attempt, waited_for_limit = 0, 0
    while True:
        try:
            result = runner(cli_model, system_prompt, parts, json_schema, effort, timeout)
            break
        except (SubscriptionLLMError, subprocess.TimeoutExpired) as e:
            message = str(e)
            if _is_usage_limit(message) and waited_for_limit < limit_max_wait:
                print(f"⏳ {backend} usage limit reached; waiting {limit_wait}s before retrying ({message[:200]})")
                time.sleep(limit_wait)
                waited_for_limit += limit_wait
                continue
            attempt += 1
            if attempt > max_retries:
                raise
            delay = min(2 ** attempt * 5, 120)
            print(f"⚠️ {backend} call failed (attempt {attempt}/{max_retries}), retrying in {delay}s: {message[:300]}")
            time.sleep(delay)

    if stop_sequences and result.text:
        cut = min((i for i in (result.text.find(s) for s in stop_sequences if s) if i != -1), default=-1)
        if cut != -1:
            result.text = result.text[:cut]
    return result
