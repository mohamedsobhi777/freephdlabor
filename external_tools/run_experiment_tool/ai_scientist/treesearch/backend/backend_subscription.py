import logging
import os
import time

from ai_scientist.llm import complete
from .utils import FunctionSpec, OutputType

logger = logging.getLogger("ai-scientist")


def query(
    system_message: str | list | None,
    user_message: str | list | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """Query a subscription-backed model (claude-code/* or codex/*) via its CLI."""
    model = model_kwargs["model"]

    # Many prompts are sent as a system message only; the CLIs need a user turn
    if user_message is None:
        system_message, user_message = None, system_message

    messages = [{"role": "user", "content": user_message or ""}]
    if isinstance(system_message, list):
        messages.insert(0, {"role": "system", "content": system_message})
        system_message = None

    t0 = time.time()
    result = complete(
        model,
        messages,
        system=system_message,
        json_schema=func_spec.json_schema if func_spec is not None else None,
        effort=os.environ.get("RUN_EXPERIMENT_REASONING_EFFORT") or None,
    )
    req_time = time.time() - t0

    if func_spec is None:
        output = result.text
    else:
        if result.structured is None:
            raise ValueError(f"{model} did not return valid JSON for {func_spec.name}: {result.text[:500]}")
        output = result.structured

    info = {"model": model, "created": int(time.time())}
    return output, req_time, result.input_tokens, result.output_tokens, info
