"""
Smoke tests for the Claude Code / Codex subscription backends.

Levels (run from the repo root inside the freephdlabor conda env):
    python scripts/test_subscription_backends.py quick        # ~1-2 min: raw CLI adapter
    python scripts/test_subscription_backends.py agents       # ~2-4 min: smolagents CodeAgent + AI-Scientist queries
    python scripts/test_subscription_backends.py experiment   # ~15-45 min: real RunExperimentTool run (stage 1, small budget)

Models default to .llm_config.yaml (main_agents.model for the agent, run_experiment_tool.code_model
for AI-Scientist). Override with --agent-model / --exp-model. API keys are removed from the environment
so a pass proves nothing is billed to API credits.
"""

import argparse
import base64
import json
import os
import struct
import sys
import time
import zlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "external_tools", "run_experiment_tool"))

for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(var, None)

import yaml

from freephdlabor.subscription_llm import complete, is_subscription_model

results = []


def check(name, fn):
    start = time.time()
    try:
        detail = fn()
        results.append((name, True, f"{detail}  ({time.time() - start:.1f}s)"))
        print(f"  PASS  {name}: {detail}  ({time.time() - start:.1f}s)")
    except Exception as e:
        results.append((name, False, repr(e)[:500]))
        print(f"  FAIL  {name}: {repr(e)[:500]}")


def red_png_data_url():
    width = height = 64
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    # Deliberately labelled as JPEG, like several AI-Scientist call sites do
    return "data:image/jpeg;base64," + base64.b64encode(png).decode()


def test_quick(model):
    print(f"\n[quick] adapter checks for {model}")

    def text():
        r = complete(model, [{"role": "user", "content": "What is 17*23? Reply with only the number."}], system="You are terse.")
        assert "391" in r.text, r.text
        return f"'{r.text.strip()}' (in={r.input_tokens}, out={r.output_tokens})"

    def multi_turn():
        r = complete(model, [
            {"role": "user", "content": "My favourite number is 42."},
            {"role": "assistant", "content": "Noted!"},
            {"role": "user", "content": "What is my favourite number? Reply with only the number."},
        ])
        assert "42" in r.text, r.text
        return r.text.strip()

    def stop_sequence():
        r = complete(model, [{"role": "user", "content": "Write 'Hello' then on a new line 'Observation: done'"}],
                     stop_sequences=["Observation:"])
        assert "Observation:" not in r.text, r.text
        return repr(r.text.strip())

    def image_and_schema():
        schema = {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]}
        r = complete(model, [{"role": "user", "content": [
            {"type": "text", "text": "What is the dominant color of this image?"},
            {"type": "image_url", "image_url": {"url": red_png_data_url()}},
        ]}], json_schema=schema)
        assert r.structured and "red" in r.structured["color"].lower(), r.text
        return r.structured

    check(f"{model} text", text)
    check(f"{model} multi-turn", multi_turn)
    check(f"{model} stop sequence", stop_sequence)
    check(f"{model} image + JSON schema", image_and_schema)


def test_agents(agent_model, exp_model):
    print(f"\n[agents] agent={agent_model}, experiment={exp_model}")
    os.environ["RUN_EXPERIMENT_REPORT_MODEL"] = exp_model

    def code_agent():
        from smolagents import CodeAgent
        from freephdlabor.utils import create_model

        agent = CodeAgent(tools=[], model=create_model(agent_model, reasoning_effort="low"),
                          code_block_tags="markdown", max_steps=4, verbosity_level=0)
        answer = agent.run("Compute the sum of the squares of the integers 1..20 with Python and return it via final_answer.")
        assert str(answer).strip() == "2870", answer
        return answer

    def treesearch_text():
        from ai_scientist.treesearch.backend import query

        out = query(system_message={"Task": "Name one sorting algorithm. One word only."}, user_message=None, model=exp_model)
        assert isinstance(out, str) and out.strip(), out
        return out.strip()

    def treesearch_function_call():
        from ai_scientist.treesearch.backend import query
        from ai_scientist.treesearch.backend.utils import FunctionSpec

        spec = FunctionSpec(
            name="submit_review",
            json_schema={"type": "object",
                         "properties": {"is_bug": {"type": "boolean"}, "summary": {"type": "string"}},
                         "required": ["is_bug", "summary"]},
            description="Submit a review of the execution output",
        )
        out = query(system_message="You review experiment logs.",
                    user_message="Log: Traceback ... ZeroDivisionError: division by zero",
                    func_spec=spec, model=exp_model)
        assert isinstance(out, dict) and out["is_bug"] is True, out
        return out

    def log_summarizer_import():
        import ai_scientist.treesearch.log_summarization as ls

        assert ls.model == exp_model, ls.model
        return f"uses {ls.model} with no OpenAI key"

    check("smolagents CodeAgent", code_agent)
    check("AI-Scientist query (text)", treesearch_text)
    check("AI-Scientist query (function call)", treesearch_function_call)
    check("AI-Scientist log summarizer", log_summarizer_import)


def test_experiment(exp_model, effort):
    print(f"\n[experiment] RunExperimentTool, stage 1 only, POC budget, model={exp_model}")
    import freephdlabor.utils  # noqa: F401  # import order matters: toolkits <-> utils are circular, as in launch_multiagent.py
    from freephdlabor.toolkits.run_experiment_tool import RunExperimentTool

    os.environ["POC_MODE"] = "true"
    os.environ["RUN_EXPERIMENT_REASONING_EFFORT"] = effort
    workspace = os.path.join(REPO_ROOT, "results", time.strftime("%Y%m%d_%H%M%S") + "_subscription_experiment_test")
    os.makedirs(workspace, exist_ok=True)
    idea = {
        "Name": "label_smoothing_digits",
        "Title": "Does Label Smoothing Improve Calibration of a Small MLP on sklearn Digits?",
        "Short Hypothesis": "Label smoothing (0.1) reduces expected calibration error of a small MLP on the "
                            "sklearn digits dataset without hurting accuracy.",
        "Related Work": "Muller et al. 2019, When Does Label Smoothing Help?",
        "Abstract": "We train a 2-layer MLP on sklearn's bundled digits dataset with and without label smoothing "
                    "and compare accuracy and expected calibration error.",
        "Experiments": "Train a 2-layer MLP (PyTorch) on sklearn.datasets.load_digits (80/20 split) for 20 epochs, "
                       "with label smoothing 0.0 and 0.1. Report validation accuracy and ECE (15 bins). "
                       "Use only the bundled sklearn dataset; no downloads. Keep runtime under 2 minutes.",
        "Risk Factors and Limitations": "Tiny dataset; differences may be within noise.",
    }

    def run():
        tool = RunExperimentTool(workspace_dir=workspace)
        out = json.loads(tool.forward(json.dumps([idea]), code_model=exp_model, feedback_model=exp_model,
                                      vlm_model=exp_model, report_model=exp_model, end_stage=1))
        assert out.get("status") != "failure", json.dumps(out)[:800]
        return f"status={out.get('status')}, results in {workspace}"

    check("RunExperimentTool stage 1", run)


def main():
    with open(os.path.join(REPO_ROOT, ".llm_config.yaml")) as f:
        config = yaml.safe_load(f) or {}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("levels", nargs="*", default=["quick", "agents"], choices=["quick", "agents", "experiment"])
    parser.add_argument("--agent-model", default=config.get("main_agents", {}).get("model", "claude-code"))
    parser.add_argument("--exp-model", default=config.get("run_experiment_tool", {}).get("code_model", "codex"))
    args = parser.parse_args()

    for model in (args.agent_model, args.exp_model):
        if not is_subscription_model(model):
            parser.error(f"{model} is not a subscription model (expected claude-code[/...] or codex[/...])")

    if "quick" in args.levels:
        for model in dict.fromkeys([args.agent_model, args.exp_model]):
            test_quick(model)
    if "agents" in args.levels:
        test_agents(args.agent_model, args.exp_model)
    if "experiment" in args.levels:
        test_experiment(args.exp_model, config.get("run_experiment_tool", {}).get("reasoning_effort", "high"))

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
