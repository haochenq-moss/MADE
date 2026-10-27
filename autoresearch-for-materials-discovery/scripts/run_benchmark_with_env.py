from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv


def install_pymatgen_compatibility() -> None:
    if importlib.util.find_spec("pymatgen.core.entries") is None:
        from pymatgen.entries import computed_entries

        sys.modules["pymatgen.core.entries"] = computed_entries


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: run_benchmark_with_env.py ENV_FILE RUNNER [ARGS...]")

    env_file = Path(sys.argv[1])
    runner = Path(sys.argv[2]).resolve()
    if not runner.is_file():
        raise SystemExit(f"Benchmark runner does not exist: {runner}")

    if env_file.is_file():
        load_dotenv(env_file, override=False)
    llm_model = os.environ.get("FARS_LLM_MODEL", "openai/gpt-5.1")
    provider = llm_model.partition("/")[0]
    provider_key = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    if provider_key and not os.environ.get(provider_key):
        raise SystemExit(f"{provider_key} is not available to the benchmark process.")

    install_pymatgen_compatibility()
    sys.argv = [str(runner), *sys.argv[3:]]
    token_callback = None
    token_usage_log = os.environ.get("FARS_TOKEN_USAGE_LOG")
    if token_usage_log:
        import dspy
        from autoresearch_for_materials_discovery.token_usage import (
            TokenUsageCallback,
        )

        token_callback = TokenUsageCallback(token_usage_log)
        callbacks = list(dspy.settings.callbacks or [])
        callbacks.append(token_callback)
        dspy.settings.configure(callbacks=callbacks)

    try:
        if os.environ.get("FARS_LOCAL_CHECKPOINTS") == "1":
            os.environ["HYDRA_FULL_ERROR"] = "1"
            import modal

            volume_class = modal.Volume
            original_commit = volume_class.commit

            def local_commit(*args, **kwargs):
                return None

            volume_class.commit = local_commit
            try:
                print(
                    "Remote Modal checkpoint commits disabled; local checkpoints remain enabled."
                )
                runpy.run_path(str(runner), run_name="__main__")
            finally:
                volume_class.commit = original_commit
        else:
            runpy.run_path(str(runner), run_name="__main__")
    finally:
        if token_callback is not None:
            token_callback.write_summary()


if __name__ == "__main__":
    main()
