from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from dspy.utils.callback import BaseCallback


class TokenUsageCallback(BaseCallback):
    """Persist provider-reported LM usage without storing prompts or responses."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._request_count = 0
        self._reported_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._api_totals = dict(self._reported_totals)
        self._usage_unavailable_count = 0
        self._cache_hit_count = 0
        self._instances: dict[str, tuple[Any, Any]] = {}

    def on_lm_start(
        self, call_id: str, instance: Any, inputs: dict[str, Any]
    ) -> None:
        del inputs
        history = getattr(instance, "history", None)
        previous_entry = history[-1] if history else None
        with self._lock:
            self._instances[call_id] = (instance, previous_entry)

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        del outputs
        with self._lock:
            call = self._instances.pop(call_id, None)
        if exception is not None:
            return
        if call is None:
            return
        instance, previous_entry = call
        history = getattr(instance, "history", None)
        if history and history[-1] is not previous_entry:
            self.record_history_entry(history[-1])
        else:
            self.record_history_entry({"model": getattr(instance, "model", None)})

    @staticmethod
    def _token_count(usage: dict[str, Any], *keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    def record_history_entry(self, entry: dict[str, Any]) -> None:
        usage = entry.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = self._token_count(usage, "prompt_tokens", "input_tokens")
        completion_tokens = self._token_count(
            usage, "completion_tokens", "output_tokens"
        )
        total_tokens = self._token_count(usage, "total_tokens")
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        response = entry.get("response")
        cache_hit = bool(getattr(response, "cache_hit", False))
        request_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        with self._lock:
            self._request_count += 1
            if cache_hit:
                self._cache_hit_count += 1
            if total_tokens is None:
                self._usage_unavailable_count += 1

            for key, value in request_usage.items():
                if value is not None:
                    self._reported_totals[key] += value
                    if not cache_hit:
                        self._api_totals[key] += value

            record = {
                "event": "lm_usage",
                "request_index": self._request_count,
                "model": entry.get("model"),
                "cache_hit": cache_hit,
                "usage_available": total_tokens is not None,
                **request_usage,
                "reported_cumulative": dict(self._reported_totals),
                "api_cumulative_excluding_cache_hits": dict(self._api_totals),
            }
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, sort_keys=True) + "\n")

    def write_summary(self) -> None:
        with self._lock:
            record = {
                "event": "lm_usage_summary",
                "request_count": self._request_count,
                "cache_hit_count": self._cache_hit_count,
                "usage_unavailable_count": self._usage_unavailable_count,
                "reported_token_totals": dict(self._reported_totals),
                "api_token_totals_excluding_cache_hits": dict(self._api_totals),
            }
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, sort_keys=True) + "\n")
