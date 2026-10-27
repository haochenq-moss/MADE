import json
from types import SimpleNamespace

from autoresearch_for_materials_discovery.token_usage import TokenUsageCallback


def test_token_usage_callback_records_provider_usage_without_payloads(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    callback = TokenUsageCallback(log_path)
    instance = SimpleNamespace(history=[], model="deepseek/deepseek-flash")

    callback.on_lm_start("call-1", instance, {"prompt": "private input"})
    instance.history.append(
        {
            "model": "deepseek/deepseek-flash",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
            "response": SimpleNamespace(cache_hit=False),
            "prompt": "must not be written",
        }
    )
    callback.on_lm_end("call-1", outputs={"response": "private output"})
    callback.write_summary()

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["total_tokens"] == 150
    assert records[0]["api_cumulative_excluding_cache_hits"]["total_tokens"] == 150
    assert records[1]["reported_token_totals"]["total_tokens"] == 150
    assert "private" not in log_path.read_text()


def test_token_usage_callback_separates_cached_usage(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    callback = TokenUsageCallback(log_path)
    instance = SimpleNamespace(history=[], model="deepseek/deepseek-flash")

    callback.on_lm_start("cached-call", instance, {})
    instance.history.append(
        {
            "model": "deepseek/deepseek-flash",
            "usage": {"input_tokens": 15, "output_tokens": 5},
            "response": SimpleNamespace(cache_hit=True),
        }
    )
    callback.on_lm_end("cached-call", outputs={})
    callback.write_summary()

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert records[0]["total_tokens"] == 20
    assert records[0]["cache_hit"] is True
    assert records[0]["api_cumulative_excluding_cache_hits"]["total_tokens"] == 0
    assert records[1]["cache_hit_count"] == 1
