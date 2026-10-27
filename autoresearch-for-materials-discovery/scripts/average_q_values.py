#!/usr/bin/env python3
"""Average independent FARS Q-tables to initialize a continued campaign."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_q_table(path: Path) -> dict[str, dict[str, float]]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Q-table root must be a JSON object: {path}")

    table: dict[str, dict[str, float]] = {}
    for state, actions in raw.items():
        if not isinstance(state, str) or not isinstance(actions, dict):
            raise ValueError(f"Invalid Q-table state entry in {path}")
        converted: dict[str, float] = {}
        for action, value in actions.items():
            if action not in {"EXPLORE", "REFINE"}:
                raise ValueError(f"Unsupported action {action!r} in {path}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Q-value must be numeric in {path}")
            q_value = float(value)
            if not math.isfinite(q_value):
                raise ValueError(f"Q-value must be finite in {path}")
            converted[action] = q_value
        table[state] = converted
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("provide at least two Q-table inputs")

    tables = [load_q_table(path) for path in args.inputs]
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"EXPLORE": 0.0, "REFINE": 0.0}
    )
    for table in tables:
        for state, actions in table.items():
            for action, value in actions.items():
                totals[state][action] += value

    merged = {
        state: {
            action: round(value / len(tables), 12)
            for action, value in actions.items()
        }
        for state, actions in sorted(totals.items())
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "method": "arithmetic mean across independent Q-tables",
        "missing_state_action_value": 0.0,
        "num_input_tables": len(tables),
        "inputs": [str(path.resolve()) for path in args.inputs],
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(merged)} scheduler states from {len(tables)} Q-tables to {args.output}")


if __name__ == "__main__":
    main()
