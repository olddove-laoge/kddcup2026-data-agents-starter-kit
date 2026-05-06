from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Reasoning policy:
- Handle the task as a small DAG: decompose into subgoals, execute branches as needed, and merge only when branch outputs are schema-compatible.
- Before calling `answer`, perform a final consistency check over metric definition, filters, time window, unit/currency, and aggregation logic.
- If any conflict appears, do not answer yet; collect more evidence and resolve conflicts first.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.
8. `action_input` must always be a JSON object. Never pass `action_input` as a plain string.
9. For `execute_python`, use this exact shape: {"action":"execute_python","action_input":{"code":"..."}}.
10. For `execute_context_sql`, only use SQLite/DB files. Never use CSV/JSON with SQL execution.
11. If an observation reports `action_input must be a JSON object` or `file is not a database`, fix the action format/file type in the next step instead of repeating the same call.

Keep reasoning concise and grounded in the observed data.
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response for branch-style execution:
```json
{"thought":"I have identified the target region; now I should query the database branch.","action":"execute_context_sql","action_input":{"path":"database.sqlite","sql":"SELECT SUM(revenue) AS total_revenue FROM sales WHERE region = 'East Asia' AND category = 'Electronics'","limit":20}}
```

Example response for Python execution:
```json
{"thought":"I need cross-file processing, so I will run Python.","action":"execute_python","action_input":{"code":"import csv\nprint('ok')"}}
```

Example response when you have the final answer:
```json
{"thought":"I verified definitions, filters, and units; I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
