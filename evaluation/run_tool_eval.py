#!/usr/bin/env python3
"""Run the tool-calling evaluation against a live knowledge base and a local model.

    run_tool_eval.py --dataset evaluation/datasets/tool_calls_v1.jsonl --credentials creds.json \
        --kb-url http://localhost:8080 --label gemma --repeats 3 --out-dir /tmp/tool-eval

Per-run captures contain answers, so they are written to --out-dir (outside the repository) and
only the aggregate is printed.

Repeating inside one model-server session at temperature 0 mostly re-measures the same decoding,
so a zero spread does not show stability. Restart the model server between runs (and say so)
before treating a spread of zero as evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag_agent.agent import Limits, run_agent
from rag_agent.graph import run_graph_agent
from rag_agent.kb_auth import login
from rag_agent.llm import ChatClient
from rag_agent.mcp_client import McpClient
from rag_agent.tool_eval import score_case


def run_once(cases, kb_url, creds, chat, limits, policy="single", engine="loop"):
    rows = []
    for case in cases:
        c = creds[case["actor"]]
        started = time.monotonic()
        with McpClient(kb_url, login(kb_url, c["username"], c["password"])) as kb:
            runner = run_graph_agent if engine == "graph" else run_agent
            result = runner(case["question"], kb, chat, limits, policy)
        failures = score_case(case["expect"], result)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": not failures,
                "failures": failures,
                "stop_reason": result.stop_reason,
                "turns": result.turns,
                "trace": [[t.name, t.outcome] for t in result.trace],
                "tool_calls": len(result.trace),
                "answer": result.answer,
                "seconds": round(time.monotonic() - started, 2),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--kb-url", default="http://localhost:8080")
    parser.add_argument("--llm-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="google/gemma-4-26b-a4b-qat")
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=["single", "decompose"], default="single")
    parser.add_argument("--engine", choices=["loop", "graph"], default="loop")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument(
        "--no-empty-retry",
        action="store_true",
        help="disable the single changed retry after an empty model reply (for A/B comparison)",
    )
    args = parser.parse_args()

    cases = [
        json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line
    ]
    creds = json.loads(args.credentials.read_text(encoding="utf-8"))
    chat = ChatClient(base_url=args.llm_base_url, model=args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_run = []
    for run in range(1, args.repeats + 1):
        limits = Limits(
            max_turns=args.max_turns,
            max_tool_calls=args.max_tool_calls,
            retry_empty_reply=not args.no_empty_retry,
        )
        rows = run_once(cases, args.kb_url, creds, chat, limits, args.policy, args.engine)
        (args.out_dir / f"{args.label}.run{run}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        per_run.append(rows)
        print(
            f"run {run}/{args.repeats}: {sum(r['passed'] for r in rows)}/{len(rows)} passed",
            flush=True,
        )

    totals = [sum(r["passed"] for r in rows) for rows in per_run]
    by_case = defaultdict(list)
    by_category = defaultdict(lambda: [0, 0])
    for rows in per_run:
        for r in rows:
            by_case[r["id"]].append(r["passed"])
            by_category[r["category"]][0] += r["passed"]
            by_category[r["category"]][1] += 1
    always_failing = sorted(i for i, v in by_case.items() if not any(v))
    flaky = sorted(i for i, v in by_case.items() if any(v) and not all(v))
    stops = Counter(r["stop_reason"] for rows in per_run for r in rows)
    tool_outcomes = Counter(o for rows in per_run for r in rows for _, o in r["trace"])

    print(f"\n### {args.label}\n")
    retry = "off" if args.no_empty_retry else "on"
    print(
        f"Dataset `{args.dataset.name}`, {len(cases)} cases, {args.repeats} run(s), "
        f"empty-reply retry {retry}, policy {args.policy}, engine {args.engine}, "
        f"limits {args.max_turns} turns / {args.max_tool_calls} tool calls."
    )
    mean = statistics.mean(totals)
    spread = f"{min(totals)}–{max(totals)}"
    print(f"\n**Passed: mean {mean:.1f}/{len(cases)}, range {spread}.**")
    print("\n| Category | Passed |\n|---|---|")
    for cat, (ok, n) in sorted(by_category.items()):
        print(f"| {cat} | {ok}/{n} |")
    print(f"\nAlways failing ({len(always_failing)}): {', '.join(always_failing) or 'none'}")
    print(f"Flaky ({len(flaky)}): {', '.join(flaky) or 'none'}")
    print(f"Stop reasons: {dict(stops)}")
    print(f"Tool-call outcomes: {dict(tool_outcomes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
