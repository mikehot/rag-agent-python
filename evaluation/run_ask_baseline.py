#!/usr/bin/env python3
"""Score the Java service's own POST /api/ask on a tool-call dataset's answer rules.

This is the baseline for "does the Python agent do better than the single-shot pipeline it sits
in front of". It sends only the question, with each case's actor's own token, and applies the same
literal rules as run_tool_eval.py (only the rules that make sense without tool calls: answer terms,
forbidden terms, declining). Captures go to --out-dir; only the aggregate is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from rag_agent.agent import AgentResult
from rag_agent.kb_auth import login
from rag_agent.tool_eval import score_case


def ask(base_url: str, token: str, question: str, timeout: float) -> str:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as http:
        response = http.post(
            "/api/ask", json={"question": question}, headers={"Authorization": f"Bearer {token}"}
        )
    data = (response.json() or {}).get("data") or {}
    return str(data.get("answer") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--kb-url", default="http://localhost:8080")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    cases = [json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines() if x]
    creds = json.loads(args.credentials.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        c = creds[case["actor"]]
        token = login(args.kb_url, c["username"], c["password"])
        answer = ask(args.kb_url, token, case["question"], args.timeout)
        # Rules that need a tool trace (must_call / no_tool / forbid_args) do not apply here.
        expect = {
            k: v
            for k, v in case["expect"].items()
            if k in {"answer_all_of_any", "answer_any", "forbid_terms", "no_answer"}
        }
        failures = score_case(expect, AgentResult(answer=answer, stop_reason="answered"))
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": not failures,
                "failures": failures,
                "answer": answer,
            }
        )
    (args.out_dir / f"{args.label}.run1.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    ok = sum(r["passed"] for r in rows)
    print(f"### {args.label} (Java POST /api/ask)\n\n**Passed: {ok}/{len(rows)}**")
    for r in rows:
        if not r["passed"]:
            print(f"- {r['id']}: {r['failures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
