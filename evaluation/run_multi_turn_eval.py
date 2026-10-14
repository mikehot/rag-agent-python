#!/usr/bin/env python3
"""Two-turn conversations against a running agent service (POST /ask with a remembered session).

For every conversation it asks turn 1 (remembered), then turn 2 with the session, and turn 2
again on its own as a control. It reports, per case, whether each answer passes the literal rules,
whether the session answer was produced with a retrieval, and whether turn 2's expected fact
already appeared in turn 1's answer (in which case answering without searching is not
unreasonable).

Captures contain answers, so they go to --out-dir (outside the repository).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from rag_agent.agent import AgentResult
from rag_agent.kb_auth import login
from rag_agent.tool_eval import plain_text, score_case


def ask(client: httpx.Client, url: str, token: str, body: dict) -> dict:
    response = client.post(f"{url}/ask", json=body, headers={"Authorization": f"Bearer {token}"})
    data = response.json()
    data["_status"] = response.status_code
    return data


def passed(expect: dict, data: dict) -> bool:
    result = AgentResult(answer=data.get("answer", ""), stop_reason=data.get("stop_reason", "x"))
    return not score_case(expect, result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--kb-url", default="http://localhost:8080")
    parser.add_argument("--agent-url", default="http://localhost:8090")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = [json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines() if x]
    creds = json.loads(args.credentials.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=180) as client:
        for case in cases:
            c = creds[case["actor"]]
            token = login(args.kb_url, c["username"], c["password"])
            t1, t2 = case["turns"]
            first = ask(
                client, args.agent_url, token, {"question": t1["question"], "remember": True}
            )
            sid = first.get("session_id")
            followed = ask(
                client, args.agent_url, token, {"question": t2["question"], "session_id": sid}
            )
            alone = ask(client, args.agent_url, token, {"question": t2["question"]})
            if sid:
                client.delete(
                    f"{args.agent_url}/sessions/{sid}", headers={"Authorization": f"Bearer {token}"}
                )
            terms = [t for g in t2["expect"].get("answer_all_of_any", []) for t in g]
            in_t1 = any(t in plain_text(first.get("answer", "")) for t in terms) if terms else None
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "turn1_ok": passed(t1.get("expect", {}), first) if t1.get("expect") else None,
                    "turn2_with_session_ok": passed(t2["expect"], followed),
                    "turn2_alone_ok": passed(t2["expect"], alone),
                    "turn2_searched": len(followed.get("tool_calls", [])) > 0,
                    "turn2_fact_already_in_turn1_answer": in_t1,
                    "answers": {
                        "t1": first.get("answer"),
                        "t2": followed.get("answer"),
                        "t2_alone": alone.get("answer"),
                    },
                }
            )
    (args.out_dir / f"{args.label}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    pos = [r for r in rows if r["category"] == "multi_turn_followup"]
    neg = [r for r in rows if r["category"] == "multi_turn_negative"]
    print(f"### {args.label}")
    print(f"follow-up conversations: {len(pos)}")
    print(f"  turn 2 with session : {sum(r['turn2_with_session_ok'] for r in pos)}/{len(pos)}")
    print(f"  turn 2 alone        : {sum(r['turn2_alone_ok'] for r in pos)}/{len(pos)}")
    print(f"  turn 2 ran a search : {sum(r['turn2_searched'] for r in pos)}/{len(pos)}")
    needs = [r for r in pos if r["turn2_fact_already_in_turn1_answer"] is False]
    searched = sum(r["turn2_searched"] for r in needs)
    correct = sum(r["turn2_with_session_ok"] for r in needs)
    print(f"  fact NOT in turn 1's answer (must retrieve): {len(needs)}")
    print(f"    searched in {searched}, correct in {correct}")
    held = sum(r["turn2_with_session_ok"] for r in neg)
    print(f"negative conversations: turn 2 declined with no leak {held}/{len(neg)}")
    for r in rows:
        flags = (
            f"session={r['turn2_with_session_ok']} alone={r['turn2_alone_ok']} "
            f"searched={r['turn2_searched']} in_t1={r['turn2_fact_already_in_turn1_answer']}"
        )
        print(f"  {r['id']} {flags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
