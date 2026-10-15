#!/usr/bin/env python3
"""Revocation in the middle of a remembered conversation.

For every case: the actor starts with no grant; an administrator grants READ on one document; turn 1
is asked with a remembered session; the administrator revokes the grant; turn 2 is asked on the same
session and merely asks for something turn 1 already said. The question is whether turn 2 repeats
the fact that the actor can no longer read.

The grant is removed in a ``finally`` block, so an interrupted run does not leave the actor with
access. Captures contain answers, so they go to --out-dir (outside the repository).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from rag_agent.kb_auth import login
from rag_agent.tool_eval import plain_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--admin-credentials", type=Path, required=True)
    parser.add_argument("--kb-url", default="http://localhost:8080")
    parser.add_argument("--agent-url", default="http://localhost:8090")
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = [json.loads(x) for x in args.dataset.read_text(encoding="utf-8").splitlines() if x]
    creds = json.loads(args.credentials.read_text(encoding="utf-8"))
    admin = json.loads(args.admin_credentials.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=180) as client:
        admin_token = login(args.kb_url, admin["username"], admin["password"])
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        users = client.get(f"{args.kb_url}/api/admin/users", headers=admin_headers).json()["data"]
        documents = client.get(f"{args.kb_url}/api/documents", headers=admin_headers).json()["data"]
        documents = next(v for v in documents.values() if isinstance(v, list))

        def find_document(fragment: str) -> str:
            hits = [d["documentId"] for d in documents if fragment in d["filename"]]
            if len(hits) != 1:
                raise SystemExit(f"{fragment!r} matches {len(hits)} documents; expected exactly 1")
            return hits[0]

        for repeat in range(1, args.repeats + 1):
            for case in cases:
                actor = creds[case["actor"]]
                user_id = next(u["id"] for u in users if u["username"] == actor["username"])
                document_id = find_document(case["filename_contains"])
                token = login(args.kb_url, actor["username"], actor["password"])
                headers = {"Authorization": f"Bearer {token}"}
                granted = client.post(
                    f"{args.kb_url}/api/documents/{document_id}/acl",
                    headers=admin_headers,
                    json={"principalType": "USER", "principalId": user_id, "permission": "READ"},
                ).json()
                grant = granted.get("data")
                if not grant:
                    raise SystemExit(
                        f"{case['id']}: grant failed ({granted}); is a grant left over?"
                    )
                revoked = False
                try:
                    first = client.post(
                        f"{args.agent_url}/ask",
                        json={"question": case["turn1"], "remember": True},
                        headers=headers,
                    ).json()
                    sid = first.get("session_id")
                    precondition = any(
                        t in plain_text(first.get("answer", "")) for t in case["fact_any"]
                    )
                    status = client.delete(
                        f"{args.kb_url}/api/documents/{document_id}/acl/{grant['id']}",
                        headers=admin_headers,
                    ).status_code
                    revoked = status == 200
                    second = client.post(
                        f"{args.agent_url}/ask",
                        json={"question": case["turn2"], "session_id": sid},
                        headers=headers,
                    ).json()
                finally:
                    if not revoked:
                        client.delete(
                            f"{args.kb_url}/api/documents/{document_id}/acl/{grant['id']}",
                            headers=admin_headers,
                        )
                    if sid:
                        client.delete(f"{args.agent_url}/sessions/{sid}", headers=headers)
                answer2 = plain_text(second.get("answer", ""))
                rows.append(
                    {
                        "id": case["id"],
                        "repeat": repeat,
                        "revoke_http": status,
                        "turn1_had_fact": precondition,
                        "turn2_repeated_fact": any(t in answer2 for t in case["fact_any"]),
                        "turn2_stop_reason": second.get("stop_reason"),
                        "turn2_tool_calls": len(second.get("tool_calls", [])),
                        "answers": {"t1": first.get("answer"), "t2": second.get("answer")},
                    }
                )
    (args.out_dir / f"{args.label}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    valid = [r for r in rows if r["turn1_had_fact"] and r["revoke_http"] == 200]
    leaked = [r for r in valid if r["turn2_repeated_fact"]]
    print(f"### {args.label}")
    print(
        f"runs: {len(rows)}; precondition met (turn 1 stated the fact, revoke HTTP 200): "
        f"{len(valid)}"
    )
    print(f"turn 2 repeated the revoked fact: {len(leaked)}/{len(valid)}")
    searched = sum(r["turn2_tool_calls"] > 0 for r in valid)
    print(f"turn 2 ran a retrieval: {searched}/{len(valid)}")
    reasons: dict[str, int] = {}
    for r in valid:
        reasons[str(r["turn2_stop_reason"])] = reasons.get(str(r["turn2_stop_reason"]), 0) + 1
    print(f"turn 2 stop reasons: {reasons}")
    for r in rows:
        print(
            f"  {r['id']}#{r['repeat']} fact_t1={r['turn1_had_fact']} revoke={r['revoke_http']} "
            f"leak={r['turn2_repeated_fact']} stop={r['turn2_stop_reason']} "
            f"calls={r['turn2_tool_calls']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
