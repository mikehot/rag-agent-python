#!/usr/bin/env python3
"""Scene driver for the administrator half of the recording (see docs/DEMO_VIDEO.md, part B).

A new synthetic document is uploaded, an employee is asked about it before any grant, the grant
is made, the employee is asked again, the grant is revoked, and the employee is asked a third
time. The point to make on camera: the answer follows the permission immediately, because
permissions are enforced in the retrieval query and not remembered by the model.

    uv run python scripts/demo_admin.py \\
        --admin-credentials admin.json --credentials creds.json --pause

Like scripts/demo.py it never prints a token or a password. It uploads one small document and,
unless --keep is given, deletes it at the end. Use a disposable knowledge base.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import httpx
from demo_captions import Pacer, check_thinking_off

from rag_agent.kb_auth import login
from rag_agent.mcp_client import McpClient

BOLD, DIM, GREEN, RED, CYAN, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[31m",
    "\033[36m",
    "\033[0m",
)

FILENAME = "demo-travel-policy.md"
EMPLOYEE = "corp.outsider"
QUESTION = "出差住宿的报销标准是多少？"
DOCUMENT = """# 差旅报销制度（演示用合成文档）

本文档是演示用的合成数据，不对应任何真实公司。

## 住宿标准

出差住宿费用按城市级别报销：一线城市每晚不超过 600 元，其他城市每晚不超过 400 元。
超出标准的部分需要部门负责人书面批准，否则不予报销。

## 提交期限

出差结束后十个工作日内提交报销申请，逾期需要说明原因。

## 餐饮补贴

出差期间餐饮补贴每天 100 元，不需要提供发票。
"""


class Admin:
    def __init__(
        self, kb_url: str, agent_url: str, admin: dict, creds: dict, pause: bool, pacer: Pacer
    ) -> None:
        self.kb, self.agent, self.pause = kb_url, agent_url, pause
        self.pacer = pacer
        self.admin_token = login(kb_url, admin["username"], admin["password"])
        self.employee_creds = creds[EMPLOYEE]
        self.http = httpx.Client(timeout=180)

    # -- presentation ----------------------------------------------------------------------
    def scene(self, label: str, title: str) -> None:
        self.pacer.begin()
        if self.pause and not self.pacer.auto:
            input(f"\n{DIM}[Enter] 进入下一幕{RESET}")
        print(f"\n{BOLD}{CYAN}━━ {label}：{title} ━━{RESET}")
        self.pacer.caption(label)

    def line(self, label: str, text: str, colour: str = "") -> None:
        body = textwrap.fill(str(text).strip(), width=78, subsequent_indent="      ")
        print(f"{BOLD}{label}{RESET}  {colour}{body}{RESET}")

    # -- calls ------------------------------------------------------------------------------
    def _admin(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.admin_token}"}

    def employee_token(self) -> str:
        c = self.employee_creds
        return login(self.kb, c["username"], c["password"])

    def employee_user_id(self) -> str:
        users = self.http.get(f"{self.kb}/api/admin/users", headers=self._admin()).json()["data"]
        return next(u["id"] for u in users if u["username"] == self.employee_creds["username"])

    def visible_documents(self) -> list[str]:
        with McpClient(self.kb, self.employee_token()) as client:
            items = client.call_tool("list_documents", {}).structured["items"]
        return sorted(item["filename"] for item in items)

    def ask(self) -> dict:
        started = time.monotonic()
        response = self.http.post(
            f"{self.agent}/ask",
            json={"question": QUESTION},
            headers={"Authorization": f"Bearer {self.employee_token()}"},
        )
        body = response.json()
        body["_seconds"] = time.monotonic() - started
        return body

    def show_employee(self) -> None:
        visible = self.visible_documents()
        print(f"{BOLD}员工{RESET}  {EMPLOYEE}  {DIM}（新员工）{RESET}")
        print(f"  能看到的文档：{visible or '（无）'}")
        self.line("问题", QUESTION)
        body = self.ask()
        self.line("回答", body["answer"], GREEN if body["cited_documents"] else DIM)
        named = body["cited_documents"] or "无"
        print(
            f"{DIM}  检索到片段：{len(body['retrieved'])} 个；回答里写出的文件名：{named}；"
            f"耗时 {body['_seconds']:.1f}s{RESET}"
        )

    # -- scenes -----------------------------------------------------------------------------
    def documents_named(self, filename: str) -> list[str]:
        response = self.http.get(f"{self.kb}/api/documents", headers=self._admin())
        data = response.json().get("data") or []
        items = data.get("items", []) if isinstance(data, dict) else data
        return [d["documentId"] for d in items if d.get("filename") == filename]

    def remove_leftovers(self) -> int:
        """Delete demo documents left behind by an interrupted earlier run (same file name)."""
        stale = self.documents_named(FILENAME)
        for document_id in stale:
            self.http.delete(f"{self.kb}/api/documents/{document_id}", headers=self._admin())
        return len(stale)

    def run(self, keep: bool) -> None:
        stale = self.remove_leftovers()
        if stale:
            print(
                f"(removed {stale} leftover demo document(s) from an earlier run)", file=sys.stderr
            )
        self._document_id: str | None = None
        try:
            self._scenes(keep)
        finally:
            # An error or Ctrl+C part-way must not leave a granted document behind.
            if self._document_id is not None and not keep:
                self.http.delete(
                    f"{self.kb}/api/documents/{self._document_id}", headers=self._admin()
                )

    def _scenes(self, keep: bool) -> None:
        self.scene("B1", "管理员上传一份新文档，等待索引完成")
        response = self.http.post(
            f"{self.kb}/api/documents/upload",
            headers=self._admin(),
            files={"file": (FILENAME, DOCUMENT.encode("utf-8"), "text/markdown")},
        )
        document_id = response.json()["data"]["documentId"]
        self._document_id = document_id
        print(
            f"上传 {BOLD}{FILENAME}{RESET}（{len(DOCUMENT.encode('utf-8'))} 字节），"
            f"documentId {document_id[:8]}…"
        )
        status, chunks = "processing", 0
        for _ in range(60):
            data = self.http.get(
                f"{self.kb}/api/documents/{document_id}", headers=self._admin()
            ).json()["data"]
            status, chunks = data["status"], data.get("chunkCount", 0)
            if status in {"ready", "failed"}:
                break
            time.sleep(1)
        colour = GREEN if status == "ready" else RED
        print(f"索引状态：{colour}{status}{RESET}，切成 {chunks} 个片段")

        self.scene("B2", "授权之前：新员工问同一个问题")
        self.show_employee()

        self.scene("B3", "管理员给这位员工授予 READ")
        user_id = self.employee_user_id()
        granted = self.http.post(
            f"{self.kb}/api/documents/{document_id}/acl",
            headers=self._admin(),
            json={"principalType": "USER", "principalId": user_id, "permission": "READ"},
        )
        grant = (granted.json() or {}).get("data")
        if not grant:
            raise SystemExit(
                f"the grant did not return an entry (HTTP {granted.status_code}); "
                "is there a leftover grant for this employee? Re-run the driver."
            )
        print(
            f"授权：principalType={grant['principalType']}  permission={grant['permission']}  "
            f"aclId {grant['id'][:8]}…"
        )

        self.scene("B4", "授权之后：同一个问题，带出处的回答")
        self.show_employee()

        self.scene("B5", "管理员撤销授权：回答立刻跟着变")
        revoked = self.http.delete(
            f"{self.kb}/api/documents/{document_id}/acl/{grant['id']}", headers=self._admin()
        )
        print(f"撤权：HTTP {revoked.status_code}")
        self.show_employee()
        note = "权限是在检索的 SQL 里过滤的；这里每次提问都不带会话，模型没有“记住”这份文档的机会。"
        print(f"{DIM}  {note}{RESET}")

        if not keep:
            self.scene("B6", "清理：删除这份演示文档")
            deleted = self.http.delete(
                f"{self.kb}/api/documents/{document_id}", headers=self._admin()
            )
            print(f"删除：HTTP {deleted.status_code}")
            self._document_id = None  # already removed; nothing for `finally` to do


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-credentials", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--kb-url", default="http://localhost:8080")
    parser.add_argument("--agent-url", default="http://localhost:8090")
    parser.add_argument("--pause", action="store_true", help="wait for Enter between scenes")
    parser.add_argument("--keep", action="store_true", help="do not delete the document at the end")
    parser.add_argument(
        "--captions", action="store_true", help="print a Chinese/English caption per scene"
    )
    parser.add_argument(
        "--auto", action="store_true", help="advance by itself after a reading delay"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="delay multiplier (0.2 for a rehearsal)"
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", help="do not check that Thinking is off"
    )
    args = parser.parse_args()
    admin = json.loads(args.admin_credentials.read_text(encoding="utf-8"))
    creds = json.loads(args.credentials.read_text(encoding="utf-8"))
    if not args.skip_preflight:
        check_thinking_off()
    pacer = Pacer(args.captions or args.auto, args.auto, args.speed)
    driver = Admin(
        args.kb_url.rstrip("/"), args.agent_url.rstrip("/"), admin, creds, args.pause, pacer
    )
    driver.run(args.keep)
    pacer.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
