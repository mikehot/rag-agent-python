"""On-screen captions and automatic pacing shared by the two demo drivers.

With ``--captions`` each scene prints a short explanation (Chinese and English) before its
result, so the recording needs no voice. With ``--auto`` the drivers advance by themselves after a
reading delay, so nobody has to press Enter at the right moment. ``--speed 0.2`` shortens every
delay for a rehearsal.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
import urllib.request

CAPTIONS: dict[str, tuple[str, str]] = {
    "A1": (
        "同一个知识库，四个人看到的文档数量不一样。这个限制是在检索的 SQL 里做的，不是靠提示词。",
        "One knowledge base, four people, four different views. The limit is enforced in the "
        "retrieval SQL, not in a prompt.",
    ),
    "A2": (
        "有权限的人得到带出处的回答。这里是 Python 智能体用这个人自己的身份，"
        "调用 Java 服务的只读工具。",
        "An authorized user gets a cited answer. The Python agent calls the Java tools with that "
        "user's own identity.",
    ),
    "A3": (
        "同一个问题，没有权限的人什么也拿不到：检索到 0 个片段，模型根本没见过这些内容。",
        "Same question, no grant: zero chunks retrieved, so the model never saw the content.",
    ),
    "A4": (
        "有权限的人问了不属于自己的主题，也是拒答。它检索到了片段，但模型判断它们回答不了。",
        "A permitted user asking about someone else's topic is declined too.",
    ),
    "A5": (
        "这是智能体层存在的理由：Java 单次检索回答不了的跨文档问题，智能体拆成两次检索再合并。",
        "Why the agent layer exists: a question that needs two documents, answered with two "
        "searches.",
    ),
    "A6": (
        "有人在问题里冒充管理员。身份只来自 token，工具的参数里根本没有 userId 这一项。",
        "Identity comes only from the token; the tool has no userId parameter to fill.",
    ),
    "A7": (
        "最后是证据，以及它不能证明的部分。这些数字不是产品指标。",
        "The evidence, and what it does not prove. These are not product metrics.",
    ),
    "B1": (
        "管理员上传一份新文档，系统解析、切片、向量化，状态变成 ready。",
        "The admin uploads a document; it is parsed, chunked and embedded, then marked ready.",
    ),
    "B2": (
        "文档已经就绪，但这位员工没有授权，所以什么也看不到，也问不出来。",
        "It is indexed, but this employee has no grant: nothing is visible and nothing is "
        "retrievable.",
    ),
    "B3": (
        "管理员给这位员工授予 READ。",
        "The admin grants READ to this user.",
    ),
    "B4": (
        "同一个问题，现在带出处地回答了。",
        "The same question is now answered with a citation.",
    ),
    "B5": (
        "撤销授权，回答立刻变回去。权限是检索查询里的过滤条件，不是模型记住的东西。",
        "Revoke it and the answer follows at once: permission is a filter in the retrieval query, "
        "not something the model remembers.",
    ),
    "B6": (
        "清理演示文档。",
        "Cleanup.",
    ),
}

HOLD_AFTER_RESULT = 7.0
# Scenes whose result is long (a long answer, a table) stay on screen longer than the default.
HOLDS: dict[str, float] = {
    "A2": 10.0,
    "A5": 16.0,
    "A6": 11.0,
    "A7": 20.0,
    "B2": 9.0,
    "B4": 10.0,
    "B5": 11.0,
}
MIN_READ = 4.0
SECONDS_PER_CHAR = 0.14
YELLOW, BOLD, DIM, RESET = "\033[93m", "\033[1m", "\033[2m", "\033[0m"


class Pacer:
    def __init__(self, captions: bool, auto: bool, speed: float) -> None:
        self.captions, self.auto, self.speed = captions, auto, speed
        self._started = False
        self._key: str | None = None

    def _sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds * self.speed))

    def begin(self) -> None:
        """Before a scene header: let the previous result be read, then clear the screen."""
        if self.auto:
            if self._started:
                self._sleep(HOLDS.get(self._key or "", HOLD_AFTER_RESULT))
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        self._started = True

    def caption(self, key: str) -> None:
        self._key = key
        if not self.captions:
            return
        zh, en = CAPTIONS[key]
        print()
        print(f"{YELLOW}{BOLD}▎{textwrap.fill(zh, width=60, subsequent_indent='  ')}{RESET}")
        print(f"{DIM}▎{textwrap.fill(en, width=84, subsequent_indent='  ')}{RESET}")
        print()
        if self.auto:
            self._sleep(max(MIN_READ, SECONDS_PER_CHAR * len(zh)))

    def finish(self) -> None:
        if self.auto:
            self._sleep(HOLDS.get(self._key or "", HOLD_AFTER_RESULT))


def check_thinking_off(
    base_url: str = "http://localhost:1234/v1", model: str = "google/gemma-4-26b-a4b-qat"
) -> None:
    """Refuse to run the demo while the model server is producing reasoning.

    A model that was unloaded and then loaded again on demand comes back with default settings,
    which for this model means Thinking is on: every answer becomes slow and the recording would
    show reasoning-length latencies. A one-sentence request with and without a tool definition is
    enough to see it.
    """
    for with_tool in (False, True):
        body: dict = {
            "model": model,
            "temperature": 0,
            "max_tokens": 120,
            "messages": [{"role": "user", "content": "用一句话说明什么是向量数据库。"}],
        }
        if with_tool:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_knowledge",
                        "description": "search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ]
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                message = json.load(response)["choices"][0]["message"]
        except Exception as exc:  # any failure means the model is not usable
            raise SystemExit(f"model server check failed: {type(exc).__name__}: {exc}") from None
        if (message.get("reasoning_content") or "").strip():
            raise SystemExit(
                "Thinking is ON for the loaded model, so every answer would be slow.\n"
                "In LM Studio: eject the model, load it from the UI, turn Enable Thinking off.\n"
                "(Loading it from the command line, or letting a request load it on demand, "
                "resets that setting.) Use --skip-preflight only if you really mean to."
            )
