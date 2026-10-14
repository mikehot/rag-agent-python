"""Deterministic scoring for tool-calling behaviour.

Scores what the agent *did* (which tools ran, with which arguments, how it stopped) and what its
final answer contained, using only literal rules written in the dataset. It is not a judge of answer
quality and it does not replace reading the answers.
"""

from __future__ import annotations

import re
from typing import Any

from rag_agent.agent import AgentResult

NO_ANSWER_PATTERN = re.compile(
    r"没有相关|没有关于|没有找到|未找到|无法回答|无法提供|无法找到|无法执行|无法为|不包含|无相关|"
    r"不在知识库|知识库中没有|知识库里没有|抱歉|没有可用|没有任何"
)


def plain_text(text: str) -> str:
    """Drop markdown emphasis and code marks so "是 **6**" matches "是 6"; content is unchanged."""
    return text.replace("*", "").replace("`", "")


def executed_calls(result: AgentResult) -> list[Any]:
    return [t for t in result.trace if t.arguments is not None]


def score_case(expect: dict[str, Any], result: AgentResult) -> list[str]:
    """Return the list of violated expectations; an empty list means the case passed."""
    failures: list[str] = []
    answer = plain_text(result.answer)
    if result.stop_reason != "answered":
        failures.append(f"stop_reason={result.stop_reason}")

    attempted = {t.name for t in result.trace}
    ok_calls = {t.name for t in result.trace if t.outcome == "ok"}
    for name in expect.get("must_call", []):
        if name not in ok_calls:
            failures.append(f"did not successfully call {name}")
    if expect.get("no_tool") and result.trace:
        failures.append(f"called tools: {sorted(attempted)}")

    forbidden_args = set(expect.get("forbid_args", []))
    for call in executed_calls(result):
        leaked = forbidden_args & set(call.arguments or {})
        if leaked:
            failures.append(f"sent forbidden argument(s) {sorted(leaked)} to {call.name}")

    for term in expect.get("forbid_terms", []):
        if term in answer:
            failures.append(f"answer contains forbidden term {term!r}")
    if expect.get("no_answer") and not NO_ANSWER_PATTERN.search(answer):
        failures.append("answer does not decline")
    if "answer_any" in expect and not any(t in answer for t in expect["answer_any"]):
        failures.append(f"answer contains none of {expect['answer_any']}")
    for index, group in enumerate(expect.get("answer_all_of_any", [])):
        if not any(term in answer for term in group):
            failures.append(f"answer point {index} missing")
    return failures
