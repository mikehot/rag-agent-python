from rag_agent.agent import AgentResult, ToolTrace
from rag_agent.tool_eval import score_case


def result(answer="好的", stop="answered", trace=None) -> AgentResult:
    return AgentResult(answer=answer, stop_reason=stop, turns=1, trace=trace or [])


def ok(name="search_knowledge", args=None) -> ToolTrace:
    return ToolTrace(name, args if args is not None else {"query": "q"}, "ok")


def test_a_correct_answer_with_the_required_tool_passes() -> None:
    expect = {"must_call": ["search_knowledge"], "answer_all_of_any": [["六个月", "6 个月"]]}
    assert score_case(expect, result("最长六个月", trace=[ok()])) == []


def test_missing_tool_call_and_missing_point_are_reported() -> None:
    expect = {"must_call": ["search_knowledge"], "answer_all_of_any": [["六个月"]]}
    failures = score_case(expect, result("不知道"))
    assert any("did not successfully call" in f for f in failures)
    assert any("answer point 0" in f for f in failures)


def test_a_tool_that_errored_does_not_count_as_called() -> None:
    errored = ToolTrace("search_knowledge", {"query": "q"}, "tool_error", True)
    assert score_case({"must_call": ["search_knowledge"]}, result(trace=[errored]))


def test_no_tool_expectation() -> None:
    assert score_case({"no_tool": True}, result("你好")) == []
    assert score_case({"no_tool": True}, result("x", trace=[ok()]))


def test_forbidden_terms_and_decline_requirement() -> None:
    expect = {"no_answer": True, "forbid_terms": ["六个月"]}
    assert score_case(expect, result("知识库中没有相关内容")) == []
    assert any("forbidden term" in f for f in score_case(expect, result("六个月")))
    assert any("does not decline" in f for f in score_case(expect, result("好的")))


def test_forbidden_arguments_are_checked_on_executed_calls_only() -> None:
    rejected = ToolTrace("list_documents", None, "rejected_arguments")
    leaked = ToolTrace("list_documents", {"tenantId": "x"}, "tool_error", True)
    expect = {"forbid_args": ["tenantId"]}
    assert score_case(expect, result(trace=[rejected])) == []
    assert any("forbidden argument" in f for f in score_case(expect, result(trace=[leaked])))


def test_non_answered_stop_reason_fails_the_case() -> None:
    assert score_case({}, result(stop="turn_limit")) == ["stop_reason=turn_limit"]


def test_markdown_emphasis_does_not_hide_a_correct_value() -> None:
    expect = {"answer_all_of_any": [["是 6", "为 6"]]}
    assert score_case(expect, result("默认值是 **6**")) == []
    assert score_case(expect, result("默认值是 `6`")) == []
