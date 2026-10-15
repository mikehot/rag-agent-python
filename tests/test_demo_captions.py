import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from demo_captions import CAPTIONS, HOLDS, Pacer


def test_every_scene_of_both_drivers_has_a_caption() -> None:
    expected = {f"A{i}" for i in range(1, 8)} | {f"B{i}" for i in range(1, 7)}
    assert set(CAPTIONS) == expected
    assert all(zh.strip() and en.strip() for zh, en in CAPTIONS.values())


def test_hold_overrides_refer_to_real_scenes() -> None:
    assert set(HOLDS) <= set(CAPTIONS)


def test_pacer_prints_a_caption_and_does_not_sleep_at_speed_zero(capsys) -> None:
    pacer = Pacer(captions=True, auto=True, speed=0.0)
    pacer.begin()
    pacer.caption("A1")
    pacer.begin()
    pacer.caption("A2")
    pacer.finish()
    printed = capsys.readouterr().out
    assert "同一个知识库" in printed and "One knowledge base" in printed


def test_pacer_is_silent_without_captions(capsys) -> None:
    pacer = Pacer(captions=False, auto=False, speed=0.0)
    pacer.begin()
    pacer.caption("A1")
    pacer.finish()
    assert capsys.readouterr().out == ""
