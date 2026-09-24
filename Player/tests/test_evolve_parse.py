"""自进化任务解析链测试：JSON/围栏/裸命令/答案行 + 强制提交 + SOP 防过期。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.memory import EvolveTaskState, GameMemory, SopRecord  # noqa: E402
from agent.protocol import Decision, Turn  # noqa: E402
from agent.tasks.evolve import (  # noqa: E402
    EvolveModule,
    extract_llm_action,
    task_signature,
)


def test_parse_json():
    assert extract_llm_action('{"executeCmd": "ls -la"}') == ("cmd", "ls -la")
    assert extract_llm_action('前言 {"taskAnswer": "579"} 后记') == ("answer", "579")


def test_parse_bash_fence():
    text = "好的，我来查一下。\n```bash\nwc -l /tmp/data.txt\n```\n以上。"
    assert extract_llm_action(text) == ("cmd", "wc -l /tmp/data.txt")


def test_parse_dollar_and_bare_command():
    assert extract_llm_action("建议执行：\n$ ls -la /tmp") == ("cmd", "ls -la /tmp")
    assert extract_llm_action("第一行\npython3 -c \"print(1)\"") == (
        "cmd", "python3 -c \"print(1)\"",
    )


def test_parse_answer_marker():
    assert extract_llm_action("推理完成，最终答案: 12") == ("answer", "12")


def test_unparseable_returns_none():
    assert extract_llm_action("今天天气不错。") is None
    assert extract_llm_action("") is None


def make_solving_module(round_no: int = 100, timeout: int = 40):
    memory = GameMemory()
    state = memory.evolve
    state.phase = "solving"
    state.task_point = (14, 14)
    state.accepted_round = round_no - 35     # elapsed=35 >= 0.85*40
    state.timeout_rounds = timeout
    module = EvolveModule(memory)
    return memory, state, module


def test_force_submit_near_timeout():
    memory, state, module = make_solving_module(round_no=136, timeout=40)
    turn = Turn.load({
        "roundNo": 136, "mapInfo": {"width": 41, "height": 32, "zones": []},
        "teamOur": {"roles": [
            {"id": 10011, "pos": {"x": 14, "y": 13}, "roleType": "pioneer",
             "health": 200, "backPackCapability": 40, "backpack": []},
        ]},
        "phaseTask": "任务中",
    })
    decision = Decision()
    used = module.plan(turn, decision, set())
    assert used
    command = decision.commands.get(10011)
    assert command is not None and command["action"] == "submitAnswer"
    assert "taskAnswer" in command


def test_sop_exact_match_auto_answer():
    text = "计算 123+456 的值，提交答案字符串。"
    memory = GameMemory()
    memory.add_sop(SopRecord(
        signature=task_signature(text), task_type="自进化类1",
        answer="579", full_text=text,
    ))
    state = memory.evolve
    state.phase = "walking"
    state.task_point = (14, 14)
    module = EvolveModule(memory)
    turn = Turn.load({
        "roundNo": 50, "mapInfo": {"width": 41, "height": 32, "zones": []},
        "teamOur": {"roles": [
            {"id": 10011, "pos": {"x": 14, "y": 13}, "roleType": "pioneer",
             "health": 200, "backPackCapability": 40, "backpack": []},
        ]},
        "phaseTask": text,
    })
    module.observe(turn)
    assert state.phase == "solving"
    assert module._pending_answer == "579"


def test_sop_signature_only_hint_mode():
    text = "计算 123+456 的值，提交答案字符串。"
    memory = GameMemory()
    memory.add_sop(SopRecord(
        signature=task_signature(text), task_type="自进化类1",
        answer="579", full_text="计算 999+1 的值，提交答案字符串。",  # 全文不同
    ))
    state = memory.evolve
    state.phase = "walking"
    state.task_point = (14, 14)
    module = EvolveModule(memory)
    turn = Turn.load({
        "roundNo": 50, "mapInfo": {"width": 41, "height": 32, "zones": []},
        "teamOur": {"roles": [
            {"id": 10011, "pos": {"x": 14, "y": 13}, "roleType": "pioneer",
             "health": 200, "backPackCapability": 40, "backpack": []},
        ]},
        "phaseTask": text,
    })
    module.observe(turn)
    assert module._pending_answer is None       # 不直接交过期答案
    assert state.sop_hint_used                  # 转为提示模式
