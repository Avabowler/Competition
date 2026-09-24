"""校验器测试：保证异常响应红线（字段缺失/动作非法必须被拦截）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.protocol import Decision, Turn  # noqa: E402
from agent.validator import sanitize  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "request.json"


def load_turn() -> Turn:
    return Turn.load(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_drops_missing_target_pos():
    turn = load_turn()
    decision = Decision(commands={
        10010: {"action": "move"},                     # 缺 targetPos
        10011: {"action": "collect"},                  # 缺 targetPos
    })
    clean = sanitize(decision, turn)
    assert clean.commands == {}


def test_drops_unknown_action():
    turn = load_turn()
    decision = Decision(commands={10010: {"action": "teleport", "targetPos": [{"x": 1, "y": 1}]}})
    assert sanitize(decision, turn).commands == {}


def test_drops_bad_attack():
    turn = load_turn()
    decision = Decision(commands={
        10020: {"action": "attack", "targetPos": [{"x": 4, "y": 4}]},   # 缺 controllerId
        10030: {"action": "attack", "controllerId": 10010, "targetPos": []},  # 空 targetPos
        10010: {"action": "attack", "controllerId": "10011", "targetPos": [{"x": 4, "y": 4}]},
        # ^ 以角色 ID 为 key 发 attack 非法
    })
    clean = sanitize(decision, turn)
    assert clean.commands == {}


def test_keeps_valid_attack():
    turn = load_turn()
    decision = Decision(commands={
        10020: {"action": "attack", "controllerId": "10010",
                "targetPos": [{"x": 4, "y": 4}]},
    })
    clean = sanitize(decision, turn)
    assert 10020 in clean.commands


def test_use_items_needing_target():
    turn = load_turn()
    decision = Decision(commands={
        10010: {"action": "use", "name": "Bomb"},                       # 炸弹必须带 targetPos
        10011: {"action": "use", "name": "Medicine"},                   # 药品不需要
    })
    clean = sanitize(decision, turn)
    assert set(clean.commands) == {10011}


def test_submit_answer_needs_string():
    turn = load_turn()
    decision = Decision(commands={
        10011: {"action": "submitAnswer", "taskAnswer": 123},           # 非字符串
    })
    assert sanitize(decision, turn).commands == {}


def test_execute_cmd_outside_task_dropped():
    turn = load_turn()                  # 样例 phase_task 为空
    decision = Decision(execute_cmd="rm -rf /")
    clean = sanitize(decision, turn)
    assert clean.execute_cmd == ""


def test_gatling_target_count_capped_by_level():
    turn = load_turn()
    gatling = next(u for u in turn.ours if u.kind == "gatling")
    decision = Decision(commands={
        gatling.unit_id: {
            "action": "attack", "controllerId": "10010",
            "targetPos": [{"x": 4, "y": 4}, {"x": 5, "y": 4}, {"x": 6, "y": 6}],
        },
    })
    # 样例加特林 level=1，3 个目标越界，应被拦截
    assert sanitize(decision, turn).commands == {}
