"""关门战术测试：黄昏封门、清晨拆门、人没齐不关门、失败回退。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.brain import Brain  # noqa: E402
from agent.build import entrance_pos  # noqa: E402
from agent.protocol import Decision, Pos, Turn  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "request.json"


def make_brain_turn(round_no: int):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["roundNo"] = round_no
    # 样例基地 (10,24)：footprint (10,23)-(11,24)，门口 = (8,22)（西侧背面）
    # 把工人 10010 变成石头工且放到门口邻格、带石头
    for role in payload["teamOur"]["roles"]:
        if role["id"] == 10010:
            role["pos"] = {"x": 9, "y": 21}
            role["backpack"] = ["stone", "stone"]
        if role["id"] in (10011, 10012):
            role["pos"] = {"x": 12, "y": 22}      # 全员在基地旁（人齐才关门）
    turn = Turn.load(payload)
    brain = Brain()
    brain.worker_jobs[10010] = "stone"
    brain.memory.wall_phase = "full"      # 封门只在整圈阶段启用
    return brain, turn


def test_gate_closed_at_dusk():
    brain, turn = make_brain_turn(69)                # rid=69 白天
    decision = Decision()
    brain._gate_evening(turn, decision, set(), set())
    command = decision.commands.get(10010)
    assert command is not None and command["action"] == "build"
    assert command["name"] == "wall"
    assert command["targetPos"][0] == {"x": 8, "y": 22}


def test_gate_opened_at_morning():
    brain, turn = make_brain_turn(3)                 # rid=3 清晨
    # 门口已有墙
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["roundNo"] = 3
    for role in payload["teamOur"]["roles"]:
        if role["id"] == 10010:
            role["pos"] = {"x": 9, "y": 21}
            role["backpack"] = ["stone"]
    payload["teamOur"]["roles"].append({
        "id": 40050, "pos": {"x": 8, "y": 22}, "roleType": "wall",
        "health": 1000, "attackPower": 0, "attackRange": 0, "level": 1,
        "backPackCapability": 0, "backpack": [],
    })
    turn = Turn.load(payload)
    brain.memory.gate.pos = (8, 22)
    decision = Decision()
    brain._gate_morning(turn, decision, set(), set())
    command = decision.commands.get(10010)
    assert command is not None and command["action"] == "remove"
    assert command["targetPos"][0] == {"x": 8, "y": 22}


def test_gate_not_closed_when_pioneer_outside_before_window():
    brain, turn = make_brain_turn(67)
    # 封门前夕（rid<69）开拓者还没回圈 -> 不封门（锁死操控手 = 防线瘫痪）
    pioneer = turn.unit_by_id(10011)
    object.__setattr__(pioneer, "pos", Pos(25, 5))
    decision = Decision()
    brain._gate_evening(turn, decision, set(), set())
    assert all(
        command.get("action") != "build" or command.get("name") != "wall"
        for command in decision.commands.values()
    )


def test_gate_closed_at_window_even_if_pioneer_outside():
    brain, turn = make_brain_turn(69)
    # 69-70 是最后窗口：保基地优先，晚归的角色门外躲避也要封门
    pioneer = turn.unit_by_id(10011)
    object.__setattr__(pioneer, "pos", Pos(25, 5))
    decision = Decision()
    brain._gate_evening(turn, decision, set(), set())
    command = decision.commands.get(10010)
    assert command is not None and command["action"] == "build"


def test_gate_closed_with_worker_outside():
    brain, turn = make_brain_turn(67)
    # 夜矿工人被锁在门外是既定方针，不阻止封门
    worker = turn.unit_by_id(10012)
    object.__setattr__(worker, "pos", Pos(25, 5))
    decision = Decision()
    brain._gate_evening(turn, decision, set(), set())
    command = decision.commands.get(10010)
    assert command is not None and command["action"] == "build"


def test_gate_not_closed_in_front_phase():
    brain, turn = make_brain_turn(69)
    brain.memory.wall_phase = "front"     # 半圈阶段不封门
    decision = Decision()
    brain._gate_evening(turn, decision, set(), set())
    assert decision.commands == {}


def test_gate_disabled_after_build_failures():
    brain, turn = make_brain_turn(69)
    brain.memory.gate.record_build_failure()
    brain.memory.gate.record_build_failure()
    assert not brain.memory.gate.enabled
    decision = Decision()
    brain._gate_evening(turn, decision, set(), set())
    assert decision.commands == {}


def test_entrance_pos_matches_wall_plan():
    brain, turn = make_brain_turn(10)
    door = entrance_pos(turn)
    assert door is not None and (door.x, door.y) == (8, 22)


def test_gate_removal_survives_economy_overwrite():
    """回归：清晨拆门指令不得被工人经济循环覆盖（覆盖=全队被困圈内冻结）。"""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["roundNo"] = 3
    for role in payload["teamOur"]["roles"]:
        if role["id"] == 10010:
            role["pos"] = {"x": 9, "y": 21}
            role["backpack"] = ["stone"]
        if role["id"] in (10011, 10012):
            role["pos"] = {"x": 12, "y": 22}
    payload["teamOur"]["roles"].append({
        "id": 40050, "pos": {"x": 8, "y": 22}, "roleType": "wall",
        "health": 1000, "attackPower": 0, "attackRange": 0, "level": 1,
        "backPackCapability": 0, "backpack": [],
    })
    turn = Turn.load(payload)
    brain = Brain()
    brain.worker_jobs[10010] = "stone"
    brain.memory.wall_phase = "full"
    brain.memory.gate.pos = (8, 22)

    decision = Decision()
    brain._day(turn, decision)          # 完整白天流程（含经济循环）
    command = decision.commands.get(10010)
    assert command is not None and command["action"] == "remove", \
        f"拆门指令被覆盖为: {command}"
    assert command["targetPos"][0] == {"x": 8, "y": 22}
