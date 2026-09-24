"""建造方向测试：来袭侧优先建墙、门口开在背面。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.build import (  # noqa: E402
    attack_side,
    entrance_pos,
    front_cells,
    wall_plan,
)
from agent.protocol import Turn  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "request.json"


def make_turn(station_x: int, station_y: int) -> Turn:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for role in payload["teamOur"]["roles"]:
        if role["id"] == 10013:
            role["pos"] = {"x": station_x, "y": station_y}
    return Turn.load(payload)


def test_left_base_attacked_from_west():
    # 样例基地 (10,24) 在地图左半 -> 机器人从左（西）来
    turn = make_turn(10, 24)
    assert attack_side(turn) == "west"
    front = front_cells(turn)
    assert front and all(p.x == 8 for p in front)   # 西排 x = xmin-2 = 8


def test_left_base_west_row_built_first():
    turn = make_turn(10, 24)
    plan = wall_plan(turn)
    assert plan, "应有建造计划"
    assert all(p.x == 8 for p in plan[:4])          # 西排 4 格排最前


def test_left_base_door_on_east_back():
    door = entrance_pos(make_turn(10, 24))
    assert (door.x, door.y) == (13, 21)             # 背面（东）角格


def test_right_base_attacked_from_east():
    turn = make_turn(30, 10)
    assert attack_side(turn) == "east"
    front = front_cells(turn)
    assert front and all(p.x == 33 for p in front)  # 东排 x = xmax+2 = 33


def test_right_base_east_row_built_first_and_door_west():
    turn = make_turn(30, 10)
    plan = wall_plan(turn)
    assert plan
    assert all(p.x == 33 for p in plan[:4])         # 东排最前
    door = entrance_pos(turn)
    assert (door.x, door.y) == (28, 8)              # 背面（西）角格


def test_ring_never_contains_door():
    for sx, sy in ((10, 24), (30, 10)):
        turn = make_turn(sx, sy)
        plan = wall_plan(turn)
        door = entrance_pos(turn)
        assert all(p != door for p in plan)
        assert len(plan) == len(set(plan))          # 无重复格
