"""物品使用服务测试：药剂/召唤令/应急炸弹/夜晚修墙。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.items import ItemService  # noqa: E402
from agent.memory import GameMemory  # noqa: E402
from agent.protocol import Decision, Turn  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "request.json"


def make_turn(**overrides) -> Turn:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload.update(overrides)
    return Turn.load(payload)


def find_role(turn: Turn, role_id: int):
    return turn.unit_by_id(role_id)


def test_medicine_used_when_hurt():
    turn = make_turn(roundNo=10)                     # 白天
    hurt = find_role(turn, 10011)
    object.__dict__  # noqa: B018  (no-op, 保持可读性)
    turn.ours[0].__class__  # noqa: B018
    # 冻结 dataclass 无法直接改字段，用 object.__setattr__
    object.__setattr__(hurt, "health", 60)
    object.__setattr__(hurt, "backpack", ("Medicine",))
    decision = Decision()
    handled = ItemService(GameMemory()).plan(turn, decision, set())
    assert 10011 in handled
    assert decision.commands[10011]["action"] == "use"
    assert decision.commands[10011]["name"] == "Medicine"


def test_summon_order_used_immediately():
    turn = make_turn(roundNo=10)
    worker = find_role(turn, 10010)
    object.__setattr__(worker, "backpack", ("BossRobotSummonOrder",))
    decision = Decision()
    handled = ItemService(GameMemory()).plan(turn, decision, set())
    assert 10010 in handled
    assert decision.commands[10010]["action"] == "use"
    assert decision.commands[10010]["name"] == "BossRobotSummonOrder"


def test_emergency_bomb_at_night():
    turn = make_turn(roundNo=85)                     # 夜晚
    worker = find_role(turn, 10010)
    object.__setattr__(worker, "backpack", ("Bomb",))
    # 在基地 footprint 旁塞一只贴脸大型机器人
    payload_boss = {
        "id": 30999, "pos": {"x": 9, "y": 22}, "roleType": "largeRobot",
        "health": 500, "abnormalState": "", "targetTeam": "challenger",
    }
    turn.robots[0].__class__  # noqa: B018
    robots = list(turn.robots) + [type(turn.robots[0]).load(payload_boss)]
    object.__setattr__(turn, "robots", tuple(robots))
    decision = Decision()
    handled = ItemService(GameMemory()).plan(turn, decision, set())
    assert handled, "大怪贴脸必须触发应急道具"
    command = next(iter(decision.commands.values()))
    assert command["action"] == "use"
    assert "targetPos" in command


def test_no_bomb_when_robots_far():
    turn = make_turn(roundNo=85)
    worker = find_role(turn, 10010)
    object.__setattr__(worker, "backpack", ("Bomb",))
    decision = Decision()
    ItemService(GameMemory()).plan(turn, decision, set())
    actions = [c.get("action") for c in decision.commands.values()]
    assert "use" not in actions
