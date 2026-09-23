"""协议解析测试：以 docs/request.txt 样例为基准。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.protocol import Pos, Turn, distance, station_footprint  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "request.json"


def load_turn() -> Turn:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return Turn.load(payload)


def test_parse_basic():
    turn = load_turn()
    assert turn.round_no == 85
    assert turn.width == 41 and turn.height == 32
    assert turn.team_type == "challenger"
    assert turn.gold == 20
    assert turn.is_day is False            # 85 % 130 = 85 -> 夜晚
    assert turn.day == 1
    assert turn.round_in_day == 85


def test_parse_units():
    turn = load_turn()
    assert turn.station() is not None
    assert len(turn.workers()) == 2
    assert turn.pioneer() is not None
    assert len(turn.weapons()) == 3
    assert len(turn.walls()) == 2
    station = turn.station()
    footprint = station_footprint(station.pos)
    assert len(footprint) == 4
    assert station.pos in footprint


def test_parse_robots_and_shops():
    turn = load_turn()
    assert len(turn.robots) == 4
    kinds = {r.kind for r in turn.robots}
    assert kinds == {"smallRobot", "middleRobot", "largeRobot", "bossRobot"}
    dizzy = [r for r in turn.robots if r.dizzy]
    assert len(dizzy) == 1
    assert turn.vendor_prices == {"stone": 1, "iron": 3, "copper": 5}
    assert turn.weapon_shop.get("Medicine") == 10
    assert turn.weapon_shop.get("BossRobotSummonOrder") == 200


def test_parse_enemies_and_tasks():
    turn = load_turn()
    assert len(turn.enemies) == 2          # 基地+围墙全图可见
    assert len(turn.player_tasks) == 2
    assert turn.player_tasks[0].is_valid
    assert turn.world_news.official
    assert turn.world_news.folk
    assert turn.last_summon_result == 0
    assert turn.last_action_results[10010] is False


def test_mines_and_blocked():
    turn = load_turn()
    assert len(turn.mines_of("stone")) == 2
    assert len(turn.mines_of("iron")) == 2
    assert len(turn.mines_of("copper")) == 2
    assert len(turn.vendors()) == 1
    blocked = turn.blocked()
    assert Pos(20, 16) in blocked          # 小贩占位阻挡
    assert Pos(4, 4) in blocked            # 机器人占位


def test_distance_chebyshev():
    assert distance(Pos(0, 0), Pos(3, 2)) == 3
    assert distance(Pos(5, 5), Pos(5, 5)) == 0
    assert distance(Pos(2, 8), Pos(1, 3)) == 5


def test_weapon_ranges():
    turn = load_turn()
    gatling = next(u for u in turn.weapons() if u.kind == "gatling")
    rocket = next(u for u in turn.weapons() if u.kind == "rocket")
    assert gatling.range_of_attack() == 4  # 请求里 attackRange=4
    assert rocket.range_of_attack() == 10 ** 9
