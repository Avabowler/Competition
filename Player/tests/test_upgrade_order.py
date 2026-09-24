"""升级序列测试：按金币梯度验证购买优先级（火箭优先于电磁、急救升级最优先）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.economy import Economy  # noqa: E402
from agent.memory import GameMemory  # noqa: E402
from agent.protocol import Turn  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "request.json"


def make_turn(gold: int) -> Turn:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["teamOur"]["goldNum"] = gold
    return Turn.load(payload)


def next_purchase(turn: Turn):
    buyer = turn.unit_by_id(10010)
    return Economy(GameMemory())._next_purchase(turn, buyer)


def test_gatling_l2_first_at_low_gold():
    # 样例：gatling L1, railgun L1, rocket L1 -> 序列首位是加特林 L2
    assert next_purchase(make_turn(gold=150)) == ("WeaponUpgradeVoucher1", 1)


def test_station_heal_upgrade_takes_priority_when_critical():
    turn = make_turn(gold=150)
    station = turn.station()
    object.__setattr__(station, "health", 500)   # <40% of 1500
    purchase = next_purchase(turn)
    assert purchase == ("StationUpgradeVoucher1", 1)


def test_wall_fixer_bought_when_wall_damaged_and_rich():
    turn = make_turn(gold=200)
    wall = turn.walls()[0]
    object.__setattr__(wall, "health", 300)      # <60% of 1000
    assert next_purchase(turn) == ("WallFixer", 1)


def test_railgun_only_after_walls():
    turn = make_turn(gold=400)
    # 加特林/火箭/基地全到顶、有 L1 围墙 -> 剩余序列中围墙 L2 券先于电磁炮
    for unit_id in (10020, 10040):
        object.__setattr__(turn.unit_by_id(unit_id), "level", 3)
    object.__setattr__(turn.station(), "level", 3)
    purchase = next_purchase(turn)
    assert purchase == ("WallUpgradeVoucher1", 1)


def test_holding_dedup_skips_voucher():
    turn = make_turn(gold=400)
    worker = turn.unit_by_id(10010)
    object.__setattr__(worker, "backpack", ("WeaponUpgradeVoucher1",))
    purchase = next_purchase(turn)
    assert purchase is not None and purchase[0] != "WeaponUpgradeVoucher1"
