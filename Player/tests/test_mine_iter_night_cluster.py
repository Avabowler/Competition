"""矿区从近到远遍历 / best-effort 逼近 / 区块记忆 / 聚控 / 夜矿 用例。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.brain import Brain  # noqa: E402
from agent.build import cluster_plan, tower_sites  # noqa: E402
from agent.combat import Combat  # noqa: E402
from agent.economy import Economy  # noqa: E402
from agent.grid import approach_step, next_step  # noqa: E402
from agent.memory import GameMemory  # noqa: E402
from agent.protocol import Decision, Pos, Turn, distance  # noqa: E402


# ---------------------------------------------------------------- 构造工具

def make_payload(round_no: int = 1, zones: list[dict] | None = None,
                 roles: list[dict] | None = None,
                 robots: list[dict] | None = None,
                 gold: int = 75) -> dict:
    return {
        "roundNo": round_no,
        "mapInfo": {"width": 41, "height": 32, "zones": zones or []},
        "teamOur": {"type": "challenger", "goldNum": gold,
                    "roles": roles or []},
        "teamEnemy": {"roles": []},
        "robot": {"roles": robots or []},
        "playerTasks": [],
        "phaseTask": "",
        "lastRoundRoleActionResults": {},
        "lastSummonTreasureResult": 0,
        "llmResp": "",
        "worldNews": {},
        "lastCmdResult": "",
        "vendorShopList": [{"name": "stone", "price": 5}],
        "weaponShopList": [],
        "errors": [],
    }


def make_role(role_id: int, kind: str, x: int, y: int,
              bag: list[str] | None = None, capacity: int = 0,
              health: int = 200, level: int = 1) -> dict:
    return {
        "id": role_id, "roleType": kind, "pos": {"x": x, "y": y},
        "health": health, "level": level, "cooldown": 0,
        "attackPower": 0, "attackRange": 0,
        "backPackCapability": capacity, "backpack": bag or [],
    }


def make_wall(role_id: int, x: int, y: int) -> dict:
    return make_role(role_id, "wall", x, y)


def make_worker(role_id: int, x: int, y: int,
                bag: list[str] | None = None) -> dict:
    return make_role(role_id, "worker", x, y, bag=bag, capacity=100)


def make_robot(robot_id: int, x: int, y: int) -> dict:
    return {"id": robot_id, "roleType": "smallRobot",
            "pos": {"x": x, "y": y}, "health": 40}


def ring_walls(role_id_start: int, center: Pos) -> list[dict]:
    """把 center 周围一圈全部用围墙堵死（模拟矿点被围死）。"""
    walls = []
    for i, n in enumerate(center.neighbours()):
        walls.append(make_wall(role_id_start + i, n.x, n.y))
    return walls


# ---------------------------------------------------------------- 寻路

def test_approach_step_presses_toward_walled_goal():
    goal = Pos(22, 22)
    payload = make_payload(roles=[make_worker(1, 19, 22)] + ring_walls(100, goal))
    turn = Turn.load(payload)
    assert next_step(turn, turn.ours[0], goal) is None   # 完全不可达
    step = approach_step(turn, turn.ours[0], goal)
    assert step is not None
    # 逼近步必须让到目标的距离严格缩短
    assert distance(step, goal) < distance(turn.ours[0].pos, goal)


def test_approach_step_none_when_already_closest():
    goal = Pos(22, 22)
    payload = make_payload(roles=[make_worker(1, 19, 22)] + ring_walls(100, goal))
    turn = Turn.load(payload)
    # 站在可达边缘：向目标的可通行侧没有更近格
    far = approach_step(turn, turn.ours[0], Pos(30, 30))
    assert far is None or distance(far, Pos(30, 30)) < distance(
        turn.ours[0].pos, Pos(30, 30))


# ---------------------------------------------------------------- 矿区遍历

def test_mine_round_skips_unreachable_nearest():
    near_mine, far_mine = Pos(22, 22), Pos(15, 15)
    payload = make_payload(
        zones=[{"pos": {"x": near_mine.x, "y": near_mine.y}, "neutralType": "stone"},
               {"pos": {"x": far_mine.x, "y": far_mine.y}, "neutralType": "stone"}],
        roles=[make_worker(1, 19, 20)] + ring_walls(100, near_mine),
    )
    turn = Turn.load(payload)
    decision = Decision()
    ok = Economy(GameMemory())._mine_round(
        turn, turn.ours[0], "stone", decision, set(),
    )
    assert ok
    target = Pos.load(decision.commands[1]["targetPos"][0])
    # 移动目标必须朝向可达的远矿，而不是扑向被围死的近矿
    assert distance(target, far_mine) < distance(turn.ours[0].pos, far_mine)


def test_fallback_move_iterates_targets():
    near_mine = Pos(22, 22)
    payload = make_payload(
        zones=[{"pos": {"x": near_mine.x, "y": near_mine.y}, "neutralType": "stone"}],
        roles=[make_worker(1, 19, 20)] + ring_walls(100, near_mine),
    )
    turn = Turn.load(payload)
    decision = Decision()
    Economy(GameMemory()).fallback_move(turn, turn.ours[0], decision, set())
    # 最近矿不可达且无小贩：best-effort 逼近也要给出移动指令（不原地发呆）
    assert 1 in decision.commands
    assert decision.commands[1]["action"] == "move"


# ---------------------------------------------------------------- 区块记忆

def test_zone_memory_full_map_world_skips_exploration():
    zones = [
        {"pos": {"x": 20, "y": 10 + i}, "neutralType": "stone"}
        for i in range(10)
    ]
    memory = GameMemory()
    turn = Turn.load(make_payload(zones=zones, roles=[make_worker(1, 5, 5)]))
    memory.observe_zones(turn)
    assert memory.exploration_mode is False
    assert memory.exploration_done(turn)


def test_zone_memory_filtered_world_remember_and_invalidate():
    memory = GameMemory()
    mine = Pos(8, 8)
    worker = make_worker(1, 5, 8)          # 与矿相距 3：视野内
    turn1 = Turn.load(make_payload(
        round_no=1, roles=[worker],
        zones=[{"pos": {"x": mine.x, "y": mine.y}, "neutralType": "stone"}],
    ))
    memory.observe_zones(turn1)
    assert memory.exploration_mode is True
    assert not memory.exploration_done(turn1)          # 缺 vendor/weaponShop

    # 下回合矿从 payload 消失但仍在视野内 -> 记忆失效
    turn2 = Turn.load(make_payload(round_no=2, roles=[worker], zones=[]))
    memory.observe_zones(turn2)
    assert (mine.x, mine.y) not in memory.zone_memory
    assert memory.effective_zones(turn2) == {}

    # 视野外的记忆项保留
    far_mine = Pos(30, 5)
    turn3 = Turn.load(make_payload(
        round_no=3, roles=[worker],
        zones=[{"pos": {"x": far_mine.x, "y": far_mine.y}, "neutralType": "iron"}],
    ))
    memory.observe_zones(turn3)
    turn4 = Turn.load(make_payload(round_no=4, roles=[worker], zones=[]))
    assert memory.effective_zones(turn4).get(far_mine) == "iron"


def test_zone_memory_merge_feeds_mines_of():
    memory = GameMemory()
    far_mine = Pos(30, 5)
    turn1 = Turn.load(make_payload(
        round_no=1, roles=[make_worker(1, 5, 5)],
        zones=[{"pos": {"x": far_mine.x, "y": far_mine.y}, "neutralType": "stone"}],
    ))
    memory.observe_zones(turn1)
    from dataclasses import replace
    turn2 = replace(
        turn1, round_no=2, zones=memory.effective_zones(
            Turn.load(make_payload(round_no=2, roles=[make_worker(1, 5, 5)])),
        ),
    )
    assert turn2.mines_of("stone") == (far_mine,)


# ---------------------------------------------------------------- 分工与探索

def test_both_workers_stone_while_front_ring_open():
    brain = Brain()
    payload = make_payload(
        round_no=6,
        roles=[
            make_role(10013, "station", 10, 24),
            make_worker(10010, 5, 23), make_worker(10012, 5, 24),
        ],
    )
    turn = Turn.load(payload)
    brain._assign_worker_jobs(turn)
    jobs = {brain.worker_jobs[w.unit_id] for w in turn.workers()}
    assert jobs == {"stone"}


def test_worker_explores_when_map_unmapped():
    brain = Brain()
    memory = brain.memory
    memory.exploration_mode = True
    memory.explored = {(x, y) for x in range(3, 9) for y in range(21, 27)}
    # 已知小贩+商店但还没找到石矿 -> 探索不应停止
    payload = make_payload(
        roles=[make_worker(1, 5, 24)],
        zones=[{"pos": {"x": 35, "y": 5}, "neutralType": "vendor"},
               {"pos": {"x": 36, "y": 5}, "neutralType": "weaponShop"}],
    )
    turn = Turn.load(payload)
    assert not brain.memory.exploration_done(turn)
    decision = Decision()
    moved = brain._explore_step(turn, turn.ours[0], decision, set())
    assert moved
    assert decision.commands[1]["action"] == "move"


# ---------------------------------------------------------------- 聚控

def test_cluster_plan_returns_seat_with_three_adjacent_towers():
    memory = GameMemory()
    turn = Turn.load(make_payload(
        roles=[make_role(10013, "station", 10, 24)],
    ))
    sites = tower_sites(turn, memory)
    seat = memory.tower_seat
    assert seat is not None, "样例基地应存在聚控布局"
    seat_pos = Pos(*seat)
    assert len(sites) == 3
    assert all(distance(site, seat_pos) <= 1 for site in sites)
    assert seat_pos not in sites
    assert len(set(sites)) == 3


def test_pioneer_controls_all_cluster_weapons():
    memory = GameMemory()
    memory.tower_seat = (10, 24)     # 与 fixture 三塔均相邻的格子
    combat = Combat(memory)
    payload = make_payload(
        round_no=80,                 # 夜晚
        roles=[
            make_role(10013, "station", 10, 24),
            make_role(10020, "gatling", 9, 24),
            make_role(10030, "railgun", 10, 25),
            make_role(10040, "rocket", 9, 25),
            make_role(10011, "pioneer", 10, 24),   # 站在控制位上
            make_worker(10010, 30, 5),
        ],
        robots=[make_robot(9001, 15, 25)],
    )
    turn = Turn.load(payload)
    decision = Decision()
    combat.plan_night(turn, decision, set())
    attacks = {
        uid: cmd for uid, cmd in decision.commands.items()
        if cmd.get("action") == "attack"
    }
    assert attacks, "聚控应驱动相邻炮台开火"
    for uid, cmd in attacks.items():
        assert cmd["controllerId"] == "10011"      # 全部由开拓者操控
        assert uid in (10020, 10030, 10040)
    assert len(attacks) == 1                       # 错峰轮发：每回合只开一门
    first = next(iter(attacks))
    # 第一门进入冷却 -> 下回合轮到另一门，火力不间断
    object.__setattr__(turn.unit_by_id(first), "cooldown", 3)
    decision2 = Decision()
    combat.plan_night(turn, decision2, set())
    attacks2 = {
        uid: cmd for uid, cmd in decision2.commands.items()
        if cmd.get("action") == "attack"
    }
    assert attacks2 and next(iter(attacks2)) != first


def test_multi_control_reverts_after_repeated_failures():
    memory = GameMemory()
    memory.tower_seat = (10, 24)
    combat = Combat(memory)
    payload = make_payload(
        round_no=80,
        roles=[
            make_role(10013, "station", 10, 24),
            make_role(10020, "gatling", 9, 24),
            make_role(10030, "railgun", 10, 25),
            make_role(10040, "rocket", 9, 25),
            make_role(10011, "pioneer", 10, 24),
            make_worker(10010, 30, 5),
        ],
        robots=[make_robot(9001, 15, 25)],
    )
    turn = Turn.load(payload)
    for _ in range(3):
        memory.last_commands = {
            10020: {"action": "attack", "controllerId": "10011"},
            10030: {"action": "attack", "controllerId": "10011"},
            10040: {"action": "attack", "controllerId": "10011"},
        }
        memory.failed_actions = dict(memory.last_commands)
        combat.plan_night(turn, Decision(), set())
    assert memory.multi_control_failed
    # 回退后 1:1 配对仍会派角色上塔位（防御不瘫痪）
    decision = Decision()
    combat.plan_night(turn, decision, set())
    assert 10010 in decision.commands


# ---------------------------------------------------------------- 夜矿与躲避

def test_night_worker_dodges_robot():
    economy = Economy(GameMemory())
    payload = make_payload(
        round_no=80,
        roles=[make_worker(1, 20, 20)],
        robots=[make_robot(9001, 22, 20)],
    )
    turn = Turn.load(payload)
    decision = Decision()
    assert economy.evade_robots(turn, turn.ours[0], decision, set())
    target = Pos.load(decision.commands[1]["targetPos"][0])
    robot = turn.robots[0].pos
    assert distance(target, robot) >= 3        # 撤出射程+余量


def test_night_worker_collects_adjacent_mine():
    economy = Economy(GameMemory())
    payload = make_payload(
        round_no=80,
        roles=[make_worker(1, 21, 22)],
        zones=[{"pos": {"x": 22, "y": 22}, "neutralType": "stone"}],
    )
    turn = Turn.load(payload)
    decision = Decision()
    economy.plan_night_worker(turn, turn.ours[0], "metal", decision, set())
    assert decision.commands[1]["action"] == "collect"


def test_dusk_regroup_sends_pioneer_to_seat_not_home():
    brain = Brain()
    brain.memory.tower_seat = (10, 24)
    payload = make_payload(
        round_no=65,                 # 白天第 65 回合（黄昏窗口）
        roles=[
            make_role(10013, "station", 10, 24, health=1500),
            make_role(10020, "gatling", 9, 24),
            make_role(10030, "railgun", 10, 25),
            make_role(10040, "rocket", 9, 25),
            make_role(10011, "pioneer", 10, 12),
            make_worker(10010, 5, 23),
        ],
    )
    turn = Turn.load(payload)
    decision = Decision()
    brain._dusk_regroup(turn, decision, set())
    assert 10011 in decision.commands
    target = Pos.load(decision.commands[10011]["targetPos"][0])
    seat = Pos(10, 24)
    assert distance(target, seat) < distance(Pos(10, 12), seat)  # 朝座位逼近
    assert 10010 not in decision.commands       # 工人不归位
