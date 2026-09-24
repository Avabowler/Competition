"""本地模拟判题器：不依赖真实环境即可回归测试两套 Brain。

模拟范围（足够经济/生存/击杀/任务回归，非完全精确）：
- 昼夜 130 回合/天、移动与碰撞、采矿(10次/矿)、贩卖/购买、建造/拆除、
  升级券与消耗品、机器人夜袭(向基地推进、攻击阻挡者)、三种武器攻击结算、
  角色死亡与复活、任务点(acceptTask/submitAnswer)、LLM 脚本化响应、
  executeCmd 脚本化响应、宝藏召唤。
- 视野：机器人全图可见、敌基地/围墙全图可见、其余敌方单位按视野4过滤。

用法：
    python tools/mock_judger.py                # 我方 Brain vs Baseline 一场
    python tools/mock_judger.py --days 5       # 只打前5天
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.protocol import (  # noqa: E402
    DAY_ROUNDS,
    ROUNDS_PER_DAY,
    ROBOT_STATS,
    WEAPON_STATS,
)


def footprint(pos) -> list:
    x, y = pos
    return [(x, y), (x + 1, y), (x, y - 1), (x + 1, y - 1)]

MAP_W, MAP_H = 41, 32

INIT_ZONES = [
    ("challengerTaskPoint1", (14, 14)), ("challengerTaskPoint2", (17, 17)),
    ("challengerTaskPoint2", (16, 17)),
    ("defenderTaskPoint1", (23, 14)), ("defenderTaskPoint2", (26, 17)),
    ("defenderTaskPoint2", (27, 17)),
    ("vendor", (20, 16)), ("weaponShop", (25, 20)),
    ("stone", (4, 24)), ("stone", (14, 3)), ("stone", (33, 20)),
    ("iron", (25, 10)), ("iron", (8, 28)), ("iron", (30, 27)),
    ("copper", (22, 26)), ("copper", (7, 2)), ("copper", (35, 8)),
]

BASE_PRICES = {"stone": 1, "iron": 3, "copper": 5}

# 脚本化 LLM：任务/宝藏的"标准答案"
MOCK_TASKS = [
    {"type": "自进化类1", "text": "计算 123+456 的值，提交答案字符串。",
     "answer": "579", "timeout": 40, "score": 50, "gold": 30,
     "cmd": 'python3 -c "print(123+456)"'},
    {"type": "自进化类2", "text": "沙盒中存在文件 /tmp/data.txt，输出其行数。提交行数字符串。",
     "answer": "12", "timeout": 60, "score": 50, "gold": 30,
     "cmd": "wc -l < /tmp/data.txt"},
]
MOCK_TREASURE = {
    "pos": (20, 25), "items": ["StarSand", "FrostPotion"],
    "day": 3, "window": (30, 80), "score": 300, "gold": 200,
}


@dataclass
class MockUnit:
    id: int
    pos: tuple[int, int]
    kind: str
    health: int
    level: int = 1
    cooldown: int = 0
    capacity: int = 0
    backpack: list[str] = field(default_factory=list)
    team: str = ""
    dead_until: int = 0          # 复活回合（0=存活）

    def dump(self) -> dict:
        power, reach = 0, 0
        if self.kind in WEAPON_STATS:
            power, reach, _hp = WEAPON_STATS[self.kind][min(self.level, 3) - 1]
        data = {
            "id": self.id, "pos": {"x": self.pos[0], "y": self.pos[1]},
            "roleType": self.kind, "health": self.health,
            "attackPower": power, "attackRange": reach, "level": self.level,
            "cooldown": self.cooldown,
        }
        if self.kind in ("worker", "pioneer"):
            data["backPackCapability"] = self.capacity
            data["backpack"] = list(self.backpack)
        return data


@dataclass
class MockRobot:
    id: int
    pos: tuple[int, int]
    kind: str
    health: int
    target: str
    dizzy: int = 0

    def dump(self) -> dict:
        return {
            "id": self.id, "pos": {"x": self.pos[0], "y": self.pos[1]},
            "roleType": self.kind, "health": self.health,
            "abnormalState": "dizzy" if self.dizzy > 0 else "",
            "targetTeam": self.target,
        }


class Side:
    def __init__(self, team: str, brain, station_pos: tuple[int, int]):
        self.team = team
        self.brain = brain
        base = 10000 if team == "challenger" else 20000
        self.station = MockUnit(base + 3, station_pos, "station", 1500, team=team)
        self.units: list[MockUnit] = [
            MockUnit(base + 0, (station_pos[0] - 5, station_pos[1] - 1), "worker", 220,
                     capacity=100, team=team),
            MockUnit(base + 1, (station_pos[0] - 5, station_pos[1] + 1), "pioneer", 200,
                     capacity=40, team=team),
            MockUnit(base + 2, (station_pos[0] - 4, station_pos[1] - 1), "worker", 220,
                     capacity=100, team=team),
            self.station,
        ]
        self.gold = 75
        self.score = 0.0
        self.kill_score = 0.0
        self.task_score = 0.0
        self.treasure_score = 0.0
        self.abnormal_count = 0
        self.llm_calls_today = 0
        self.awaiting_llm = False
        self.llm_resp_queue: list[str] = []
        self.cmd_result_queue: list[str] = []
        self.last_cmd_result = ""
        self.last_llm_resp = ""
        self.last_summon_result = 0
        self.last_action_results: dict[str, bool] = {}
        self.phase_task = ""
        self.task_state: dict = {}          # 当前任务进度
        self.task_cooldown: dict[str, int] = {}   # 任务点 -> 冷却截止回合
        self.errors: list[dict] = []
        self.station_alive_day = 0          # 基地存活天数积分累计
        self.used_treasure = False

    # ---- 视野过滤后的敌方单位

    def visible_enemies(self, world: "World") -> list[MockUnit]:
        enemy = world.other(self)
        visible: list[MockUnit] = []
        my_cells = [u.pos for u in self.units if u.health > 0]
        for unit in enemy.units:
            if unit.health <= 0:
                continue
            if unit.kind in ("station", "wall"):
                visible.append(unit)
                continue
            if any(cheb(a, unit.pos) <= 4 for a in my_cells):
                visible.append(unit)
        return visible


class World:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.zones: dict[tuple[int, int], str] = {
            pos: kind for kind, pos in INIT_ZONES
        }
        self.mine_uses: dict[tuple[int, int], int] = {}
        self.robots: list[MockRobot] = []
        self.robot_seq = 30000
        self.round = 0
        self.treasure_used = False
        self.pending_summons: list = []

    def other(self, side: Side) -> Side:
        return self.sides[1] if side is self.sides[0] else self.sides[0]

    def blocked_cells(self, mover_team: str) -> set[tuple[int, int]]:
        cells = set(self.zones.keys())
        for side in self.sides:
            for unit in side.units:
                if unit.health <= 0:
                    continue
                if unit.kind == "station":
                    cells.update(footprint(unit.pos))
                else:
                    cells.add(unit.pos)
        for robot in self.robots:
            if robot.health > 0:
                cells.add(robot.pos)
        return cells


def cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def line_cells(start, end) -> list[tuple[int, int]]:
    cells = []
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return [start]
    for i in range(steps + 1):
        t = i / steps
        cells.append((round(x0 + t * (x1 - x0)), round(y0 + t * (y1 - y0))))
    seen = set()
    unique = []
    for c in cells:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


class MockJudger:
    def __init__(self, brain_a, brain_b, seed: int = 42, verbose: bool = False,
                 hard: bool = False, llm_prose: bool = False):
        self.world = World(seed)
        self.world.sides = [
            Side("challenger", brain_a, (10, 24)),
            Side("defender", brain_b, (30, 10)),
        ]
        self.verbose = verbose
        self.hard = hard
        self.llm_prose = llm_prose
        self.total_rounds = 10 * ROUNDS_PER_DAY

    # ------------------------------------------------ 回合主循环

    def run(self, max_days: int = 10) -> dict:
        report = {"sides": {}, "rounds": 0}
        end_round = min(max_days * ROUNDS_PER_DAY, self.total_rounds)
        for round_no in range(1, end_round + 1):
            self.world.round = round_no
            round_in_day = (round_no - 1) % ROUNDS_PER_DAY + 1
            day = (round_no - 1) // ROUNDS_PER_DAY + 1
            is_day = round_in_day <= DAY_ROUNDS

            if round_in_day == 1:
                self._on_new_day(day)
            if round_in_day == DAY_ROUNDS + 1 and not is_day:
                self._spawn_robots(day)

            responses = {}
            for side in self.world.sides:
                request = self._build_request(side, round_no, day, round_in_day)
                responses[id(side)] = self._call_brain(side, request)

            for side in self.world.sides:
                self._apply(side, responses[id(side)], round_no, is_day)

            self._settle_robots(round_no, is_day)

            # 结束判定：双基地均毁
            if all(s.station.health <= 0 for s in self.world.sides):
                break

        for side in self.world.sides:
            survived_days = 0
            # 简化：按基地当前血量>0 记满存活天数
            if side.station.health > 0:
                survived_days = day
            report["sides"][side.team] = {
                "brain": type(side.brain).__name__,
                "score": round(side.score + side.kill_score + side.task_score
                               + side.treasure_score + 10 * sum(range(1, survived_days + 1)), 1),
                "kills": side.kill_score,
                "task": side.task_score,
                "treasure": side.treasure_score,
                "gold_left": side.gold,
                "abnormal": side.abnormal_count,
                "station_hp": side.station.health,
                "weapons": len([u for u in side.units if u.kind in ("gatling", "railgun", "rocket")]),
                "walls": len([u for u in side.units if u.kind == "wall"]),
            }
        report["rounds"] = self.world.round
        return report

    # ------------------------------------------------ 事件

    def _on_new_day(self, day: int) -> None:
        for side in self.world.sides:
            side.llm_calls_today = 0
            side.errors = []
        # 脚本化价格波动：第3天铁涨价
        for side in self.world.sides:
            side.day_price_mult = {"iron": 2.0} if day == 3 else {}

    def _spawn_robots(self, day: int) -> None:
        world = self.world
        for target_team, corner in (("challenger", (0, 31)), ("defender", (40, 0)),
                                    ("challenger", (0, 0)), ("defender", (40, 31))):
            side = next(s for s in world.sides if s.team == target_team)
            if side.station.health <= 0:
                continue
            mult = 2 if self.hard else 1
            waves = [
                ("smallRobot", (2 + day) * mult),
                ("middleRobot", max(0, day - 1) * mult),
                ("largeRobot", max(0, day - 3) * mult),
                ("bossRobot", (1 if day >= 7 else 0) * mult),
            ]
            for kind, count in waves:
                for _ in range(count):
                    spawn = (corner[0] + world.rng.randrange(3),
                             corner[1] + world.rng.randrange(3))
                    spawn = (min(max(spawn[0], 0), MAP_W - 1),
                             min(max(spawn[1], 0), MAP_H - 1))
                    world.robots.append(MockRobot(
                        world.robot_seq, spawn, kind,
                        ROBOT_STATS[kind][2], target_team,
                    ))
                    world.robot_seq += 1
            for team, kind in list(getattr(world, "pending_summons", [])):
                spawn = (corner[0], corner[1])
                world.robots.append(MockRobot(
                    world.robot_seq, spawn, kind, ROBOT_STATS[kind][2], team))
                world.robot_seq += 1
            world.pending_summons = []

    # ------------------------------------------------ 请求构建

    def _build_request(self, side: Side, round_no: int, day: int, round_in_day: int):
        world = self.world
        zones = [
            {"neutralType": kind, "pos": {"x": p[0], "y": p[1]}}
            for p, kind in world.zones.items()
        ]
        units = [u.dump() for u in side.units if u.health > 0 or u.kind == "station"]
        tasks = []
        for name in ("TaskPoint1", "TaskPoint2"):
            key = f"{side.team}{name}"
            task = MOCK_TASKS[0] if name.endswith("1") else MOCK_TASKS[1]
            cooldown = max(0, side.task_cooldown.get(key, 0) - round_no)
            tasks.append({
                "taskType": task["type"],
                "taskPosition": self._task_point_pos(key),
                "coldDownRounds": cooldown,
                "scoreReward": task["score"],
                "goldReward": task["gold"],
                "isValid": cooldown == 0 and not side.phase_task,
                "timeoutRounds": task["timeout"],
            })
        request = {
            "roundNo": round_no,
            "mapInfo": {"width": MAP_W, "height": MAP_H, "zones": zones},
            "teamOur": {
                "type": side.team, "teamId": side.team, "teamName": side.team,
                "goldNum": side.gold, "totalScore": int(side.score),
                "playerTasks": tasks,
                "roles": units,
            },
            "teamEnemy": {"roles": [u.dump() for u in side.visible_enemies(world)]},
            "robot": {"roles": [r.dump() for r in world.robots if r.health > 0]},
            "phaseTask": side.phase_task,
            "lastRoundRoleActionResults": side.last_action_results,
            "lastSummonTreasureResult": side.last_summon_result,
            "llmResp": side.last_llm_resp,
            "worldNews": {
                "officialNews": "今日无重大新闻",
                "folkLegends": f"第{day}天：传说中的线索指向北方高地的古老祭坛。",
            },
            "lastCmdResult": side.last_cmd_result,
            "vendorShopList": [
                {"name": ore, "price": self._price(side, ore)} for ore in BASE_PRICES
            ],
            "weaponShopList": [
                {"name": n, "price": p} for n, p in [
                    ("WeaponUpgradeVoucher1", 100), ("WeaponUpgradeVoucher2", 150),
                    ("WallUpgradeVoucher1", 20), ("WallUpgradeVoucher2", 30),
                    ("StationUpgradeVoucher1", 100), ("StationUpgradeVoucher2", 150),
                    ("WallFixer", 10), ("Medicine", 10), ("DizzyWeapon", 100),
                    ("Bomb", 100), ("SmallRobotSummonOrder", 20),
                    ("MiddleRobotSummonOrder", 30), ("LargeRobotSummonOrder", 100),
                    ("BossRobotSummonOrder", 200), ("AcientTablet", 15),
                    ("StarSand", 15), ("FlameBreath", 15), ("FrostPotion", 15),
                    ("ThornAmulet", 15), ("IronWhistle", 15),
                ]
            ],
            "errors": side.errors,
        }
        side.last_llm_resp = ""
        side.last_cmd_result = ""
        side.errors = []
        side.last_summon_result = 0
        return request

    def _price(self, side: Side, ore: str) -> int:
        mult = getattr(side, "day_price_mult", {}).get(ore, 1.0)
        return max(1, round(BASE_PRICES[ore] * mult))

    def _task_point_pos(self, key: str) -> dict:
        for kind, pos in INIT_ZONES:
            if kind.lower().startswith(key.lower().replace("taskpoint", "taskpoint")):
                pass
        for kind, pos in INIT_ZONES:
            if kind == key.replace("TaskPoint", "TaskPoint"):
                return {"x": pos[0], "y": pos[1]}
        # challengerTaskPoint1 等
        for kind, pos in INIT_ZONES:
            if kind.lower() == key.lower():
                return {"x": pos[0], "y": pos[1]}
        return {"x": 0, "y": 0}

    # ------------------------------------------------ 调用 Brain

    def _call_brain(self, side: Side, request: dict) -> dict:
        try:
            brain = side.brain.decide if hasattr(side.brain, "decide") else side.brain
            response = brain(copy.deepcopy(request))
            if not isinstance(response, dict) or "roleCommandMap" not in response:
                side.abnormal_count += 1
                return {}
            # 格式硬校验（模拟判题器口径）
            for unit_id, command in response["roleCommandMap"].items():
                if not isinstance(command, dict) or "action" not in command:
                    side.abnormal_count += 1
                    response["roleCommandMap"][unit_id] = {}
            return response
        except Exception as exc:  # noqa: BLE001
            side.abnormal_count += 1
            if self.verbose:
                print(f"[{side.team}] brain crash: {exc}")
            return {}

    # ------------------------------------------------ 指令结算

    def _apply(self, side: Side, response: dict, round_no: int, is_day: bool) -> None:
        world = self.world
        commands = response.get("roleCommandMap", {})
        prompt = response.get("prompt", "")
        execute_cmd = response.get("executeCmd", "")
        results: dict[str, bool] = {}

        # LLM/沙盒脚本化响应
        if prompt:
            if side.phase_task:
                if self.llm_prose and "严格只输出" not in prompt:
                    side.llm_resp_queue.append(self._prose_task_response(side, prompt))
                else:
                    side.llm_resp_queue.append(json.dumps(
                        {"taskAnswer": self._mock_task_answer(side)}))
            elif "民间传闻" in prompt or "祭坛" in prompt or "宝藏" in prompt:
                side.llm_resp_queue.append(json.dumps({
                    "pos": {"x": MOCK_TREASURE["pos"][0], "y": MOCK_TREASURE["pos"][1]},
                    "items": MOCK_TREASURE["items"],
                    "day": MOCK_TREASURE["day"],
                    "round_window": list(MOCK_TREASURE["window"]),
                    "confidence": 0.9,
                }))
            else:
                side.llm_resp_queue.append(json.dumps({"forecasts": []}))
            side.llm_calls_today += 1
            if side.llm_calls_today > 3 and not side.phase_task:
                side.errors.append({"errorCode": 5, "description": "LLM额度超限"})
        if side.llm_resp_queue:
            side.last_llm_resp = side.llm_resp_queue.pop(0)
        if execute_cmd:
            side.cmd_result_queue.append("[exitCode:0]\nmock sandbox output")
        if side.cmd_result_queue:
            side.last_cmd_result = side.cmd_result_queue.pop(0)

        our_map = {u.id: u for u in side.units}
        moves: dict[int, tuple[int, int]] = {}
        char_cmds: dict[int, dict] = {}
        weapon_cmds: dict[int, dict] = {}

        for key, command in commands.items():
            try:
                unit_id = int(key)
            except (TypeError, ValueError):
                side.abnormal_count += 1
                continue
            action = command.get("action")
            if action == "attack":
                weapon_cmds[unit_id] = command
            else:
                char_cmds[unit_id] = command

        # 角色指令
        for unit_id, command in char_cmds.items():
            unit = our_map.get(unit_id)
            if unit is None or unit.health <= 0 or unit.kind not in ("worker", "pioneer"):
                continue
            ok = self._apply_character(side, unit, command, round_no, is_day, moves)
            results[str(unit_id)] = ok

        # 移动碰撞结算（同回合同时移动）
        self._resolve_moves(world, moves)

        # 武器攻击（先于机器人移动）
        for weapon_id, command in weapon_cmds.items():
            weapon = our_map.get(weapon_id)
            if weapon is None or weapon.health <= 0 or not is_day is False:
                pass
            if weapon is None or weapon.health <= 0:
                continue
            ok = self._apply_attack(side, weapon, command, our_map)
            results[str(weapon_id)] = ok

        # 冷却推进
        for unit in side.units:
            if unit.cooldown > 0:
                unit.cooldown -= 1

        side.last_action_results = results

    def _fake_shell(self, cmd: str) -> str:
        """迷你假 shell：覆盖 ls/cat/wc/python3 -c 的模式化输出。"""
        low = cmd.lower()
        if "wc -l" in low:
            return "[exitCode:0]\n12 /tmp/data.txt"
        if "123+456" in cmd.replace(" ", ""):
            return "[exitCode:0]\n579"
        if low.startswith("ls") or " ls " in low:
            return "[exitCode:0]\nREADME.md\ndata.txt\napp.py"
        if "cat" in low:
            return "[exitCode:0]\nmock file content line1\nline2"
        if "python3 -c" in low or "python -c" in low:
            return "[exitCode:0]\n42"
        return "[exitCode:0]\n(ok) " + cmd[:40]

    def _prose_task_response(self, side: Side, prompt: str) -> str:
        """prose 模式：模拟不守格式约定的 LLM（围栏命令/自然语言答案）。"""
        task = side.task_state.get("task")
        if task is None:
            return "我不知道该做什么。"
        if "尚未执行任何命令" in prompt:
            return ("好的，我先探索一下沙盒。\n"
                    "```bash\n" + task["cmd"] + "\n```\n"
                    "执行后再看结果。")
        return f"根据命令输出，本题最终答案: {task['answer']}"

    def _mock_task_answer(self, side: Side) -> str:
        task = side.task_state.get("task")
        return task["answer"] if task else ""

    def _apply_character(self, side: Side, unit: MockUnit, command: dict,
                         round_no: int, is_day: bool,
                         moves: dict) -> bool:
        world = self.world
        action = command.get("action")
        targets = command.get("targetPos") or []
        target = (targets[0]["x"], targets[0]["y"]) if targets else None
        name = command.get("name")

        if action == "move":
            if target and cheb(unit.pos, target) == 1:
                moves[unit.id] = target
                return True
            return False
        if action == "collect":
            if unit.kind != "worker" or target is None:
                return False
            if world.zones.get(target) not in ("stone", "iron", "copper"):
                return False
            if cheb(unit.pos, target) > 1 or len(unit.backpack) >= unit.capacity:
                return False
            ore = world.zones[target]
            unit.backpack.append(ore)
            used = world.mine_uses.get(target, 0) + 1
            world.mine_uses[target] = used
            if used >= 10:
                del world.zones[target]
                self._respawn_mine(ore)
            return True
        if action == "sell":
            if name not in BASE_PRICES or name not in unit.backpack:
                return False
            vendor = next((p for p, k in world.zones.items() if k == "vendor"), None)
            if vendor is None or cheb(unit.pos, vendor) > 1:
                return False
            num = max(1, int(command.get("num", 1)))
            have = unit.backpack.count(name)
            num = min(num, have)
            for _ in range(num):
                unit.backpack.remove(name)
            side.gold += num * self._price(side, name)
            return True
        if action == "buy":
            shop = next((p for p, k in world.zones.items() if k == "weaponShop"), None)
            if shop is None or cheb(unit.pos, shop) > 1:
                return False
            num = max(1, int(command.get("num", 1)))
            price = dict((n, p) for n, p in [
                ("WeaponUpgradeVoucher1", 100), ("WeaponUpgradeVoucher2", 150),
                ("WallUpgradeVoucher1", 20), ("WallUpgradeVoucher2", 30),
                ("StationUpgradeVoucher1", 100), ("StationUpgradeVoucher2", 150),
                ("WallFixer", 10), ("Medicine", 10), ("DizzyWeapon", 100),
                ("Bomb", 100), ("SmallRobotSummonOrder", 20),
                ("MiddleRobotSummonOrder", 30), ("LargeRobotSummonOrder", 100),
                ("BossRobotSummonOrder", 200), ("AcientTablet", 15),
                ("StarSand", 15), ("FlameBreath", 15), ("FrostPotion", 15),
                ("ThornAmulet", 15), ("IronWhistle", 15),
            ]).get(name)
            if price is None or side.gold < price * num:
                return False
            if len(unit.backpack) + num > unit.capacity:
                return False
            side.gold -= price * num
            unit.backpack.extend([name] * num)
            return True
        if action == "build":
            if not is_day or unit.kind != "worker" or target is None:
                return False
            if cheb(unit.pos, target) > 1:
                return False
            station = side.station
            ring = self._ring_distance(target, station.pos)
            if name in ("gatling", "railgun", "rocket"):
                if side.gold < 25:
                    return False
                existing = [u for u in side.units if u.kind in ("gatling", "railgun", "rocket")]
                if len(existing) >= 3 and target not in [u.pos for u in existing]:
                    return False
                side.gold -= 25
                for u in existing:
                    if u.pos == target:
                        side.units.remove(u)
                new_id = self._next_building_id(side, name)
                side.units.append(MockUnit(new_id, target, name, 1000, team=side.team))
                return True
            if name == "wall":
                if "stone" not in unit.backpack:
                    return False
                if ring is None or ring < 2 or ring > 3:
                    return False
                unit.backpack.remove("stone")
                new_id = self._next_building_id(side, "wall")
                side.units.append(MockUnit(new_id, target, "wall", 1000, team=side.team))
                return True
            return False
        if action == "remove":
            if unit.kind != "worker" or target is None:
                return False
            wall = next((u for u in side.units if u.kind == "wall" and u.pos == target), None)
            if wall is None or cheb(unit.pos, target) > 1:
                return False
            side.units.remove(wall)
            return True
        if action == "acceptTask":
            if unit.kind != "pioneer" or side.phase_task:
                return False
            point = self._near_task_point(side, unit.pos)
            if point is None:
                return False
            task = MOCK_TASKS[0] if "TaskPoint1" in point else MOCK_TASKS[1]
            side.phase_task = task["text"]
            side.task_state = {
                "task": task, "key": point, "accept_round": round_no,
                "pos": unit.pos,
            }
            return True
        if action == "submitAnswer":
            if unit.kind != "pioneer" or not side.phase_task:
                return False
            answer = str(command.get("taskAnswer", ""))
            task = side.task_state["task"]
            accept_round = side.task_state["accept_round"]
            if answer.strip() == task["answer"]:
                speed = 5 * task["timeout"] / max(1, round_no - accept_round)
                side.task_score += task["score"] + speed
                side.gold += task["gold"]
            else:
                side.errors.append({"errorCode": 2, "description": "答案错误"})
            side.task_cooldown[side.task_state["key"]] = round_no + 30
            side.phase_task = ""
            side.task_state = {}
            return True
        if action == "summonTreasure":
            if unit.kind != "pioneer" or target is None or cheb(unit.pos, target) > 1:
                return False
            items = list(command.get("item", []))
            if any(i not in unit.backpack for i in items):
                side.last_summon_result = 3
                return True
            day = (round_no - 1) // ROUNDS_PER_DAY + 1
            round_in_day = (round_no - 1) % ROUNDS_PER_DAY + 1
            if self.world.treasure_used:
                side.last_summon_result = 4
            elif (tuple(target) == MOCK_TREASURE["pos"] and
                  items == MOCK_TREASURE["items"] and
                  day == MOCK_TREASURE["day"] and
                  MOCK_TREASURE["window"][0] <= round_in_day <= MOCK_TREASURE["window"][1]):
                for i in items:
                    unit.backpack.remove(i)
                side.treasure_score += MOCK_TREASURE["score"]
                side.gold += MOCK_TREASURE["gold"]
                side.last_summon_result = 1
                self.world.treasure_used = True
            else:
                side.last_summon_result = 2
                for i in items:
                    unit.backpack.remove(i)
            return True
        if action == "use":
            item = name
            if item not in unit.backpack:
                return False
            if item == "Medicine":
                unit.backpack.remove(item)
                unit.health = {"worker": 220, "pioneer": 200}[unit.kind]
                return True
            if item and item.endswith("Voucher1") or item and item.endswith("Voucher2"):
                if target is None:
                    return False
                building = next(
                    (u for u in side.units
                     if u.pos == target and u.kind in ("station", "wall", "gatling", "railgun", "rocket")),
                    None)
                if building is None or cheb(unit.pos, target) > 1:
                    return False
                want = 1 if item.endswith("1") else 2
                if building.level != want:
                    return False
                unit.backpack.remove(item)
                building.level += 1
                stats = {"station": ((0, 0, 1500), (0, 0, 3000), (0, 0, 4500)),
                         "wall": ((0, 0, 1000), (0, 0, 1500), (0, 0, 2000)),
                         "gatling": WEAPON_STATS["gatling"],
                         "railgun": WEAPON_STATS["railgun"],
                         "rocket": WEAPON_STATS["rocket"]}
                table = stats[building.kind]
                building.health = table[min(building.level, 3) - 1][2]
                return True
            if item == "WallFixer":
                wall = next((u for u in side.units if u.kind == "wall" and u.pos == target), None)
                if wall is None:
                    return False
                unit.backpack.remove(item)
                wall.health = (1000, 1500, 2000)[wall.level - 1]
                return True
            if item in ("DizzyWeapon", "Bomb"):
                if target is None:
                    return False
                unit.backpack.remove(item)
                for robot in self.world.robots:
                    if robot.health > 0 and cheb(robot.pos, target) <= 1:
                        if item == "Bomb":
                            robot.health -= 100
                        else:
                            robot.dizzy = 5
                return True
            if item and item.endswith("SummonOrder"):
                unit.backpack.remove(item)
                enemy = self.world.other(side)
                self.world.pending_summons = getattr(self.world, "pending_summons", [])
                kind = {"SmallRobotSummonOrder": "smallRobot",
                        "MiddleRobotSummonOrder": "middleRobot",
                        "LargeRobotSummonOrder": "largeRobot",
                        "BossRobotSummonOrder": "bossRobot"}[item]
                self.world.pending_summons.append((enemy.team, kind))
                return True
            return False
        if action == "drop":
            if name in unit.backpack:
                unit.backpack.remove(name)
                return True
            return False
        if action == "acceptTaskFake":
            return False
        return False

    def _near_task_point(self, side: Side, pos) -> str | None:
        for key in (f"{side.team}TaskPoint1", f"{side.team}TaskPoint2"):
            point = self._task_point_pos(key)
            if cheb(pos, (point["x"], point["y"])) <= 1:
                return key
        return None

    def _ring_distance(self, target, station_pos) -> int | None:
        fp = footprint(station_pos)
        ds = [cheb(target, c) for c in fp]
        return min(ds)

    def _next_building_id(self, side: Side, kind: str) -> int:
        offsets = {"gatling": 20, "railgun": 30, "rocket": 40}
        base = 10000 if side.team == "challenger" else 20000
        if kind in offsets:
            used = [u.id for u in side.units if u.kind == kind]
            return base + offsets[kind] + len(used)
        base = 40000 if side.team == "challenger" else 41000
        used = [u.id for u in side.units if u.kind == "wall"]
        return base + len(used)

    def _respawn_mine(self, ore: str) -> None:
        world = self.world
        for _ in range(50):
            x = world.rng.randrange(MAP_W)
            y = world.rng.randrange(MAP_H)
            if (x, y) not in world.zones and not self._near_base((x, y)):
                world.zones[(x, y)] = ore
                return

    def _near_base(self, pos) -> bool:
        for side in self.world.sides:
            if self._ring_distance(pos, side.station.pos) <= 3:
                return True
        return False

    def _resolve_moves(self, world: World, moves: dict[int, tuple[int, int]]) -> None:
        # 目标被占/被争抢则取消
        blocked = world.blocked_cells("")
        claimed: dict[tuple[int, int], int] = {}
        for unit_id, target in sorted(moves.items()):
            if target in blocked or target in claimed:
                continue
            claimed[target] = unit_id
        final: dict[int, tuple[int, int]] = {
            unit_id: target for target, unit_id in claimed.items()
        }
        # 位置互换取消
        id_to_unit = {u.id: u for s in world.sides for u in s.units}
        for unit_id, target in list(final.items()):
            unit = id_to_unit.get(unit_id)
            if unit is None:
                continue
            other_id = next((oid for oid, t in final.items() if t == unit.pos), None)
            if other_id is not None and other_id != unit_id:
                other = id_to_unit.get(other_id)
                if other is not None and final.get(other_id) == unit.pos and target == other.pos:
                    del final[unit_id]
                    del final[other_id]
        for unit_id, target in final.items():
            unit = id_to_unit.get(unit_id)
            if unit is not None:
                unit.pos = target

    def _apply_attack(self, side: Side, weapon: MockUnit, command: dict,
                      our_map: dict[int, MockUnit]) -> bool:
        world = self.world
        controller_id = command.get("controllerId")
        try:
            controller = our_map.get(int(controller_id))
        except (TypeError, ValueError):
            controller = None
        if controller is None or controller.health <= 0:
            return False
        if cheb(controller.pos, weapon.pos) > 1:
            return False
        targets = command.get("targetPos") or []
        positions = [(t["x"], t["y"]) for t in targets]
        if not positions:
            return False
        reach = WEAPON_STATS[weapon.kind][weapon.level - 1][1]
        for p in positions:
            if cheb(weapon.pos, p) > min(reach, 10**6):
                return False
        robots_by_pos = {r.pos: r for r in world.robots if r.health > 0}

        if weapon.kind == "gatling":
            for target in positions:
                path = [c for c in line_cells(weapon.pos, target) if c != weapon.pos]
                for cell in path:
                    robot = robots_by_pos.get(cell)
                    if robot:
                        robot.health -= 10 * weapon.level
                        break
            return True
        if weapon.kind == "railgun":
            energy = 10 * weapon.level
            path = [c for c in line_cells(weapon.pos, positions[0]) if c != weapon.pos]
            for cell in path:
                robot = robots_by_pos.get(cell)
                if robot and robot.health > 0:
                    dmg = min(energy, robot.health)
                    robot.health -= dmg
                    energy -= dmg
                    if energy <= 0:
                        break
            return True
        if weapon.kind == "rocket":
            if weapon.cooldown > 0:
                return False
            weapon.cooldown = 3
            for target in positions:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cell = (target[0] + dx, target[1] + dy)
                        robot = robots_by_pos.get(cell)
                        if robot:
                            robot.health -= 20 if (dx, dy) == (0, 0) else 10
            return True
        return False

    # ------------------------------------------------ 机器人结算

    def _settle_robots(self, round_no: int, is_day: bool) -> None:
        world = self.world
        if is_day and (round_no - 1) % ROUNDS_PER_DAY + 1 == 1:
            world.robots = []          # 天亮清场
            return

        for robot in world.robots:
            if robot.health <= 0:
                continue
            if robot.dizzy > 0:
                robot.dizzy -= 1
                continue
            side = next(s for s in world.sides if s.team == robot.target)
            station = side.station
            if station.health <= 0:
                continue
            fp = footprint(station.pos)
            # 基地进入攻击距离 -> 直接打基地
            if min(cheb(robot.pos, c) for c in fp) <= 3:
                station.health -= ROBOT_STATS[robot.kind][0]
                continue
            # 向基地走一步；下一步被敌方单位占据则攻击该阻挡者
            step = self._robot_step(robot, fp, hostile=True)
            if step is None:
                continue
            blocker = next(
                (u for u in side.units
                 if u.health > 0 and u.kind != "station" and u.pos == step), None)
            if blocker is not None:
                blocker.health -= ROBOT_STATS[robot.kind][0]
                if blocker.health <= 0 and blocker.kind in ("worker", "pioneer"):
                    blocker.dead_until = round_no + 130
                continue
            other_robot = next(
                (r for r in world.robots if r is not robot and r.health > 0 and r.pos == step),
                None)
            if other_robot is not None:
                continue          # 被己方机器人挡住，原地等待
            robot.pos = step

        # 击杀积分：机器人死在其进攻方的防线前（简化：记给被进攻方）
        for robot in world.robots:
            if robot.health <= 0:
                killer = next(s for s in world.sides if s.team == robot.target)
                killer.kill_score += ROBOT_STATS[robot.kind][3]
        world.robots = [r for r in world.robots if r.health > 0]

        # 角色复活
        for side in world.sides:
            for unit in side.units:
                if unit.kind in ("worker", "pioneer") and unit.health <= 0:
                    if round_no >= unit.dead_until and unit.dead_until:
                        unit.health = {"worker": 220, "pioneer": 200}[unit.kind]
                        unit.pos = (side.station.pos[0], side.station.pos[1] - 2)

    def _robot_step(self, robot: MockRobot, goal_cells, hostile: bool = False):
        """贪心一步逼近目标；hostile=True 时敌方单位所在格视为可走（返回该格表示要攻击它）。"""
        side = next(s for s in self.world.sides if s.team == robot.target)
        blocked = self.world.blocked_cells(robot.target)
        if hostile:
            enemy_cells = set()
            for u in side.units:
                if u.health > 0 and u.kind != "station":
                    enemy_cells.add(u.pos)
            blocked -= enemy_cells
        best = None
        best_dist = cheb(robot.pos, min(goal_cells, key=lambda g: cheb(robot.pos, g)))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                step = (robot.pos[0] + dx, robot.pos[1] + dy)
                if not (0 <= step[0] < MAP_W and 0 <= step[1] < MAP_H):
                    continue
                if step in blocked:
                    continue
                dist = min(cheb(step, g) for g in goal_cells)
                if dist < best_dist:
                    best, best_dist = step, dist
        return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swap", action="store_true", help="双方换边再打一场")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hard", action="store_true", help="机器人波次x2")
    parser.add_argument("--llm-prose", action="store_true",
                        help="mock LLM 用散文+围栏格式回复（验证解析兼容）")
    args = parser.parse_args()

    from agent.brain import Brain
    from agent.brain_baseline import BaselineBrain

    def play(brain_a, brain_b, seed):
        judger = MockJudger(brain_a(), brain_b(), seed=seed, verbose=args.verbose,
                            hard=args.hard, llm_prose=args.llm_prose)
        return judger.run(max_days=args.days)

    report1 = play(Brain, BaselineBrain, args.seed)
    print(json.dumps(report1, ensure_ascii=False, indent=2))
    if args.swap:
        report2 = play(BaselineBrain, Brain, args.seed)
        print(json.dumps(report2, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
