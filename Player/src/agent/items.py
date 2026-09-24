"""物品使用服务：让背包里的东西真正被用掉（买→用闭环）。

由 brain 在昼夜规划之前调用，物品动作抢占战斗/经济 duty；
被物品占用本回合的角色（handled）不再被 economy/combat 重复调度。

优先级：
1. 夜晚应急：机器人逼近基地 -> Bomb(群伤) / DizzyWeapon(>=6只改冻结)
2. 生命药剂：持有者血量 <100 -> 立即用（防死，武器不能整夜没人操控）
3. 召唤令：持有即用（无距离限制）——修复"买了不用"的乌龙
4. 围墙修复包（白天）：残墙 <60% -> 走到墙旁使用
"""
import logging

from .grid import next_step
from .memory import GameMemory
from .protocol import (
    Decision,
    Pos,
    Turn,
    Unit,
    WALL_STATS,
    WEAPON_STATS,
    cmd_use,
    distance,
    station_footprint,
)

LOGGER = logging.getLogger(__name__)

MEDICINE_HP_LINE = 100          # 血量低于该值立即嗑药
WALL_FIX_RATIO = 0.6            # 残墙修复阈值（占当前等级满血比）
EMERGENCY_RADIUS = 1            # 小/中型：贴脸咬到基地才算应急
BIG_ROBOT_RADIUS = 4            # large/boss：逼近到 4 格(啃墙段)算应急
EMERGENCY_WEIGHT = {"smallRobot": 1, "middleRobot": 1, "largeRobot": 2, "bossRobot": 3}
DIZZY_SWARM_SIZE = 6            # 等效数量达到该值用眩晕代替炸弹
EMERGENCY_MIN_WEIGHT = 5        # 等效数量达到该值才触发（啃墙不算）
EMERGENCY_COOLDOWN = 8          # 两次应急之间至少隔 N 回合

SUMMON_ORDERS = (
    "SmallRobotSummonOrder", "MiddleRobotSummonOrder",
    "LargeRobotSummonOrder", "BossRobotSummonOrder",
)


def wall_max_health(level: int) -> int:
    return WALL_STATS[min(max(level, 1), 3) - 1][2]


def building_max_health(unit: Unit) -> int:
    if unit.kind == "wall":
        return wall_max_health(unit.level)
    if unit.kind == "station":
        return (1500, 3000, 4500)[min(max(unit.level, 1), 3) - 1]
    if unit.kind in WEAPON_STATS:
        return WEAPON_STATS[unit.kind][min(max(unit.level, 1), 3) - 1][2]
    return unit.health


class ItemService:
    def __init__(self, memory: GameMemory):
        self.memory = memory

    # ---------------------------------------------------------------- 主入口

    def plan(self, turn: Turn, decision: Decision, claimed: set[Pos]) -> set[int]:
        """返回本回合被物品动作占用的角色 ID 集合。"""
        handled: set[int] = set()
        skip_items = self._last_failed_use_items(turn)
        solving_pioneer = self._solving_pioneer_id(turn)

        # 2) 药剂（最优先：保命）
        self._use_medicine(turn, decision, handled, skip_items, solving_pioneer)
        # 3) 召唤令
        self._use_summon_orders(turn, decision, handled, skip_items, solving_pioneer)
        # 1) 夜晚应急炸弹/眩晕
        if not turn.is_day:
            self._emergency_bomb(turn, decision, handled, skip_items, solving_pioneer)
            self._night_wall_fix(turn, decision, handled, skip_items)
        # 4) 围墙修复（白天）
        else:
            self._repair_walls(turn, decision, claimed, handled, skip_items)
        return handled

    # ---------------------------------------------------------------- 药剂

    def _use_medicine(self, turn: Turn, decision: Decision, handled: set[int],
                      skip_items: set[str], solving_pioneer: int | None) -> None:
        if "Medicine" in skip_items:
            return
        for role in turn.controllables():
            if role.unit_id in handled:
                continue
            if role.health >= MEDICINE_HP_LINE or "Medicine" not in role.backpack:
                continue
            if role.unit_id == solving_pioneer and role.health > 60:
                continue       # 解题中的开拓者除非濒死否则不打断
            decision.commands[role.unit_id] = cmd_use("Medicine")
            handled.add(role.unit_id)
            LOGGER.info("medicine used by %s (hp=%d)", role.unit_id, role.health)
            return

    # ---------------------------------------------------------------- 召唤令

    def _use_summon_orders(self, turn: Turn, decision: Decision, handled: set[int],
                           skip_items: set[str], solving_pioneer: int | None) -> None:
        roles = list(turn.controllables())
        if turn.is_day:
            roles.sort(key=lambda r: 0 if r.kind == "pioneer" else 1)
        else:
            roles.sort(key=lambda r: 0 if r.kind == "worker" else 1)
        for role in roles:
            if role.unit_id in handled or role.unit_id == solving_pioneer:
                continue
            for item in role.backpack:
                if item in SUMMON_ORDERS and item not in skip_items:
                    decision.commands[role.unit_id] = cmd_use(item)
                    handled.add(role.unit_id)
                    LOGGER.info("summon order %s used by %s", item, role.unit_id)
                    return

    # ---------------------------------------------------------------- 应急炸弹/眩晕

    def _emergency_bomb(self, turn: Turn, decision: Decision, handled: set[int],
                        skip_items: set[str], solving_pioneer: int | None) -> None:
        station = turn.station()
        if station is None or station.health <= 0:
            return
        footprint = station_footprint(station.pos)
        threats = []
        for robot in turn.robots:
            if robot.health <= 0 or robot.dizzy:
                continue
            gap = min(distance(robot.pos, cell) for cell in footprint)
            limit = BIG_ROBOT_RADIUS if robot.kind in ("largeRobot", "bossRobot") \
                else EMERGENCY_RADIUS
            if gap <= limit:
                threats.append(robot)
        weight = sum(EMERGENCY_WEIGHT.get(r.kind, 1) for r in threats)
        big_near = any(
            r.kind in ("largeRobot", "bossRobot") for r in threats
        )
        if not big_near and weight < EMERGENCY_MIN_WEIGHT:
            return
        # 冷却：丢一枚要顶 8 回合，避免整晚丢炸弹不开火
        if turn.round_no - self.memory.last_emergency_round < EMERGENCY_COOLDOWN:
            return

        use_dizzy = weight >= DIZZY_SWARM_SIZE
        name = "DizzyWeapon" if use_dizzy else "Bomb"
        if name in skip_items:
            name = "Bomb" if use_dizzy else "DizzyWeapon"
            if name in skip_items:
                return
        centroid = Pos(
            round(sum(r.pos.x for r in threats) / len(threats)),
            round(sum(r.pos.y for r in threats) / len(threats)),
        )
        # 距机群最近的可用操控者；眩晕/炸弹无使用距离限制
        candidates = [
            r for r in turn.controllables()
            if r.unit_id not in handled and r.unit_id != solving_pioneer
            and name in r.backpack
        ]
        if not candidates:
            other = "Bomb" if use_dizzy else "DizzyWeapon"
            if other in skip_items:
                return
            candidates = [
                r for r in turn.controllables()
                if r.unit_id not in handled and r.unit_id != solving_pioneer
                and other in r.backpack
            ]
            name = other
            if not candidates:
                return
        user = min(candidates, key=lambda r: distance(r.pos, centroid))
        decision.commands[user.unit_id] = cmd_use(name, centroid)
        handled.add(user.unit_id)
        self.memory.last_emergency_round = turn.round_no
        LOGGER.info("emergency %s at %s by %s", name, centroid, user.unit_id)

    # ---------------------------------------------------------------- 围墙修复

    def _repair_walls(self, turn: Turn, decision: Decision, claimed: set[Pos],
                      handled: set[int], skip_items: set[str]) -> None:
        if "WallFixer" in skip_items:
            return
        damaged = [
            wall for wall in turn.walls()
            if wall.health < wall_max_health(wall.level) * WALL_FIX_RATIO
        ]
        if not damaged:
            return
        for role in turn.workers():
            if role.unit_id in handled or "WallFixer" not in role.backpack:
                continue
            target = min(
                damaged,
                key=lambda w: (distance(role.pos, w.pos), w.pos.x, w.pos.y),
            )
            if distance(role.pos, target.pos) <= 1:
                decision.commands[role.unit_id] = cmd_use("WallFixer", target.pos)
            else:
                step = next_step(turn, role, target.pos)
                if step is None or step in claimed:
                    continue
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
            handled.add(role.unit_id)
            return

    # ---------------------------------------------------------------- 夜晚修墙

    def _night_wall_fix(self, turn: Turn, decision: Decision, handled: set[int],
                        skip_items: set[str]) -> None:
        """夜晚 WallFixer 无使用限制：残墙 <30% 时最近的操控者抢修一回合，
        防止缺口在火力下被啃穿（比让墙塌了再补的代价小得多）。"""
        if "WallFixer" in skip_items:
            return
        critical = [
            wall for wall in turn.walls()
            if wall.health < wall_max_health(wall.level) * 0.3
        ]
        if not critical:
            return
        for role in turn.controllables():
            if role.unit_id in handled or "WallFixer" not in role.backpack:
                continue
            target = min(
                critical,
                key=lambda w: (distance(role.pos, w.pos), w.pos.x, w.pos.y),
            )
            decision.commands[role.unit_id] = cmd_use("WallFixer", target.pos)
            handled.add(role.unit_id)
            LOGGER.info("night wall fix at %s by %s", target.pos, role.unit_id)
            return

    # ---------------------------------------------------------------- 工具

    def _solving_pioneer_id(self, turn: Turn) -> int | None:
        if self.memory.evolve.phase == "solving":
            pioneer = turn.pioneer()
            if pioneer is not None:
                return pioneer.unit_id
        return None

    def _last_failed_use_items(self, turn: Turn) -> set[str]:
        skip: set[str] = set()
        for command in self.memory.failed_actions.values():
            if command.get("action") == "use":
                name = command.get("name")
                if isinstance(name, str):
                    skip.add(name)
        return skip
