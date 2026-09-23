"""夜战模块：角色操控武器开火。

规则要点：
- attack 以武器 ID 为 key，controllerId 为站在武器旁一格的角色；
- 加特林：等级=子弹数，目标须落在同一 90° 锥内，每颗子弹命中弹道上最近机器人（10 伤害/颗）；
- 电磁炮：单目标，能量=10×等级，沿弹道穿透机器人，每只受 min(剩余能量,自身血量)；
- 火箭：等级=导弹数，落点中心 20、周围 8 格溅射 10，可叠加，发射后冷却 3 回合，弹道无阻挡。

目标优先级：打我方的 > 打敌方的；大/BOSS 优先（击杀积分高、威胁大）；
同时避免过量伤害浪费（如电磁炮能量不砸在小机器人堆里）。
"""
import logging
import math

from .grid import next_step
from .protocol import (
    Decision,
    Pos,
    Robot,
    Turn,
    Unit,
    cmd_attack,
    distance,
)

LOGGER = logging.getLogger(__name__)


def line_cells(start: Pos, end: Pos) -> list[Pos]:
    """两中心连线经过的格子（含端点），用于弹道命中判定。"""
    cells: list[Pos] = []
    x0, y0 = start.x, start.y
    x1, y1 = end.x, end.y
    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return [start]
    for i in range(steps + 1):
        t = i / steps
        # 四舍五入取样；切比雪夫几何下足以覆盖直线主格子
        cx = x0 + t * dx
        cy = y0 + t * dy
        cells.append(Pos(round(cx), round(cy)))
    # 去重保序
    seen: set[Pos] = set()
    unique: list[Pos] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


def robots_on_path(turn: Turn, start: Pos, end: Pos) -> list[Robot]:
    """弹道 start->end 上按距离排序的机器人（不含 start 处）。"""
    path = set(line_cells(start, end))
    path.discard(start)
    hits = [r for r in turn.robots if r.pos in path and r.health > 0]
    hits.sort(key=lambda r: distance(start, r.pos))
    return hits


def angle_between(base: Pos, a: Pos, b: Pos) -> float:
    """以 base 为顶点，a/b 两方向夹角（度）。"""
    v1 = (a.x - base.x, a.y - base.y)
    v2 = (b.x - base.x, b.y - base.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / (m1 * m2)))
    return math.degrees(math.acos(cos))


def _threat_score(turn: Turn, robot: Robot, station_pos: Pos | None) -> float:
    score = robot.score * 10.0
    if robot.target_team == turn.team_type:
        score += 30.0
        if station_pos is not None:
            score += max(0.0, 40 - distance(robot.pos, station_pos))
    if robot.dizzy:
        score -= 5.0            # 眩晕目标仍可打，但优先级略降
    return score


class Combat:
    def plan_night(self, turn: Turn, decision: Decision, claimed: set[Pos],
                   exclude: set[int] | None = None) -> None:
        station = turn.station()
        station_pos = station.pos if station else None
        weapons = turn.weapons()
        exclude = exclude or set()
        roles = [r for r in turn.controllables() if r.unit_id not in exclude]

        if not weapons:
            self._hide(turn, roles, decision, claimed)
            return

        # 角色-武器配对：按序号稳定绑定，角色数可能多于/少于武器数
        pairs: list[tuple[Unit, Unit]] = []
        for index, weapon in enumerate(weapons):
            if index < len(roles):
                pairs.append((roles[index], weapon))
        # 多出来的角色跟随第一个武器附近（备用操控手）
        for role in roles[len(weapons):]:
            pairs.append((role, weapons[0]))

        used_roles: set[int] = set()
        for role, weapon in pairs:
            if role.unit_id in used_roles:
                continue
            if distance(role.pos, weapon.pos) <= 1:
                used_roles.add(role.unit_id)
                if weapon.cooldown > 0:
                    continue          # 火箭冷却中
                targets = self._targets_for(turn, weapon, station_pos)
                if targets:
                    decision.commands[weapon.unit_id] = cmd_attack(
                        role.unit_id, targets,
                    )
            else:
                used_roles.add(role.unit_id)
                step = next_step(turn, role, weapon.pos)
                if step is not None and step not in claimed:
                    claimed.add(step)
                    decision.commands[role.unit_id] = {
                        "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                    }

    # ---- 各武器目标选择

    def _targets_for(self, turn: Turn, weapon: Unit, station_pos: Pos | None) -> list[Pos]:
        if weapon.kind == "gatling":
            return self._gatling_targets(turn, weapon, station_pos)
        if weapon.kind == "railgun":
            return self._railgun_targets(turn, weapon, station_pos)
        if weapon.kind == "rocket":
            return self._rocket_targets(turn, weapon, station_pos)
        return []

    def _in_range_robots(self, turn: Turn, weapon: Unit) -> list[Robot]:
        reach = weapon.range_of_attack()
        return [
            r for r in turn.robots
            if r.health > 0 and distance(weapon.pos, r.pos) <= reach
        ]

    def _gatling_targets(self, turn: Turn, weapon: Unit, station_pos: Pos | None) -> list[Pos]:
        max_targets = weapon.max_targets
        robots = self._in_range_robots(turn, weapon)
        if not robots:
            return []
        robots.sort(key=lambda r: (-_threat_score(turn, r, station_pos), distance(weapon.pos, r.pos)))

        picked: list[Robot] = []
        for robot in robots:
            if len(picked) >= max_targets:
                break
            if all(
                angle_between(weapon.pos, robot.pos, other.pos) <= 90.0
                for other in picked
            ):
                picked.append(robot)
        return [r.pos for r in picked]

    def _railgun_targets(self, turn: Turn, weapon: Unit, station_pos: Pos | None) -> list[Pos]:
        energy = max(weapon.level, 1) * 10
        robots = self._in_range_robots(turn, weapon)
        best: tuple[float, Pos] | None = None
        for robot in robots:
            path_robots = robots_on_path(turn, weapon.pos, robot.pos)
            # 沿途可被能量覆盖的击杀积分期望
            gain = 0.0
            left = energy
            for hit in path_robots:
                dmg = min(left, hit.health)
                left -= dmg
                if dmg >= hit.health:
                    gain += hit.score
                else:
                    gain += 0.5        # 未击杀也算削血收益
                if left <= 0:
                    break
            gain += _threat_score(turn, robot, station_pos) * 0.1
            if best is None or gain > best[0]:
                best = (gain, robot.pos)
        return [best[1]] if best else []

    def _rocket_targets(self, turn: Turn, weapon: Unit, station_pos: Pos | None) -> list[Pos]:
        missiles = weapon.max_targets
        robots = self._in_range_robots(turn, weapon)
        if not robots:
            return []
        # 候选落点 = 机器人所在格，按 3x3 溅射覆盖的威胁期望排序
        scored: list[tuple[float, Pos]] = []
        for robot in robots:
            gain = 0.0
            for other in robots:
                d = distance(robot.pos, other.pos)
                if d == 0:
                    gain += min(20, other.health) / max(other.health, 1) * other.score * 10
                elif d <= 1:
                    gain += min(10, other.health) / max(other.health, 1) * other.score * 5
            gain += _threat_score(turn, robot, station_pos) * 0.2
            scored.append((gain, robot.pos))
        scored.sort(key=lambda t: -t[0])

        targets: list[Pos] = []
        for gain, pos in scored:
            if len(targets) >= missiles:
                break
            targets.append(pos)
            if gain >= 40:
                targets.append(pos)     # 高价值大目标叠弹速杀
        return targets[:missiles]

    # ---- 无武器时：角色撤到基地旁躲避

    def _hide(self, turn: Turn, roles: list[Unit], decision: Decision,
              claimed: set[Pos]) -> None:
        station = turn.station()
        if station is None:
            return
        for role in roles:
            if distance(role.pos, station.pos) <= 2:
                continue
            step = next_step(turn, role, station.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
