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

from .grid import approach_step, next_step
from .memory import GameMemory
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

MULTI_CONTROL_MAX_STREAKS = 3   # 聚控攻击连续整轮全失败 N 次后回退 1:1


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
    def __init__(self, memory: GameMemory) -> None:
        self.memory = memory

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

        # 聚控优先：开拓者一人操控 cluster 内全部炮台（工人腾出去夜矿）
        if self._cluster_control(turn, decision, claimed, weapons, roles,
                                 exclude, station_pos):
            return

        # 每座武器的可站位格（相邻且可通行），操控配对按距离贪心：
        # 关键约束是一座武器的可站位格可能只剩 1 个，固定 zip 配对会让
        # 某座炮台永远无人可控（真实事故：加特林整夜哑火）。
        stand_cells: dict[int, list[Pos]] = {}
        for weapon in weapons:
            blocked = turn.blocked(None)
            cells = [
                n for n in weapon.pos.neighbours()
                if turn.on_map(n) and n not in blocked
            ]
            cells.sort(key=lambda p: (p.x, p.y))
            stand_cells[weapon.unit_id] = cells

        assignments: list[tuple[Unit, Unit, Pos | None]] = []  # (role, weapon, stand)
        used_weapons: set[int] = set()
        used_stands: set[Pos] = set()
        unassigned: list[Unit] = []
        for role in roles:
            best: tuple[int, int, Unit, Pos | None] | None = None
            for w_index, weapon in enumerate(weapons):
                if weapon.unit_id in used_weapons:
                    continue
                here = distance(role.pos, weapon.pos) <= 1
                candidates: list[Pos] = []
                if here:
                    candidates = [role.pos]
                else:
                    candidates = [
                        c for c in stand_cells[weapon.unit_id]
                        if c not in used_stands and c not in claimed
                    ]
                if not candidates:
                    continue
                stand = min(candidates, key=lambda p: (distance(role.pos, p), p.x, p.y))
                cost = (distance(role.pos, stand), w_index)
                if best is None or cost < best[:2]:
                    best = (cost[0], cost[1], weapon, stand)
            if best is None:
                unassigned.append(role)
                continue
            _, _, weapon, stand = best
            used_weapons.add(weapon.unit_id)
            used_stands.add(stand)
            assignments.append((role, weapon, stand))

        # 多余角色跟随第一座武器（备用操控手，不重复开火）
        for role in unassigned:
            assignments.append((role, weapons[0], None))

        fired_weapons: set[int] = set()
        for role, weapon, stand in assignments:
            if distance(role.pos, weapon.pos) <= 1:
                if weapon.unit_id in fired_weapons or weapon.cooldown > 0:
                    continue
                targets = self._targets_for(turn, weapon, station_pos)
                if targets:
                    fired_weapons.add(weapon.unit_id)
                    decision.commands[weapon.unit_id] = cmd_attack(
                        role.unit_id, targets,
                    )
                continue
            goal = stand if stand is not None else weapon.pos
            step = next_step(turn, role, goal)
            if step is None:
                step = next_step(turn, role, weapon.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }

    # ---- 聚控：开拓者一人操控三塔

    def _cluster_control(self, turn: Turn, decision: Decision, claimed: set[Pos],
                         weapons: tuple[Unit, ...], roles: list[Unit],
                         exclude: set[int], station_pos: Pos | None) -> bool:
        """开拓者站控制位同时操控相邻的多座炮台。返回是否接管了本回合夜战。"""
        if self.memory.multi_control_failed:
            return False
        seat = self.memory.tower_seat
        if seat is None:
            return False
        seat_pos = Pos(*seat)
        pioneer = next(
            (r for r in roles if r.kind == "pioneer"), None,
        )
        if pioneer is None:
            return False
        cluster = [w for w in weapons if distance(w.pos, seat_pos) <= 1]
        if len(cluster) < 2:
            return False       # 聚控布局不存在（少于 2 座塔相邻控制位）

        self._track_multi_control_health(cluster)

        if distance(pioneer.pos, seat_pos) > 1 \
                and pioneer.unit_id not in decision.commands:
            step = next_step(turn, pioneer, seat_pos) \
                or approach_step(turn, pioneer, seat_pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[pioneer.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
        # 错峰轮发：火箭冷却同为 3 回合，每回合只开一门（轮转），
        # 形成"每回合一发"的不间断火力，也避免多弹同时砸同一目标造成过量
        ready: list[tuple[Unit, list[Pos]]] = []
        for weapon in cluster:
            if weapon.cooldown > 0 or distance(pioneer.pos, weapon.pos) > 1:
                continue
            targets = self._targets_for(turn, weapon, station_pos)
            if targets:
                ready.append((weapon, targets))
        if ready:
            weapon, targets = ready[self.memory.fire_rotate % len(ready)]
            self.memory.fire_rotate += 1
            decision.commands[weapon.unit_id] = cmd_attack(
                pioneer.unit_id, targets,
            )
        # 其余未被 exclude 的角色（如 multi_ok 下没去夜矿的工人）跟随第一座塔待命
        for role in roles:
            if role.unit_id == pioneer.unit_id or role.unit_id in decision.commands:
                continue
            goal = cluster[0].pos
            if distance(role.pos, goal) <= 1:
                continue
            step = next_step(turn, role, goal)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
        return True

    def _track_multi_control_health(self, cluster: list[Unit]) -> None:
        """上回合聚控攻击若"发了却全失败"计一次连续失败；连续 N 次回退 1:1。"""
        issued = [
            w.unit_id for w in cluster
            if self.memory.last_commands.get(w.unit_id, {}).get("action") == "attack"
        ]
        if not issued:
            return
        if any(unit_id in self.memory.failed_actions for unit_id in issued):
            self.memory.multi_control_failures += 1
        else:
            self.memory.multi_control_failures = 0
        if self.memory.multi_control_failures >= MULTI_CONTROL_MAX_STREAKS:
            self.memory.multi_control_failed = True
            self.memory.multi_control_failures = 0
            LOGGER.warning("multi-control reverted to 1:1 after repeated failures")

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
