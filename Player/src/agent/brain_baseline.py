"""Demo(CoreGeek) 原版策略的移植版，仅作 A/B 回归基线，不参与正式决策。

逻辑与 Demo/CoreGeek/src/agent/brain.py 一致：
白天建3炮台+围墙圈+采石；夜晚角色配对武器攻击最近机器人。
"""
from typing import Any

from .build import tower_sites, wall_plan
from .grid import next_step
from .protocol import (
    Decision,
    Pos,
    Turn,
    Unit,
    WEAPON_BUILD_COST,
    cmd_attack,
    cmd_build,
    cmd_collect,
    distance,
    station_footprint,
)

TOWER_LOADOUT = ("gatling", "railgun", "rocket")
STONE_BATCH = 6


class BaselineBrain:
    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        turn = Turn.load(payload)
        decision = Decision()
        if turn.is_day:
            self._day(turn, decision)
        else:
            self._night(turn, decision)
        return decision.dump()

    def _day(self, turn: Turn, decision: Decision) -> None:
        sites = tower_sites(turn)
        order = wall_plan(turn)
        standing_towers = {u.pos for u in turn.weapons()}
        standing_walls = {u.pos for u in turn.walls()}
        occupied = turn.occupied_cells()
        towers_missing = [p for p in sites if p not in standing_towers]
        walls_missing = [p for p in order if p not in standing_walls]

        for role in turn.workers():
            self._worker_day(
                turn, role, sites, towers_missing, walls_missing, decision,
            )

    def _worker_day(self, turn: Turn, role: Unit, sites, towers_missing,
                    walls_missing, decision: Decision) -> None:
        if towers_missing and turn.gold >= WEAPON_BUILD_COST:
            for index, site in enumerate(sites):
                if site in towers_missing:
                    self._build_or_walk(turn, role, site, TOWER_LOADOUT[index], decision)
                    return
        if not walls_missing:
            return
        stones = role.backpack.count("stone")
        mine = self._adjacent_mine(turn, role)
        if mine is not None and stones < STONE_BATCH:
            decision.commands[role.unit_id] = cmd_collect(mine)
            return
        if stones:
            for site in walls_missing:
                self._build_or_walk(turn, role, site, "wall", decision)
                return
            return
        self._mine(turn, role, decision)

    def _adjacent_mine(self, turn: Turn, role: Unit) -> Pos | None:
        mines = sorted(
            (m for m in turn.mines_of("stone")
             if role.pos != m and distance(role.pos, m) <= 1),
            key=lambda p: (distance(role.pos, p), p.x, p.y),
        )
        return mines[0] if mines else None

    def _night(self, turn: Turn, decision: Decision) -> None:
        roles = turn.controllables()
        weapons = turn.weapons()
        for role, tower in zip(roles, weapons):
            if distance(role.pos, tower.pos) <= 1:
                if tower.cooldown > 0:
                    continue
                target = self._attack_target(turn, tower)
                if target is not None:
                    decision.commands[tower.unit_id] = cmd_attack(role.unit_id, target)
                continue
            step = next_step(turn, role, tower.pos)
            if step is not None:
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }

    def _attack_target(self, turn: Turn, tower: Unit) -> Pos | None:
        reach = tower.range_of_attack()
        targets = [
            r for r in turn.robots
            if r.health > 0 and distance(tower.pos, r.pos) <= reach
        ]
        if not targets:
            return None
        nearest = min(targets, key=lambda r: (distance(tower.pos, r.pos), r.robot_id))
        return nearest.pos

    def _build_or_walk(self, turn: Turn, role: Unit, target: Pos, name: str,
                       decision: Decision) -> None:
        if role.pos != target and distance(role.pos, target) <= 1:
            decision.commands[role.unit_id] = cmd_build(name, target)
            return
        step = next_step(turn, role, target)
        if step is not None:
            decision.commands[role.unit_id] = {
                "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
            }

    def _mine(self, turn: Turn, role: Unit, decision: Decision) -> None:
        if role.capacity is not None and len(role.backpack) >= role.capacity:
            return
        mines = sorted(
            turn.mines_of("stone"),
            key=lambda p: (distance(role.pos, p), p.x, p.y),
        )
        for mine in mines:
            if role.pos != mine and distance(role.pos, mine) <= 1:
                decision.commands[role.unit_id] = cmd_collect(mine)
                return
            step = next_step(turn, role, mine)
            if step is not None:
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
                return
