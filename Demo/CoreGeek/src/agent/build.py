"""建造规划：炮台位（基地旁一圈）与围墙防线（外圈两格，留门口）。

布局依据：
- 武器建在基地旁一格（可建造区蓝色区域推断），围墙在外圈两格（黄色区域推断）；
  试建造失败的格子记入 memory.build_failures，后续跳过。
- 任务书规定子弹/电磁能量只被机器人吸收，围墙不挡弹道，
  因此"围墙圈+圈内炮台"可行：炮台可越过围墙打击啃墙的机器人。
"""
from .memory import GameMemory
from .protocol import (
    Pos,
    Turn,
    distance,
    station_footprint,
)


def footprint_distance(pos: Pos, footprint: tuple[Pos, ...]) -> int:
    return min(distance(pos, cell) for cell in footprint)


def tower_sites(turn: Turn, memory: GameMemory | None = None) -> list[Pos]:
    """基地周围一格的炮台位（固定锚点：不管是否已建成，列表保持稳定）。

    稳定性很重要：炮台 loadout(gatling/railgun/rocket) 按本列表索引分配，
    若已建成的炮台把格子挤出去，会导致后续建造搭配错位。
    """
    station = turn.station()
    if station is None:
        return []
    footprint = station_footprint(station.pos)
    candidates = [
        n for cell in footprint for n in cell.neighbours()
        if turn.on_map(n)
        and footprint_distance(n, footprint) == 1
        and n not in turn.zones
    ]
    seen: set[Pos] = set()
    unique: list[Pos] = []
    for pos in sorted(candidates, key=lambda p: (p.x, p.y)):
        if pos in seen:
            continue
        seen.add(pos)
        if memory is not None and (pos.x, pos.y) in memory.build_failures:
            continue
        unique.append(pos)
    unique.sort(key=lambda p: (footprint_distance(p, footprint), p.x, p.y))
    return unique[:3]


def wall_plan(turn: Turn, memory: GameMemory | None = None) -> list[Pos]:
    """基地外圈两格的围墙防线，顺时针列出，留一个门口。"""
    station = turn.station()
    if station is None:
        return []
    footprint = station_footprint(station.pos)
    xs = [p.x for p in footprint]
    ys = [p.y for p in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    order = [
        *(Pos(x, ymin - 2) for x in range(xmin - 2, xmax + 3)),        # 南
        *(Pos(xmax + 2, y) for y in range(ymin - 1, ymax + 2)),        # 东
        *(Pos(x, ymax + 2) for x in range(xmin - 2, xmax + 3)),        # 北
        *(Pos(xmin - 2, y) for y in range(ymin - 1, ymax + 2)),        # 西
    ]
    # 东南角留门（机器人来袭方向未知，固定门口便于角色进出与防守集中）
    entrance = Pos(xmax + 2, ymin - 2)
    occupied = turn.occupied_cells()
    planned: list[Pos] = []
    for pos in order:
        if pos == entrance:
            continue
        if not turn.on_map(pos):
            continue
        if pos in turn.zones:
            continue
        if pos in occupied:
            continue
        if memory is not None and (pos.x, pos.y) in memory.build_failures:
            continue
        planned.append(pos)
    return planned


def walls_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    plan = wall_plan(turn, memory)
    standing = {unit.pos for unit in turn.walls()}
    return [pos for pos in plan if pos not in standing]


def towers_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    sites = tower_sites(turn, memory)
    standing = {unit.pos for unit in turn.weapons()}
    return [pos for pos in sites if pos not in standing]
