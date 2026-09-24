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


def wall_ring(turn: Turn) -> frozenset[Pos]:
    """围墙圈静态几何（不含门口、不含中立/出界格），供建造与选址共用。"""
    station = turn.station()
    if station is None:
        return frozenset()
    footprint = station_footprint(station.pos)
    xs = [p.x for p in footprint]
    ys = [p.y for p in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    entrance = Pos(xmax + 2, ymin - 2)
    cells: set[Pos] = set()
    for x in range(xmin - 2, xmax + 3):
        for y in range(ymin - 2, ymax + 3):
            pos = Pos(x, y)
            ring = footprint_distance(pos, footprint)
            if ring != 2 or pos == entrance or not turn.on_map(pos):
                continue
            if pos in turn.zones:
                continue
            cells.add(pos)
    return frozenset(cells)


def tower_sites(turn: Turn, memory: GameMemory | None = None) -> list[Pos]:
    """基地周围一格的炮台位（固定锚点）。

    选址准则（连通性优先）：三座塔的"可站位格"必须从门口经内部走廊全部可达，
    否则会出现第三座炮台永远无人可控的结构性死位。在所有 3-组合里选
    (连通可行, 可站位格最多, 打散度大) 的最优解；几何静态，结果稳定。
    """
    from itertools import combinations

    station = turn.station()
    if station is None:
        return []
    footprint = station_footprint(station.pos)
    ring2 = wall_ring(turn)
    candidates = sorted({
        n for cell in footprint for n in cell.neighbours()
        if turn.on_map(n)
        and footprint_distance(n, footprint) == 1
        and n not in turn.zones
        and n not in ring2
    }, key=lambda p: (p.x, p.y))
    if memory is not None:
        candidates = [
            p for p in candidates if (p.x, p.y) not in memory.build_failures
        ]
    if len(candidates) < 3:
        return candidates

    door = Pos(max(p.x for p in footprint) + 2, min(p.y for p in footprint) - 2)
    door_adjacent = {
        n for n in door.neighbours()
        if footprint_distance(n, footprint) == 1 and n in candidates
    }

    def evaluate(combo: tuple[Pos, ...]) -> tuple[int, int, int] | None:
        blocked = set(footprint) | set(combo) | set(ring2) | set(turn.zones)
        free = {
            p for p in candidates
            if p not in combo and p not in blocked
        }
        starts = door_adjacent & free
        visited: set[Pos] = set(starts)
        frontier = list(starts)
        while frontier:
            cur = frontier.pop()
            for n in cur.neighbours():
                if n in free and n not in visited:
                    visited.add(n)
                    frontier.append(n)
        stands: set[Pos] = set()
        reachable_stands = 0
        for site in combo:
            mine = {n for n in site.neighbours() if n in free}
            stands |= mine
            if mine & visited:
                reachable_stands += 1
        connected = int(len(stands) == len(stands & visited) and reachable_stands == 3
                        and bool(starts))
        spread = min(
            (distance(a, b) for a, b in combinations(combo, 2)), default=0,
        )
        return (connected, reachable_stands, len(stands), spread)

    best: tuple[Pos, ...] | None = None
    best_score: tuple[int, int, int, int] | None = None
    for combo in combinations(candidates, 3):
        score = evaluate(combo)
        if score is None:
            continue
        full = (score[0], score[1], score[2], score[3])
        if best_score is None or full > best_score:
            best_score = full
            best = combo
    if best is None:
        return candidates[:3]
    return list(best)

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


def entrance_pos(turn: Turn) -> Pos | None:
    """围墙圈的门口位置（与 wall_plan 排除的是同一格），关门战术目标点。"""
    station = turn.station()
    if station is None:
        return None
    footprint = station_footprint(station.pos)
    xs = [p.x for p in footprint]
    ys = [p.y for p in footprint]
    return Pos(max(xs) + 2, min(ys) - 2)


def walls_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    plan = wall_plan(turn, memory)
    standing = {unit.pos for unit in turn.walls()}
    return [pos for pos in plan if pos not in standing]


def towers_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    sites = tower_sites(turn, memory)
    standing = {unit.pos for unit in turn.weapons()}
    return [pos for pos in sites if pos not in standing]
