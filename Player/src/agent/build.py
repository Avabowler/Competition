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


def attack_side(turn: Turn) -> str:
    """机器人来袭方向："west"（基地在地图左半）/"east"（右半）。

    实战情报：红方（基地方位偏左）机器人从左来，蓝方（偏右）从右来；
    半场换边后基地坐标互换，此推断自动适配。
    """
    station = turn.station()
    if station is None:
        return "west"
    return "west" if station.pos.x * 2 < turn.width else "east"


def entrance_side(turn: Turn) -> str:
    """门口开在来袭方向的背面。"""
    return "east" if attack_side(turn) == "west" else "west"


def wall_rows(turn: Turn) -> dict[str, list[Pos]]:
    """围墙圈四排格子（不含门口，门口由调用方排除）。"""
    station = turn.station()
    if station is None:
        return {}
    footprint = station_footprint(station.pos)
    xs = [p.x for p in footprint]
    ys = [p.y for p in footprint]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return {
        "south": [Pos(x, ymin - 2) for x in range(xmin - 2, xmax + 3)],
        "north": [Pos(x, ymax + 2) for x in range(xmin - 2, xmax + 3)],
        "west": [Pos(xmin - 2, y) for y in range(ymin - 1, ymax + 2)],
        "east": [Pos(xmax + 2, y) for y in range(ymin - 1, ymax + 2)],
    }


def front_cells(turn: Turn) -> frozenset[Pos]:
    """来袭正面的墙排（修复/升级优先级最高）。"""
    rows = wall_rows(turn)
    return frozenset(rows.get(attack_side(turn), ()))


def wall_ring(turn: Turn) -> frozenset[Pos]:
    """围墙圈静态几何（不含门口、不含中立/出界格），供建造与选址共用。"""
    door = entrance_pos(turn)
    cells: set[Pos] = set()
    for row in wall_rows(turn).values():
        for pos in row:
            if pos == door or not turn.on_map(pos) or pos in turn.zones:
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

    door = entrance_pos(turn)
    door_adjacent = set()
    if door is not None:
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
    """围墙防线：**来袭正面优先**建造，门口开在背面。

    来袭方向由基地在地图的左右半区推断（左半区基地 -> 机器人从左来）。
    建造顺序：正面排 -> 南 -> 北 -> 背面排（门口在背面排上，最后合拢）。
    """
    station = turn.station()
    if station is None:
        return []
    rows = wall_rows(turn)
    front = attack_side(turn)
    back = entrance_side(turn)
    order = [*rows[front], *rows["south"], *rows["north"], *rows[back]]
    door = entrance_pos(turn)
    occupied = turn.occupied_cells()
    planned: list[Pos] = []
    for pos in order:
        if door is not None and pos == door:
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
    """围墙圈的门口位置：来袭方向背面的角格。

    基地在左半区（机器人从左来）-> 门开东侧 (xmax+2, ymin-2)；
    基地在右半区（机器人从右来）-> 门开西侧 (xmin-2, ymin-1)。
    """
    station = turn.station()
    if station is None:
        return None
    footprint = station_footprint(station.pos)
    xs = [p.x for p in footprint]
    ys = [p.y for p in footprint]
    if entrance_side(turn) == "east":
        return Pos(max(xs) + 2, min(ys) - 2)
    return Pos(min(xs) - 2, min(ys) - 1)


def walls_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    plan = wall_plan(turn, memory)
    standing = {unit.pos for unit in turn.walls()}
    return [pos for pos in plan if pos not in standing]


def towers_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    sites = tower_sites(turn, memory)
    standing = {unit.pos for unit in turn.weapons()}
    return [pos for pos in sites if pos not in standing]
