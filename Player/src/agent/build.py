"""建造规划：炮台位（基地旁一圈）与围墙防线（外圈两格，留门口）。

布局依据：
- 武器建在基地旁一格（可建造区蓝色区域推断），围墙在外圈两格（黄色区域推断）；
  试建造失败的格子记入 memory.build_failures，后续跳过。
- 任务书规定子弹/电磁能量只被机器人吸收，围墙不挡弹道，
  因此"围墙圈+圈内炮台"可行：炮台可越过围墙打击啃墙的机器人。
- 聚控布局：控制位 S 也是 ring1 空格（不建东西），三座塔建在 S 周围的
  ring1 相邻格上——开拓者站 S 即与三塔相邻，一人可同时操控三座炮台。
"""
from itertools import combinations

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
    """机器人来袭方向："east"（基地在地图左半）/"west"（右半）。

    实战情报（真实环境校准）：机器人从远离基地的一侧压过来——
    左半区基地的机器人从右（东）来，右半区基地的从左（西）来。
    半场换边后基地坐标互换，此推断自动适配。
    """
    station = turn.station()
    if station is None:
        return "east"
    return "east" if station.pos.x * 2 < turn.width else "west"


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


def _ring1_candidates(turn: Turn, memory: GameMemory | None) -> list[Pos]:
    """基地旁一圈的可选建造格（排除中立区/围墙圈/失败格），按坐标排序。"""
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
        and (memory is None or (n.x, n.y) not in memory.build_failures)
    }, key=lambda p: (p.x, p.y))
    return candidates


def _seat_corridor_ok(seat: Pos, towers: set[Pos], candidates: list[Pos],
                      door: Pos | None) -> bool:
    """控制位是否经 ring1 空格走廊与门口连通（圈内的角色要能走到座位）。"""
    free = {c for c in candidates if c not in towers}
    if door is None:
        return True
    starts = {c for c in free if distance(c, door) <= 1}
    if not starts:
        return False
    visited: set[Pos] = set(starts)
    frontier = list(starts)
    while frontier:
        cur = frontier.pop()
        for n in cur.neighbours():
            if n in free and n not in visited:
                visited.add(n)
                frontier.append(n)
    return seat in visited


def cluster_plan(turn: Turn,
                 memory: GameMemory | None) -> tuple[Pos, list[Pos]] | None:
    """聚控布局：返回 (控制位 S, 塔位 2~3 座)。

    约束：S 与塔均为 ring1 候选、塔与 S 切比雪夫距离 <=1、
    S 经 ring1 空格从门口可达。几何上与同一格相邻的塔必然呈三角/弧形
    包围 S（三塔共线则不存在公共邻格），评分：
    塔数(3>2) > 延续旧座位（稳定）> 正交邻格数（三角形"两翼各一塔，
    开拓者居中打两边"）> 座位贴近门口（背侧，远离来袭方向保开拓者）。
    """
    candidates = _ring1_candidates(turn, memory)
    if len(candidates) < 3:
        return None
    occupied = turn.occupied_cells()
    door = entrance_pos(turn)
    prev_seat = (
        Pos(*memory.tower_seat)
        if memory is not None and memory.tower_seat else None
    )
    best: tuple[Pos, list[Pos]] | None = None
    best_score: tuple[int, int, int, int] | None = None
    orth4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
    for seat in candidates:
        if seat in occupied:
            continue          # 控制位必须可站立（不能压在已有建筑上）
        adjacent = [
            c for c in candidates if c != seat and distance(c, seat) <= 1
        ]
        if len(adjacent) < 2:
            continue
        for size in (3, 2):
            if len(adjacent) < size:
                continue
            for towers in combinations(adjacent, size):
                if not _seat_corridor_ok(seat, set(towers), candidates, door):
                    continue
                stable = 1 if prev_seat is not None and seat == prev_seat else 0
                orth = sum(
                    1 for t in towers if (t.x - seat.x, t.y - seat.y) in orth4
                )
                door_affinity = -distance(seat, door) if door is not None else 0
                score = (len(towers), stable, orth, door_affinity)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (seat, list(towers))
    return best


def tower_sites(turn: Turn, memory: GameMemory | None = None) -> list[Pos]:
    """炮台选址：聚控优先（顺带持久化控制位），无解回退分散布局。"""
    cluster = cluster_plan(turn, memory)
    if cluster is not None:
        seat, sites = cluster
        if memory is not None:
            memory.tower_seat = (seat.x, seat.y)
        return sorted(sites, key=lambda p: (p.x, p.y))
    if memory is not None:
        memory.tower_seat = None
    return _scattered_sites(turn, memory)


def _scattered_sites(turn: Turn, memory: GameMemory | None) -> list[Pos]:
    """（回退）分散选址：三座塔的可站位格须从门口经内部走廊全部可达。"""
    station = turn.station()
    if station is None:
        return []
    footprint = station_footprint(station.pos)
    ring2 = wall_ring(turn)
    candidates = _ring1_candidates(turn, memory)
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

FRONT_FLANK_CELLS = 2        # 半圈方案中南/北排各向正面延伸的格数


def wall_plan(turn: Turn, memory: GameMemory | None = None,
              full: bool = False) -> list[Pos]:
    """围墙防线（分阶段）：
    - 前期（full=False）：只围**面向机器人的半圈**（正面排 + 南/北排靠正面
      两格），石头开销约一半，省下的产能转经济；
    - 后期（full=True）：补齐南/北剩余与背面排（门口在背面）。
    来袭方向由基地左右半区推断，正面排始终最优先。
    """
    station = turn.station()
    if station is None:
        return []
    rows = wall_rows(turn)
    front = attack_side(turn)
    back = entrance_side(turn)
    door = entrance_pos(turn)
    occupied = turn.occupied_cells()

    def usable(pos: Pos) -> bool:
        if door is not None and pos == door:
            return False
        if not turn.on_map(pos) or pos in turn.zones or pos in occupied:
            return False
        if memory is not None and (pos.x, pos.y) in memory.build_failures:
            return False
        return True

    front_row = [p for p in rows[front] if usable(p)]

    def flank_side(side: str) -> list[Pos]:
        row = sorted(
            rows[side],
            key=lambda p: (-p.x if front == "east" else p.x),
        )
        return [p for p in row[:FRONT_FLANK_CELLS] if usable(p)]

    planned = front_row + flank_side("south") + flank_side("north")
    if full:
        seen = set(planned)
        for side in ("south", "north", back):
            for pos in rows[side]:
                if pos in seen or not usable(pos):
                    continue
                seen.add(pos)
                planned.append(pos)
    return planned


def entrance_pos(turn: Turn) -> Pos | None:
    """围墙圈的门口位置：来袭方向背面的角格。

    基地在左半区（机器人从东来）-> 门开西侧 (xmin-2, ymin-1)；
    基地在右半区（机器人从西来）-> 门开东侧 (xmax+2, ymin-2)。
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
    full = getattr(memory, "wall_phase", "front") == "full"
    plan = wall_plan(turn, memory, full=full)
    standing = {unit.pos for unit in turn.walls()}
    return [pos for pos in plan if pos not in standing]


def towers_pending(turn: Turn, memory: GameMemory) -> list[Pos]:
    sites = tower_sites(turn, memory)
    standing = {unit.pos for unit in turn.weapons()}
    return [pos for pos in sites if pos not in standing]
