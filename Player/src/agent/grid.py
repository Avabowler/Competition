"""A* 寻路：8 方向移动、切比雪夫距离启发式，与任务书移动规则一致。"""
from heapq import heappop, heappush
from itertools import count

from .protocol import Pos, Turn, Unit, distance

_STEPS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def next_step(turn: Turn, moving: Unit, goal: Pos | set[Pos] | frozenset[Pos]) -> Pos | None:
    """返回从 moving 出发走向 goal（单格或一组目标格）的第一步。

    目标格本身可以是障碍（例如走到小贩旁边一格时以障碍集合为目标），
    此时路径终点是距目标最近的可达格旁；实现上目标格不可通行则以
    其八邻格为终点集合。
    """
    if isinstance(goal, Pos):
        goals: frozenset[Pos] = frozenset({goal})
    else:
        goals = frozenset(goal)
    if not goals:
        return None
    if moving.pos in goals:
        return None

    blocked = turn.blocked(moving)
    passable_goals = {g for g in goals if g not in blocked and turn.on_map(g)}
    if not passable_goals:
        # 目标全是障碍：以目标的可通行邻格为终点
        passable_goals = {
            n for g in goals for n in g.neighbours()
            if n not in blocked and turn.on_map(n)
        }
        if not passable_goals:
            return None
    if moving.pos in passable_goals:
        return None          # 已站上目标邻格（目标本身不可通行），无需移动

    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [
        (_min_dist(moving.pos, passable_goals), 0, next(order), moving.pos)
    ]
    came_from: dict[Pos, Pos] = {}
    best = {moving.pos: 0}
    seen: set[Pos] = set()

    while frontier:
        _, cost, _, current = heappop(frontier)
        if current in passable_goals:
            return _first_step(came_from, moving.pos, current)
        if current in seen:
            continue
        seen.add(current)
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.on_map(step) or step in seen:
                continue
            new_cost = cost + 1
            if new_cost >= best.get(step, new_cost + 1):
                continue
            best[step] = new_cost
            came_from[step] = current
            heappush(
                frontier,
                (new_cost + _min_dist(step, passable_goals), new_cost, next(order), step),
            )
    return None


def approach_step(turn: Turn, moving: Unit, goal: Pos) -> Pos | None:
    """next_step 的 best-effort 版本：目标完全不可达时，返回通往
    "可达区域中离 goal 最近格" 的第一步（贴墙逼近；堵塞一解除即接上）。"""
    step = next_step(turn, moving, goal)
    if step is not None or moving.pos == goal:
        return step

    blocked = turn.blocked(moving)
    order = count()
    frontier: list[tuple[int, int, int, Pos]] = [(0, 0, next(order), moving.pos)]
    best: dict[Pos, tuple[int, int]] = {moving.pos: (0, 0)}
    came_from: dict[Pos, Pos] = {}
    seen: set[Pos] = set()
    best_cell: tuple[int, int, Pos] = (distance(moving.pos, goal), 0, moving.pos)

    while frontier:
        _, cost, _, current = heappop(frontier)
        if current in seen:
            continue
        seen.add(current)
        heuristic = distance(current, goal)
        if (heuristic, cost) < best_cell[:2]:
            best_cell = (heuristic, cost, current)
        for dx, dy in _STEPS:
            step = Pos(current.x + dx, current.y + dy)
            if step in blocked or not turn.on_map(step) or step in seen:
                continue
            new_cost = cost + 1
            if (new_cost, new_cost) >= best.get(step, (new_cost + 1, new_cost)):
                continue
            best[step] = (new_cost, new_cost)
            came_from[step] = current
            heappush(frontier, (new_cost + heuristic, new_cost, next(order), step))

    if best_cell[2] == moving.pos:
        return None
    return _first_step(came_from, moving.pos, best_cell[2])


def adjacent_cells(turn: Turn, center: Pos, moving: Unit | None = None) -> list[Pos]:
    """center 周围一圈中可站立的格子（不越界；默认排除障碍）。"""
    blocked = turn.blocked(moving)
    cells = [
        n for n in center.neighbours()
        if turn.on_map(n) and n not in blocked
    ]
    cells.sort(key=lambda p: (p.x, p.y))
    return cells


def _min_dist(pos: Pos, goals: frozenset[Pos]) -> int:
    return min(distance(pos, g) for g in goals)


def _first_step(came_from: dict[Pos, Pos], start: Pos, goal: Pos) -> Pos:
    current = goal
    while came_from[current] != start:
        current = came_from[current]
    return current
