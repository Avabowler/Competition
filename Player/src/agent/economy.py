"""经济系统：工人分工采矿、贩卖、按购物清单购买与使用。

分工（Brain.assignments 每回合重算）：
- stone：专职石头（围墙材料 + 修复储备）
- metal：铜/铁（按当前价格与新闻预测选矿，满载后去小贩贩卖顺路购物）

白天 70 回合的节奏：采矿 -> 背包达阈值 -> 去小贩 -> 卖 -> （金币够且有购物单）去武器商店买 -> 使用/返程。
夜晚工人不归位：躲避机器人后继续 夜间经济链（plan_night_worker），通宵采矿。
"""
import logging

from .build import entrance_pos, front_cells, walls_pending
from .grid import approach_step, next_step
from .items import building_max_health
from .memory import GameMemory
from .protocol import (
    Decision,
    Pos,
    Turn,
    Unit,
    WEAPON_BUILD_COST,
    cmd_build,
    cmd_buy,
    cmd_collect,
    cmd_sell,
    cmd_use,
    distance,
)

LOGGER = logging.getLogger(__name__)

SELL_BATCH = 10            # 金属矿满 N 个回城贩卖
STONE_RESERVE = 8          # 石头工人的常备库存（低于则优先补采）
MEDICINE_KEEP = 1          # 每人常备药品
WALL_UPGRADE_HP = 1500     # 血量低于此的 L1 围墙值得升级/修复


def _ore_value(turn: Turn, memory: GameMemory, ore: str) -> int:
    """矿石优先级：单价为主，新闻预测上调/下调。"""
    price = turn.ore_price(ore)
    if memory.mining_blocked(ore, turn):
        return -1                      # 预测停工，不采
    if memory.price_up_soon(ore, turn):
        price *= 2                     # 即将涨价，提前囤
    return price


def choose_metal_ore(turn: Turn, memory: GameMemory, worker: Unit) -> str:
    """为金属工人挑选当前最优矿种，考虑距离与价值。"""
    best_ore, best_score = "", -1.0
    for ore in ("copper", "iron"):
        value = _ore_value(turn, memory, ore)
        if value <= 0:
            continue
        mines = turn.mines_of(ore)
        if not mines:
            continue
        nearest = min(distance(worker.pos, m) for m in mines)
        score = value * 10 / (nearest + 5)
        if score > best_score:
            best_ore, best_score = ore, score
    if best_ore:
        return best_ore
    # 兜底：石头总有市场
    return "stone"


def _nearest(turn: Turn, worker: Unit, positions: list[Pos] | tuple[Pos, ...]) -> Pos | None:
    if not positions:
        return None
    return min(positions, key=lambda p: (distance(worker.pos, p), p.x, p.y))


def _walk_or_reach(
    turn: Turn, worker: Unit, goal_cells: list[Pos], decision: Decision,
    claimed: set[Pos],
) -> bool:
    """向目标格集合走一步；已到达（相邻可交互）返回 False 表示无需再走。"""
    step = next_step(turn, worker, frozenset(goal_cells))
    if step is None:
        return False
    if step in claimed:
        return False
    claimed.add(step)
    decision.commands[worker.unit_id] = {
        "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
    }
    return True


def _try_interact_adjacent(
    turn: Turn, worker: Unit, targets: list[Pos], command: dict,
    decision: Decision,
) -> bool:
    """若 worker 与任一目标格相邻则执行 command，返回是否执行。"""
    for target in targets:
        if distance(worker.pos, target) <= 1:
            decision.commands[worker.unit_id] = command
            return True
    return False


class Economy:
    """产出两名工人的白天指令。"""

    def __init__(self, memory: GameMemory):
        self.memory = memory

    # ---- 主入口

    def plan_worker(self, turn: Turn, worker: Unit, role: str, decision: Decision,
                    claimed: set[Pos], build_slots: list[tuple[Pos, str]]) -> None:
        # 携带升级券的人最优先送券到目标建筑并使用（否则券会一直压在背包里）
        if self._deliver_vouchers(turn, worker, decision, claimed):
            return
        if role == "stone":
            self._stone_worker(turn, worker, decision, claimed, build_slots)
        else:
            self._metal_worker(turn, worker, decision, claimed, build_slots)

    def plan_night_worker(self, turn: Turn, worker: Unit, role: str,
                          decision: Decision, claimed: set[Pos]) -> None:
        """夜间经济链：躲避 > 送券 > 满载贩卖购物 > 近处采矿（build 夜间非法）。

        夜矿只采基地 NIGHT_MINE_RADIUS 半径内的矿；近处无矿时向基地收拢，
        不为远矿整夜脱离防守圈。"""
        if self.evade_robots(turn, worker, decision, claimed):
            return
        if self._deliver_vouchers(turn, worker, decision, claimed):
            return
        load = sum(
            worker.backpack.count(o) for o in ("copper", "iron", "stone")
        )
        night_ending = turn.round_in_day >= NIGHT_SELL_DEADLINE
        if load >= SELL_BATCH or (night_ending and load > 0) or \
                (worker.capacity and load >= worker.capacity - 2):
            keep = "stone" if role == "stone" else None
            if self._sell_trip(turn, worker, decision, claimed, keep_ore=keep):
                return
        if role == "stone":
            ore = "stone"
        else:
            ore = choose_metal_ore(turn, self.memory, worker)
        station = turn.station()
        center = station.pos if station is not None else None
        if self._mine_round(turn, worker, ore, decision, claimed,
                            center=center, max_dist=NIGHT_MINE_RADIUS):
            return
        # 近处无矿：向基地收拢（抵达防守圈附近后交给战斗模块收编守塔）
        if station is not None and distance(worker.pos, station.pos) > 4:
            step = next_step(turn, worker, station.pos) \
                or approach_step(turn, worker, station.pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[worker.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
            return
        self.fallback_move(turn, worker, decision, claimed)

    def evade_robots(self, turn: Turn, worker: Unit, decision: Decision,
                     claimed: set[Pos]) -> bool:
        """机器人进入警戒圈（射程 3 + 1 格余量）时远离其包围，返回是否规避。"""
        threats = [
            r for r in turn.robots
            if r.health > 0 and distance(worker.pos, r.pos) <= EVADE_ROBOT_RANGE
        ]
        if not threats:
            return False
        blocked = turn.blocked(worker)
        best: tuple[tuple[int, int, int], Pos] | None = None
        for n in worker.pos.neighbours():
            if not turn.on_map(n) or n in blocked or n in claimed:
                continue
            clearance = min(distance(n, t.pos) for t in threats)
            key = (-clearance, n.x, n.y)
            if best is None or key < best[0]:
                best = (key, n)
        if best is None:
            return False
        decision.commands[worker.unit_id] = {
            "action": "move", "targetPos": [{"x": best[1].x, "y": best[1].y}],
        }
        return True

    def _deliver_vouchers(self, turn: Turn, worker: Unit, decision: Decision,
                          claimed: set[Pos]) -> bool:
        for item in worker.backpack:
            if not (item.endswith("Voucher1") or item.endswith("Voucher2")):
                continue
            target = self._voucher_target(turn, worker, item)
            if target is None:
                continue
            if distance(worker.pos, target) <= 1:
                decision.commands[worker.unit_id] = cmd_use(item, target)
            else:
                step = next_step(turn, worker, target)
                if step is not None and step not in claimed:
                    claimed.add(step)
                    decision.commands[worker.unit_id] = {
                        "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                    }
                else:
                    return False   # 走不过去就先干别的
            return True
        return False

    # ---- 石头工人：半圈未合拢时墙先于塔，其余时间采石

    def _stone_worker(self, turn: Turn, worker: Unit, decision: Decision,
                      claimed: set[Pos], build_slots: list[tuple[Pos, str]]) -> None:
        memory = self.memory
        stones = worker.backpack.count("stone")
        # 封门 deadline：石头工是封门建造者，时间不够走到门口就立刻动身
        # （金属工保持通宵夜矿不受影响；黄昏 64-70 brain._gate_evening 优先接管，
        #   它提前返回时这里继续兜底走位）
        if memory.gate.enabled and memory.wall_phase == "full" \
                and turn.round_in_day >= GATE_BUILDER_ROUND:
            gate_pos = entrance_pos(turn)
            if gate_pos is not None and distance(worker.pos, gate_pos) > 1:
                gap = distance(worker.pos, gate_pos)
                if turn.round_in_day + gap >= GATE_CLOSE_ROUND - 1:
                    step = next_step(turn, worker, gate_pos) \
                        or approach_step(turn, worker, gate_pos)
                    if step is not None and step not in claimed:
                        claimed.add(step)
                        decision.commands[worker.unit_id] = {
                            "action": "move",
                            "targetPos": [{"x": step.x, "y": step.y}],
                        }
                        return
        pending_towers = [slot for slot in build_slots if slot[1] != "wall"]
        pending_walls = [slot for slot in build_slots if slot[1] == "wall"]
        # 正面半圈未合拢 -> 防线缺口是最大威胁，围墙优先于炮台
        ring_open = not memory.ring_completed and memory.wall_phase == "front"
        first, second = (
            (self._try_walls, self._try_towers) if ring_open
            else (self._try_towers, self._try_walls)
        )
        if first(turn, worker, decision, claimed, pending_towers,
                 pending_walls, stones):
            return
        if second(turn, worker, decision, claimed, pending_towers,
                  pending_walls, stones):
            return

        # 3) 采石 / 修墙物资（无墙可建或走不过去也采矿，防原地发呆）
        if self._mine_round(turn, worker, "stone", decision, claimed):
            return

        # 4) 围墙已齐且石头富余 -> 去卖掉多余的石头
        self._sell_trip(turn, worker, decision, claimed, keep_ore="stone")

    def _try_towers(self, turn: Turn, worker: Unit, decision: Decision,
                    claimed: set[Pos], pending_towers: list[tuple[Pos, str]],
                    pending_walls: list[tuple[Pos, str]], stones: int) -> bool:
        if not (pending_towers and turn.gold >= WEAPON_BUILD_COST):
            return False
        for site, name in pending_towers:
            if distance(worker.pos, site) <= 1 and worker.pos != site:
                decision.commands[worker.unit_id] = cmd_build(name, site)
                return True
        site, name = pending_towers[0]
        return _walk_or_reach(turn, worker, [site], decision, claimed)

    def _try_walls(self, turn: Turn, worker: Unit, decision: Decision,
                   claimed: set[Pos], pending_towers: list[tuple[Pos, str]],
                   pending_walls: list[tuple[Pos, str]], stones: int) -> bool:
        if not (pending_walls and stones > 0):
            return False
        for site, name in pending_walls:
            if distance(worker.pos, site) <= 1:
                decision.commands[worker.unit_id] = cmd_build(name, site)
                return True
        target = pending_walls[0][0]
        return _walk_or_reach(turn, worker, [target], decision, claimed)

    # ---- 金属工人：高价矿 -> 满载贩卖 -> 购物

    def _metal_worker(self, turn: Turn, worker: Unit, decision: Decision,
                      claimed: set[Pos],
                      build_slots: list[tuple[Pos, str]] | None = None) -> None:
        memory = self.memory
        ores = [o for o in ("copper", "iron", "stone") if worker.backpack.count(o)]
        load = sum(worker.backpack.count(o) for o in ("copper", "iron", "stone"))

        # 防线缺口大时帮忙建墙（机器人会从缺口直灌基地）
        walls_pending = [slot for slot in build_slots if slot[1] == "wall"]
        if len(walls_pending) >= 4 and worker.backpack.count("stone") > 0:
            for site, name in walls_pending:
                if distance(worker.pos, site) <= 1:
                    decision.commands[worker.unit_id] = cmd_build(name, site)
                    return

        # 背包将满 / 当天临近结束 / 金币闲置需要消费 -> 贩卖+购物之旅
        day_ending = turn.round_in_day >= DAY_SELL_DEADLINE
        gold_burning = turn.gold >= GOLD_TRIP_LINE
        if load >= SELL_BATCH or (day_ending and load > 0) or gold_burning or \
                worker.capacity and load >= worker.capacity - 2:
            if self._sell_trip(turn, worker, decision, claimed, keep_ore=None):
                return

        ore = choose_metal_ore(turn, memory, worker)
        if self._mine_round(turn, worker, ore, decision, claimed):
            return
        # 无矿可采：随大流去小贩（顺路购物）
        self._sell_trip(turn, worker, decision, claimed, keep_ore=None)

    # ---- 通用动作

    def _mine_round(self, turn: Turn, worker: Unit, ore: str, decision: Decision,
                    claimed: set[Pos] | None = None,
                    center: Pos | None = None, max_dist: int | None = None) -> bool:
        claimed = claimed if claimed is not None else set()
        mines = turn.mines_of(ore)
        if center is not None and max_dist is not None:
            mines = [m for m in mines if distance(center, m) <= max_dist]
        if not mines:
            return False
        if worker.capacity is not None and len(worker.backpack) >= worker.capacity:
            return False
        # 从近到远逐个尝试：最近的矿不可达（邻格被占/被隔）就换下一个，
        # 绝不因单个矿点而整体放弃；全部真路径失败才 best-effort 逼近最近的
        ordered = sorted(mines, key=lambda m: (distance(worker.pos, m), m.x, m.y))
        adjacent = [m for m in ordered if distance(worker.pos, m) <= 1]
        if adjacent:
            decision.commands[worker.unit_id] = cmd_collect(adjacent[0])
            return True
        for target in ordered:
            step = next_step(turn, worker, target)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[worker.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
                return True
        step = approach_step(turn, worker, ordered[0])
        if step is not None and step not in claimed:
            claimed.add(step)
            decision.commands[worker.unit_id] = {
                "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
            }
            return True
        return False

    def _sell_trip(self, turn: Turn, worker: Unit, decision: Decision,
                   claimed: set[Pos], keep_ore: str | None) -> bool:
        """去小贩卖矿；顺路（相邻武器商店）按购物单购买。返回是否发出了指令。"""
        vendors = list(turn.vendors())
        if not vendors:
            return False
        load = sum(worker.backpack.count(o) for o in ("copper", "iron", "stone"))
        vendor = _nearest(turn, worker, vendors)
        if vendor is None:
            return False

        near_vendor = distance(worker.pos, vendor) <= 1
        if near_vendor and load > 0:
            # 卖出：优先高价矿，保留 keep_ore 数量
            for ore in ("copper", "iron", "stone"):
                count = worker.backpack.count(ore)
                if keep_ore == "stone" and ore == "stone":
                    count = max(0, count - STONE_RESERVE)
                if count > 0:
                    decision.commands[worker.unit_id] = cmd_sell(ore, count)
                    return True
            # 无可卖（只剩保留量）-> 转去购物
        elif not near_vendor and load > 0:
            if _walk_or_reach(turn, worker, [vendor], decision, claimed):
                return True

        # 购物：相邻武器商店时按清单买
        shops = list(turn.weapon_shops())
        shop = _nearest(turn, worker, shops) if shops else None
        if shop is not None:
            if distance(worker.pos, shop) <= 1:
                purchase = self._next_purchase(turn, worker)
                if purchase is not None:
                    name, num = purchase
                    decision.commands[worker.unit_id] = cmd_buy(name, num)
                    return True
                # 无需购物 -> 回矿区
            else:
                if _walk_or_reach(turn, worker, [shop], decision, claimed):
                    return True
        return False

    # ---- 购物清单（有序：取第一项买得起的）

    def _next_purchase(self, turn: Turn, buyer: Unit) -> tuple[str, int] | None:
        gold = turn.gold
        shop = turn.weapon_shop
        weapons = turn.weapons()
        station = turn.station()
        # 全队已在背包中的券/消耗品，避免重复购买
        holding: set[str] = set()
        team_count: dict[str, int] = {}
        for unit in turn.ours:
            for item in unit.backpack:
                if "Voucher" in item or item in RESTOCK_ITEMS:
                    holding.add(item)
                team_count[item] = team_count.get(item, 0) + 1

        reserve = WEAPON_BUILD_COST     # 保底重建储备：买完不能低于 25
        def can(price: int, soft: bool = False) -> bool:
            if price <= 0 or gold < price:
                return False
            return soft or gold - price >= reserve

        # 1) 残墙修复包 + 应急道具补货 + 药剂
        if team_count.get("Medicine", 0) < MEDICINE_KEEP and any(
            u.is_character and u.health < 120 for u in turn.ours
        ):
            if can(shop.get("Medicine", 10), soft=True):
                return ("Medicine", 1)
        damaged_wall = any(
            w.health < building_max_health(w) * 0.6 for w in turn.walls()
        )
        if damaged_wall and gold >= 150 and team_count.get("WallFixer", 0) < 2:
            if can(shop.get("WallFixer", 10), soft=True):
                return ("WallFixer", 1)
        if gold >= 500:
            for item in ("Bomb", "DizzyWeapon"):
                if team_count.get(item, 0) < 1 and can(shop.get(item, 100)):
                    return (item, 1)

        # 2) 急救升级：残血建筑用升级券"回满血"（当治疗用，优先级最高）
        if station is not None and station.health < building_max_health(station) * 0.4:
            voucher = f"StationUpgradeVoucher{station.level}"
            if can(shop.get(voucher, 0), soft=True) and voucher not in holding:
                return (voucher, 1)
        for weapon in weapons:
            if weapon.health < building_max_health(weapon) * 0.4 and weapon.level < 3:
                voucher = f"WeaponUpgradeVoucher{weapon.level}"
                if can(shop.get(voucher, 0), soft=True) and voucher not in holding:
                    return (voucher, 1)

        # 3) 主升级序列：加特林L2 -> 火箭L2 -> 基地L2 -> 火箭L3 -> 加特林L3
        #    -> 围墙L2券 -> 电磁L2 -> 基地L3 -> 围墙L3券 -> 电磁L3
        sequence: list[tuple[str, str]] = []
        gatlings = [w for w in weapons if w.kind == "gatling"]
        rockets = [w for w in weapons if w.kind == "rocket"]
        railguns = [w for w in weapons if w.kind == "railgun"]
        if any(w.level == 1 for w in gatlings):
            sequence.append(("WeaponUpgradeVoucher1", "gatling-L2"))
        if any(w.level == 1 for w in rockets):
            sequence.append(("WeaponUpgradeVoucher1", "rocket-L2"))
        if station is not None and station.level == 1:
            sequence.append(("StationUpgradeVoucher1", "station-L2"))
        if any(w.level == 2 for w in rockets):
            sequence.append(("WeaponUpgradeVoucher2", "rocket-L3"))
        if any(w.level == 2 for w in gatlings):
            sequence.append(("WeaponUpgradeVoucher2", "gatling-L3"))
        if any(w.level == 1 for w in turn.walls()):
            sequence.append(("WallUpgradeVoucher1", "wall-L2"))
        if any(w.level == 1 for w in railguns):
            sequence.append(("WeaponUpgradeVoucher1", "railgun-L2"))
        if station is not None and station.level == 2:
            sequence.append(("StationUpgradeVoucher2", "station-L3"))
        if any(w.level == 2 for w in turn.walls()):
            sequence.append(("WallUpgradeVoucher2", "wall-L3"))
        if any(w.level == 2 for w in railguns):
            sequence.append(("WeaponUpgradeVoucher2", "railgun-L3"))

        # 同一张券名可能在序列中出现多次（如火箭/加特林共用 WeaponUpgradeVoucher1），
        # 目标选择交给 _voucher_target（挑残血目标），这里只判断"存在可升目标"。
        for voucher, tag in sequence:
            if voucher in holding:
                continue
            price = shop.get(voucher, 0)
            if not can(price):
                continue
            if self._voucher_exists(turn, voucher):
                return (voucher, 1)
        return None

    def _voucher_exists(self, turn: Turn, voucher: str) -> bool:
        if voucher.startswith("Weapon"):
            wanted = 1 if voucher.endswith("1") else 2
            return any(w.level == wanted for w in turn.weapons())
        if voucher.startswith("Station"):
            station = turn.station()
            return station is not None
        if voucher.startswith("Wall"):
            wanted = 1 if voucher.endswith("1") else 2
            return any(w.level == wanted for w in turn.walls())
        return False

    def fallback_move(self, turn: Turn, worker: Unit, decision: Decision,
                      claimed: set[Pos]) -> None:
        """空闲工人保底移动：从近到远遍历矿区 -> 小贩，绝不原地发呆。"""
        groups = ((turn.mines(), True), (list(turn.vendors()), False))
        for targets, skip_if_adjacent in groups:
            if not targets:
                continue
            ordered = sorted(
                targets, key=lambda p: (distance(worker.pos, p), p.x, p.y),
            )
            if skip_if_adjacent and distance(worker.pos, ordered[0]) <= 1:
                break              # 已就位（下回合会有具体动作），转下一类目标
            for target in ordered:
                step = next_step(turn, worker, target)
                if step is not None and step not in claimed:
                    claimed.add(step)
                    decision.commands[worker.unit_id] = {
                        "action": "move",
                        "targetPos": [{"x": step.x, "y": step.y}],
                    }
                    return
        # 全部目标不可达：朝最近的矿区 best-effort 逼近（堵塞一解除即接上）
        mines = sorted(
            turn.mines(), key=lambda p: (distance(worker.pos, p), p.x, p.y),
        )
        if mines:
            step = approach_step(turn, worker, mines[0])
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[worker.unit_id] = {
                    "action": "move",
                    "targetPos": [{"x": step.x, "y": step.y}],
                }

    # ---- 升级券使用（走到目标建筑旁使用；目标选血量最低的）

    def use_vouchers(self, turn: Turn, roles: list[Unit], decision: Decision,
                     claimed: set[Pos]) -> None:
        for role in roles:
            for item in role.backpack:
                if item.endswith("Voucher1") or item.endswith("Voucher2"):
                    target = self._voucher_target(turn, role, item)
                    if target is None:
                        continue
                    if distance(role.pos, target) <= 1:
                        decision.commands[role.unit_id] = cmd_use(item, target)
                    else:
                        step = next_step(turn, role, target)
                        if step is not None and step not in claimed:
                            claimed.add(step)
                            decision.commands[role.unit_id] = {
                                "action": "move",
                                "targetPos": [{"x": step.x, "y": step.y}],
                            }
                    break  # 每回合每人最多处理一张券

    def _voucher_target(self, turn: Turn, role: Unit, item: str) -> Pos | None:
        """券的使用目标：符合等级的候选中挑血量最低的（升级回满血收益最大）。"""
        if item.startswith("Weapon"):
            wanted_level = 1 if item.endswith("1") else 2
            candidates = [w for w in turn.weapons() if w.level == wanted_level]
            if not candidates:
                return None
            return min(candidates, key=lambda w: w.health).pos
        if item.startswith("Station"):
            station = turn.station()
            return station.pos if station is not None else None
        if item.startswith("Wall"):
            wanted_level = 1 if item.endswith("1") else 2
            candidates = [w for w in turn.walls() if w.level == wanted_level]
            if not candidates:
                return None
            front = front_cells(turn)
            # 正面墙优先（吃伤害最多），同级里再挑血量最低的
            candidates.sort(key=lambda w: (w.pos not in front, w.health))
            return candidates[0].pos
        return None


DAY_SELL_DEADLINE = 60     # 白天第 60 回合后金属工人开始收尾贩卖
GOLD_TRIP_LINE = 175       # 金币到该值就跑商店（=L3 升级券 150 + 保底 25，避免在
                           # 100 档反复小采购、攒不下 L3 券；首日 L2 由富余自然触发）
RESTOCK_ITEMS = ("WallFixer", "Bomb", "DizzyWeapon", "Medicine")
NIGHT_SELL_DEADLINE = 124  # 夜晚（71-130）该回合起收尾贩卖，别把矿背过夜
EVADE_ROBOT_RANGE = 4      # 机器人射程 3 + 1 格余量，进入即规避
NIGHT_MINE_RADIUS = 12     # 夜矿范围：只采基地此半径内的矿，远矿留给白天
GATE_BUILDER_ROUND = 58    # 石头工（封门建造者）该回合起按"能否赶到门口"决定动身
GATE_CLOSE_ROUND = 69      # 封门时间窗（69-70），与 brain.GATE_CLOSE_ROUND 一致
