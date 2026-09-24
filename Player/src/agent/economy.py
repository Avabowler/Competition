"""经济系统：工人分工采矿、贩卖回城、按购物清单购买与使用。

分工：每名工人有固定职责（存于 Brain.assignments）：
- stone：专职石头（围墙材料 + 修复储备）
- metal：铜/铁（按当前价格与新闻预测选矿，满载后去小贩贩卖顺路购物）

白天 70 回合的节奏：采矿 -> 背包达阈值 -> 去小贩 -> 卖 -> （金币够且有购物单）去武器商店买 -> 使用/返程。
"""
import logging

from .build import front_cells, towers_pending, walls_pending
from .grid import next_step
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
        # 关门前的回家 deadline：时间不够走回去就立刻动身（关门战术的保命前提）
        station = turn.station()
        if station is not None and turn.round_in_day >= HOMEBOUND_ROUND:
            from .build import footprint_distance
            from .protocol import station_footprint
            home_gap = footprint_distance(worker.pos, station_footprint(station.pos))
            if turn.round_in_day + home_gap >= GATE_CLOSE_ROUND:
                step = next_step(turn, worker, station.pos)
                if step is not None and step not in claimed:
                    claimed.add(step)
                    decision.commands[worker.unit_id] = {
                        "action": "move",
                        "targetPos": [{"x": step.x, "y": step.y}],
                    }
                    return
        if role == "stone":
            self._stone_worker(turn, worker, decision, claimed, build_slots)
        else:
            self._metal_worker(turn, worker, decision, claimed, build_slots)

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

    # ---- 石头工人：先炮台后围墙，其余时间采石

    def _stone_worker(self, turn: Turn, worker: Unit, decision: Decision,
                      claimed: set[Pos], build_slots: list[tuple[Pos, str]]) -> None:
        memory = self.memory
        stones = worker.backpack.count("stone")

        # 1) 炮台优先（金币足够）
        pending_towers = [slot for slot in build_slots if slot[1] != "wall"]
        if pending_towers and turn.gold >= WEAPON_BUILD_COST:
            for site, name in pending_towers:
                if distance(worker.pos, site) <= 1 and worker.pos != site:
                    decision.commands[worker.unit_id] = cmd_build(name, site)
                    return
            site, name = pending_towers[0]
            if _walk_or_reach(turn, worker, [site], decision, claimed):
                return

        # 2) 围墙补建（建"与工人相邻"的那段，而不是列表第一段）
        pending_walls = [slot for slot in build_slots if slot[1] == "wall"]
        if pending_walls and stones > 0:
            for site, name in pending_walls:
                if distance(worker.pos, site) <= 1:
                    decision.commands[worker.unit_id] = cmd_build(name, site)
                    return
            target = pending_walls[0][0]
            if _walk_or_reach(turn, worker, [target], decision, claimed):
                return

        # 3) 采石 / 修墙物资：攒一小批就回去建（快周转）
        if stones < STONE_RESERVE or not pending_walls:
            if self._mine_round(turn, worker, "stone", decision):
                return

        # 4) 围墙已齐且石头富余 -> 去卖掉多余的石头
        self._sell_trip(turn, worker, decision, claimed, keep_ore="stone")

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
        if self._mine_round(turn, worker, ore, decision):
            return
        # 无矿可采：随大流去小贩（顺路购物）
        self._sell_trip(turn, worker, decision, claimed, keep_ore=None)

    # ---- 通用动作

    def _mine_round(self, turn: Turn, worker: Unit, ore: str, decision: Decision) -> bool:
        mines = turn.mines_of(ore)
        if not mines:
            return False
        if worker.capacity is not None and len(worker.backpack) >= worker.capacity:
            return False
        adjacent = [m for m in mines if distance(worker.pos, m) <= 1]
        if adjacent:
            target = min(adjacent, key=lambda m: (m.x, m.y))
            decision.commands[worker.unit_id] = cmd_collect(target)
            return True
        target = min(mines, key=lambda m: (distance(worker.pos, m), m.x, m.y))
        step = next_step(turn, worker, target)
        if step is not None:
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
GOLD_TRIP_LINE = 120       # 金币闲置到该值就提前跑一趟商店消费
RESTOCK_ITEMS = ("WallFixer", "Bomb", "DizzyWeapon", "Medicine")
HOMEBOUND_ROUND = 60       # 白天该回合起按"能否在封门前到家"决定是否动身
GATE_CLOSE_ROUND = 69      # 与 brain.GATE_CLOSE_ROUND 保持一致
