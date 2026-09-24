"""总调度：昼夜流程编排、开拓者优先级、prompt 槽仲裁。

优先级约定：
- 物品动作（items）抢占一切 duty：应急炸弹/眩晕 > 药剂 > 召唤令 > 修墙；
- 开拓者白天：自进化任务 > 宝藏 > 升级券使用/待命；夜晚聚控三塔开火；
- 关门战术：人齐才在黄昏封门（工人通宵夜矿时自然不封）；清晨 1-5 拆门出门；
- 工人：白天采矿建造，夜晚躲避机器人继续采矿/贩卖；
- prompt 槽每回合仅一个：evolve（任务期免费）> news > treasure，受每日 3 次预算约束。
"""
import logging
from dataclasses import replace
from typing import Any

from .build import entrance_pos, tower_sites, walls_pending
from .combat import Combat
from .economy import Economy
from .grid import approach_step, next_step
from .items import ItemService, building_max_health
from .memory import GameMemory
from .protocol import (
    Decision,
    Pos,
    Turn,
    Unit,
    cmd_build,
    cmd_remove,
    distance,
    station_footprint,
)
from .tasks.evolve import EvolveModule
from .tasks.news import NewsModule
from .tasks.treasure import TreasureModule
from .validator import sanitize

LOGGER = logging.getLogger(__name__)

HARASS_GOLD_LINE = 600        # 金币富余线（防御优先，之后才骚扰）
DUSK_REGROUP_ROUND = 64       # 白天该回合起角色归位到夜间武器旁
GATE_CLOSE_ROUND = 69         # 封门时间窗（69-70）
GATE_OPEN_DEADLINE = 5        # 清晨拆门时间窗（1-5）
NIGHT_RECALL_DAY = 7          # 该天起夜晚全员回防（大波夜袭，经济让位防守）
STATION_RECALL_HP = 0.5       # 基地血量低于此比例时提前召回回防


class Brain:
    def __init__(self) -> None:
        self.memory = GameMemory()
        self.economy = Economy(self.memory)
        self.combat = Combat(self.memory)
        self.items = ItemService(self.memory)
        self.evolve = EvolveModule(self.memory)
        self.news = NewsModule(self.memory)
        self.treasure = TreasureModule(self.memory)
        self.worker_jobs: dict[int, str] = {}

    # ---------------------------------------------------------------- 主入口

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        turn = Turn.load(payload)
        memory = self.memory
        memory.begin_round(turn)
        memory.observe_llm(turn)
        memory.observe_zones(turn)
        memory.observe_explored(turn)
        memory.record_news(turn)
        # 决策层使用"payload ∪ 区块记忆"的合并中立元素视图
        turn = replace(turn, zones=memory.effective_zones(turn))
        self.evolve.observe(turn)
        self.news.observe(turn)
        self.treasure.observe(turn)

        decision = Decision()
        try:
            if turn.is_day:
                self._day(turn, decision)
            else:
                self._night(turn, decision)
            self._arbitrate_llm(turn, decision)
        except Exception:
            LOGGER.exception("planning failed, partial decision kept")

        decision = sanitize(decision, turn)
        memory.remember_commands(turn, decision.commands)
        if turn.round_no % 50 == 0:
            LOGGER.info(
                "round %d day %d gold %d score %d cmds %d",
                turn.round_no, turn.day, turn.gold, turn.total_score,
                len(decision.commands),
            )
        return decision.dump()

    # ---------------------------------------------------------------- 白天

    def _day(self, turn: Turn, decision: Decision) -> None:
        claimed: set[Pos] = set()
        # 物品动作抢占一切 duty
        handled = self.items.plan(turn, decision, claimed)
        workers = turn.workers()
        self._assign_worker_jobs(turn)

        # 黄昏：开拓者去聚控座位；工人继续白天最后一段经济/建造（建造合法至 70 回合）
        if turn.round_in_day >= DUSK_REGROUP_ROUND:
            self._gate_evening(turn, decision, claimed, handled)
            self._dusk_regroup(turn, decision, claimed, handled)
            self._worker_day_loop(turn, decision, claimed, handled)
            return

        # 清晨：先拆门再干活
        if turn.round_in_day <= GATE_OPEN_DEADLINE:
            self._gate_morning(turn, decision, claimed, handled)

        self._worker_day_loop(turn, decision, claimed, handled)

        # 开拓者：任务 > 宝藏 > 升级券（始终以任务为先，冷却空窗就提前驻守）
        pioneer = turn.pioneer()
        if pioneer is not None and pioneer.unit_id not in handled \
                and pioneer.unit_id not in decision.commands:
            used = self.evolve.plan(turn, decision, claimed)
            if not used and pioneer.unit_id not in decision.commands:
                used = self.treasure.plan(turn, decision, claimed)
            if not used and pioneer.unit_id not in decision.commands:
                self.economy.use_vouchers(turn, [pioneer], decision, claimed)

        # 开拓者空闲兜底：地图未探明 -> 前沿探索；否则无指令且远离基地时走回基地
        if pioneer is not None and pioneer.unit_id not in decision.commands \
                and pioneer.unit_id not in handled \
                and self.memory.evolve.phase == "idle":
            if not self._explore_step(turn, pioneer, decision, claimed):
                station = turn.station()
                if station is not None:
                    fp = station_footprint(station.pos)
                    if min(distance(pioneer.pos, c) for c in fp) > 2:
                        step = next_step(turn, pioneer, station.pos)
                        if step is not None and step not in claimed:
                            claimed.add(step)
                            decision.commands[pioneer.unit_id] = {
                                "action": "move",
                                "targetPos": [{"x": step.x, "y": step.y}],
                            }

        # 工人空闲时把身上的升级券用掉
        idle_roles = [
            w for w in workers
            if w.unit_id not in decision.commands and w.unit_id not in handled
        ]
        if idle_roles:
            self.economy.use_vouchers(turn, idle_roles, decision, claimed)
        # 保底：仍无动作的工人先补探地图，再走向最近矿区/小贩（防原地发呆）
        for worker in workers:
            if worker.unit_id not in decision.commands \
                    and worker.unit_id not in handled:
                if self._explore_step(turn, worker, decision, claimed):
                    continue
                self.economy.fallback_move(turn, worker, decision, claimed)

        # 骚扰：金币富余时买召唤令（使用由 items 完成）
        self._maybe_harass(turn, decision, handled)

    def _worker_day_loop(self, turn: Turn, decision: Decision,
                         claimed: set[Pos], handled: set[int]) -> None:
        """逐工人下发白天经济/建造指令，并维护围墙阶段状态机。"""
        all_sites = tower_sites(turn, self.memory)
        standing_towers = {u.pos for u in turn.weapons()}
        tower_slots = self._tower_slots(turn, all_sites, standing_towers)
        wall_slots = [(pos, "wall") for pos in walls_pending(turn, self.memory)]
        build_slots = tower_slots + wall_slots

        for worker in turn.workers():
            # handled=物品动作占用；已有指令=清晨拆门等前置动作已规划
            # （不跳过会被 plan_worker 覆盖，导致门墙永远拆不掉、全队被困圈内）
            if worker.unit_id in handled or worker.unit_id in decision.commands:
                continue
            job = self.worker_jobs.get(worker.unit_id, "metal")
            self.economy.plan_worker(
                turn, worker, job, decision, claimed, build_slots,
            )

        # 正面半圈合拢检测 -> 锁存（释放金属工去经济线）
        if not self.memory.ring_completed:
            if len(walls_pending(turn, self.memory)) <= 1:
                self.memory.ring_completed = True
                LOGGER.info("front wall ring completed")

        # 阶段流转：有塔升到 2 级或第 4 天起，从"半圈"扩为"整圈"
        if self.memory.wall_phase == "front" and (
            any(w.level >= 2 for w in turn.weapons()) or turn.day >= 4
        ):
            self.memory.wall_phase = "full"
            LOGGER.info("wall phase -> full")

    def _explore_step(self, turn: Turn, role: Unit, decision: Decision,
                      claimed: set[Pos]) -> bool:
        """地图未探明时的前沿探索：走向与未探索格相邻的可通行格。"""
        if self.memory.exploration_done(turn):
            return False
        targets = self.memory.explore_targets(turn)
        ordered = sorted(
            targets, key=lambda p: (distance(role.pos, p), p.x, p.y),
        )
        for target in ordered:
            step = next_step(turn, role, target)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
                return True
        if ordered:
            step = approach_step(turn, role, ordered[0])
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
                return True
        return False

    def _tower_slots(self, turn: Turn, all_sites: list[Pos],
                     standing: set[Pos]) -> list[tuple[Pos, str]]:
        """站点->武器的持久映射：全部火箭发射台（射程穿墙、冷却 3 回合，
        三座错峰轮发形成不间断火力；L3 射程无限全场覆盖）。
        站点建造连续失败 >=2 次则释放其槽位给新站点复用。"""
        plan = self.memory.tower_plan
        site_failures = self.memory.site_failures
        valid = {(s.x, s.y) for s in all_sites}
        # 站点已从候选中消失（被拉黑）或连续失败 -> 释放映射
        for key in list(plan):
            if key not in valid or site_failures.get(key, 0) >= 2:
                del plan[key]
        # 记录炮台建造失败次数
        for command in self.memory.failed_actions.values():
            if command.get("action") != "build":
                continue
            name = command.get("name")
            targets = command.get("targetPos") or []
            if name in ("gatling", "railgun", "rocket") and targets:
                key = (targets[0]["x"], targets[0]["y"])
                site_failures[key] = site_failures.get(key, 0) + 1
        if not plan:
            for site in all_sites:
                plan[(site.x, site.y)] = "rocket"
        for site in all_sites:
            key = (site.x, site.y)
            if key not in plan:
                plan[key] = "rocket"
        return [
            (site, plan[(site.x, site.y)])
            for site in all_sites
            if site not in standing
        ]

    def _assign_worker_jobs(self, turn: Turn) -> None:
        """每回合重算分工（无粘滞状态）：
        - 正面半圈未合拢：全员采石+建墙（防线缺口是最大威胁，首日两人齐上）；
        - 合拢后 1 人石头工（修墙/备封门石），其余金属工（保升级资金）。"""
        workers = turn.workers()
        if not workers:
            return
        if not self.memory.ring_completed and self.memory.wall_phase == "front":
            for worker in workers:
                self.worker_jobs[worker.unit_id] = "stone"
            return
        # 关门战术启用时石头工必须常备石头（每天 1 块封门）
        stone_needed = walls_pending(turn, self.memory) or bool(
            tower_sites(turn, self.memory)
        ) or self.memory.gate.enabled
        for index, worker in enumerate(workers):
            if index == 0 and stone_needed:
                self.worker_jobs[worker.unit_id] = "stone"
            else:
                self.worker_jobs[worker.unit_id] = "metal"

    # ---------------------------------------------------------------- 关门战术

    def _night_recall(self, turn: Turn) -> bool:
        """后期/基地残血时召回全员回防：通宵夜矿让位于生存。"""
        station = turn.station()
        if station is None:
            return True
        if turn.day >= NIGHT_RECALL_DAY:
            return True
        return station.health < building_max_health(station) * STATION_RECALL_HP

    def _gate_evening(self, turn: Turn, decision: Decision, claimed: set[Pos],
                      handled: set[int]) -> None:
        """黄昏 69-70 回合：石头工在门口建墙封门。"""
        gate = self.memory.gate
        if not gate.enabled:
            return
        if self.memory.wall_phase != "full":
            return              # 半圈阶段没有完整门框，不封门
        gate_pos = entrance_pos(turn)
        if gate_pos is None:
            gate.enabled = False
            return
        gate.pos = (gate_pos.x, gate_pos.y)
        # 上回合封门建造失败 -> 记录失败，连续 2 次禁用战术
        for command in self.memory.failed_actions.values():
            if command.get("action") != "build":
                continue
            targets = command.get("targetPos") or []
            if targets and (targets[0]["x"], targets[0]["y"]) == gate.pos:
                gate.record_build_failure()
                if not gate.enabled:
                    LOGGER.info("gate tactic disabled after build failures")
                    return

        standing = {u.pos for u in turn.walls()}
        if gate_pos in standing:
            return
        # 归位要求只在前夕（rid < 69）生效：开拓者必须回圈内（夜晚操控炮台），
        # 夜矿工人允许被锁在门外（通宵夜矿方针）；69-70 无论如何封门——
        # 保基地优先，晚归的角色门外躲避，清晨拆门后照常作业
        workers_free = (not self.memory.multi_control_failed) \
            and not self._night_recall(turn)
        fp = station_footprint(turn.station().pos) if turn.station() else ()
        if turn.round_in_day < GATE_CLOSE_ROUND:
            for role in turn.controllables():
                if fp and min(distance(role.pos, cell) for cell in fp) > 3:
                    if workers_free and role.kind == "worker":
                        continue
                    return
        builder = self._stone_worker(turn)
        if builder is None or builder.unit_id in handled:
            return
        stones = builder.backpack.count("stone")
        if stones < 1:
            return
        if distance(builder.pos, gate_pos) == 1:
            decision.commands[builder.unit_id] = cmd_build("wall", gate_pos)
        elif distance(builder.pos, gate_pos) > 1:
            # 站位目标：门口的"邻格"而非门洞本身（站进门洞会被围墙圈封死）；
            # 目标取全部邻格走"最近可达格"——门外/门内相邻均可施工
            blocked = turn.blocked(builder)
            stand_cells = [
                n for n in gate_pos.neighbours()
                if turn.on_map(n) and n not in blocked and n != builder.pos
            ]
            if not stand_cells:
                return
            nearest = min(stand_cells, key=lambda p: (distance(builder.pos, p), p.x, p.y))
            step = next_step(turn, builder, frozenset(stand_cells)) \
                or approach_step(turn, builder, nearest)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[builder.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }

    def _gate_morning(self, turn: Turn, decision: Decision, claimed: set[Pos],
                      handled: set[int]) -> None:
        """清晨 1-5 回合：石头工拆掉门口的墙出门。"""
        gate = self.memory.gate
        if not gate.enabled or gate.pos is None:
            return
        gate_pos = Pos(*gate.pos)
        gate_wall = next((w for w in turn.walls() if w.pos == gate_pos), None)
        if gate_wall is None:
            return               # 夜里被打掉/已拆，无需处理
        builder = self._stone_worker(turn)
        if builder is None or builder.unit_id in handled:
            return
        if distance(builder.pos, gate_pos) <= 1:
            decision.commands[builder.unit_id] = cmd_remove(gate_pos)
            handled.add(builder.unit_id)   # 防止被工人经济循环覆盖
        else:
            step = next_step(turn, builder, gate_pos)
            if step is not None and step not in claimed:
                claimed.add(step)
                handled.add(builder.unit_id)
                decision.commands[builder.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }

    def _stone_worker(self, turn: Turn) -> Any:
        for worker in turn.workers():
            if self.worker_jobs.get(worker.unit_id) == "stone":
                return worker
        return turn.workers()[0] if turn.workers() else None

    def _dusk_regroup(self, turn: Turn, decision: Decision, claimed: set[Pos],
                      handled: set[int] | None = None) -> None:
        """黄昏：开拓者提前走到聚控控制位（无座位则塔旁/基地）。
        通宵夜矿方针下工人不归位；召回夜（后期/基地残血）工人回武器旁协防。"""
        handled = handled or set()
        recall = self._night_recall(turn)
        pioneer = turn.pioneer()
        if pioneer is not None and pioneer.unit_id not in handled \
                and pioneer.unit_id not in decision.commands:
            goal: Pos | None = None
            if self.memory.tower_seat is not None:
                goal = Pos(*self.memory.tower_seat)
            if goal is None:
                weapons = turn.weapons()
                if weapons:
                    goal = weapons[0].pos
            if goal is None:
                station = turn.station()
                goal = station.pos if station is not None else None
            if goal is not None and distance(pioneer.pos, goal) > 1:
                step = next_step(turn, pioneer, goal) \
                    or approach_step(turn, pioneer, goal)
                if step is not None and step not in claimed:
                    claimed.add(step)
                    decision.commands[pioneer.unit_id] = {
                        "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                    }
        if not recall:
            return
        # 召回夜：工人各自归位到夜间武器旁（无武器则回基地）
        weapons = turn.weapons()
        station = turn.station()
        for index, worker in enumerate(turn.workers()):
            if worker.unit_id in handled or worker.unit_id in decision.commands:
                continue
            goal = weapons[index].pos if index < len(weapons) else (
                station.pos if station is not None else None)
            if goal is None or distance(worker.pos, goal) <= 1:
                continue
            step = next_step(turn, worker, goal) \
                or approach_step(turn, worker, goal)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[worker.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }

    def _maybe_harass(self, turn: Turn, decision: Decision,
                      handled: set[int] | None = None) -> None:
        """金币富余时购买机器人召唤令，给对方夜晚加压（使用由 items 完成）。"""
        handled = handled or set()
        if turn.gold < HARASS_GOLD_LINE:
            return
        order = turn.weapon_shop.get("BossRobotSummonOrder", 200)
        if turn.gold < order:
            order = turn.weapon_shop.get("LargeRobotSummonOrder", 100)
        buyer = None
        for role in (*turn.workers(), turn.pioneer() or None):
            if role is None or role.unit_id in handled:
                continue
            shops = list(turn.weapon_shops())
            if shops and distance(role.pos, min(
                shops, key=lambda s: distance(role.pos, s),
            )) <= 1:
                buyer = role
                break
        if buyer is not None and buyer.unit_id not in decision.commands:
            name = ("BossRobotSummonOrder" if order == 200 else "LargeRobotSummonOrder")
            decision.commands[buyer.unit_id] = cmd_buy(name, 1)

    # ---------------------------------------------------------------- 夜晚

    def _night(self, turn: Turn, decision: Decision) -> None:
        claimed: set[Pos] = set()
        state = self.memory.evolve
        pioneer = turn.pioneer()

        # 物品动作抢占（应急炸弹/药剂/召唤令）
        handled = self.items.plan(turn, decision, claimed)

        # 工人夜间经济（近处夜矿+躲避+收拢）；召回夜/聚控回退时工人上塔位
        busy: set[int] = set()
        if not self.memory.multi_control_failed and not self._night_recall(turn):
            for worker in turn.workers():
                if worker.unit_id in handled or worker.unit_id in decision.commands:
                    continue
                job = self.worker_jobs.get(worker.unit_id, "metal")
                self.economy.plan_night_worker(
                    turn, worker, job, decision, claimed,
                )
                if worker.unit_id in decision.commands:
                    busy.add(worker.unit_id)

        # 解题中的开拓者继续驻守任务点（危险撤离由 evolve 内部处理）
        if pioneer is not None and state.phase == "solving" \
                and pioneer.unit_id not in handled:
            pioneer_busy = self.evolve.plan(turn, decision, claimed)
            if pioneer_busy and pioneer.unit_id in decision.commands:
                self.combat.plan_night(turn, decision, claimed,
                                       exclude=handled | busy)
                return

        self.combat.plan_night(turn, decision, claimed, exclude=handled | busy)

    # ---------------------------------------------------------------- LLM 仲裁

    def _arbitrate_llm(self, turn: Turn, decision: Decision) -> None:
        # executeCmd：仅 evolve 解题期使用
        cmd = self.evolve.wants_execute_cmd(turn)
        if cmd:
            decision.execute_cmd = cmd
            return   # 执行命令的回合不再发 prompt（结果下回合回来再问）

        # prompt：evolve（免费）> news > treasure
        prompt = self.evolve.wants_prompt(turn)
        if prompt:
            decision.prompt = prompt
            self.memory.mark_prompt_sent(turn)
            return
        if not self.memory.spend_llm(turn):
            return
        prompt = self.news.wants_prompt(turn)
        if prompt:
            decision.prompt = prompt
            self.memory.mark_prompt_sent(turn)
            return
        prompt = self.treasure.wants_prompt(turn)
        if prompt:
            decision.prompt = prompt
            self.memory.mark_prompt_sent(turn)
            return
        # 没用掉的预算退回
        self.memory.llm_calls_today = max(0, self.memory.llm_calls_today - 1)
