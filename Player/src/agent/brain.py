"""总调度：昼夜流程编排、开拓者优先级、prompt 槽仲裁。

优先级约定：
- 开拓者白天：自进化任务 > 宝藏 > 升级券使用/待命；
- 开拓者夜晚：解题中驻守任务点（危险则弃守），否则参战；
- prompt 槽每回合仅一个：evolve（任务期免费）> news > treasure，受每日 3 次预算约束。
"""
import logging
from typing import Any

from .build import tower_sites, walls_pending
from .combat import Combat
from .economy import Economy
from .grid import next_step
from .memory import GameMemory
from .protocol import (
    Decision,
    Pos,
    Turn,
    cmd_buy,
    distance,
)
from .tasks.evolve import EvolveModule
from .tasks.news import NewsModule
from .tasks.treasure import TreasureModule
from .validator import sanitize

LOGGER = logging.getLogger(__name__)

HARASS_GOLD_LINE = 400        # 金币超过该线购买召唤令骚扰对方
DUSK_REGROUP_ROUND = 66       # 白天该回合起角色归位到夜间武器旁


class Brain:
    def __init__(self) -> None:
        self.memory = GameMemory()
        self.economy = Economy(self.memory)
        self.combat = Combat()
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
        memory.record_news(turn)
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
        workers = turn.workers()
        self._assign_worker_jobs(turn)

        # 黄昏归位：白天最后几回合全员走向夜间操控位，避免入夜长跑
        if turn.round_in_day >= DUSK_REGROUP_ROUND:
            self._dusk_regroup(turn, decision, claimed)
            return

        all_sites = tower_sites(turn, self.memory)
        standing_towers = {u.pos for u in turn.weapons()}
        loadouts = ("gatling", "railgun", "rocket")
        tower_slots = [
            (site, loadouts[index])
            for index, site in enumerate(all_sites)
            if site not in standing_towers
        ]
        wall_slots = [(pos, "wall") for pos in walls_pending(turn, self.memory)]
        build_slots = tower_slots + wall_slots

        for worker in workers:
            job = self.worker_jobs.get(worker.unit_id, "metal")
            self.economy.plan_worker(
                turn, worker, job, decision, claimed, build_slots,
            )

        # 开拓者：任务 > 宝藏 > 升级券 > 待命
        pioneer = turn.pioneer()
        if pioneer is not None and pioneer.unit_id not in decision.commands:
            used = self.evolve.plan(turn, decision, claimed)
            if not used and pioneer.unit_id not in decision.commands:
                used = self.treasure.plan(turn, decision, claimed)
            if not used and pioneer.unit_id not in decision.commands:
                self.economy.use_vouchers(turn, [pioneer], decision, claimed)

        # 工人空闲时把身上的升级券用掉
        idle_roles = [
            w for w in workers if w.unit_id not in decision.commands
        ]
        if idle_roles:
            self.economy.use_vouchers(turn, idle_roles, decision, claimed)

        # 骚扰：金币富余时让最近商店的人捎召唤令（在购物清单里体现）
        self._maybe_harass(turn, decision)

    def _dusk_regroup(self, turn: Turn, decision: Decision, claimed: set[Pos]) -> None:
        """白天最后 5 回合：角色提前走到各自夜间武器旁。"""
        weapons = turn.weapons()
        roles = turn.controllables()
        pairs = list(zip(roles, list(weapons) + [None] * max(0, len(roles) - len(weapons))))
        for role, weapon in pairs:
            if role.unit_id in decision.commands:
                continue
            goal = weapon.pos if weapon is not None else None
            if goal is not None and distance(role.pos, goal) <= 1:
                continue
            if goal is None:
                station = turn.station()
                goal = station.pos if station is not None else role.pos
            step = next_step(turn, role, goal)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[role.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }

    def _assign_worker_jobs(self, turn: Turn) -> None:
        workers = turn.workers()
        if not workers:
            return
        stone_needed = bool(walls_pending(turn, self.memory)) or bool(
            tower_sites(turn, self.memory)
        )
        for worker in workers:
            job = self.worker_jobs.get(worker.unit_id)
            if job is None:
                job = "stone" if stone_needed and len(
                    [j for j in self.worker_jobs.values() if j == "stone"]
                ) == 0 else "metal"
                self.worker_jobs[worker.unit_id] = job
        # 防线完工后石头工转金属
        if not stone_needed:
            for worker in workers:
                self.worker_jobs[worker.unit_id] = "metal"

    def _maybe_harass(self, turn: Turn, decision: Decision) -> None:
        """金币富余时购买机器人召唤令，给对方夜晚加压（对方少赚生存/击杀分）。"""
        if turn.gold < HARASS_GOLD_LINE:
            return
        order = turn.weapon_shop.get("BossRobotSummonOrder", 200)
        if turn.gold < order:
            order = turn.weapon_shop.get("LargeRobotSummonOrder", 100)
        buyer = None
        for role in (*turn.workers(), turn.pioneer() or None):
            if role is None:
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

        # 解题中的开拓者继续驻守任务点（危险撤离由 evolve 内部处理）
        pioneer_busy = False
        if pioneer is not None and state.phase == "solving":
            pioneer_busy = self.evolve.plan(turn, decision, claimed)
            if pioneer_busy and pioneer.unit_id in decision.commands:
                # 任务优先：combat 只调度其余角色
                self.combat.plan_night(turn, decision, claimed, exclude={pioneer.unit_id})
                return

        self.combat.plan_night(turn, decision, claimed)

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
