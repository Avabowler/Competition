"""跨回合记忆：判题器每回合发来全新快照，任务/传闻/预算等必须自己记。

GameMemory 由 Brain 持有并在每回合 begin_round() 时更新；
HTTP 层是多线程的，Brain 外层有全局锁，这里不再加锁。
"""
from dataclasses import dataclass, field
from typing import Any

from .protocol import LLM_FREE_PER_DAY, Turn


@dataclass(slots=True)
class OreForecast:
    """news 模块产出的某矿石预测：relative_day 从当天起算 1=今天。"""
    ore: str
    mining_blocked: set[int] = field(default_factory=set)   # 相对天数，不可采集
    price_up: set[int] = field(default_factory=set)         # 相对天数，价格上涨
    price_down: set[int] = field(default_factory=set)


@dataclass(slots=True)
class TreasureState:
    """treasure 模块推理出的宝藏信息。"""
    opened: bool = False                 # 已成功开启（全场仅一个宝藏）
    deduced_items: list[str] = field(default_factory=list)
    deduced_pos: tuple[int, int] | None = None
    deduced_day: int | None = None       # 绝对天数
    deduced_round_window: tuple[int, int] | None = None
    confidence: float = 0.0
    attempts: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    tried_items: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class EvolveTaskState:
    """自进化任务状态机数据（evolve 模块读写）。"""
    phase: str = "idle"                  # idle/walking/accepted/solving/done
    task_point: tuple[int, int] | None = None
    accepted_round: int = 0
    walk_start_round: int = 0
    task_type: str = ""
    signature: str = ""                  # 任务文本签名（SOP 匹配用）
    timeout_rounds: int = 0
    cmd_history: list[tuple[str, str]] = field(default_factory=list)  # (命令, 结果)
    llm_history: list[tuple[str, str]] = field(default_factory=list)  # (prompt, resp)
    last_answer: str = ""
    best_answer: str = ""
    full_task_text: str = ""             # 当前任务原文（SOP 全文比对用）
    accept_retries: int = 0              # acceptTask 已发但未下发任务的连续次数
    parse_failures: int = 0              # llmResp 连续解析失败次数
    corrected_prompt_sent: bool = False  # 是否已发过"严格 JSON"修正 prompt
    explore_index: int = 0               # 解析彻底失败时的兜底探索命令游标
    sop_hint_used: bool = False          # 本任务是否命中 SOP（提示模式）
    sop_hint_injected: bool = False      # SOP 提示是否已注入过 prompt
    submit_attempts: int = 0
    last_point_index: int = 0


@dataclass(slots=True)
class SopRecord:
    """已完成任务的解法沉淀，供同类任务复用。"""
    signature: str                       # 任务文本签名（前若干字符）
    task_type: str
    answer: str = ""
    commands: list[str] = field(default_factory=list)
    full_text: str = ""                  # 完整任务原文（防过期答案：全文一致才直接复用）
    notes: str = ""


@dataclass(slots=True)
class GateState:
    """关门战术状态：黄昏在基地围墙圈门口建墙，清晨拆除。"""
    pos: tuple[int, int] | None = None
    fail_count: int = 0                  # 连续建造失败次数
    enabled: bool = True                 # 连续失败>=2 自动禁用（回退留门）

    def record_build_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= 2:
            self.enabled = False


@dataclass(slots=True)
class GameMemory:
    last_round_no: int = 0
    day: int = 0

    # LLM 预算：每日重置；任务期间调用不计数
    llm_calls_today: int = 0
    llm_quota_error_today: bool = False
    awaiting_llm: bool = False           # 已发 prompt 未收到 resp
    awaiting_llm_round: int = 0

    # 新闻与经济
    news_archive: dict[int, tuple[str, str]] = field(default_factory=dict)  # day -> (官方, 民间)
    price_history: list[tuple[int, dict[str, int]]] = field(default_factory=list)
    forecasts: list[OreForecast] = field(default_factory=list)
    forecasts_day: int = 0

    # 任务/宝藏
    evolve: EvolveTaskState = field(default_factory=EvolveTaskState)
    treasure: TreasureState = field(default_factory=TreasureState)
    sop_library: list[SopRecord] = field(default_factory=list)
    gate: GateState = field(default_factory=GateState)
    last_emergency_round: int = -999       # 上次应急道具使用回合（冷却用）
    tower_plan: dict[tuple[int, int], str] = field(default_factory=dict)
    site_failures: dict[tuple[int, int], int] = field(default_factory=dict)
    ring_completed: bool = False
    # ^ 围墙圈是否曾合拢过（合拢后缺口只由石头工单修，金属工回归采矿）
    # ^ 炮台位->武器类型的持久映射（loadout 锚定，防站点列表漂移导致重复建同一武器）

    # 指令反馈
    last_commands: dict[int, dict[str, Any]] = field(default_factory=dict)
    failed_actions: dict[int, dict[str, Any]] = field(default_factory=dict)  # 上回合失败的指令
    build_failures: set[tuple[int, int]] = field(default_factory=set)  # 试过不可建造的格子

    def begin_round(self, turn: Turn) -> None:
        new_day = turn.day != self.day
        if new_day:
            self.day = turn.day
            self.llm_calls_today = 0
            self.llm_quota_error_today = False
        self.failed_actions = {
            unit_id: self.last_commands.get(unit_id, {})
            for unit_id, ok in turn.last_action_results.items()
            if not ok and unit_id in self.last_commands
        }
        for command in self.failed_actions.values():
            if command.get("action") == "build":
                targets = command.get("targetPos") or []
                if targets:
                    self.build_failures.add((targets[0]["x"], targets[0]["y"]))
        self.price_history.append(
            (turn.round_no, dict(turn.vendor_prices))
        )
        self._trim()

    def remember_commands(self, turn: Turn, commands: dict[int, dict[str, Any]]) -> None:
        self.last_commands = dict(commands)

    def record_news(self, turn: Turn) -> None:
        news = turn.world_news
        if news.official or news.folk:
            self.news_archive.setdefault(
                turn.day, (news.official, news.folk),
            )

    # ---- LLM 预算

    @property
    def task_active(self) -> bool:
        return self.evolve.phase in ("accepted", "solving")

    def llm_budget_left(self) -> int:
        if self.llm_quota_error_today:
            return 0
        return max(0, LLM_FREE_PER_DAY - self.llm_calls_today)

    def spend_llm(self, turn: Turn) -> bool:
        """决定是否允许本回合发出 prompt；允许则记账。"""
        if self.awaiting_llm:
            return False
        if self.task_active:
            return True                   # 任务期间免费
        if self.llm_budget_left() <= 0:
            return False
        self.llm_calls_today += 1
        return True

    def observe_llm(self, turn: Turn) -> None:
        """每回合开头消化 llmResp / errorCode=5。"""
        if turn.llm_resp:
            self.awaiting_llm = False
        elif self.awaiting_llm and turn.round_no > self.awaiting_llm_round + 1:
            # 发出后两回合仍无响应，放弃等待避免死锁
            self.awaiting_llm = False
        for err in turn.errors:
            if err.code == 5:
                self.llm_quota_error_today = True
                self.awaiting_llm = False

    def mark_prompt_sent(self, turn: Turn) -> None:
        self.awaiting_llm = True
        self.awaiting_llm_round = turn.round_no

    # ---- SOP

    def find_sop(self, signature: str) -> SopRecord | None:
        for record in self.sop_library:
            if record.signature == signature:
                return record
        return None

    def add_sop(self, record: SopRecord) -> None:
        if not self.find_sop(record.signature):
            self.sop_library.append(record)

    # ---- 价格预测查询

    def mining_blocked(self, ore: str, turn: Turn) -> bool:
        rel = self._relative_day(turn)
        for forecast in self.forecasts:
            if forecast.ore == ore and rel in forecast.mining_blocked:
                return True
        return False

    def price_up_soon(self, ore: str, turn: Turn) -> bool:
        rel = self._relative_day(turn)
        for forecast in self.forecasts:
            if forecast.ore == ore and (rel in forecast.price_up or rel + 1 in forecast.price_up):
                return True
        return False

    def _relative_day(self, turn: Turn) -> int:
        return turn.day - self.forecasts_day if self.forecasts_day else 1

    def _trim(self) -> None:
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-100:]
        state = self.evolve
        if len(state.cmd_history) > 40:
            state.cmd_history = state.cmd_history[-20:]
        if len(state.llm_history) > 20:
            state.llm_history = state.llm_history[-10:]
