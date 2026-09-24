"""自进化任务模块：开拓者领取任务点任务，LLM+沙盒迭代解题。

异步交互模型（接口文档）：
- 本回合响应携带 prompt / executeCmd，下一回合请求带回 llmResp / lastCmdResult；
- 任务期间（领取->结束）LLM 调用不计数、不限次；
- 每回合至多 1 条 executeCmd，单条 ≤15s；
- 领取任务的开拓者离开任务点一格内任务即结束，因此解题期间必须驻守。

状态机：idle -> walking -> solving -> idle
- walking：走向任务点，到达相邻即 acceptTask；
- solving：phase_task 已下发，循环 prompt/executeCmd，产出答案即 submitAnswer；
- 任务结束（phase_task 消失）回到 idle，答案未被判错则沉淀 SOP。
"""
import json
import logging
import re

from ..grid import next_step
from ..memory import GameMemory, SopRecord
from ..protocol import (
    Decision,
    Pos,
    Turn,
    Unit,
    cmd_accept_task,
    cmd_submit_answer,
    distance,
)

LOGGER = logging.getLogger(__name__)

ACCEPT_DEADLINE = 55          # 白天第 55 回合后不再接新任务（保证有解题时间）
TIMEOUT_SAFETY = 0.85         # 超时回合的 85% 处强制提交最优答案
HISTORY_CHAR_LIMIT = 1200     # 喂给 LLM 的单条输出截断长度
DANGER_DISTANCE = 7           # 夜晚机器人逼近该距离则弃任务保命
WALK_GIVE_UP = 30             # 走了 30 回合还没到任务点则放弃本次
SOP_SIGNATURE_LEN = 40


def task_signature(text: str) -> str:
    return re.sub(r"\s+", "", text)[:SOP_SIGNATURE_LEN]


def _extract_json(text: str) -> dict | None:
    """从 LLM 回复中提取第一个 JSON 对象，容忍代码围栏与前后杂文。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        end = -1
        for index in range(start, len(cleaned)):
            ch = cleaned[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end != -1:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        start = cleaned.find("{", start + 1)
    return None


class EvolveModule:
    def __init__(self, memory: GameMemory):
        self.memory = memory
        self._pending_cmd: str | None = None
        self._pending_answer: str | None = None

    # ---------------------------------------------------------------- 观测

    def observe(self, turn: Turn) -> None:
        """回合开头消化 llmResp / lastCmdResult / errors，推进状态机。"""
        state = self.memory.evolve
        self._pending_cmd = None
        self._pending_answer = None

        if state.phase == "walking":
            self._observe_walking(turn)
        elif state.phase == "solving":
            self._observe_solving(turn)

        if turn.last_cmd_result:
            # 命令历史：与上一轮发出的命令配对由 brain 记录，这里只存结果
            state.cmd_history.append(("", turn.last_cmd_result[:HISTORY_CHAR_LIMIT]))

    def _observe_walking(self, turn: Turn) -> None:
        state = self.memory.evolve
        if turn.phase_task:
            self._enter_solving(turn)
            return
        pioneer = turn.pioneer()
        if pioneer is None:
            state.phase = "idle"
            return
        if turn.round_no - state.walk_start_round > WALK_GIVE_UP:
            LOGGER.info("give up walking to task point")
            state.phase = "idle"
            state.task_point = None

    def _enter_solving(self, turn: Turn) -> None:
        state = self.memory.evolve
        state.phase = "solving"
        state.signature = task_signature(turn.phase_task)
        for tp in turn.player_tasks:
            if state.task_point and tp.pos == Pos(*state.task_point):
                state.task_type = tp.task_type
                state.timeout_rounds = tp.timeout_rounds or 60
                break
        sop = self.memory.find_sop(state.signature)
        if sop is not None and sop.answer:
            LOGGER.info("SOP hit: %r", state.signature)
            self._pending_answer = sop.answer

    def _observe_solving(self, turn: Turn) -> None:
        state = self.memory.evolve
        if not turn.phase_task:
            self._finish(turn)
            return
        if turn.llm_resp:
            state.llm_history.append(("", turn.llm_resp[:HISTORY_CHAR_LIMIT]))
            parsed = _extract_json(turn.llm_resp)
            if parsed:
                cmd = parsed.get("executeCmd")
                answer = parsed.get("taskAnswer")
                if isinstance(cmd, str) and cmd.strip():
                    self._pending_cmd = cmd.strip()
                elif isinstance(answer, str) and answer.strip():
                    self._pending_answer = answer.strip()

    def _finish(self, turn: Turn) -> None:
        state = self.memory.evolve
        rejected = any(e.code == 2 for e in turn.errors) or \
            any(e.code == 2 for e in turn.errors)
        if state.last_answer and not rejected and state.signature:
            self.memory.add_sop(SopRecord(
                signature=state.signature,
                task_type=state.task_type,
                answer=state.last_answer,
                commands=[],
            ))
            LOGGER.info("task finished, SOP recorded: %r", state.signature)
        state.phase = "idle"
        state.task_point = None
        state.accepted_round = 0
        state.signature = ""
        state.last_answer = ""

    # ---------------------------------------------------------------- 规划

    def plan(self, turn: Turn, decision: Decision, claimed: set[Pos]) -> bool:
        """产出开拓者本回合指令；返回 True 表示已占用开拓者。"""
        state = self.memory.evolve
        pioneer = turn.pioneer()
        if pioneer is None:
            state.phase = "idle"
            return False
        if state.phase == "idle":
            return self._plan_goto_point(turn, pioneer, decision, claimed)
        if state.phase == "walking":
            return self._plan_goto_point(turn, pioneer, decision, claimed)
        if state.phase == "solving":
            return self._plan_solving(turn, pioneer, decision, claimed)
        return False

    def _plan_goto_point(self, turn: Turn, pioneer: Unit, decision: Decision,
                         claimed: set[Pos]) -> bool:
        state = self.memory.evolve
        if not turn.is_day:
            return False
        if turn.round_in_day > ACCEPT_DEADLINE:
            return False
        if state.phase == "idle":
            target = self._choose_task_point(turn)
            if target is None:
                return False
            state.task_point = (target.x, target.y)
            state.phase = "walking"
            state.walk_start_round = turn.round_no
        point = Pos(*state.task_point)  # type: ignore[arg-type]

        if distance(pioneer.pos, point) <= 1:
            decision.commands[pioneer.unit_id] = cmd_accept_task()
            state.accepted_round = turn.round_no
            return True
        step = next_step(turn, pioneer, point)
        if step is not None and step not in claimed:
            claimed.add(step)
            decision.commands[pioneer.unit_id] = {
                "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
            }
            return True
        return True   # 卡住也保留状态，下回合继续尝试

    def _choose_task_point(self, turn: Turn) -> Pos | None:
        state = self.memory.evolve
        pioneer = turn.pioneer()
        if pioneer is None:
            return None
        points = [
            tp for tp in turn.player_tasks
            if tp.is_valid and tp.cooldown_rounds == 0 and tp.timeout_rounds > 0
        ]
        if not points:
            return None
        points.sort(key=lambda tp: (distance(pioneer.pos, tp.pos), tp.pos.x, tp.pos.y))
        index = state.last_point_index % min(len(points), 2)
        state.last_point_index += 1
        return points[index].pos

    def _plan_solving(self, turn: Turn, pioneer: Unit, decision: Decision,
                      claimed: set[Pos]) -> bool:
        state = self.memory.evolve
        if state.task_point is None:
            state.phase = "idle"
            return False
        point = Pos(*state.task_point)

        # 夜晚机器人逼近：提交最优答案后放弃（保命）
        if not turn.is_day:
            threat = min(
                (distance(pioneer.pos, r.pos) for r in turn.robots), default=99,
            )
            if threat <= DANGER_DISTANCE:
                if state.best_answer:
                    decision.commands[pioneer.unit_id] = cmd_submit_answer(state.best_answer)
                    state.last_answer = state.best_answer
                state.phase = "idle"     # 移动交给 combat；离开即任务结束
                return False

        # 超时保护：接近超时强制提交
        if state.timeout_rounds:
            elapsed = turn.round_no - state.accepted_round
            if elapsed >= int(state.timeout_rounds * TIMEOUT_SAFETY):
                answer = state.best_answer
                if answer:
                    decision.commands[pioneer.unit_id] = cmd_submit_answer(answer)
                    state.last_answer = answer
                    state.phase = "idle"
                    return True

        # 有答案就提交
        if self._pending_answer is not None:
            decision.commands[pioneer.unit_id] = cmd_submit_answer(self._pending_answer)
            state.last_answer = self._pending_answer
            state.best_answer = self._pending_answer
            self._pending_answer = None
            return True

        # 驻守任务点一格内（离开即任务结束）
        if distance(pioneer.pos, point) > 1:
            step = next_step(turn, pioneer, point)
            if step is not None:
                decision.commands[pioneer.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
                return True
        # 原地等待 LLM/沙盒结果（executeCmd/prompt 由 brain 填充）
        return True

    # ---------------------------------------------------------------- LLM/沙盒

    def wants_execute_cmd(self, turn: Turn) -> str | None:
        if self.memory.evolve.phase != "solving":
            return None
        if self._pending_cmd:
            cmd = self._pending_cmd
            self._pending_cmd = None
            return cmd
        return None

    def wants_prompt(self, turn: Turn) -> str | None:
        state = self.memory.evolve
        if state.phase != "solving":
            return None
        if self._pending_cmd or self._pending_answer:
            return None
        return self._build_prompt(turn)

    def _build_prompt(self, turn: Turn) -> str:
        state = self.memory.evolve
        pairs: list[str] = []
        for cmd, result in state.cmd_history[-6:]:
            snippet = result.replace("\n", " | ")[:300]
            pairs.append(f"$ {cmd[:120]}\n{snippet}" if cmd else snippet)
        rejected = any(e.code == 2 for e in turn.errors)
        feedback = "注意：上次提交的答案被判错误，请修正后重新作答。" if rejected else ""
        return (
            "你是编程竞赛AI，在沙盒中解题。沙盒为Linux，可执行bash命令与python3，无网络，"
            "每回合只能执行一条命令，单条命令限时15秒，请避免长循环。\n"
            f"任务描述：\n{turn.phase_task}\n\n"
            f"最近执行记录：\n{chr(10).join(pairs) or '（尚未执行任何命令）'}\n"
            f"{feedback}\n"
            "请只输出一个JSON对象，不要输出任何其他内容，格式二选一：\n"
            '{"executeCmd": "<要执行的一条bash命令，用于探索或求解>"}\n'
            '{"taskAnswer": "<已确定的最终答案字符串>"}'
        )
