"""自进化任务模块：开拓者领取任务点任务，LLM 主导 + 沙盒迭代解题。

异步交互模型（接口文档）：
- 本回合响应携带 prompt / executeCmd，下一回合请求带回 llmResp / lastCmdResult；
- 任务期间（领取->结束）LLM 调用不计数、不限次；
- 每回合至多 1 条 executeCmd，单条 ≤15s；
- 领取任务的开拓者离开任务点一格内任务即结束，因此解题期间必须驻守。

稳健化设计（真实环境 LLM 输出格式不可控）：
- 多格式解析链：JSON(含围栏) -> ```bash/```sh 围栏 -> "$ cmd"/常见命令行 -> "答案:" 行；
- 连续解析失败 3 次 -> 发严格 JSON 修正 prompt；6 次 -> 转为本地兜底探索命令序列；
- 超时 85% 处强制 submitAnswer（best_answer -> 末次命令输出末 token -> 空串），
  保证部分通过率分；
- SOP 防过期：任务全文一致才直接复用答案；仅签名一致则把命令序列作为提示注入 prompt。
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
WALK_GIVE_UP = 10             # 走了 10 回合还没到任务点则换点/放弃
ACCEPT_MAX_RETRIES = 2        # acceptTask 已发但任务未下发的最大重试次数
PARSE_FAIL_CORRECT = 3        # 连续解析失败 N 次发修正 prompt
PARSE_FAIL_EXPLORE = 6        # 连续解析失败 N 次转本地兜底探索
SOP_SIGNATURE_LEN = 40

COMMAND_HINTS = (
    "ls", "cat ", "python3", "python ", "wc ", "grep ", "find ", "head ",
    "tail ", "pwd", "echo ", "sed ", "awk ", "curl ",
)
EXPLORE_FALLBACK = (
    "ls -la",
    "ls -la /tmp 2>/dev/null; ls -la .",
    "cat README* *.md 2>/dev/null | head -60",
)


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


def extract_llm_action(text: str) -> tuple[str, str] | None:
    """多格式解析链：返回 ("cmd", 命令) / ("answer", 答案) / None。

    顺序：JSON -> ```bash/```sh 围栏 -> "$ cmd"或常见命令行 -> 答案标记行。
    """
    if not text:
        return None
    parsed = _extract_json(text)
    if parsed:
        # 键名别名：判题器 LLM 未必按我们的示例用 executeCmd/taskAnswer
        cmd = (parsed.get("executeCmd") or parsed.get("command")
               or parsed.get("cmd"))
        answer = (parsed.get("taskAnswer") or parsed.get("answer")
                  or parsed.get("final_answer"))
        if isinstance(cmd, str) and cmd.strip():
            return ("cmd", cmd.strip())
        if isinstance(answer, str) and answer.strip():
            return ("answer", answer.strip())
        return None

    fence = re.search(r"```(?:bash|sh|shell)?\s*\n(.*?)```", text, re.S)
    if fence:
        for line in fence.group(1).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return ("cmd", line.lstrip("$ ").strip())

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("$ ") and len(s) > 2:
            return ("cmd", s[2:].strip())
        if any(s.startswith(hint) for hint in COMMAND_HINTS):
            return ("cmd", s)

    marked = re.search(r"(?:最终答案|答案|answer)\s*[:：]\s*(\S[^\n]*)", text, re.I)
    if marked:
        value = marked.group(1).strip()
        # 太长或含叙述性标点说明是分析过程（如"初步答案是42，让我再验证"），
        # 不是答案本身
        if len(value) <= 24 and not re.search(r"[，。；！？、“”‘’]", value):
            return ("answer", value)
    return None


def last_token_of(text: str) -> str:
    """取命令输出最后一个 token，作为强制提交时的猜答。"""
    tokens = (text or "").replace("[TRUNCATED]", "").split()
    return tokens[-1] if tokens else ""


class EvolveModule:
    def __init__(self, memory: GameMemory):
        self.memory = memory
        self._pending_cmd: str | None = None
        self._pending_answer: str | None = None
        self._sent_cmd_log: list[str] = []   # 已发往沙盒的命令（配对 lastCmdResult）

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
            state.cmd_history.append(
                (self._last_sent_cmd(), turn.last_cmd_result[:HISTORY_CHAR_LIMIT]),
            )

    def _last_sent_cmd(self) -> str:
        """最近一次发往沙盒的命令。"""
        return self._sent_cmd_log[-1] if self._sent_cmd_log else ""

    def note_execute_cmd(self, cmd: str) -> None:
        self._sent_cmd_log.append(cmd)
        if len(self._sent_cmd_log) > 20:
            self._sent_cmd_log = self._sent_cmd_log[-10:]

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
        state.full_task_text = turn.phase_task
        state.accept_retries = 0
        state.parse_failures = 0
        state.corrected_prompt_sent = False
        state.sop_hint_used = False
        state.sop_hint_injected = False
        # 每个任务独立的干净上下文：清掉上一个任务遗留的命令/响应历史，
        # 否则上一任务的输出会滚进本任务 prompt，LLM 会被无关上下文带偏
        state.cmd_history = []
        state.llm_history = []
        state.explore_index = 0
        state.llm_retry_round = 0
        for tp in turn.player_tasks:
            if state.task_point and tp.pos == Pos(*state.task_point):
                state.task_type = tp.task_type
                state.timeout_rounds = tp.timeout_rounds or 60
                break
        # SOP：全文一致才直接复用答案；仅签名一致则注入提示
        sop = self.memory.find_sop(state.signature)
        if sop is not None and sop.answer:
            if sop.full_text == turn.phase_task:
                LOGGER.info("SOP exact hit: %r", state.signature)
                self._pending_answer = sop.answer
            else:
                state.sop_hint_used = True      # 提示在 _build_prompt 中注入
                LOGGER.info("SOP signature hit (hint mode): %r", state.signature)

    def _observe_solving(self, turn: Turn) -> None:
        state = self.memory.evolve
        if not turn.phase_task:
            self._finish(turn)
            return
        if not turn.llm_resp:
            return
        state.llm_history.append(("", turn.llm_resp[:HISTORY_CHAR_LIMIT]))
        action = extract_llm_action(turn.llm_resp)
        if action is None:
            state.parse_failures += 1
            if state.parse_failures >= PARSE_FAIL_CORRECT:
                state.corrected_prompt_sent = False   # 下回合请求发修正 prompt
            LOGGER.warning("llm resp unparseable (#%d): %.120s",
                           state.parse_failures, turn.llm_resp)
            return
        state.parse_failures = 0
        kind, value = action
        if kind == "cmd":
            self._pending_cmd = value
        else:
            self._pending_answer = value

    def _finish(self, turn: Turn) -> None:
        state = self.memory.evolve
        rejected = any(e.code == 2 for e in turn.errors)
        if state.last_answer and not rejected and state.signature:
            self.memory.add_sop(SopRecord(
                signature=state.signature,
                task_type=state.task_type,
                answer=state.last_answer,
                full_text=state.full_task_text,
            ))
            LOGGER.info("task finished, SOP recorded: %r", state.signature)
        state.phase = "idle"
        state.task_point = None
        state.accepted_round = 0
        state.signature = ""
        state.full_task_text = ""
        state.last_answer = ""
        state.best_answer = ""
        state.accept_retries = 0
        state.parse_failures = 0
        state.cmd_history = []
        state.llm_history = []
        state.explore_index = 0
        state.llm_retry_round = 0
        state.corrected_prompt_sent = False

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
            target_task = self._choose_task_point(turn)
            if target_task is None:
                return False
            state.task_point = (target_task.pos.x, target_task.pos.y)
            state.phase = "walking"
            state.walk_start_round = turn.round_no
            state.accepted_round = 0
            state.accept_retries = 0
        point = Pos(*state.task_point)  # type: ignore[arg-type]

        if distance(pioneer.pos, point) <= 1:
            # 冷却中的任务点：驻守等待（不发 acceptTask），冷却结束立即接
            tp = next((t for t in turn.player_tasks if t.pos == point), None)
            can_accept = tp is None or (tp.is_valid and tp.cooldown_rounds == 0)
            if not can_accept:
                return True
            if state.accepted_round == 0:
                decision.commands[pioneer.unit_id] = cmd_accept_task()
                state.accepted_round = turn.round_no
                state.accept_retries = 1
            elif state.accept_retries <= ACCEPT_MAX_RETRIES:
                # 任务还没下发，谨慎重试（可能上一条 accept 失败）
                decision.commands[pioneer.unit_id] = cmd_accept_task()
                state.accept_retries += 1
            else:
                # 任务点坏了：换点
                LOGGER.info("task point %s not responding, rotating", point)
                state.phase = "idle"
                state.task_point = None
                state.accepted_round = 0
                state.accept_retries = 0
            return True
        step = next_step(turn, pioneer, point)
        if step is not None and step not in claimed:
            claimed.add(step)
            decision.commands[pioneer.unit_id] = {
                "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
            }
            return True
        return True   # 卡住也保留状态，下回合继续尝试

    def _choose_task_point(self, turn: Turn):
        """选任务点：可接的优先；全在冷却时选最快恢复的（提前驻守等待）。"""
        state = self.memory.evolve
        pioneer = turn.pioneer()
        if pioneer is None:
            return None
        points = [tp for tp in turn.player_tasks if tp.timeout_rounds > 0]
        if not points:
            return None
        ready = [tp for tp in points if tp.is_valid and tp.cooldown_rounds == 0]
        pool = ready or points
        pool.sort(key=lambda tp: (
            tp.cooldown_rounds, distance(pioneer.pos, tp.pos), tp.pos.x, tp.pos.y,
        ))
        index = state.last_point_index % min(len(pool), 2)
        state.last_point_index += 1
        return pool[index]

    def _forced_answer(self) -> str:
        state = self.memory.evolve
        if state.best_answer:
            return state.best_answer
        if state.cmd_history:
            return last_token_of(state.cmd_history[-1][1])
        return ""

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
                forced = self._forced_answer()
                if forced:
                    decision.commands[pioneer.unit_id] = cmd_submit_answer(forced)
                    state.last_answer = forced
                state.phase = "idle"     # 移动交给 combat；离开即任务结束
                return False

        # 超时保护：接近超时必提交（哪怕空串也拿部分通过率机会）
        if state.timeout_rounds:
            elapsed = turn.round_no - state.accepted_round
            if elapsed >= int(state.timeout_rounds * TIMEOUT_SAFETY):
                answer = self._forced_answer()
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
        state = self.memory.evolve
        if state.phase != "solving":
            return None
        if self._pending_cmd:
            cmd = self._pending_cmd
            self._pending_cmd = None
            self.note_execute_cmd(cmd)
            return cmd
        # LLM 连续解析失败：转为本地兜底探索命令推进信息收集
        if state.parse_failures >= PARSE_FAIL_EXPLORE \
                and state.explore_index < len(EXPLORE_FALLBACK):
            cmd = EXPLORE_FALLBACK[state.explore_index]
            state.explore_index += 1
            self.note_execute_cmd(cmd)
            return cmd
        return None

    def wants_prompt(self, turn: Turn) -> str | None:
        state = self.memory.evolve
        if state.phase != "solving":
            return None
        if self._pending_cmd or self._pending_answer:
            return None
        if state.parse_failures >= PARSE_FAIL_EXPLORE:
            # 兜底探索期间每 10 回合重试一次 LLM
            if turn.round_no - state.llm_retry_round >= 10:
                state.llm_retry_round = turn.round_no
                return self._build_prompt(turn)
            return None
        return self._build_prompt(turn)

    def _build_prompt(self, turn: Turn) -> str:
        state = self.memory.evolve
        pairs: list[str] = []
        for cmd, result in state.cmd_history[-6:]:
            snippet = result.replace("\n", " | ")
            if len(snippet) > 400:
                # 只保留末尾：报错信息/计算结果/文档关键段几乎都在输出的后半段
                snippet = "…" + snippet[-400:]
            pairs.append(f"$ {cmd[:120]}\n{snippet}" if cmd else snippet)
        rejected = any(e.code == 2 for e in turn.errors)

        # 解析失败修正 prompt
        if state.parse_failures >= PARSE_FAIL_CORRECT and not state.corrected_prompt_sent:
            state.corrected_prompt_sent = True
            return (
                "你上次的输出无法解析。请严格只输出一个 JSON 对象，不要任何多余文字：\n"
                '{"executeCmd": "<一条bash命令>"} 或 {"taskAnswer": "<最终答案>"}'
            )

        sop_hint = ""
        if state.sop_hint_used and not state.sop_hint_injected:
            sop = self.memory.find_sop(state.signature)
            if sop is not None and sop.commands:
                sop_hint = (
                    "\n参考：曾解决过同模板任务，当时使用的命令序列：\n"
                    + "\n".join(f"- {c}" for c in sop.commands[:6])
                    + "\n答案形态类似：" + (sop.answer[:40] or "（见命令输出）") + "\n"
                )
            state.sop_hint_injected = True

        feedback = "注意：上次提交的答案被判错误，请修正后重新作答。" if rejected else ""
        return (
            "你是编程竞赛AI，在沙盒中解题。沙盒为Linux，可执行bash命令与python3，无网络，"
            "每回合只能执行一条命令，单条命令限时15秒，请避免长循环。\n"
            f"任务描述：\n{turn.phase_task}\n\n"
            f"最近执行记录：\n{chr(10).join(pairs) or '（尚未执行任何命令）'}\n"
            f"{sop_hint}{feedback}\n"
            "请只输出一个JSON对象，不要输出任何其他内容，格式二选一：\n"
            '{"executeCmd": "<要执行的一条bash命令，用于探索或求解>"}\n'
            '{"taskAnswer": "<已确定的最终答案字符串>"}\n'
            '示例：{"executeCmd": "cat /home/README.md"}'
        )
