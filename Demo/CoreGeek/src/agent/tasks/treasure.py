"""长上下文任务：民间传闻 -> 宝藏三元组（地点/时间/祭品）推理与召唤。

流程：
1. 每日归档民间传闻（memory.news_archive）；
2. 累计 >=2 天线索后花 1 次 LLM 预算推理，要求输出严格 JSON 三元组；
3. 按推理结果购买祭品（武器商店 15 金币/个）、开拓者携带、
   在时间窗内走到宝藏点相邻召唤；
4. 用 lastSummonTreasureResult 反馈修正（3=祭品错 2=地点/时间错）。

宝藏全场仅一个，开启成功后停止。
"""
import json
import logging

from ..grid import next_step
from ..memory import GameMemory
from ..protocol import (
    Decision,
    Pos,
    Turn,
    Unit,
    cmd_buy,
    cmd_summon_treasure,
    distance,
)

LOGGER = logging.getLogger(__name__)

MIN_LEGEND_DAYS = 2          # 至少积累两天传闻再推理
CONFIDENCE_THRESHOLD = 0.5
RETHINK_COOLDOWN = 20        # 反馈失败后至少隔 20 回合再问 LLM
TASK_ITEM_PRICES = {"AcientTablet", "StarSand", "FlameBreath",
                    "FrostPotion", "ThornAmulet", "IronWhistle"}


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:index + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return None
    return None


class TreasureModule:
    def __init__(self, memory: GameMemory):
        self.memory = memory
        self._await_parse = False
        self._last_ask_round = 0
        self._summon_round = 0

    # ---------------------------------------------------------------- 观测

    def observe(self, turn: Turn) -> None:
        treasure = self.memory.treasure

        # 召唤反馈
        if turn.last_summon_result == 1:
            treasure.opened = True
            LOGGER.info("treasure opened!")
        elif turn.last_summon_result == 3:
            treasure.attempts.append((
                turn.round_no,
                {"items": list(treasure.deduced_items), "code": 3},
            ))
            treasure.tried_items.append(list(treasure.deduced_items))
            treasure.confidence = 0.0          # 祭品错误 -> 重新推理
        elif turn.last_summon_result == 2:
            treasure.attempts.append((
                turn.round_no,
                {"pos": treasure.deduced_pos, "code": 2},
            ))
        elif turn.last_summon_result == 4:
            treasure.opened = True              # 已被开过，放弃

        if not self._await_parse:
            return
        self._await_parse = False
        parsed = _extract_json(turn.llm_resp)
        if not parsed:
            LOGGER.warning("treasure llm response not JSON: %.200s", turn.llm_resp)
            return
        pos = parsed.get("pos") or parsed.get("position")
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            treasure.deduced_pos = (int(pos["x"]), int(pos["y"]))
        items = parsed.get("items")
        if isinstance(items, list) and items:
            treasure.deduced_items = [str(i) for i in items]
        day = parsed.get("day")
        if isinstance(day, int) and day > 0:
            treasure.deduced_day = day
        window = parsed.get("round_window") or parsed.get("rounds")
        if isinstance(window, list) and len(window) == 2:
            treasure.deduced_round_window = (int(window[0]), int(window[1]))
        try:
            treasure.confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            treasure.confidence = 0.0
        LOGGER.info(
            "treasure deduced: pos=%s items=%s day=%s window=%s conf=%.2f",
            treasure.deduced_pos, treasure.deduced_items, treasure.deduced_day,
            treasure.deduced_round_window, treasure.confidence,
        )

    # ---------------------------------------------------------------- LLM

    def wants_prompt(self, turn: Turn) -> str | None:
        treasure = self.memory.treasure
        if treasure.opened:
            return None
        legends = [
            (day, folk) for day, (_, folk) in sorted(self.memory.news_archive.items())
            if folk
        ]
        if len(legends) < MIN_LEGEND_DAYS:
            return None
        if treasure.confidence >= CONFIDENCE_THRESHOLD and treasure.deduced_items:
            return None
        if turn.round_no - self._last_ask_round < RETHINK_COOLDOWN and self._last_ask_round:
            return None
        self._last_ask_round = turn.round_no
        self._await_parse = True
        return self._build_prompt(turn, legends)

    def _build_prompt(self, turn: Turn, legends: list[tuple[int, str]]) -> str:
        lines = [f"第{day}天传闻：{folk}" for day, folk in legends]
        feedback = ""
        treasure = self.memory.treasure
        if treasure.tried_items:
            feedback = f"\n以下祭品组合已被证明错误：{treasure.tried_items}，请换一组。"
        if treasure.attempts:
            failed_pos = [a[1].get("pos") for a in treasure.attempts if a[1].get("code") == 2]
            if failed_pos:
                feedback += f"\n以下地点/时间召唤失败（无宝藏或未到时间）：{failed_pos}。"
        return (
            "你是解谜分析师。游戏每天发布民间传闻，全部传闻共同指向一个宝藏的"
            "位置、开启时间、开启所需祭品物品。请综合全部线索推理。\n"
            f"今天是第{turn.day}天，当前回合{turn.round_in_day}（每天130回合：白天1-70，夜晚71-130）。\n"
            f"地图大小 {turn.width}x{turn.height}，坐标原点在左下角。\n"
            "可购买的祭品物品名：AcientTablet, StarSand, FlameBreath, "
            "FrostPotion, ThornAmulet, IronWhistle。\n\n"
            f"全部传闻：\n{chr(10).join(lines)}\n{feedback}\n"
            "只输出一个JSON对象，不要其他内容，格式：\n"
            '{"pos": {"x": 0, "y": 0}, "items": ["物品名"], '
            '"day": <绝对天数>, "round_window": [<起>, <止>], "confidence": <0~1>}\n'
            "信息不足的字段填 null，confidence 如实估计。"
        )

    # ---------------------------------------------------------------- 行动

    def plan(self, turn: Turn, decision: Decision, claimed: set[Pos]) -> bool:
        """开拓者空闲时的宝藏行动；返回 True 表示已占用开拓者。"""
        treasure = self.memory.treasure
        pioneer = turn.pioneer()
        if pioneer is None or treasure.opened:
            return False
        if not treasure.deduced_items or treasure.deduced_pos is None:
            return False
        if treasure.confidence < CONFIDENCE_THRESHOLD:
            return False

        target_pos = Pos(*treasure.deduced_pos)
        items_needed = [i for i in treasure.deduced_items if i not in pioneer.backpack]
        shops = list(turn.weapon_shops())
        shop = min(shops, key=lambda s: distance(pioneer.pos, s)) if shops else None

        # 1) 补齐祭品
        if items_needed and shop is not None:
            affordable = [
                i for i in items_needed
                if turn.weapon_shop.get(i, 15) <= turn.gold
            ]
            if distance(pioneer.pos, shop) <= 1 and affordable:
                decision.commands[pioneer.unit_id] = cmd_buy(affordable[0], 1)
                return True
            step = next_step(turn, pioneer, shop)
            if step is not None and step not in claimed:
                claimed.add(step)
                decision.commands[pioneer.unit_id] = {
                    "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                }
                return True
            return True   # 钱不够/到不了：等待（经济会赚金币）

        # 2) 祭品齐了 -> 判断时间窗
        in_window = self._in_window(turn)
        if not in_window:
            # 时间未到：先到目标附近待命（白天）
            if turn.is_day and distance(pioneer.pos, target_pos) > 2:
                step = next_step(turn, pioneer, target_pos)
                if step is not None and step not in claimed:
                    claimed.add(step)
                    decision.commands[pioneer.unit_id] = {
                        "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
                    }
                    return True
            return turn.is_day   # 夜晚让位给 combat

        # 3) 在窗口内且祭品齐 -> 前往召唤
        if distance(pioneer.pos, target_pos) <= 1:
            if turn.round_no - self._summon_round >= 2:
                self._summon_round = turn.round_no
                decision.commands[pioneer.unit_id] = cmd_summon_treasure(
                    target_pos, list(treasure.deduced_items),
                )
                return True
            return True
        step = next_step(turn, pioneer, target_pos)
        if step is not None:
            decision.commands[pioneer.unit_id] = {
                "action": "move", "targetPos": [{"x": step.x, "y": step.y}],
            }
            return True
        return True

    def _in_window(self, turn: Turn) -> bool:
        treasure = self.memory.treasure
        if treasure.deduced_day and treasure.deduced_day != turn.day:
            return False
        if treasure.deduced_round_window is None:
            return True
        start, end = treasure.deduced_round_window
        return start <= turn.round_in_day <= end
