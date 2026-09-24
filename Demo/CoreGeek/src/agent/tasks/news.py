"""推理类任务：官方消息 -> 矿石价格波动 / 停工窗口预测，联动经济决策。

每日白天首回合发布官方消息；每天 LLM 预算 3 次，这里消耗 1 次。
LLM 输出严格 JSON 预测，写入 memory.forecasts 供 economy 选矿使用。
"""
import json
import logging

from ..memory import GameMemory, OreForecast
from ..protocol import Turn

LOGGER = logging.getLogger(__name__)

NO_NEWS_MARKERS = ("今日无重大新闻", "无重大新闻", "无新闻")


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


class NewsModule:
    def __init__(self, memory: GameMemory):
        self.memory = memory
        self._last_asked_day = 0
        self._await_parse = False

    def observe(self, turn: Turn) -> None:
        """消化 llmResp 中的预测 JSON。"""
        if not self._await_parse:
            return
        self._await_parse = False
        parsed = _extract_json(turn.llm_resp)
        if not parsed:
            LOGGER.warning("news llm response not JSON: %.200s", turn.llm_resp)
            return
        forecasts: list[OreForecast] = []
        for item in parsed.get("forecasts", []):
            if not isinstance(item, dict):
                continue
            ore = str(item.get("ore") or "")
            if ore not in ("stone", "iron", "copper"):
                continue
            forecasts.append(OreForecast(
                ore=ore,
                mining_blocked={int(d) for d in item.get("mining_blocked", []) if isinstance(d, int)},
                price_up={int(d) for d in item.get("price_up", []) if isinstance(d, int)},
                price_down={int(d) for d in item.get("price_down", []) if isinstance(d, int)},
            ))
        if forecasts:
            self.memory.forecasts = forecasts
            self.memory.forecasts_day = turn.day
            LOGGER.info("forecasts updated: %s", forecasts)

    def wants_prompt(self, turn: Turn) -> str | None:
        if not turn.is_day or turn.round_in_day > 5:
            return None
        if self._last_asked_day == turn.day:
            return None
        news = turn.world_news.official
        if not news or any(marker in news for marker in NO_NEWS_MARKERS):
            return None
        self._last_asked_day = turn.day
        self._await_parse = True
        return self._build_prompt(turn)

    def _build_prompt(self, turn: Turn) -> str:
        prices = ", ".join(f"{ore}={turn.ore_price(ore)}" for ore in ("stone", "iron", "copper"))
        return (
            "你是游戏经济分析师。游戏每天发布官方消息，可能引起矿石供需变化：\n"
            f"今天官方消息：{turn.world_news.official}\n\n"
            f"当前小贩回收价（金币/个）：{prices}。今天是第{turn.day}天。\n"
            "请推断该消息对未来4天（相对天数：1=今天, 2=明天, ...）的影响，"
            "只输出一个JSON对象，不要其他内容，格式：\n"
            '{"forecasts": [{"ore": "iron", "mining_blocked": [2,3], '
            '"price_up": [2,3], "price_down": [4]}]}\n'
            "字段均为相对天数数组，可省略无影响矿种；价格不变的矿种不要列出。"
        )
