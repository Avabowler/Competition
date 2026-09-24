"""出站指令校验器：保证响应永不触发"异常响应"。

任务书第八章：异常响应含"响应格式错误"与"指令错误"（指令字段缺失、无法识别），
累计 5 次则不再被调度。因此任何指令在发送前必须通过这里的硬校验；
不满足游戏规则的指令（如攻击超射程）只算"执行失败"不计异常，做软校验仅记日志。
"""
import logging
from typing import Any

from .protocol import (
    ORES,
    TOWER_TYPES,
    USE_NEEDS_TARGET,
    Decision,
    Pos,
    Turn,
    distance,
)

LOGGER = logging.getLogger(__name__)

VALID_ACTIONS = frozenset({
    "move", "attack", "sell", "buy", "build", "remove", "acceptTask",
    "submitAnswer", "summonTreasure", "use", "drop", "collect",
})


def _valid_pos_list(raw: Any, expected: int | None = None) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    if expected is not None and len(raw) != expected:
        return False
    for item in raw:
        if not isinstance(item, dict):
            return False
        try:
            x, y = int(item.get("x")), int(item.get("y"))
        except (TypeError, ValueError):
            return False
        if x != item.get("x") and not isinstance(item.get("x"), int):
            return False
        if y != item.get("y") and not isinstance(item.get("y"), int):
            return False
    return True


def _hard_check(command: dict[str, Any], turn: Turn, unit_id: int) -> str | None:
    """返回 None 表示通过硬校验；返回字符串为拒绝原因（格式层面非法）。"""
    action = command.get("action")
    if action not in VALID_ACTIONS:
        return f"unknown action {action!r}"

    unit = turn.unit_by_id(unit_id)

    if action == "move":
        if not _valid_pos_list(command.get("targetPos"), 1):
            return "move needs targetPos[1]"
    elif action == "attack":
        if not isinstance(command.get("controllerId"), str) or not command["controllerId"]:
            return "attack needs string controllerId"
        if not _valid_pos_list(command.get("targetPos")):
            return "attack needs non-empty targetPos"
        if unit is None or not unit.is_weapon:
            return "attack must be keyed by a weapon id"
        targets = len(command["targetPos"])
        if unit.kind == "railgun" and targets != 1:
            return "railgun needs exactly 1 target"
        if unit.kind in ("gatling", "rocket") and targets > max(unit.level, 1):
            return f"{unit.kind} targets {targets} > level {unit.level}"
    elif action == "sell":
        if command.get("name") not in ORES:
            return "sell needs ore name"
        if "num" in command and not isinstance(command.get("num"), int):
            return "sell num must be int"
    elif action == "buy":
        if not isinstance(command.get("name"), str) or not command["name"]:
            return "buy needs name"
        if "num" in command and not isinstance(command.get("num"), int):
            return "buy num must be int"
    elif action == "build":
        if command.get("name") not in (*TOWER_TYPES, "wall"):
            return "build needs building name"
        if not _valid_pos_list(command.get("targetPos"), 1):
            return "build needs targetPos[1]"
    elif action == "remove":
        if not _valid_pos_list(command.get("targetPos"), 1):
            return "remove needs targetPos[1]"
    elif action == "acceptTask":
        pass
    elif action == "submitAnswer":
        if not isinstance(command.get("taskAnswer"), str):
            return "submitAnswer needs string taskAnswer"
    elif action == "summonTreasure":
        if not _valid_pos_list(command.get("targetPos"), 1):
            return "summonTreasure needs targetPos[1]"
        items = command.get("item")
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            return "summonTreasure needs string item list"
    elif action == "use":
        name = command.get("name")
        if not isinstance(name, str) or not name:
            return "use needs name"
        if name in USE_NEEDS_TARGET and not _valid_pos_list(command.get("targetPos"), 1):
            return f"use {name} needs targetPos[1]"
    elif action == "drop":
        if not isinstance(command.get("name"), str) or not command["name"]:
            return "drop needs name"
    elif action == "collect":
        if not _valid_pos_list(command.get("targetPos"), 1):
            return "collect needs targetPos[1]"

    if unit is not None and unit.kind not in ("worker", "pioneer") and action != "attack":
        # 非攻击指令只能由角色发出（attack 以武器 ID 为 key）
        return f"action {action} keyed by non-character unit {unit_id}"
    return None


def _soft_check(command: dict[str, Any], turn: Turn, unit_id: int) -> None:
    """游戏规则层面大概率失败的指令：仅记录日志，仍照发（不计异常）。"""
    unit = turn.unit_by_id(unit_id)
    action = command["action"]
    if unit is None:
        return
    raw_targets = command.get("targetPos") or []
    targets = [Pos(int(p["x"]), int(p["y"])) for p in raw_targets]

    if action == "move":
        if targets and distance(unit.pos, targets[0]) != 1:
            LOGGER.warning("soft: %s move not adjacent %s->%s", unit_id, unit.pos, targets[0])
        if targets and not turn.on_map(targets[0]):
            LOGGER.warning("soft: %s move off map %s", unit_id, targets[0])
    elif action == "attack":
        reach = unit.range_of_attack()
        for target in targets:
            if distance(unit.pos, target) > reach:
                LOGGER.warning("soft: weapon %s target %s out of range %d", unit_id, target, reach)
    elif action == "collect":
        if targets and turn.zones.get(targets[0]) not in ORES:
            LOGGER.warning("soft: %s collect at non-mine %s", unit_id, targets[0])
        if targets and distance(unit.pos, targets[0]) > 1:
            LOGGER.warning("soft: %s collect not adjacent", unit_id)
    elif action == "build":
        if targets and distance(unit.pos, targets[0]) > 1:
            LOGGER.warning("soft: %s build not adjacent", unit_id)
    elif action == "use":
        name = command.get("name")
        if name not in unit.backpack:
            LOGGER.warning("soft: %s use %s not in backpack", unit_id, name)


def sanitize(decision: Decision, turn: Turn) -> Decision:
    """过滤掉硬校验不过的指令；同一角色重复 key 只保留最后一个。"""
    clean: dict[int, dict[str, Any]] = {}
    for unit_id, command in decision.commands.items():
        if not isinstance(command, dict):
            LOGGER.error("drop non-dict command for %s", unit_id)
            continue
        try:
            reason = _hard_check(command, turn, unit_id)
        except Exception:  # 校验器自身异常也要兜住
            LOGGER.exception("hard check crashed for %s", unit_id)
            reason = "checker error"
        if reason is not None:
            LOGGER.error("drop command unit=%s: %s (%r)", unit_id, reason, command)
            continue
        _soft_check(command, turn, unit_id)
        clean[unit_id] = command

    prompt = decision.prompt if isinstance(decision.prompt, str) else ""
    execute_cmd = decision.execute_cmd if isinstance(decision.execute_cmd, str) else ""
    if execute_cmd and not turn.phase_task:
        # 沙盒命令仅在执行任务期间可用，其余场景丢弃防异常
        LOGGER.error("drop executeCmd outside task")
        execute_cmd = ""
    return Decision(commands=clean, prompt=prompt, execute_cmd=execute_cmd)
