"""协议层：request 全字段解析、游戏常量表、出站指令构造器。

接口约定见 docs/接口文档.md：
- 我方 roleType: station / gatling / railgun / rocket / wall / pioneer / worker
- 机器人 roleType: smallRobot / middleRobot / largeRobot / bossRobot
- 距离一律为切比雪夫距离 max(|dx|, |dy|)；基地 2x2，pos 为左上角。
"""
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------- 常量表

MAP_WIDTH = 41
MAP_HEIGHT = 32

DAY_ROUNDS = 70
NIGHT_ROUNDS = 60
ROUNDS_PER_DAY = DAY_ROUNDS + NIGHT_ROUNDS
MAX_ROUNDS = 10 * ROUNDS_PER_DAY

VISION_RANGE = 4

ORES = ("stone", "iron", "copper")
TOWER_TYPES = ("gatling", "railgun", "rocket")
BUILDING_TYPES = ("station", "gatling", "railgun", "rocket", "wall")
CONTROLLABLE_TYPES = ("worker", "pioneer")

WEAPON_BUILD_COST = 25
MAX_WEAPON_COUNT = 3
INITIAL_GOLD = 75

# (攻击力, 攻击距离, 血量) 按等级 1/2/3
WEAPON_STATS = {
    "gatling": ((10, 3, 1000), (20, 5, 1500), (30, 7, 2000)),
    "railgun": ((10, 6, 1000), (20, 8, 1500), (30, 10, 2000)),
    "rocket": ((20, 10, 1000), (40, 15, 1500), (60, 10**9, 2000)),
}
WALL_STATS = ((0, 0, 1000), (0, 0, 1500), (0, 0, 2000))
STATION_STATS = ((0, 0, 1500), (0, 0, 3000), (0, 0, 4500))

ROBOT_STATS = {
    # roleType: (攻击力, 攻击距离, 血量, 击杀积分)
    "smallRobot": (5, 3, 40, 1),
    "middleRobot": (10, 3, 60, 2),
    "largeRobot": (20, 3, 500, 4),
    "bossRobot": (40, 3, 800, 10),
}

ROBOT_SCORE = {name: stats[3] for name, stats in ROBOT_STATS.items()}
ROCKET_COOLDOWN = 3
ROBOT_RESPAWN_DELAY = 20          # 角色死亡后次日白天开始 20 回合复活
TASK_POINT_COOLDOWN = 30          # 任务点刷新回合数
LLM_FREE_PER_DAY = 3              # 每游戏日 LLM 调用上限（自进化任务期间不计数）
SANDBOX_CMD_TIMEOUT_SEC = 15

# 商店物品名（武器商店）
UPGRADE_ITEMS = (
    "WeaponUpgradeVoucher1", "WeaponUpgradeVoucher2",
    "WallUpgradeVoucher1", "WallUpgradeVoucher2",
    "StationUpgradeVoucher1", "StationUpgradeVoucher2",
)
CONSUMABLES = (
    "WallFixer", "Medicine", "DizzyWeapon", "Bomb",
    "SmallRobotSummonOrder", "MiddleRobotSummonOrder",
    "LargeRobotSummonOrder", "BossRobotSummonOrder",
)
# use 时必须带 targetPos 的物品
USE_NEEDS_TARGET = frozenset({
    "WallFixer", "DizzyWeapon", "Bomb",
    *UPGRADE_ITEMS,
})
TASK_ITEMS = (
    "AcientTablet", "StarSand", "FlameBreath",
    "FrostPotion", "ThornAmulet", "IronWhistle",
)

SUMMON_RESULT = {
    0: "未探测",
    1: "成功获取宝藏",
    2: "无宝藏或未到开启时间",
    3: "献祭物品错误",
    4: "宝藏已空",
}

NEUTRAL_MINES = ("stone", "iron", "copper")
NEUTRAL_SHOPS = ("vendor", "weaponShop")
NEUTRAL_TASK_POINTS = (
    "challengerTaskPoint1", "challengerTaskPoint2",
    "defenderTaskPoint1", "defenderTaskPoint2",
)

# ---------------------------------------------------------------- 基础结构


@dataclass(frozen=True, slots=True)
class Pos:
    x: int
    y: int

    @classmethod
    def load(cls, raw: Any) -> "Pos":
        return cls(int(raw["x"]), int(raw["y"]))

    def dump(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}

    def within(self, width: int, height: int) -> bool:
        return 0 <= self.x < width and 0 <= self.y < height

    def chebyshev_to(self, other: "Pos") -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def neighbours(self) -> tuple["Pos", ...]:
        return tuple(
            Pos(self.x + dx, self.y + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if dx or dy
        )


def distance(first: Pos, second: Pos) -> int:
    """切比雪夫距离。"""
    return first.chebyshev_to(second)


def station_footprint(pos: Pos) -> tuple[Pos, ...]:
    """基地 2x2，pos 为左上角（y 向上，故向下展开）。"""
    return (
        pos,
        Pos(pos.x + 1, pos.y),
        Pos(pos.x, pos.y - 1),
        Pos(pos.x + 1, pos.y - 1),
    )


# ---------------------------------------------------------------- 单位


@dataclass(frozen=True, slots=True)
class Unit:
    unit_id: int
    pos: Pos
    kind: str
    health: int
    level: int
    cooldown: int
    attack_power: int
    attack_range: int
    capacity: int | None
    backpack: tuple[str, ...]

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Unit":
        capacity = raw.get("backPackCapability")
        return cls(
            int(raw.get("id") or 0),
            Pos.load(raw["pos"]),
            str(raw["roleType"]),
            int(raw.get("health") or 0),
            int(raw.get("level") or 0),
            int(raw.get("cooldown") or 0),
            int(raw.get("attackPower") or 0),
            int(raw.get("attackRange") or 0),
            int(capacity) if capacity is not None else None,
            tuple(str(item) for item in raw.get("backpack") or ()),
        )

    @property
    def is_station(self) -> bool:
        return self.kind == "station"

    @property
    def is_weapon(self) -> bool:
        return self.kind in TOWER_TYPES

    @property
    def is_character(self) -> bool:
        return self.kind in CONTROLLABLE_TYPES

    @property
    def max_targets(self) -> int:
        """该武器单次攻击可指定的目标数 = 等级（电磁狙击炮恒为 1）。"""
        if self.kind == "railgun":
            return 1
        if self.kind in ("gatling", "rocket"):
            return max(self.level, 1)
        return 1

    def range_of_attack(self) -> int:
        if self.attack_range and self.attack_range < 10**9:
            return self.attack_range
        if self.attack_range >= 10**9:
            return 10**9
        if self.kind in WEAPON_STATS:
            level = min(max(self.level, 1), 3)
            return WEAPON_STATS[self.kind][level - 1][1]
        return 0


@dataclass(frozen=True, slots=True)
class Robot:
    robot_id: int
    pos: Pos
    kind: str
    health: int
    abnormal_state: str
    target_team: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "Robot":
        return cls(
            int(raw["id"]),
            Pos.load(raw["pos"]),
            str(raw.get("roleType") or "smallRobot"),
            int(raw.get("health") or 0),
            str(raw.get("abnormalState") or ""),
            str(raw.get("targetTeam") or ""),
        )

    @property
    def attack_power(self) -> int:
        return ROBOT_STATS.get(self.kind, (5, 3, 40, 1))[0]

    @property
    def attack_range(self) -> int:
        return ROBOT_STATS.get(self.kind, (5, 3, 40, 1))[1]

    @property
    def max_health(self) -> int:
        return ROBOT_STATS.get(self.kind, (5, 3, 40, 1))[2]

    @property
    def score(self) -> int:
        return ROBOT_STATS.get(self.kind, (5, 3, 40, 1))[3]

    @property
    def dizzy(self) -> bool:
        return self.abnormal_state == "dizzy"


@dataclass(frozen=True, slots=True)
class TaskPoint:
    task_type: str
    pos: Pos
    cooldown_rounds: int
    score_reward: int
    gold_reward: int
    is_valid: bool
    timeout_rounds: int

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "TaskPoint":
        return cls(
            str(raw.get("taskType") or ""),
            Pos.load(raw["taskPosition"]),
            int(raw.get("coldDownRounds") or 0),
            int(raw.get("scoreReward") or 0),
            int(raw.get("goldReward") or 0),
            bool(raw.get("isValid")),
            int(raw.get("timeoutRounds") or 0),
        )


@dataclass(frozen=True, slots=True)
class WorldNews:
    official: str
    folk: str

    @classmethod
    def load(cls, raw: dict[str, Any] | None) -> "WorldNews":
        raw = raw or {}
        return cls(str(raw.get("officialNews") or ""), str(raw.get("folkLegends") or ""))


@dataclass(frozen=True, slots=True)
class GameError:
    code: int
    description: str

    @classmethod
    def load(cls, raw: dict[str, Any]) -> "GameError":
        return cls(int(raw.get("errorCode") or 0), str(raw.get("description") or ""))


# ---------------------------------------------------------------- 回合快照


@dataclass(frozen=True, slots=True)
class Turn:
    round_no: int
    day: int                       # 1..10
    round_in_day: int              # 1..130
    is_day: bool
    width: int
    height: int
    team_type: str                 # challenger / defender
    gold: int
    total_score: int
    zones: dict[Pos, str]          # 中立元素（矿区/商店/任务点）
    ours: tuple[Unit, ...]
    enemies: tuple[Unit, ...]
    robots: tuple[Robot, ...]
    player_tasks: tuple[TaskPoint, ...]
    phase_task: str
    last_action_results: dict[int, bool]
    last_summon_result: int
    llm_resp: str
    world_news: WorldNews
    last_cmd_result: str
    vendor_prices: dict[str, int]
    weapon_shop: dict[str, int]
    errors: tuple[GameError, ...]

    # ---- 解析

    @classmethod
    def load(cls, payload: dict[str, Any]) -> "Turn":
        round_no = int(payload.get("roundNo") or 0)
        round_in_day = (round_no - 1) % ROUNDS_PER_DAY + 1
        info = payload.get("mapInfo") or {}
        team = payload.get("teamOur") or {}
        width = int(info.get("width") or MAP_WIDTH)
        height = int(info.get("height") or MAP_HEIGHT)
        return cls(
            round_no=round_no,
            day=(round_no - 1) // ROUNDS_PER_DAY + 1,
            round_in_day=round_in_day,
            is_day=round_in_day <= DAY_ROUNDS,
            width=width,
            height=height,
            team_type=str(team.get("type") or ""),
            gold=int(team.get("goldNum") or 0),
            total_score=int(team.get("totalScore") or 0),
            zones={
                Pos.load(zone["pos"]): str(zone["neutralType"])
                for zone in info.get("zones") or ()
            },
            ours=tuple(Unit.load(role) for role in team.get("roles") or ()),
            enemies=tuple(
                Unit.load(role)
                for role in (payload.get("teamEnemy") or {}).get("roles") or ()
            ),
            robots=tuple(
                Robot.load(robot)
                for robot in (payload.get("robot") or {}).get("roles") or ()
            ),
            player_tasks=tuple(
                TaskPoint.load(task) for task in team.get("playerTasks") or ()
            ),
            phase_task=str(payload.get("phaseTask") or ""),
            last_action_results={
                int(role_id): bool(ok)
                for role_id, ok in (payload.get("lastRoundRoleActionResults") or {}).items()
            },
            last_summon_result=int(payload.get("lastSummonTreasureResult") or 0),
            llm_resp=str(payload.get("llmResp") or ""),
            world_news=WorldNews.load(payload.get("worldNews")),
            last_cmd_result=str(payload.get("lastCmdResult") or ""),
            vendor_prices={
                str(item["name"]): int(item.get("price") or 0)
                for item in payload.get("vendorShopList") or ()
            },
            weapon_shop={
                str(item["name"]): int(item.get("price") or 0)
                for item in payload.get("weaponShopList") or ()
            },
            errors=tuple(
                GameError.load(err) for err in payload.get("errors") or ()
            ),
        )

    # ---- 我方单位查询

    def unit_by_id(self, unit_id: int) -> Unit | None:
        for unit in self.ours:
            if unit.unit_id == unit_id:
                return unit
        return None

    def alive(self, kinds: Iterable[str]) -> tuple[Unit, ...]:
        wanted = set(kinds)
        return tuple(u for u in self.ours if u.kind in wanted and u.health > 0)

    def station(self) -> Unit | None:
        for unit in self.ours:
            if unit.is_station:
                return unit
        return None

    def workers(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive(("worker",)), key=lambda u: u.unit_id))

    def pioneer(self) -> Unit | None:
        for unit in self.alive(("pioneer",)):
            return unit
        return None

    def weapons(self) -> tuple[Unit, ...]:
        return tuple(sorted(
            self.alive(TOWER_TYPES), key=lambda u: (u.pos.x, u.pos.y),
        ))

    def walls(self) -> tuple[Unit, ...]:
        return self.alive(("wall",))

    def controllables(self) -> tuple[Unit, ...]:
        return tuple(sorted(self.alive(CONTROLLABLE_TYPES), key=lambda u: u.unit_id))

    # ---- 地图查询

    def footprint(self, unit: Unit) -> tuple[Pos, ...]:
        if unit.is_station:
            return station_footprint(unit.pos)
        return (unit.pos,)

    def mines_of(self, ore: str) -> tuple[Pos, ...]:
        return tuple(pos for pos, kind in self.zones.items() if kind == ore)

    def mines(self) -> tuple[Pos, ...]:
        return tuple(
            pos for pos, kind in self.zones.items() if kind in NEUTRAL_MINES
        )

    def vendors(self) -> tuple[Pos, ...]:
        return tuple(pos for pos, kind in self.zones.items() if kind == "vendor")

    def weapon_shops(self) -> tuple[Pos, ...]:
        return tuple(pos for pos, kind in self.zones.items() if kind == "weaponShop")

    def our_task_points(self) -> tuple[Pos, ...]:
        wanted = (
            f"{self.team_type}TaskPoint1", f"{self.team_type}TaskPoint2",
        )
        return tuple(pos for pos, kind in self.zones.items() if kind in wanted)

    def on_map(self, pos: Pos) -> bool:
        return pos.within(self.width, self.height)

    def occupied_cells(self, ours_only: bool = True) -> frozenset[Pos]:
        cells: set[Pos] = set()
        for unit in self.ours:
            cells.update(self.footprint(unit))
        if not ours_only:
            for unit in self.enemies:
                cells.update(self.footprint(unit))
        return frozenset(cells)

    def blocked(self, moving: Unit | None = None) -> frozenset[Pos]:
        """对移动而言被阻挡的格子：中立元素、双方建筑/角色、机器人。"""
        cells = set(self.zones.keys())
        cells.update(self.occupied_cells(ours_only=False))
        if moving is not None:
            cells.discard(moving.pos)
            if moving.is_station:
                cells.difference_update(station_footprint(moving.pos))
        for robot in self.robots:
            cells.add(robot.pos)
        return frozenset(cells)

    # ---- 经济

    def ore_price(self, ore: str) -> int:
        return self.vendor_prices.get(ore, 0)


# ---------------------------------------------------------------- 指令构造器
# 只负责拼结构，字段合法性由 validator 统一把关。


def _pos_list(pos: Pos) -> list[dict[str, int]]:
    return [pos.dump()]


def cmd_move(pos: Pos) -> dict[str, Any]:
    return {"action": "move", "targetPos": _pos_list(pos)}


def cmd_attack(controller_id: int, targets: list[Pos] | Pos) -> dict[str, Any]:
    if isinstance(targets, Pos):
        targets = [targets]
    return {
        "action": "attack",
        "controllerId": str(controller_id),
        "targetPos": [pos.dump() for pos in targets],
    }


def cmd_sell(name: str, num: int = 1) -> dict[str, Any]:
    return {"action": "sell", "name": name, "num": int(num)}


def cmd_buy(name: str, num: int = 1) -> dict[str, Any]:
    return {"action": "buy", "name": name, "num": int(num)}


def cmd_build(name: str, pos: Pos) -> dict[str, Any]:
    return {"action": "build", "name": name, "targetPos": _pos_list(pos)}


def cmd_remove(pos: Pos) -> dict[str, Any]:
    return {"action": "remove", "targetPos": _pos_list(pos)}


def cmd_accept_task() -> dict[str, Any]:
    return {"action": "acceptTask"}


def cmd_submit_answer(answer: str) -> dict[str, Any]:
    return {"action": "submitAnswer", "taskAnswer": str(answer)}


def cmd_summon_treasure(pos: Pos, items: list[str]) -> dict[str, Any]:
    return {
        "action": "summonTreasure",
        "targetPos": _pos_list(pos),
        "item": list(items),
    }


def cmd_use(name: str, pos: Pos | None = None) -> dict[str, Any]:
    command: dict[str, Any] = {"action": "use", "name": name}
    if pos is not None:
        command["targetPos"] = _pos_list(pos)
    return command


def cmd_drop(name: str) -> dict[str, Any]:
    return {"action": "drop", "name": name}


def cmd_collect(pos: Pos) -> dict[str, Any]:
    return {"action": "collect", "targetPos": _pos_list(pos)}


@dataclass(slots=True)
class Decision:
    """一回合的完整出站响应。"""
    commands: dict[int, dict[str, Any]] = field(default_factory=dict)
    prompt: str = ""
    execute_cmd: str = ""

    def dump(self) -> dict[str, Any]:
        return {
            "roleCommandMap": {
                str(unit_id): command
                for unit_id, command in self.commands.items()
            },
            "prompt": self.prompt,
            "executeCmd": self.execute_cmd,
        }
