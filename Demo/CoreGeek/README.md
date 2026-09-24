# 《未来战争》参赛程序

基于官方 Demo（CoreGeek）框架层重建的参赛 Bot，Python 3.11+ 纯标准库实现。

## 启动

```bash
bash run.sh <port>     # 判题器标准启动方式
```

## 架构

```
src/agent/
├── server.py       # HTTP 层：3.5s 决策看门狗，异常兜底返回空指令
├── brain.py        # 总调度：昼夜编排、开拓者优先级、LLM prompt 槽仲裁
├── protocol.py     # request 全字段解析 + 游戏常量表 + 指令构造器
├── validator.py    # 出站指令硬校验（防"异常响应"红线的最后防线）
├── memory.py       # 跨回合状态：LLM 预算/新闻归档/价格史/SOP 库/失败记录
├── grid.py         # A* 寻路（8 向、切比雪夫）
├── build.py        # 炮台位锚定 + 围墙防线规划
├── economy.py      # 采矿分工/贩卖/购物优先级/升级券配送
├── combat.py       # 夜战：加特林锥形多目标/电磁炮穿透/火箭溅射/角色走位
├── brain_baseline.py  # Demo 原版策略（A/B 回归基线，不参与正式决策）
└── tasks/
    ├── evolve.py   # 自进化任务状态机（LLM+沙盒异步迭代，SOP 沉淀复用）
    ├── news.py     # 官方消息 → 矿价/停工预测（每日 3 次 LLM 预算）
    └── treasure.py # 民间传闻 → 宝藏三元组推理与召唤
```

### 关键设计

- **三层防出局**：validator 硬校验（字段缺失/动作非法直接丢弃）→ server 看门狗
  （决策 3.5s 未完成降级空响应）→ 异常兜底。指令"执行失败"（碰撞等）不算异常，
  照常发送不拦截。
- **LLM/沙盒异步模型**：本回合发 `prompt`/`executeCmd`，下回合收
  `llmResp`/`lastCmdResult`；任务模块按状态机消化反馈推进。
- **LLM 预算**：每日 3 次（news 1 次 + treasure 按需），自进化任务期间免费不限次。
- **SOP 沉淀**：任务文本签名 → 答案模板，同题复用压缩回合数（速度积分加成）。

## 测试与回归

```bash
python -m pytest tests/ -q              # 协议解析 + 校验器单测
python tools/mock_judger.py --days 10 --swap   # 本地模拟判题器：vs Demo 基线 A/B
```

模拟器对局结果（10 天、双方换边，seed 42）：
Brain 总分 8638/8664 vs Baseline 60/68，异常响应 0，
单回合决策平均 1.7ms / 最差 26ms（判题器限额 5s）。

## 已知边界

- 可建造区域（蓝/黄）不在协议中，炮台按"基地邻格"、围墙按"外圈 2-3 格"推断，
  建造失败会记入 memory.build_failures 并跳过（build 失败不计异常，可安全试错）。
- 加特林/电磁炮弹道只被机器人吸收（围墙不挡），布局按此设计。
