"""HTTP 层：收 request JSON -> Brain 决策 -> 回 roleCommandMap/prompt/executeCmd。

安全设计（对应任务书"5 次异常响应出局"红线）：
1. 决策放在工作线程里跑，主线程 join(DECIDE_TIMEOUT)；超时返回空指令（合法降级）。
2. 决策抛异常同样降级为空指令，由日志排查。
3. Brain 内部有全局锁，串行化多线程请求。
"""
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

DECIDE_TIMEOUT = 3.5  # 秒；判题器 5 秒超时，留足网络余量

LOGGER = logging.getLogger(__name__)

_BRAIN = None
_BRAIN_LOCK = threading.Lock()


def get_brain():
    global _BRAIN
    if _BRAIN is None:
        from agent.brain import Brain
        _BRAIN = Brain()
    return _BRAIN


def decide_safe(payload: dict[str, Any]) -> dict[str, Any]:
    """带超时与异常兜底的决策入口，永远返回合法响应结构。"""
    try:
        brain = get_brain()
    except Exception:
        LOGGER.exception("brain init failed")
        return {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}

    result: dict[str, Any] = {}

    def run() -> None:
        try:
            with _BRAIN_LOCK:
                result.update(brain.decide(payload))
        except Exception:
            LOGGER.exception("decide failed for round %s", payload.get("roundNo"))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(DECIDE_TIMEOUT)
    if worker.is_alive():
        LOGGER.error("decide timeout on round %s, fallback empty", payload.get("roundNo"))
        return {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}
    if not result:
        # 决策线程异常退出，同样降级
        return {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}
    return result


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            response = decide_safe(payload)
            LOGGER.info("round %s -> %s", payload.get("roundNo"), response)
        except Exception:
            LOGGER.exception("request handling failed")
            response = {"roleCommandMap": {}, "prompt": "", "executeCmd": ""}
        body = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(port: int) -> None:
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
