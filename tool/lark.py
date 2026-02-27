import json
from datetime import datetime
from urllib import request


WEBHOOK_URL = "https://www.feishu.cn/flow/api/trigger-webhook/0dab5484c669f8dbc898a052f19efad3"
LARK_LOG_WEBHOOK_URL = "https://www.feishu.cn/flow/api/trigger-webhook/c09f9f2d91646c670e5d1a6b46b81567"


def generate_run_id() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def _post_json(url: str, payload: dict) -> str:
    data = json.dumps(payload).encode("utf-8")

    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except Exception:
        return ""


def lark_notify(title: str, msg: str) -> str:
    payload = {"title": title, "msg": msg}
    return _post_json(WEBHOOK_URL, payload)


def lark_log(project: str, file: str, run_id: str, log: dict) -> str:
    payload = {"project": project, "file": file, "run_id": run_id, "log": log}
    return _post_json(LARK_LOG_WEBHOOK_URL, payload)
