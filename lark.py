import requests


def send_message(title: str, msg: str) -> None:
    """
    发送飞书消息

    Args:
        title: 消息标题
        msg: 消息内容
    """
    url = "https://www.feishu.cn/flow/api/trigger-webhook/0dab5484c669f8dbc898a052f19efad3"

    try:
        requests.post(url, json={"title": title, "msg": msg}, timeout=10)
    except Exception:
        pass


def log_info(file: str, run_id: str, log: dict) -> None:
    """
    记录日志信息到飞书

    Args:
        file: 运行的文件名
        run_id: 运行ID
        log: JSON格式的日志信息
    """
    url = "https://www.feishu.cn/flow/api/trigger-webhook/c09f9f2d91646c670e5d1a6b46b81567"

    try:
        requests.post(url, json={"project": "Person_reID", "file": file, "run_id": run_id, "log": log}, timeout=10)
    except Exception:
        pass
