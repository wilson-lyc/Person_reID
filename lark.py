import requests
import threading
import time


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


class TrainingMonitor:
    """训练监控类，用于检测训练是否卡住或异常中断"""

    def __init__(self, run_id: str, timeout_minutes: int = 15):
        """
        初始化监控器

        Args:
            run_id: 运行ID
            timeout_minutes: 超时时间（分钟），超过此时间无进度则告警
        """
        self.run_id = run_id
        self.timeout_seconds = timeout_minutes * 60
        self.last_update_time = time.time()
        self.is_running = True
        self.alert_sent = False
        self._thread = None

    def start(self):
        """启动监控线程"""
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def update(self):
        """更新最后活跃时间"""
        self.last_update_time = time.time()
        self.alert_sent = False

    def stop(self):
        """停止监控"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor(self):
        """监控线程主函数"""
        while self.is_running:
            time.sleep(60)  # 每分钟检查一次
            if not self.is_running:
                break

            elapsed = time.time() - self.last_update_time
            if elapsed > self.timeout_seconds and not self.alert_sent:
                # 发送超时告警
                send_message(
                    "训练告警",
                    f"Run ID: {self.run_id}\n训练已无响应超过 {int(elapsed // 60)} 分钟，请检查是否卡住！"
                )
                self.alert_sent = True


# 全局监控器实例
_monitor = None


def start_monitoring(run_id: str, timeout_minutes: int = 15):
    """启动训练监控"""
    global _monitor
    _monitor = TrainingMonitor(run_id, timeout_minutes)
    _monitor.start()


def update_monitoring():
    """更新监控状态"""
    global _monitor
    if _monitor:
        _monitor.update()


def stop_monitoring():
    """停止训练监控"""
    global _monitor
    if _monitor:
        _monitor.stop()


def send_error_notification(run_id: str, error_msg: str):
    """发送训练错误通知"""
    send_message("训练异常", f"Run ID: {run_id}\n错误信息: {error_msg}")
