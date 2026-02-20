import requests


def test_send_message():
    """
    测试飞书消息发送功能
    """
    url = "https://www.feishu.cn/flow/api/trigger-webhook/0dab5484c669f8dbc898a052f19efad3"

    try:
        response = requests.post(url, json={"title": "测试", "msg": "这是一条测试消息"}, timeout=10)
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

        if response.status_code == 200:
            print("✓ 飞书消息发送成功")
            return True
        else:
            print("✗ 飞书消息发送失败")
            return False
    except requests.exceptions.Timeout:
        print("✗ 请求超时")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求异常: {e}")
        return False


if __name__ == "__main__":
    test_send_message()
