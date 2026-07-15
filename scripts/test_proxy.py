"""测试可选动态代理，不输出代理账号、密码或 API 响应正文。"""

import os

import requests


PROXY_API_URL = os.environ.get("JULIANG_PROXY_API_URL", "").strip()


def main() -> None:
    if not PROXY_API_URL:
        raise SystemExit("未配置环境变量 JULIANG_PROXY_API_URL")

    response = requests.get(PROXY_API_URL, timeout=15)
    response.raise_for_status()
    first_line = response.text.strip().splitlines()[0]
    parts = first_line.split(":")
    if len(parts) < 2:
        raise SystemExit("代理 API 返回格式异常")

    host, port = parts[0].strip(), parts[1].strip()
    print(f"代理 API 状态: HTTP {response.status_code}")
    print(f"代理端点: {host}:{port}")
    print(f"认证模式: {'账号密码' if len(parts) >= 4 else 'IP白名单'}")
    print("安全提示: 未输出代理用户名、密码或 API URL")


if __name__ == "__main__":
    main()
