"""兼容入口：qihuang 高级爬虫已合并到 crawl_qihuang.py。

旧参数 ``--step 1`` 和 ``--step 2`` 分别映射到方剂、药材。
古籍模式 ``--step 3`` 已永久停用。
"""

import argparse
import json

from crawl_qihuang import crawl_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if args.step == 3:
        raise SystemExit("qihuang 古籍抓取已停用，请使用 jicheng/gushiwen 原始层爬虫")

    kind = "formulas" if args.step == 1 else "herbs"
    print(json.dumps(crawl_dataset(kind), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
