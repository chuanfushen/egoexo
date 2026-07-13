#!/usr/bin/env python3
"""
列出 TOS 上指定前缀的所有文件，支持并发加速。
Usage:
    python list_tos_files.py tos://bucket/prefix --suffix .pt -o output.txt
"""
import argparse
import concurrent.futures
import os
import sys
from typing import List, Optional
from urllib.parse import urlparse

import tos
from tqdm import tqdm


def parse_tos_url(url: str) -> tuple:
    """Parse tos://bucket/prefix URL."""
    if url.startswith("tos://"):
        url = url[6:]  # Remove tos:// prefix
    parts = url.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def create_tos_client() -> tos.TosClientV2:
    """Create TOS client from environment variables."""
    ak = os.environ.get("VOLC_ACCESSKEY", "")
    sk = os.environ.get("VOLC_SECRETKEY", "")
    endpoint = os.environ.get("VOLC_ENDPOINT", "tos-cn-beijing.ivolces.com")
    region = os.environ.get("VOLC_REGION", "cn-beijing")

    if not ak or not sk:
        raise ValueError(
            "请设置环境变量 VOLC_ACCESSKEY 和 VOLC_SECRETKEY\n"
            "export VOLC_ACCESSKEY=your_access_key\n"
            "export VOLC_SECRETKEY=your_secret_key"
        )

    return tos.TosClientV2(ak, sk, endpoint, region)


def list_objects_page(
    client: tos.TosClientV2,
    bucket: str,
    prefix: str,
    marker: Optional[str] = None,
    max_keys: int = 1000,
) -> tuple:
    """List a single page of objects."""
    resp = client.list_objects(
        bucket=bucket,
        prefix=prefix,
        max_keys=max_keys,
        marker=marker,
    )
    return resp.contents, resp.next_marker


def list_all_objects(
    client: tos.TosClientV2,
    bucket: str,
    prefix: str,
    suffix: Optional[str] = None,
    max_workers: int = 32,
) -> List[str]:
    """
    List all objects with given prefix using concurrent pagination.
    Returns list of object keys.
    """
    all_keys = []

    # First, get initial page to determine total count
    print(f"正在获取文件列表: tos://{bucket}/{prefix}")

    contents, next_marker = list_objects_page(client, bucket, prefix)

    # Filter keys
    for obj in contents:
        if suffix is None or obj.key.endswith(suffix):
            all_keys.append(obj.key)

    # If there's only one page, return early
    if not next_marker:
        return all_keys

    # Collect all markers for concurrent fetching
    markers = []
    current_marker = next_marker

    # We need to sequentially get all markers first
    with tqdm(desc="扫描页面", unit="page") as pbar:
        while current_marker:
            markers.append(current_marker)
            contents, current_marker = list_objects_page(
                client, bucket, prefix, marker=current_marker
            )
            pbar.update(1)

    print(f"发现 {len(markers)} 个分页，开始并发获取...")

    # Concurrently fetch all pages
    def fetch_page(marker):
        try:
            contents, _ = list_objects_page(client, bucket, prefix, marker=marker)
            return [obj.key for obj in contents if suffix is None or obj.key.endswith(suffix)]
        except Exception as e:
            print(f"获取页面失败 (marker={marker[:50]}...): {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_page, marker): marker for marker in markers}

        with tqdm(total=len(markers), desc="下载页面", unit="page") as pbar:
            for future in concurrent.futures.as_completed(futures):
                keys = future.result()
                all_keys.extend(keys)
                pbar.update(1)

    return all_keys


def main():
    parser = argparse.ArgumentParser(description="列出 TOS 文件")
    parser.add_argument("url", help="TOS URL (e.g., tos://bucket/prefix)")
    parser.add_argument("-o", "--output", required=True, help="输出文件路径")
    parser.add_argument("--suffix", default=".pt", help="文件后缀过滤 (默认: .pt)")
    parser.add_argument("--workers", type=int, default=32, help="并发数 (默认: 32)")
    parser.add_argument("--add-prefix", action="store_true", help="在输出中添加 tos:// 前缀")

    args = parser.parse_args()

    # Parse URL
    bucket, prefix = parse_tos_url(args.url)
    print(f"Bucket: {bucket}")
    print(f"Prefix: {prefix}")
    print(f"Suffix filter: {args.suffix if args.suffix else 'None'}")

    # Create client
    client = create_tos_client()

    # List all objects
    keys = list_all_objects(
        client,
        bucket,
        prefix,
        suffix=args.suffix,
        max_workers=args.workers,
    )

    # Write to file
    prefix_str = f"tos://{bucket}/" if args.add_prefix else ""
    with open(args.output, "w") as f:
        for key in sorted(keys):
            f.write(f"{prefix_str}{key}\n")

    print(f"\n完成！共找到 {len(keys)} 个文件")
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()