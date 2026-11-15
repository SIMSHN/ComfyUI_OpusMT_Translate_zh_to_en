# -*- coding: utf-8 -*-
"""Install this folder as a ComfyUI custom node package.

Usage
-----
    python install_comfyui.py "D:\\ComfyUI_windows_portable\\ComfyUI"
    python install_comfyui.py --check
    python install_comfyui.py "D:\\ComfyUI" --copy

By default a directory junction / symlink is created, so the 300 MB model
weights stay where they are and are NOT duplicated. Use --copy if you prefer
a standalone copy (only the PyTorch weights are copied; the TensorFlow and
Rust weights are skipped).

This script never deletes anything. If the target name already exists it
stops and asks you to handle it manually.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "ComfyUI_OpusMT_Translate"

NODE_FILES = (
    "__init__.py",
    "opus_mt_nodes.py",
    "requirements.txt",
    "install_comfyui.py",
    "convert_to_safetensors.py",
    "README.md",
    "COMFYUI_README.md",
)
MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "source.spm",
    "target.spm",
    "pytorch_model.bin",
    "metadata.json",
)


def check_files() -> bool:
    print("== 检查本地模型/节点文件 ==")
    ok = True
    for name in ("config.json", "source.spm", "target.spm", "vocab.json", "pytorch_model.bin"):
        path = HERE / name
        exists = path.is_file()
        size = f"{path.stat().st_size / 1024 / 1024:.1f} MB" if exists else "-"
        print(f"  {'OK ' if exists else 'MISS'} {name:<24} {size}")
        ok &= exists
    for name in NODE_FILES:
        if not (HERE / name).is_file():
            print(f"  MISS {name}")
            ok &= False

    print("== 检查 Python 依赖（当前解释器）==")
    for module in ("torch", "transformers", "sentencepiece"):
        try:
            __import__(module)
            print(f"  OK  {module}")
        except Exception as exc:
            print(f"  MISS {module}: {exc}")
    print()
    return bool(ok)


def find_comfyui_root() -> Path | None:
    env = os.environ.get("COMFYUI_PATH", "").strip()
    if env and (Path(env) / "custom_nodes").is_dir():
        return Path(env).resolve()

    candidates = [
        HERE / "ComfyUI",
        HERE.parent / "ComfyUI",
        Path("C:/ComfyUI"),
        Path("D:/ComfyUI"),
        Path("E:/ComfyUI"),
        Path("C:/ComfyUI_windows_portable/ComfyUI"),
        Path("D:/ComfyUI_windows_portable/ComfyUI"),
        Path("E:/ComfyUI_windows_portable/ComfyUI"),
    ]
    for path in candidates:
        if (path / "custom_nodes").is_dir():
            return path.resolve()
    return None


def portable_python_hint(root: Path) -> str:
    embedded = root.parent / "python_embeded" / "python.exe"
    if embedded.is_file():
        return str(embedded)
    return sys.executable


def link_into(root: Path) -> bool:
    target = root / "custom_nodes" / PACKAGE_NAME
    if target.exists() or target.is_symlink():
        print(f"[跳过] 目标已存在: {target}")
        print("       如果它就是本节点包，无需重复安装；否则请手动改名或删除后再运行。")
        return True

    print(f"创建目录链接: {target}  ->  {HERE}")
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(HERE)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("mklink 失败:", (result.stdout + result.stderr).strip())
            print("可改用 --copy 参数以复制方式安装。")
            return False
    else:
        try:
            os.symlink(str(HERE), str(target), target_is_directory=True)
        except OSError as exc:
            print("symlink 失败:", exc)
            print("可改用 --copy 参数以复制方式安装。")
            return False
    return True


def copy_into(root: Path) -> bool:
    target = root / "custom_nodes" / PACKAGE_NAME
    if target.exists():
        print(f"[跳过] 目标已存在: {target}")
        print("       请手动改名或删除后再运行，本脚本不会删除任何已有内容。")
        return True
    target.mkdir(parents=True)
    print(f"复制到: {target}")

    for name in NODE_FILES + MODEL_FILES:
        src = HERE / name
        if not src.is_file():
            print(f"  [跳过] 缺少 {name}")
            continue
        size_mb = src.stat().st_size / 1024 / 1024
        print(f"  {name} ({size_mb:.1f} MB) ...", flush=True)
        shutil.copy2(src, target / name)
    print("已跳过 tf_model.h5 / rust_model.ot（PyTorch 推理不需要）。")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="把本目录安装为 ComfyUI 自定义节点")
    parser.add_argument("comfyui_root", nargs="?", help="ComfyUI 根目录（含 custom_nodes 文件夹）")
    parser.add_argument("--copy", action="store_true", help="复制文件而不是创建目录链接")
    parser.add_argument("--check", action="store_true", help="只检查文件与依赖，不安装")
    args = parser.parse_args()

    files_ok = check_files()
    if args.check:
        return 0 if files_ok else 1
    if not files_ok:
        print("模型文件不完整，安装中止。请确认下载完整后再试。")
        return 1

    root = Path(args.comfyui_root).expanduser().resolve() if args.comfyui_root else find_comfyui_root()
    if root is None or not (root / "custom_nodes").is_dir():
        print("找不到 ComfyUI 根目录。请把路径作为参数传入，例如：")
        print('  python install_comfyui.py "D:\\ComfyUI_windows_portable\\ComfyUI"')
        return 1

    print(f"ComfyUI 根目录: {root}")
    ok = copy_into(root) if args.copy else link_into(root)
    if not ok:
        return 1

    py = portable_python_hint(root)
    print("\n== 下一步 ==")
    print(f'1) 安装依赖:  "{py}" -m pip install -r "{HERE / "requirements.txt"}"')
    print("   （整合包用户请用 python_embeded 里的 python.exe，不要用系统 Python）")
    print("2) 重启 / 刷新 ComfyUI，在节点菜单 text > translation 下找到三个节点。")
    print("3) 首次执行会加载约 300 MB 权重，之后常驻内存复用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
