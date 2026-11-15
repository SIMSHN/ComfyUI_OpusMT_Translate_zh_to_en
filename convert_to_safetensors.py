# -*- coding: utf-8 -*-
"""Optional: convert the legacy ``pytorch_model.bin`` to ``model.safetensors``.

Why: this OPUS-MT repo was exported in 2020 with the legacy (non-zip) torch
pickle format. Recent transformers/torch versions may refuse to load .bin
weights or only support safetensors. Converting once makes loading faster and
future-proof. transformers automatically prefers model.safetensors when present.

Run with the Python that ComfyUI uses (torch is already installed there):

    python convert_to_safetensors.py

Weights that share storage (Marian ties encoder/decoder embeddings) are cloned
so safetensors' no-shared-memory rule is satisfied; transformers re-ties them
on load. The original pytorch_model.bin is left untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "pytorch_model.bin"


def main() -> int:
    parser = argparse.ArgumentParser(description="把 pytorch_model.bin 转成 model.safetensors")
    parser.add_argument("--force", action="store_true", help="已存在时也重新生成")
    parser.add_argument(
        "--output",
        default=str(HERE / "model.safetensors"),
        help="输出路径（默认与本脚本同目录的 model.safetensors）",
    )
    args = parser.parse_args()

    dst = Path(args.output).expanduser().resolve()
    if dst.is_file() and not args.force:
        print(f"已存在 {dst}，无需转换（如需重建请加 --force）。")
        return 0
    if not SRC.is_file():
        print(f"找不到 {SRC}")
        return 1

    try:
        import torch
        from safetensors.torch import save_file, load_file
    except ImportError as exc:
        print(f"缺少依赖: {exc}\n请执行: python -m pip install safetensors")
        return 1

    print(f"读取 {SRC} ({SRC.stat().st_size / 1024 / 1024:.1f} MB) ...", flush=True)
    try:
        state = torch.load(str(SRC), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only argument
        state = torch.load(str(SRC), map_location="cpu")
    except Exception as exc:
        print(f"读取失败: {exc}")
        print("若提示 weights_only 相关错误，可尝试把 transformers/torch 降到较旧版本，")
        print("或直接从 Hugging Face 重新下载该模型的 safetensors 版本。")
        return 1

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Marian ties encoder/decoder embeddings to `model.shared.weight`. Those
    # keys point at the SAME storage, which safetensors forbids. Keeping all of
    # them would blow the file up from ~300 MB to ~550 MB, so keep one copy
    # (transformers re-ties the weights on load via tie_weights()).
    tensors = {}
    seen_storage = {}
    merged = []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        try:
            storage = value.untyped_storage() if hasattr(value, "untyped_storage") else value.storage()
            signature = (
                storage.data_ptr(),
                tuple(value.shape),
                str(value.dtype),
                value.storage_offset(),
            )
        except Exception:
            signature = None

        if signature is not None and signature in seen_storage:
            previous = seen_storage[signature]
            if "shared" in key and "shared" not in previous:
                tensors.pop(previous, None)
                tensors[key] = value.detach().cpu().contiguous().clone()
                seen_storage[signature] = key
            else:
                merged.append(key)
            continue

        tensors[key] = value.detach().cpu().contiguous().clone()
        if signature is not None:
            seen_storage[signature] = key

    if merged:
        print(f"合并 {len(merged)} 个共享存储的张量，例如: {merged[:3]}")

    print(f"共 {len(tensors)} 个张量，写入 {dst} ...", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst), metadata={"format": "pt", "source": "opus-mt-zh-en"})
    del tensors, state

    check = load_file(str(dst))
    print(f"校验通过：{len(check)} 个张量，{dst.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"完成。把 {dst.name} 放进模型目录并重启 ComfyUI，节点会优先使用它；"
          "原始 pytorch_model.bin 保留作为后备。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
