# -*- coding: utf-8 -*-
"""
ComfyUI custom nodes for offline OPUS-MT (MarianMT) text translation.

Default model shipped next to this file: Helsinki-NLP/opus-mt-zh-en (Chinese -> English).
The loader also accepts any other OPUS-MT / MarianMT model directory, so you can
swap in e.g. opus-mt-en-zh for reverse translation.

Design notes
------------
* No network access: everything is loaded with ``local_files_only=True``.
* Dependencies are imported lazily/defensively so a missing package never breaks
  ComfyUI startup (the node still registers and shows a helpful error when run).
* The model is cached in memory, keyed by (model path, device).
* Long input is split on line breaks / punctuation to respect Marian's 512 tokens.
"""

from __future__ import annotations

import gc
import json
import os
import re
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

    _IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - depends on the ComfyUI environment
    torch = None
    AutoTokenizer = AutoModelForSeq2SeqLM = GenerationConfig = None
    _IMPORT_ERROR = _exc

NODE_DIR = Path(__file__).resolve().parent
MODEL_ENV_VAR = "OPUS_MT_MODEL_PATH"
DEFAULT_MODEL_DIRNAME = "opus-mt-zh-en"

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
_PIP_HINT = (
    'python -m pip install "transformers>=4.30" sentencepiece protobuf\n'
    "（torch 已随 ComfyUI 提供，请不要重装，以免覆盖 CUDA 版本）"
)

# ---------------------------------------------------------------------------
# Model directory discovery
# ---------------------------------------------------------------------------


def _model_dir_is_valid(path: Path) -> bool:
    """A usable MarianMT repo needs a config, a tokenizer and at least one weight file."""
    try:
        if not (path / "config.json").is_file():
            return False
        has_tokenizer = any(
            (path / name).is_file() for name in ("source.spm", "tokenizer.json", "spiece.model")
        )
        has_weights = any((path / name).is_file() for name in _WEIGHT_FILES)
        return has_tokenizer and has_weights
    except OSError:
        return False


def _candidate_model_dirs():
    """Ordered list of places where the model may live."""
    candidates = []

    env_value = os.environ.get(MODEL_ENV_VAR, "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())

    # Most important case: this node package IS the model repository (files sit
    # next to __init__.py), which is the layout used by this download.
    candidates += [
        NODE_DIR,
        NODE_DIR / "models" / DEFAULT_MODEL_DIRNAME,
        NODE_DIR / DEFAULT_MODEL_DIRNAME,
    ]

    # Standard ComfyUI models folder, when the package was installed as a link
    # or the model was moved out.
    try:
        import folder_paths  # provided by ComfyUI

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            base = Path(models_dir)
            candidates += [
                base / "translation" / DEFAULT_MODEL_DIRNAME,
                base / "text_translation" / DEFAULT_MODEL_DIRNAME,
                base / DEFAULT_MODEL_DIRNAME,
                base / "translation",
            ]
        try:
            for folder in folder_paths.get_folder_paths("translation"):
                candidates += [Path(folder) / DEFAULT_MODEL_DIRNAME, Path(folder)]
        except Exception:
            pass
    except Exception:
        pass

    return candidates


def find_model_dir(explicit_path: str = "") -> Path:
    """Resolve and validate the model directory."""
    explicit_path = (explicit_path or "").strip().strip('"').strip("'")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(
                f"[OPUS-MT] 指定的模型目录不存在: {path}\n"
                "请填写包含 config.json / source.spm / target.spm / vocab.json / "
                "pytorch_model.bin 的文件夹绝对路径。"
            )
        if not _model_dir_is_valid(path):
            raise FileNotFoundError(f"[OPUS-MT] 目录 {path} 里没有找到完整的模型文件。")
        return path.resolve()

    candidates = _candidate_model_dirs()
    for path in candidates:
        if _model_dir_is_valid(path):
            return path.resolve()

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        "[OPUS-MT] 找不到模型目录。已搜索:\n"
        f"{searched}\n"
        "解决办法（任选其一）：\n"
        "  1) 在节点的 model_path 里填写模型文件夹的绝对路径；\n"
        f"  2) 设置环境变量 {MODEL_ENV_VAR} 指向模型文件夹；\n"
        '  3) 把模型文件放到 ComfyUI/models/translation/opus-mt-zh-en/ 。\n'
        "模型文件应保持与本节点在一起（本下载包即如此），不要只复制单个 .bin 文件。"
    )


def _require_deps():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            f"[OPUS-MT] 翻译依赖未安装或不兼容: {_IMPORT_ERROR}\n"
            f"请在 ComfyUI 使用的 Python 环境中执行:\n  {_PIP_HINT}"
        )


def _resolve_device(device: str) -> str:
    device = (device or "cpu").lower().strip()
    if device == "auto":
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    if device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        print("[OPUS-MT] CUDA 不可用，已回退到 CPU。")
        return "cpu"
    return device


def _describe_direction(path: Path) -> str:
    try:
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return f"{meta.get('src_name', '?')} -> {meta.get('tgt_name', '?')}"
    except Exception:
        return "以模型权重为准"


# ---------------------------------------------------------------------------
# Model wrapper + cache
# ---------------------------------------------------------------------------


class TranslationModel:
    """Lightweight handle passed between the loader node and the translate node."""

    __slots__ = ("path", "device", "tokenizer", "model")

    def __init__(self, path, device, tokenizer, model):
        self.path = path
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

    def __repr__(self):
        return f"<OPUS-MT {self.path.name} on {self.device}>"


_CACHE = {}
_CACHE_LOCK = threading.RLock()


def load_translation_model(model_path: str = "", device: str = "cpu") -> TranslationModel:
    """Load (or reuse) a MarianMT model. Thread-safe, offline only."""
    _require_deps()
    path = find_model_dir(model_path)
    dev = _resolve_device(device)
    key = (str(path), dev)

    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        print(f"[OPUS-MT] 首次加载模型 ({_describe_direction(path)}): {path} -> {dev}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"[OPUS-MT] 模型加载失败: {exc}\n"
                "常见原因：transformers 版本过新/过旧、缺少 sentencepiece 或 protobuf、"
                "权重文件损坏。\n"
                f"可尝试：{_PIP_HINT}\n"
                "或把 transformers 固定在 4.4x 后重启 ComfyUI。"
            ) from exc

        model.to(dev)
        model.eval()
        bundle = TranslationModel(path, dev, tokenizer, model)
        _CACHE[key] = bundle
        print(f"[OPUS-MT] 模型就绪，常驻内存直到 ComfyUI 重启或卸载。{bundle!r}")
        return bundle


def unload_models():
    """Drop cached models and release (CPU / GPU) memory."""
    with _CACHE_LOCK:
        _CACHE.clear()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print("[OPUS-MT] 已卸载翻译模型并释放显存。")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# Zero-width split: cut right after punctuation but keep every following
# whitespace character, so re-joining the translated chunks is lossless.
_SPLIT_RE = re.compile(r"(?<=[。！？!?；;，,、：:])(?=\s*\S)")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _chunk_line(line: str, max_chars: int):
    """Split one line into chunks that comfortably fit Marian's 512-token window."""
    if len(line) <= max_chars:
        return [line]

    pieces = [p for p in _SPLIT_RE.split(line) if p]
    chunks, buf = [], ""
    for piece in pieces:
        while len(piece) > max_chars:  # punctuation-free wall of text
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(piece[:max_chars])
            piece = piece[max_chars:]
        if buf and len(buf) + len(piece) > max_chars:
            chunks.append(buf)
            buf = ""
        buf += piece
    if buf:
        chunks.append(buf)
    return chunks


def _split_text(text: str, max_chars: int):
    """Return (flat chunk list, chunk count per original line)."""
    max_chars = max(20, int(max_chars))
    chunks, per_line = [], []
    for line in (text or "").split("\n"):
        line_chunks = _chunk_line(line, max_chars) if line.strip() else [""]
        per_line.append(len(line_chunks))
        chunks.extend(line_chunks)
    return chunks, per_line


def _join_text(translated, per_line) -> str:
    lines, cursor = [], 0
    for count in per_line:
        lines.append("".join(translated[cursor : cursor + count]).strip())
        cursor += count
    return "\n".join(lines)


def _generation_config(bundle: TranslationModel, num_beams: int):
    """Base generation config (length is set per chunk inside the loop).

    OPUS-MT ships ``max_length=512`` in generation_config.json. Combining that
    with ``max_new_tokens`` triggers warnings on newer transformers, because
    generate() back-fills the model's max_length into any config where it is
    None. We therefore drive length with an explicit ``max_length`` instead.
    """
    try:
        raw = bundle.model.generation_config.to_dict()
        raw.pop("_from_model_config", None)
        cfg = GenerationConfig(**raw)
        cfg.num_beams = num_beams
        cfg.do_sample = False
        cfg.max_new_tokens = None
        return cfg
    except Exception:
        return None


def _translate_chunks(bundle: TranslationModel, chunks, num_beams: int, max_new_tokens: int):
    import copy

    results = []
    base_cfg = _generation_config(bundle, num_beams)
    max_positions = getattr(bundle.model.config, "max_position_embeddings", 512) or 512

    for chunk in chunks:
        if not chunk.strip():
            results.append("")
            continue
        encoded = bundle.tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=max_positions
        )
        encoded = {k: v.to(bundle.device) for k, v in encoded.items()}

        # "max_new_tokens" is exposed in the UI, but internally we cap the total
        # sequence length: input length + requested new tokens, hard-capped by
        # the model's positional limit. This avoids the max_length clash for
        # transformers 4.x and 5.x alike.
        total_length = min(int(max_positions), int(encoded["input_ids"].shape[1]) + int(max_new_tokens))
        if base_cfg is not None:
            cfg = copy.deepcopy(base_cfg)
            cfg.max_length = total_length
            kwargs = {"generation_config": cfg}
        else:  # very old transformers
            kwargs = {
                "num_beams": num_beams,
                "max_length": total_length,
                "do_sample": False,
            }
        with torch.inference_mode():
            output = bundle.model.generate(**encoded, **kwargs)
        results.append(bundle.tokenizer.decode(output[0], skip_special_tokens=True).strip())

    return results


# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------

CATEGORY = "text/translation"
MODEL_TYPE = "OPUS_MT_MODEL"


class OpusMTModelLoader:
    """Load an OPUS-MT model once and share it with several translate nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cpu", "cuda", "auto"],),
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "留空 = 自动查找本目录 / ComfyUI models/translation",
                    },
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "离线加载 OPUS-MT (MarianMT) 翻译模型，默认使用同目录的 opus-mt-zh-en。"

    def load(self, device, model_path=""):
        return (load_translation_model(model_path, device),)


class OpusMTTextTranslate:
    """Translate text with OPUS-MT. Works standalone or with a loader node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "一只戴眼镜的橘猫，坐在窗台上，阳光，写实照片",
                        "dynamicPrompts": False,
                        "placeholder": "输入中文文本/提示词，也可以从其他节点连线输入",
                    },
                ),
                "num_beams": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 512, "step": 8}),
                "chunk_size": ("INT", {"default": 180, "min": 40, "max": 400, "step": 10}),
                "passthrough_english": ("BOOLEAN", {"default": True}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                "device": (["cpu", "cuda", "auto"],),
            },
            "optional": {
                "model": (MODEL_TYPE,),
                "model_path": (
                    "STRING",
                    {"default": "", "placeholder": "可选：自定义模型目录绝对路径"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "离线中译英（默认 opus-mt-zh-en）。输出 STRING，可直接连到 CLIP Text Encode 的 text 输入。"
    )

    def translate(
        self,
        text,
        num_beams=4,
        max_new_tokens=256,
        chunk_size=180,
        passthrough_english=True,
        keep_loaded=True,
        device="cpu",
        model=None,
        model_path="",
    ):
        text = "" if text is None else str(text)
        if not text.strip():
            return ("",)

        if passthrough_english and not _has_cjk(text):
            print("[OPUS-MT] 输入不含中文，已原样输出（关闭 passthrough_english 可强制翻译）。")
            return (text,)

        auto_loaded = not isinstance(model, TranslationModel)
        bundle = (
            load_translation_model(model_path, device)
            if auto_loaded
            else model
        )

        chunks, per_line = _split_text(text, chunk_size)
        print(f"[OPUS-MT] 翻译 {len(chunks)} 段文本，num_beams={num_beams}，设备={bundle.device}")
        try:
            translated = _translate_chunks(bundle, chunks, int(num_beams), int(max_new_tokens))
        finally:
            if auto_loaded and not keep_loaded:
                unload_models()

        return (_join_text(translated, per_line),)


class OpusMTModelUnloader:
    """Optional utility node to free memory from a workflow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"unload": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = CATEGORY
    DESCRIPTION = "清空翻译模型缓存，释放内存/显存。"

    def unload(self, unload=True):
        if unload:
            unload_models()
        return ()
