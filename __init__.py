# -*- coding: utf-8 -*-
"""ComfyUI custom node package: offline OPUS-MT text translation (zh -> en by default)."""

try:  # normal ComfyUI import (as a package)
    from .opus_mt_nodes import (
        OpusMTModelLoader,
        OpusMTTextTranslate,
        OpusMTModelUnloader,
    )
except ImportError:  # some launchers load __init__.py directly by file location
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from opus_mt_nodes import (  # type: ignore
        OpusMTModelLoader,
        OpusMTTextTranslate,
        OpusMTModelUnloader,
    )

NODE_CLASS_MAPPINGS = {
    "OpusMTTextTranslate": OpusMTTextTranslate,
    "OpusMTModelLoader": OpusMTModelLoader,
    "OpusMTModelUnloader": OpusMTModelUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpusMTTextTranslate": "文本翻译 (中→英 · OPUS-MT)",
    "OpusMTModelLoader": "OPUS-MT 翻译模型加载器",
    "OpusMTModelUnloader": "卸载 OPUS-MT 翻译模型",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
