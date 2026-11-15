# ComfyUI 文本翻译节点（OPUS-MT / 中译英）

把当前目录里的模型仓库直接变成 ComfyUI 自定义节点：**完全离线**地把中文提示词翻译成英文，输出 `STRING`，可以直接连到 `CLIP Text Encode (Prompt)` 的 text 输入。

模型为 `Helsinki-NLP/opus-mt-zh-en`（MarianMT，6 层 / 512 维，约 300 MB 权重，CC-BY-4.0），方向固定为 **中文 → 英文**。目录里的 `config.json`、`source.spm`、`target.spm`、`vocab.json`、`pytorch_model.bin` 就是模型本体，节点会就地读取，不需要重新下载。

## 目录里新增的文件

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | ComfyUI 节点注册入口 |
| `opus_mt_nodes.py` | 节点实现：模型发现/加载、长文本切分、生成参数、缓存 |
| `requirements.txt` | 额外依赖（不含 torch） |
| `install_comfyui.py` | 一键安装脚本：把本目录链接进 ComfyUI，或 `--check` 体检 |
| `convert_to_safetensors.py` | 可选工具：把旧版权重转成 safetensors（见下方"兼容性"） |
| `COMFYUI_README.md` | 本文档 |

原有的 `tf_model.h5`（TensorFlow）和 `rust_model.ot`（Rust）用不上，可保留；`--copy` 安装方式会自动跳过它们。

## 安装

### 方式 A：一键链接（推荐，不复制 300 MB 模型）

```bash
python install_comfyui.py "D:\ComfyUI_windows_portable\ComfyUI"
```

参数是**含 `custom_nodes` 文件夹的 ComfyUI 根目录**。脚本会在 `custom_nodes/ComfyUI_OpusMT_Translate` 创建一个目录联接（Windows `mklink /J`，其他系统 symlink），模型文件仍留在原地。省略参数时脚本会尝试自动探测常见路径，也可设置环境变量 `COMFYUI_PATH`。

加 `--copy` 则改成完整复制（只复制 PyTorch 需要的 8 个模型文件，约 314 MB）。脚本**不会删除或覆盖**任何已有内容。

### 方式 B：手动

把整个 `E:\translator` 文件夹复制（或移动）到 `ComfyUI\custom_nodes\` 下，并改名为 `ComfyUI_OpusMT_Translate`（用下划线，别用中文或短横线，避免旧版 ComfyUI 导入失败）。

### 方式 C：模型与节点分开放

节点文件放进 `custom_nodes/`，模型文件放进 `ComfyUI/models/translation/opus-mt-zh-en/`；或设置环境变量 `OPUS_MT_MODEL_PATH` 指向模型目录；或直接在节点的 `model_path` 输入框里填绝对路径。三者优先级：**节点填写的路径 > 环境变量 > 节点同目录 > ComfyUI models 目录**。

### 依赖

用 **ComfyUI 实际运行的那个 Python** 安装（整合包一般是 `ComfyUI_windows_portable\python_embeded\python.exe`）：

```bash
python -m pip install -r requirements.txt
```

即 `transformers`、`sentencepiece`、`protobuf`、`sacremoses`（最后一个是 MarianTokenizer 的推荐依赖，装了可消除分词器警告）。**不要**在这里安装 torch —— 会和 ComfyUI 自带的 CUDA 版 torch 冲突。安装完重启 ComfyUI，在节点菜单 `text > translation` 下就能找到节点。

随时可以用 `python install_comfyui.py --check` 体检：列出模型文件是否齐全、当前 Python 里依赖是否可用。

## 三个节点

| 节点 | 说明 |
| --- | --- |
| `文本翻译 (中→英 · OPUS-MT)` | 主节点。可单独使用（自动加载并缓存模型），也可接 `model` 输入复用加载器的模型 |
| `OPUS-MT 翻译模型加载器` | 提前加载一次模型，输出 `OPUS_MT_MODEL`，适合多个翻译节点共用 |
| `卸载 OPUS-MT 翻译模型` | 清空模型缓存，释放内存/显存 |

主节点参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `text` | 示例中文提示词 | 多行文本；支持从其他 STRING 节点连线输入 |
| `num_beams` | 4 | Beam search 宽度，越大越慢、通常略好；模型原始配置是 6 |
| `max_new_tokens` | 256 | 单段译文的长度上限；内部按「输入长度 + 该值」截断，总长不超过模型的 512 token 限制 |
| `chunk_size` | 180 | 长文本按标点/换行切分的每块字符数，防止超过 Marian 的 512 token 上限 |
| `passthrough_english` | True | 输入不含汉字时原样输出，避免把已有英文提示词再"翻译"一遍 |
| `keep_loaded` | True | 翻译后保持模型常驻；关掉则每次用完卸载（会变慢） |
| `device` | cpu | `cpu` / `cuda` / `auto`。默认 CPU：这个小模型 CPU 上单句约 1 秒内，且不占用出图显存 |
| `model`（可选） | — | 接加载器输出 |
| `model_path`（可选） | 空 | 自定义模型目录绝对路径，留空自动查找 |

## 典型工作流

```
[中文提示词] → [文本翻译 (中→英 · OPUS-MT)] → [CLIP Text Encode (Prompt)] → [KSampler] …
```

在翻译节点的 `text` 里直接写中文，或把任意 STRING 输出连进来即可。首次执行会打印 `[OPUS-MT] 首次加载模型...`，之后复用内存中的模型，不会重复读盘。

实测（CPU，torch 2.15 + transformers 5.23）：首次加载约 2–5 秒，单句 beam=4 约 0.3–0.5 秒。例如 `一只戴眼镜的橘猫，坐在窗台上，阳光，写实照片` → `An orange cat with glasses, sitting on the window stand, sunshine, real pictures.`

## 注意事项与排错

- **方向是单向的**：只能中译英。需要英译中请另外下载 `Helsinki-NLP/opus-mt-en-zh`，把它的文件放到一个文件夹，再在 `model_path` 里指定（代码本身是通用的 MarianMT 加载器）。
- **找不到模型**：节点会打印所有搜索过的路径。填 `model_path` 绝对路径最直接。
- **报 `ImportError` / 依赖缺失**：确认用的是 ComfyUI 的那个 Python 装的包；`sentencepiece` 是 Marian 分词器的必需项。
- **旧版权重的兼容性**：本仓库的 `pytorch_model.bin` 是 2020 年导出的 legacy pickle 格式（不是普通 zip 容器）。已在 torch 2.15 / transformers 5.23（CPU）实测可直接加载，节点对 4.x / 5.x 的生成参数也做了兼容，控制台无警告。万一你的环境报 `.bin` 加载失败：用 ComfyUI 的 Python 执行 `python convert_to_safetensors.py`，生成 `model.safetensors`（约 298 MB），节点自动优先使用它、原 `.bin` 保留作为后备；或把 transformers 固定在 4.4x。转换脚本会合并 Marian 的共享嵌入权重（否则文件会膨胀到 550 MB），实测转换后译文与原权重完全一致。
- **长文本**：会按行和标点切块逐段翻译再拼回，保留原有换行；逗号分隔的提示词标签串可以整段输入。
- **输出为空/太短**：调大 `max_new_tokens`，或把 `num_beams` 调到 6。
- 模型是 2020 年 OPUS 语料训练的轻量翻译模型，偏直译，适合提示词场景，不适合长篇文学翻译。

## 许可

模型权重来自 Helsinki-NLP OPUS-MT（CC-BY-4.0，详见 `README.md`）。节点代码可自由使用与修改。
