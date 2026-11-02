# lab：只设置一个输入路径

此分支来自 `experiment/post-rerank-context`，保留原来的完整 Markdown 校对、分块、建库断点和后重排上下文扩展功能。

## 硬件与默认方案

| 项目 | 本机配置及含义 | 本分支的用法 |
|---|---|---|
| CPU | i9-12900 | 音频解码、CPU INT8 后备、embedding 与重排 |
| 系统内存 | 32GB | 模型、音频、向量与程序共同使用 |
| T400 独立显存 | Dedicated GPU memory 4GB | 优先量化转录，逐窗口处理 |
| T400 共享内存 | Shared GPU memory 15.8GB | 属于上述 32GB 系统内存；不是新增独立显存 |
| GPU memory 19.8GB | 4GB 独立显存与 15.8GB 共享内存的合计显示 | 不能按 20GB 独显选择 CUDA 模型 |
| UHD 770 | 使用系统共享内存的核显 | 当前入口未接入 OpenVINO，不与 T400 拼接显存 |

Windows 的 dedicated/shared 含义见 [Microsoft 说明](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)，核显共享内存见 [Intel 说明](https://www.intel.com/content/www/us/en/support/articles/000020962/graphics.html)。

默认选用 faster-whisper **medium**，CUDA 时尝试 INT8/FP16 混合计算，CPU 时使用 INT8。
它替代旧入口默认的 small，但更大的模型能否改善当前领域术语，仍须用同一段音频人工对比。
CUDA 转录失败时明确记录错误并尝试同一模型的 CPU INT8；第二次失败就停止。
不会把模型自动降到 small，也不会把失败伪装成转录完成。

Embedding 沿用 **BAAI/bge-m3**，重排沿用 **BAAI/bge-reranker-v2-m3**。
这两个模型默认在 CPU 上运行（batch size 8，embedding 上限 1024 tokens），保留 T400 的显存给转录。
ASR、分块、建库分进程顺序运行，阶段结束后释放模型。
增强使用 extractive：摘录原文，不调用 Ollama、不生成补充事实；摘要不参与向量计算。
检索沿用 Dense + BM25 → RRF → cross-encoder，并在重排选中后扩展前后 15 秒证据。

32GB 内存提高了 CPU 运行和建库的余量，**没有把 T400 的独立显存提高到 20GB**。
large-v3 是可选实验：不承诺 4GB 显卡必定能放下或达到某个速度。
[faster-whisper 官方基准](https://github.com/SYSTRAN/faster-whisper#benchmark)的显存数字来自其他硬件及特定参数，不能当作本机实测。
先稳定完成转录并抽查，再比较大模型和检索指标，仍是本项目的推进顺序。

## 第一次使用

前提：已有 64 位 Python 3.11 或 3.12。Windows 安装时启用 Python launcher / PATH；GPU 路径需要正常工作的 NVIDIA 驱动。
无需单独安装 ffmpeg、ffprobe、CUDA Toolkit、Ollama 或 Docker：音频解码由 PyAV 提供，GPU 运行库从 CUDA torch wheel 复用。
CUDA 能否工作仍取决于本机驱动与 DLL 兼容性，入口提供 CPU 后备，尚未在这台 T400 上实测。

1. 打开仓库根目录的 **`lab_input.txt`**，只填一行视频文件或文件夹路径，例如：

   ```text
   C:\Users\你的名字\OneDrive - Ericsson\Videos
   ```

   中文、空格均可；输入也可以是一段 `.mp4` 或 `.mp3`。不需要修改其他目录。

2. 在 Git Bash 运行：

   ```bash
   bash run_lab.sh
   ```

   也可以双击 **`run_lab.cmd`**。两个入口执行同一流程。

3. 首次会自动创建 `.venv-lab`、安装依赖，并从 Hugging Face 准备三个模型。
   后续复用环境、pip 缓存和 Hugging Face 缓存；不会反复下载已经缓存的模型。
   下载阶段需要能连接 PyPI、PyTorch wheel 源及 Hugging Face；失败会保留进度并停止，不改镜像、不关闭证书验证。
   推理阶段使用本地模型且启用离线模式。视频/转录不会发送到模型服务。

输入文件夹只按媒体扩展名扫描，忽略 PPT/PDF/Word，不读取这些文档正文。
OneDrive 仅联机媒体在打开时可能由 Windows 自动下载；本入口不自动释放其本地副本，也不覆盖原视频。

## 自动执行的流程

```text
准备环境与模型 → 扫描媒体 → 转录及窗口断点 → 每个视频一份完整 .review.md
→ 导入该校对稿的当前正文 → 语义与时间窗分块 → 原文摘录 → embedding / BM25 建库
```

第一次运行会直接用未修改的校对稿建库。需要人工修正时，在整份 `.review.md` 内修改正文后，再执行同一个启动命令。
它会保留原稿与人工修改，生成对应的新校正版 JSONL，再更新分块和索引，不用手工改零散 JSON。
未修改段落保留原 ASR 片段；修改段落仅保留整段时间范围并标记 `estimated`，不会伪造逐句时间戳。
段落标题、编号、时间和文件头必须保留；原始转录改变后，旧校对稿会被拒绝导回，需要重新导出并迁移校对内容。

所有结果自动存入 `data/lab/<输入路径短哈希>/`，路径由程序确定：

| 路径 | 内容 |
|---|---|
| `config.json` | 此输入的配置（JSON 同时是合法 YAML） |
| `videos.json` | 当前输入的视频清单 |
| `transcripts/` | 原始 JSONL、质量日志、逐窗口断点及每视频一份完整 `.review.md` |
| `corrected/` | 按校对内容哈希保存的校正版 JSONL，各版保留 |
| `effective-transcripts.jsonl` | 本次实际用于分块的全部视频正文 |
| `chunks.jsonl` / `enriched.jsonl` | 分块及原文摘录结果 |
| `index/` | 本地 Qdrant、BM25、向量缓存与 `CURRENT` 快照指针 |
| `logs/` | 各阶段日志 |

目录名相同但位置不同的输入使用不同工作目录；同名视频也不会覆盖彼此。
不读取先前部署的 `RAG_CONFIG` / `RAG_*__*` 路径设置，避免误写旧索引。
完整转录在输入文件大小/修改时间及设置未变、输出校验通过时直接复用；中断时复用已验证的窗口。
文件内容改过但刻意保留相同大小与修改时间的情况不在媒体变更检测保证内。
修改校对正文会使下游阶段失效重建；未改变的完整阶段可复用，建库中断仍利用已有向量断点。
复制源码目录到其他位置会改变路径身份；输出不会自动迁移到新目录。

## 其他入口

只查看计划，不安装环境、不下载或加载模型：

```bash
bash run_lab.sh --plan
```

仅准备环境和模型：

```bash
bash run_lab.sh --setup-only
```

建库完成后打开已有知识库：

```bash
bash run_lab.sh --search
```

建完立即打开网页：

```bash
bash run_lab.sh --serve
```

可选对比 large-v3（仅实验，不是必改项）：

```bash
LAB_ASR_MODEL=large-v3 bash run_lab.sh
```

语言默认自动检测；如果视频语言固定，可选择 `LAB_LANGUAGE=en` 或 `LAB_LANGUAGE=zh` 后运行。
模型或语言改变会重新转录；已有校对稿仍保留，若不再匹配原始 JSONL，导回会停止，请先备份校对稿并重新导出迁移。
断点和缓存只证明处理步骤已完成，**不证明转录准确或检索召回率达到 95%**。

## 验证范围

新增测试覆盖单一路径、中文空格路径、只扫描媒体、同名视频隔离、人工校对进入下游、
阶段缓存失效、PyAV 音频窗口、转录断点及输入变更检测、完整转录复用、保留人工修改、旧路径环境隔离。
云端验证使用测试音频解码和模型替身；不声称在用户的 T400 上完成了模型推理、性能或准确率验收。
