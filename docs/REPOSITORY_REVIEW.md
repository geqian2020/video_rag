# 现有仓库审查与迁移建议

审查来源：`geqian2020/video_rag` 的 main 分支；读取了转录脚本、分块、清洗、建库、检索和桌面 demo。未读取真实 chunks 内容或数据库中的业务正文。现有内容保留在原 `workflow/`；此升级包不删除、不打包旧业务数据，也不替换旧脚本。

| 项目 | 当前实现与具体位置 | 修改或补充方案 |
|---|---|---|
| 转录 | `workflow/kownledge/transcribe_meetings.py` 已有 ffmpeg + Whisper 本地处理 | 保留作为上游；新项目不强制安装 Whisper。未来优先同时输出 Whisper segment 的 start/end JSONL，当前 Markdown 用转换器迁移 |
| 分块 | `build_chunks.py` 的 `merge_lines_into_chunks` 主要由 300/400 字触发 | 新增 `ingest/chunk.py`，句子 embedding 相似度 + 最短/最长时间窗 + 静音间隔；token 上限仅为防截断硬约束 |
| 时间戳 | 原 `end_seconds=current[-1][0]` 是最后一行的开始，不是真正结束 | JSONL 显式 end；旧 Markdown 使用下一条 start 和明确传入的视频总时长，标记 estimated |
| 原文追溯 | `clean_text.py` 会直接替换术语、转换简繁，旧 chunk 只存清洗结果 | 新链路保留句子原文和来源时间；审核过的术语映射可未来作为独立检索字段，不自动改写证据 |
| ID | 原 ID=`md_path.stem + 块序号`，不同文件夹同名视频可能冲突 | 使用稳定 video_id + 时间 + 文本的 UUID；调用方保证 video_id 唯一且跨机器稳定 |
| Embedding | 原默认 bge-small-zh-v1.5，CPU | 默认 BGE-M3，可切换 Qwen3-Embedding；统一模型配置、revision 和离线加载 |
| 距离 | 原建库没有指定 Chroma cosine metric，查询却按 `1-distance` 展示相似度 | 新库明确 cosine；RRF 与 reranker 分数分别展示，不声称为概率 |
| 增强 | 当前无 enrichment | 新增本地 Ollama、结构化 JSON、原文摘要摘录、3-5 个证据问题、验证与失败策略 |
| 混合检索 | 当前只有 dense | jieba + 标准 BM25 编为 Qdrant sparse；两个通道都先 filter，再分别取 30，RRF 后保留 30 |
| 重排 | 当前无 | 必须 cross-encoder；30→5；失败报错，不退回假装成功 |
| API/评估 | 当前是 CLI 和 Tkinter 界面 | FastAPI `/search`、API key、gold set、Recall/MRR、自动化测试 |
| 更新一致性 | 原 manifest 只看文件 mtime，模型/配置变化可能被跳过；删除源文件也不会自动删其旧块 | 第一版完整快照重建，原子切换 CURRENT，模型指纹校验，旧快照保留；后续按规模再实现内容哈希增量 |
| 数据管理 | Git tree 已包含 `index_data/chunks.jsonl`、Chroma 数据库、日志、pyc | 新增 `.gitignore`，仅使用虚构示例；已跟踪文件不会被 ignore 自动清除，历史版本也仍可能包含数据 |

## 迁移顺序

1. 将升级包解压覆盖到仓库根目录；保留 `workflow/` 作为历史流程。
2. 使用同一批转录建立新索引到 `data/index`，不要复用旧 Chroma 数据库。
3. 先跑样例和测试，再在公司内网使用真实 gold set 比较检索效果。
4. 工作台改接 `/search`；原桌面 demo 仍连接旧库，可保留过渡使用。新 API 不直接兼容该 Chroma demo。
5. 审核后再停用旧检索脚本；转录上游可继续使用。

## Git 中已有数据需单独处理

本次不会自动删除远端已有文件。请在本地备份后排查：

- `workflow/video_kb/index_data/`：含持久化数据库和转录块。
- `workflow/kownledge/*.log`、`*_result.txt`、`*_err.txt`：可能包含问题、结果和本机路径。
- 所有 `__pycache__/`：不属于源码。
- `config/glossary.json`、`term_corrections.json`：可能包含内部产品代号，应审核后决定放私有仓库还是外置。

如确有敏感数据，删除最新版本不等于清除 Git 历史；按公司流程处理仓库可见性、历史清理与必要的凭据轮换。本升级包仅包含新增通用代码和虚构样例，没有把旧数据再打包分发。

## 补充：OCR 词汇辅助转录

新增 `ocr.extract`、`ocr.vocab` 和 `transcribe.run`。保留旧转录脚本作为独立入口；新入口通过已确认的局部术语提示 Whisper，并直接输出 JSONL。之前的“仅保留上游”升级说明现在包含此可选扩展。操作与边界见 OCR_ASSISTED_ASR.md。
