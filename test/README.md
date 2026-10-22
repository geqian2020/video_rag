# 公开 agent 视频测试集

本目录保存原始视频页面链接、可核对的标题/作者/时长、下载工具和评测操作。**`test/` 是仓库目录**；代码仍在 main 分支。媒体、完整字幕、实际问题/答案/证据、索引与逐题记录均放 data/public_benchmark/，不提交 Git。

## 来源

原始地址一行一个见 [urls.txt](urls.txt)，机器可读元信息见 [videos.json](videos.json)。时长是 2026-10-22 从 YouTube watch-page 的 videoDetails.lengthSeconds 读取，不用搜索摘要猜测；列出字幕轨道不代表每个环境都能成功下载。

| ID | 视频/作者 | 时长 | 方向 |
|---|---|---|---|
| xzRJOaqDMCg | LangChain Tutorial for Beginners: Build Production-Ready AI Agents from Scratch in 99 Minutes — Zahiruddin Tavargere | 1:38:42 | 工具、记忆、MCP、人工审批与动态行为 |
| 7WC12HHC55E | 1 Hour LangGraph Crash Course — Machine Learning With Hamza | 1:04:20 | 图状态、路由、检查点、HITL、ReAct |
| 95XrWA6-UFM | Building Agentic and Multi-Agent Systems with LangGraph — Open Data Science and AI Conference | 1:59:27 | Agentic RAG 与多智能体 |
| kBXYFaZ0EN0 | CrewAI Step-by-Step — Alejandro AO | 1:07:43 | 任务依赖、工具和会议助理 |
| s-jYxgKMqRc | Master CrewAI: Your Ultimate Beginner's Guide! — Sam Witteveen | 1:00:18 | 角色、任务、委派和进程 |

videos.json 的 alternates 另外保留了 MCP workshop 来源；该来源在本次云端下载尝试中被 HTTP 429 限流。访问失败如实记录，不绕过登录、付费或下载保护。公开可观看不等于允许重新分发；按原站规则及作者授权获取，Git 只保存链接与工具。

## 1. 下载视频或先用字幕

先复用现有 Python/ffmpeg/模型缓存。在有网络并获准下载的机器运行：

```bash
pip install -r requirements.txt
pip install -r test/requirements.txt
python -m test.download --mode subtitles
# 要测视频 OCR/本地 ASR，显式下载媒体：
python -m test.download --mode both
```

默认仅下载英文字幕，减少数 GB 媒体传输；manual/auto 字幕由 yt-dlp 选择，下载后仍需抽查标准术语。字幕来自平台而非项目 Whisper，这条路径只验证检索，**不能证明 OCR 词库改善了 ASR**。

公司代理 CA 在操作系统可信库、而默认 certifi 不认识时可加 `--system-ca`，仍然验证 TLS。不要关闭证书校验。工具复用已有有效字幕；失败写本地 downloads.json 和日志并返回非零。遇到 429 按平台提示稍后重试，不反复轰击接口；本次云端有一条主来源因此未进入测试。yt-dlp 的 YouTube 视频格式提取可能另需其官方支持的 JavaScript runtime，查看官方 README。

输出：

- data/public_benchmark/raw/视频ID/：原始 JSON3 字幕/可选媒体。
- data/public_benchmark/downloads.json：各来源成功/失败与文件哈希。

## 2. 字幕标准化

```bash
python -m test.prepare_corpus
# 已明确有来源缺失，只想测已取得的子集时：
python -m test.prepare_corpus --allow-missing
```

转换后有 per-video segments/、合并 transcripts.jsonl 和 corpus_stats.json。字幕滚动显示会让 display duration 重叠，转换时将每个事件的结束时间限制到下一条正文起点，并标记 timestamp_quality=estimated；不用显示时长伪装逐字口播对齐。统计明确列出缺失来源，部分语料不会伪装成完整 manifest。

## 3. 造题、证据定位与人工审核

每个视频只给 agent 相应视频内容，使用 [QUESTION_PROMPT.md](QUESTION_PROMPT.md)。建议 60-100 题并混入标准术语手写题，含相关无答案题。每题 video_id 是标签来源；**不把它复制成检索参数 video_ids**，评测时所有视频均参与竞争。

默认需要人工 reviewed=true 才生成 gold：

```bash
python -m eval.make_gold --segments data/public_benchmark/transcripts.jsonl \
  --draft data/public_benchmark/draft.jsonl --output data/public_benchmark/gold.jsonl
```

只做流程试跑可显式加 `--allow-unreviewed`；结果会带 agent_draft，不能当最终验收分数。脚本检查逐字证据、视频范围、空/重复问题、重复证据位置与时间连续性；不能用字面匹配替代对问题/答案真实性和无答案范围的人工审核。

gold 的时间字段统一为当前仓库格式：

```json
{"query":"换一种说法的问题","type":"howto","video_id":"来源视频ID","reviewed":true,"relevant":[{"video_id":"来源视频ID","start_ts":120.0,"end_ts":135.0,"evidence":"逐字原文"}]}
```

证据重复出现时默认拒绝；draft 可提供 evidence_start_hint 指向经检查的出现时间。不使用首个 find 结果随意定位。无答案题 relevant=[]，需人工确认在**整个检索语料**无答案。标签只引用时间范围，换分块参数后仍可使用。

不采用“交叠 1 秒即可命中”的规则，它会放宽判定。本项目仍要求单块覆盖至少 50% 的 gold 时间段；多段证据各自计 Recall，同一证据多次命中不重复计数。跨块完整覆盖暂需逐题检查，比较分块方案时应同时看 chunk 时长与失败实例。

## 4. 实际检索评估

根目录 config.yaml 为正式 BGE-M3 + bge-reranker-v2-m3 方案；默认需要已准备的本地权重与 Ollama。按根 README 建库，然后：

```bash
python -m eval.run_eval --gold data/public_benchmark/gold.jsonl \
  --output data/public_benchmark/eval.json --details data/public_benchmark/details.jsonl
```

按视频、题型汇总；可回答题统计 Recall@30、Recall@5、Hit@5 和 MRR，无答案题单独计算误接受率。逐题记录含候选和最终证据，真实数据不可提交。可用 --min-score 评估一个**已在独立验证集选定**的阈值；不要在这批测试题上调完阈值又报告同批分数。此参数为离线评估开关，尚未自动修改生产 API 的拒答策略。

### 可选轻量英文 CPU 基线

本次云端无 GPU，使用真实小模型跑英文字幕，不依赖 Ollama：

- embedding：BAAI/bge-small-en-v1.5，固定 revision 在 config.cpu.yaml。
- cross-encoder：cross-encoder/ms-marco-MiniLM-L-6-v2，固定 revision。
- Dense 只编码原文；摘要为摘录，问题增强关闭，状态明确标为 extractive_fallback。
- 所有视频全局检索；dense/BM25/hybrid 各比较前 30 与前 5，另测 hybrid+rerank。

先用批准的方式准备以上模型到本地缓存，或将 test/config.cpu.yaml 改为完整本地目录（不提交公司路径），再运行：

```bash
python -m test.run_benchmark --build --gold data/public_benchmark/gold.jsonl
# 语料/分块/模型没有变化时不重复建库：
python -m test.run_benchmark --gold data/public_benchmark/gold.jsonl
```

offline=true，不会因模型缺失自动下载。仅有 CPU 时，可先从 PyTorch 官方 CPU wheel 渠道准备 torch，避免误装大型 CUDA 依赖；已有 torch 直接复用。

产出 report.json、details.jsonl。最新实际运行范围和指标见 [CLOUD_RUN.md](CLOUD_RUN.md)。未启用生成问题增强，所以本次不能衡量 enrichment 的收益；英文基线不能替代中文专有名词测试或正式 BGE-M3 的评估。

## 5. 失败分析与后续

从 details.jsonl 筛选 failed=true，人工标记：证据在 30 条外、重排丢失、分块边界、ASR 术语错词、缩写/同义词、跨段推理、标签错误。先修标签与数据，再调参。无答案 score 分布只用于设计验证集；不直接写生产阈值。

视频 OCR/Whisper 要另走 docs/OCR_ASSISTED_ASR.md，比较同段音频无词表/有词表的 CER、术语命中与误插词；完成后再对照检索效果。待检索稳定，最后由批准的工作台 LLM 做端到端正确性/忠实度/拒答抽测。

官方工具资料：[yt-dlp](https://github.com/yt-dlp/yt-dlp)、[BGE small](https://huggingface.co/BAAI/bge-small-en-v1.5)、[MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)。
