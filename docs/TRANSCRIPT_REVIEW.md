# 人工校对转录稿

逐片段 JSONL 用来保留 ASR 的时间信息。**一个视频对应一份完整 Markdown，从头到尾保留所有已转录正文，不摘要、不省略；段落都在同一个文件内。** 文件顶部标明视频 ID、原始片段数和段落数，无需编辑 JSON 字段。完整性以输入转录为准，不能补回 ASR 漏听的内容。

## 已有转录：直接导出，无需重新转录

从仓库根目录、原有 Python 环境运行。目录和文件路径可以包含中文及空格。

```powershell
# 单文件：在原文件旁生成 demo.review.md；支持 JSONL 或顶层数组 JSON
python -m transcribe.review export --input "D:/video_rag_data/transcripts/demo.jsonl"

# 批量：递归处理目录中的全部 JSONL，按视频导出完整校对稿
python -m transcribe.review export --input "D:/video_rag_data/transcripts"

# 一个 JSONL 混有多个视频时，也会每视频生成一份 .md
python -m transcribe.review export --input "D:/video_rag_data/merged.jsonl" --output "D:/video_rag_data/review"
```

导出只读取本地转录，不打开视频，不加载 Whisper 或 embedding 模型，不调用大模型。已有校对稿会保留。
单视频输入默认生成旁边的 `demo.review.md`；多视频输入默认生成 `demo.reviews/` 文件夹，每个视频一份稿，文件名包含安全处理后的视频 ID 末级名称与短哈希，避免同名覆盖。多视频输入的 `--output` 必须是文件夹。每份稿包含该视频所有原始片段，包括相互交错存储的记录及最后一个片段。
`python -m transcribe.run` 在完成转录后也会尝试生成旁边的 `.review.md`。已被批处理跳过或通过注册表复用的历史转录，可用上述批量命令补齐。

## 修改正文

用 VS Code、记事本或 Markdown 编辑器打开 `.review.md`，保持 Markdown 格式并以 UTF-8 保存。示例（虚构）：

```markdown
## P000001 | 00:00:00.000 → 00:00:32.000 | "training/radio-001"

PC 软件负责编排整个测试流程。它控制外部仪表，向设备发送测试命令，并收集测试结果。
```

只修改段落下的正文，可以纠错、调整标点、换行或合并句子。保留文件头注释、全部段落标题、编号、时间和视频标识。
不清楚的地方回听音频；不要按常识补写事实。整段无法辨认可写 `[无法识别]`，后续分块会跳过该段。

默认在至少 30 秒且遇到句末标点后换段，最长 60 秒；视频变化、时间回退、超过 10 秒的间隔会换段。缺标点时仍可能在上限处分段，不保证一个段落恰好是完整句子或话题。
可调整 `--target-seconds 40 --max-seconds 60 --gap-seconds 10`。单个原始片段超过上限会报错，不截断或伪造时间。段落上限应不超过后续 `chunking.max_seconds`（项目默认 90 秒）。

## 导回新文件，再分块

```powershell
python -m transcribe.review import --source "D:/video_rag_data/transcripts/demo.jsonl" --review "D:/video_rag_data/transcripts/demo.review.md" --output "D:/video_rag_data/corrected/demo.jsonl"
python -m ingest.chunk --input "D:/video_rag_data/corrected/demo.jsonl" --output "D:/video_rag_data/corrected/demo.chunks.jsonl"
```

导回时校验原件哈希以及全部段落编号、顺序、时间和视频标识。原件改变、标题丢失或正文被清空时停止，不发布输出。输出必须为新的路径，不覆盖原始转录或已有文件。二次校对可继续修改同一 Markdown，导出另一个新 JSONL。
如果稿件来自合并 JSONL，`--source` 仍指定这个合并原件；文件头记录当前视频 ID，导回只生成该视频的校正版，不混入其他视频。每份稿分别导回自己的 `.jsonl`，全量建库前再按既有流程汇总全部有效视频。旧版无视频过滤标记的稿件仍按其原有范围导回。

未改动段落恢复原始小片段及时间。修改过的段落作为一条新记录，使用原始片段的最早开始、最晚结束时间，标记 `timestamp_quality: estimated`。这是段落定位，不能认为每个修改后的句子都重新完成了音频对齐；不会把新增文字按比例分摊到虚构时间点。

原始 JSONL、窗口断点和 ASR audit 都保留；校正版另存到 `corrected` 目录，不自动写回 ASR 缓存或完成注册表，以免后续续跑覆盖修订。导入分块、增强和建库时显式选择校正版；更新全量索引时仍需提供所有有效视频。校对段落与最终检索 chunk 是两步，模型分块仍在校对之后进行。
