# OCR 建词汇库，辅助本地视频转录

上一版 RAG 只消费转录稿，没有 OCR 辅助识别。本次新增一个可选上游：视频抽帧 → OCR 记录 → 术语候选 CSV → 人工确认标准拼写 → Whisper 按时间窗口使用术语提示 → 带 start/end 的转录 JSONL → 原有 RAG。

该词汇库用于帮助识别“声音对应哪个拼写”，不为原文补写产品解释或背景事实。OCR 文字不会直接混入转录或 Dense embedding。

## 为什么需要确认

专有名词既可能被 ASR 听错，也可能被 OCR 看错。比如画面原词为 `Qdrant`，OCR 错为 `Qdranl`，直接把它加入热词会让语音识别也偏向错字。处理时必须保留画面证据，由人把 `canonical_term` 改为正确拼写后确认。

OCR 置信度和多帧重复只是审核线索，不是词条正确性的保证。同一静态 PPT 被抽到多次不算独立真实性证据；代码不会据此自动批准。`similar_candidates` 提醒英文相近词供审核，不自动合并 O/0、I/l、产品名和缩写。

## 安装与已有环境复用

先检查现有 ffmpeg、Whisper、PyTorch、OCR 和模型缓存，已经可用的环境直接复用。只需要重新转录时才装 Whisper，已经有正确转录稿时不需要。

```powershell
ffmpeg -version
ffprobe -version
python -m pip show openai-whisper rapidocr-onnxruntime torch
# 缺少时在经批准的准备环境安装；离线工作机使用同平台 wheelhouse
pip install -r requirements-ocr.txt
# 仅缺少 Whisper 时：
pip install -r requirements-asr.txt
```

本适配器固定使用 `rapidocr-onnxruntime==1.4.4`，其 wheel 内含三个小型 PP-OCR ONNX 模型，可直接本地运行。此选择是为了提供简单且不在运行时下载模型的兼容入口，不声称其准确率优于新版。已经使用新版 `rapidocr`、PaddleOCR 或更好 OCR 时，可将识别结果导出为下述 observations 格式复用后续流程；两类 RapidOCR 包不是同一个 API，不能只改依赖版本。

`config.yaml` 的 `ocr.det_model/rec_model/cls_model/rec_keys` 可指定已有本地 ONNX 与配套字典；不指定时用固定包自带模型。字典必须与识别模型配套，不能简单往 rec_keys 字符表里塞产品名就当作增加词汇知识。`asr.model` 指定已有 `small.pt`、`medium.pt` 等路径，或已缓存的 Whisper 模型名。新入口在模型缺失时直接报错，不自动下载。

## 第一步：抽帧与 OCR

```powershell
python -m ocr.extract --video "D:/approved/demo.mp4" --video-id "training/demo" --output data/ocr
```

命令显示本次输出目录，例如 `data/ocr/<本次运行ID>/`，内含：

- `frames/`：实际送入 OCR 的裁剪/放大后截图。
- `observations.jsonl`：每条识别的 video_id、采样目标 timestamp、frame_path、text、confidence、box。
- `manifest.json`：采样间隔和裁剪参数。OCR timestamp 用于选择相邻画面术语，不是口播对齐时间戳。

默认每 5 秒采样，固定间隔可能漏过短暂画面；可改为 2 秒，代价是更多计算和截图磁盘占用。当前版本没有自动场景切换检测。

OCR 对专有名词效果差时，先改善来源，再建词表：

1. 优先使用原始高分辨率录像，避免聊天软件再次压缩。放大不能恢复已丢失的笔画。
2. 在 `ocr.crop` 只保留 PPT/代码/软件窗口，移除人脸、小窗、字幕与菜单。坐标为归一化 `x0,y0,x1,y1`，例如 `[0.05,0.05,0.95,0.85]`。
3. `ocr.upscale` 默认 1.5，长边上限 `max_side: 2560`。高分辨率和裁剪会增加计算，不保证每个词都识别正确。
4. `ocr.min_confidence` 默认 0.7；被过滤的原始 OCR 仍保存在 observations，可降低阈值重新生成候选。`min_frames` 只影响证据强弱标记，不自动批准。
5. 如果有原始 PPT/PDF、产品词表或公司已审核术语表，直接核对其文字比从模糊画面猜拼写可靠；可手工维护 vocabulary.json 的已确认词条。不要让 LLM 凭背景常识猜产品名。

这些参数也可通过环境变量设置，例如 `$env:RAG_OCR__INTERVAL_SECONDS="2"`；嵌套配置遵循项目统一规则。

## 第二步：生成并审核候选词表

```powershell
python -m ocr.vocab candidates --input "data/ocr/本次运行ID/observations.jsonl" --output data/vocab_candidates.csv
```

多个视频可重复传 `--input`。CSV 已存在会拒绝覆盖，避免抹掉之前的人工审核结果。中文候选来自名词分词与短中文文本，英文候选保留缩写/数字/连字符；它只是保守的候选提取器，仍会出现普通词、漏掉长复合词，可人工添加词汇。

用本地 Excel 或文本编辑器打开 CSV：

| 列 | 操作 |
|---|---|
| ocr_term | OCR 提出的原始候选，保留供追溯 |
| canonical_term | 改为人工确认的准确拼写，例如 Qdranl → Qdrant |
| approved | 正确且适合作为术语的填写 `yes`，未确认保持空白 |
| scope | 默认 video，仅适用该视频；明确需要跨视频使用才改 global |
| video_id | video 范围必填；改 global 时清空此列 |
| priority | 同一作用域内优先级，越大越优先，范围 -100 到 100 |
| frame_count/max_confidence | 审核参考，不是正确性认证 |
| evidence_strength/similar_candidates | 重复出现程度与相近英文词提示 |
| timestamps_json/evidence_json | 保留原始取证时间与截图路径，通常不修改 |

导入确认结果：

```powershell
python -m ocr.vocab approve --review data/vocab_candidates.csv --output data/vocabulary.json
# 后续另一批审核结果可合并到同一词库：
python -m ocr.vocab approve --review data/next_review.csv --merge data/vocabulary.json --output data/vocabulary.json
```

仅 `approved=yes/true/1` 的词条进入词库；默认全部不确认。合并不删除已有词条，需停用时手工改 JSON 的 `approved:false`。不同视频同名词可分别保留时间范围。确认后的标准拼写与原始 OCR 证据分别保存，不用错字污染 prompt。

JSON 格式见 `examples/vocabulary.json`。人工词条无需伪造 OCR 证据：可以用 `evidence:[{"kind":"manual","note":"已核对的原始PPT或术语表"}]`，并填正确的作用域；整段视频适用的人工词条可以用空 `timestamps`。一般优先视频范围，少用巨大的全局词表。

## 第三步：带词汇提示重新转录

```powershell
python -m transcribe.run --video "D:/approved/demo.mp4" --video-id "training/demo" --vocabulary data/vocabulary.json --output data/assisted.jsonl
```

新增入口直接输出 RAG 所需的 `video_id/text/start/end`，以及独立 `.audit.json` 记录每个窗口实际选中的术语。原先 `workflow/kownledge/transcribe_meetings.py` 仍是旧入口；运行旧脚本不会自动启用新增词汇辅助。要使用本功能，使用上面的新命令。

Whisper 每次处理默认 30 秒音频。代码只选择当前视频中、距该窗口默认前后 30 秒内的已确认画面术语，再补充明确批准的全局词；最多 30 个词、180 个 Whisper tokenizer token。具体优先级可在 `asr` 配置中调整。

通过 `initial_prompt` 和 `carry_initial_prompt` 将这些词作为识别上下文，改善拼写偏好。它不是自定义声学词典或硬性约束，不保证识别正确；提示词还可能诱发额外术语，因此限制范围并做对照评估。代码不进行 ASR 结果字符串替换，也不把没有说出的 OCR 字词直接插入文本。

外部分窗有边界截断和上下文损失的取舍，边界处应抽查；本版本没有宣称实现重叠音频的精确词级合并。所有口播时间仍来自 Whisper 的 segment，不来自 OCR。

同一个 `video_id` 必须和 OCR、词库保持一致。`.audit.json` 的 `selected_terms` 若为空，表示没有词汇实际生效，应检查 approved、video_id、时间范围和 token 预算。

## 第四步：同段音频比较效果

```powershell
python -m transcribe.run --video "D:/approved/demo.mp4" --video-id "training/demo" --no-vocab --output data/baseline.jsonl
```

准备一小段人工校对参考 JSONL（同一个短视频/片段、同 video_id）后：

```powershell
python -m eval.run_asr_eval --reference data/reference.jsonl --baseline data/baseline.jsonl --assisted data/assisted.jsonl --vocabulary data/vocabulary.json
```

报告整体 CER（越低越好）、标准术语召回（越高越好），以及“识别出了但参考里没有的已知术语”。后者用于发现 prompt 造成的误插词，不能只看术语命中率。CER 忽略空白、标点并统一大小写，每个评估 video_id 限制 5000 字，适合先验证短样本。词汇匹配只按确认后的标准拼写，不把 OCR 错字算作正确。

比较使用同模型、同参数、同一音频，唯一差异是词表。未完成这一步前，不声称公司专有名词准确率已经提高。

## 第五步：进入现有 RAG

```powershell
python -m ingest.chunk --input data/assisted.jsonl
python -m ingest.enrich
python -m ingest.index
```

重新转录后需要重新分块建库，不能继续用旧 Chroma 或旧向量。多个视频的输出需要先合并成完整 JSONL 快照，再交给 ingest.index，避免只建入一个视频。

## 本地运行与依据

所有 OCR/截图/词库/转录默认在 data 下，均不提交 Git；它们可能包含内部屏幕内容，应使用和视频相同的本地权限。OCR 与新转录入口只接受本地文件，ffmpeg/ffprobe 限制为 file/pipe 协议。新 OCR/ASR 为可选原生 Python 上游，本项目默认 Docker API 镜像未打入这些可选依赖。

官方依据：[本地 Whisper transcribe 参数](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)、[RapidOCR 项目](https://github.com/RapidAI/RapidOCR)。适配器另依据固定 1.4.4 wheel 中实际接口与内置模型验证，避免将新版文档接口混用于旧包。
