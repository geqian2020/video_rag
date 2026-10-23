# 评估规则与验收计划

## 数据与标签

目标 150–200 道人工审核题，包含独立手写问题、标准术语/缩写及全语料无答案题。已有 80 题属于已曝光草稿，独立人工审核为 0，不得再次当作未见测试。按视频和相似问题家族隔离 dev / validation / test；跨视频等价来源也须一起分组。

六道旧失败见 [逐题核对](../test/FAILURE_REVIEW.md)。先人工精简为能完整支持答案的 1–3 句，补等价来源，消除 this example 等歧义；旧成绩与旧 gold 不回填。

`evidence_policy=all`（默认）表示多段证据缺一不可；`alternatives` 表示任一位置即可完整回答。Hit 表示至少命中一处；Recall 按策略计分。时间命中保持同视频且覆盖标注时间至少 50%，不放宽规则追分。复杂的“多组必需证据且每组又有多个来源”暂不支持。

跨来源出题稿示例（虚构内容）：

```json
{"video_id":"fiction_a","query":"如何启动示例？","answer":"调用 start。","type":"howto","reviewed":false,"split":"dev","evidence_policy":"alternatives","evidence":[{"video_id":"fiction_a","quote":"调用 start 启动。"},{"video_id":"fiction_b","quote":"运行 start 即可启动。"}]}
```

## 冻结与配对

`consumed.jsonl` 每行 `{"video_id":"已看过结果的视频"}`；须完整记录已曝光来源。人工审核所有题后：

```bash
python -m eval.freeze_gold --gold data/evaluation/gold.v2.jsonl \
  --consumed data/evaluation/consumed.jsonl --output data/evaluation/lock.v2.json
python -m eval.run_eval --gold data/evaluation/gold.v2.jsonl --split dev \
  --lock data/evaluation/lock.v2.json --output data/evaluation/report.json \
  --details data/evaluation/details.jsonl
python -m eval.statistics --before data/evaluation/before.jsonl \
  --after data/evaluation/after.jsonl --output data/evaluation/paired.json
```

冻结器要求至少 150 题、reviewed=true、显式划分、视频不跨划分、test 视频未曝光，记录 gold 哈希并拒绝覆盖旧锁。评估前 `--lock` 校验哈希；开发时只选 dev，锁定模型/阈值后才选 test。工具不能验证人工身份或题目语义质量。

Hit 使用 Wilson 95% 区间，Recall/MRR 使用逐题 bootstrap；配对工具校验完整问题与标签，拒绝重复、缺题或改标签，不静默取交集。区间不覆盖标签错误、视频内相关性和反复调参偏差，视频过少时不能声称稳定总体收益。

## 实验顺序与目标

1. 修标签后固定分块、候选上限和预算。模型、标题、术语查询扩展、候选扩容、去重分别比较，一次只改一处。
2. 原小块 Recall@30、扩展证据覆盖、最终 Hit@5 与 MRR 分开报告，同时记录上下文长度和延迟。补邻句不等于原向量召回提升。
3. validation 标定拒绝阈值，锁定后测 test；报告误接受与错误拒绝。不同重排器分数不能直接共用阈值。即使 16 道负题零误接受，Wilson 95% 上限仍约 19.4%，不能证明误接受率低于 5%。
4. 按工作台实际 LLM tokenizer 核验系统提示、问题、top-5 与输出的总预算；embedding tokenizer 不能替代 LLM tokenizer。
5. 最后由批准的工作台模型检验答案正确、证据忠实及无答案拒答。

用户目标：原小块 Recall@30 ≥95%。暂定工程建议：最终 Hit@5 ≥90%、端到端正确率 ≥90%、无答案误接受率 ≤5%；这些尚非已达成的正式验收指标。

## OCR / Whisper

I/l/1、O/0 仅分组待审，`ocr.vocab candidates --vocabulary` 给已有确认拼写建议，不自动批准。截图与识别原文保留，不凭拼写相似或 ASR 初稿直接认定发音正确。

同时统计 CER、术语种类召回、出现次数召回、参考外种类及超出次数。次数只是片段级误插入代理，不能定位具体句子的词是否用对。可用 baseline 错词样本调参，仍须用未按 baseline 选出的代表性片段测试。

公司中英混合原视频只在批准的本地环境测试。英语公开课与受控图/TTS 只能补流程验证，不能代替公司数据。
