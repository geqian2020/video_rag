# 公司 agent 出题提示词（每次只提供一个视频）

输入 video_id、字幕/转录，每行 [start-end] 原文。只在已批准环境处理公司资料。

请生成 12-16 道可回答问题，另加 3 道相关但当前语料没有答案的问题。

- 可回答题只能凭 1-3 条相邻转录片段回答，不依赖外部知识。题型混合 factual/howto/why/compare。
- 不照抄问题措辞，至少三分之一使用口语、缩写或同义表达；不要看到 ASR 错词就跟着错写标准术语。
- evidence 逐字摘录，不能伪造。多段可用 evidence 数组列出各自逐字摘录，限相邻来源。
- 不可回答题 type=unanswerable，evidence=[]，answer=""；相关性不能取代人工确认。应检查整个检索语料而非只看单视频。
- 所有题最初 reviewed=false。不要把生成答案当作已验证事实。
- 输出 JSONL，不加 Markdown。每行：

```json
{"video_id":"视频ID","query":"换一种说法的问题","answer":"仅由证据支持的参考答案","evidence":["逐字摘录"],"type":"factual","reviewed":false}
```

收到稿件后，人工核对问题、答案、证据与语料范围，将合格题的 reviewed 改 true。再加入独立手写真实问题，特别检查标准术语/缩写。不要把整份评测集当作提示词喂给 enrichment。
