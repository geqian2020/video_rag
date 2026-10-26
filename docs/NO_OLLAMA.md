# 公司电脑无法安装 Ollama：使用原文模式

当前公司部署的 `config.yaml` 默认设置 `enrichment.mode: extractive`。在此模式下，
`ingest.enrich` 不创建 HTTP 客户端，不检查/启动 Ollama，不加载生成模型，不生成问题。
输出仍兼容现有索引与 API，可以直接完成本地检索主流程。

## 能力变化

| 内容 | 原文模式 |
|---|---|
| 转录原文、来源片段、ID、时间戳 | 完整保留 |
| 摘要 | 逐字摘录开头一句；无句界时保留对应原文摘录 |
| 生成问题 | 空数组 `questions: []` |
| 状态 | 显式标为 `extractive_fallback`，不伪装成模型增强成功 |
| Dense | BGE-M3 编码块原文，与原有逻辑相同 |
| BM25 | 仅块原文；缺少生成问题带来的额外关键词覆盖 |
| RRF、Cross-Encoder、API | 继续使用现有实现 |

不增加外部事实、不把摘要或生成问题写入 Dense。检索效果与带 doc2query 的模式可能不同，
需要独立评测，不能直接沿用其他模式的成绩。仍需现有 embedding 和 reranker 权重，
无需 Qwen 生成模型或 Ollama。

## 手动更新公司电脑

1. 覆盖 `core/config.py` 和 `ingest/enrich.py`。
2. 在本地 `config.yaml` 已有的 `enrichment:` 下新增一行，保留其余本地配置：

```yaml
enrichment:
  mode: extractive
```

这是局部配置片段，不要用这两行替换整个文件。已有配置未设置 `mode` 时保留历史行为
`ollama`，因此需要明确设置此项。仓库随附配置已经选择 `extractive`。

不需要更新或调用 `core/ollama.py`。外层 `.sh` 仍调用同一增强入口，无须修改命令。
也可以使用环境变量 `RAG_ENRICHMENT__MODE=extractive`；不建议仅依赖临时终端变量
代替长期部署配置。

## 已完成分块后继续

在代码仓库目录执行：

```bash
python -m ingest.enrich --mode extractive \
  --input ../video_rag_data/chunks/chunks.jsonl \
  --output ../video_rag_data/enriched/enriched.jsonl
```

相对路径要求数据目录与代码目录同级。CLI 的 `--mode` 优先于配置中的模式。
这一步无需重新转录或分块。完成后可运行 `ingest.index`，但单独手动建库时要确保
`storage.root` 与 V7 及后续 API 使用相同位置；V7 入口可能通过环境变量设置该路径。
不要在不知道有效索引路径时另外创建一套默认索引。

程序每处理 500 块打印进度，最后原子发布完整输出。原文模式没有生成推理过程，仍然
全量做轻量摘录；它不新增增强阶段的逐块检查点。源块校验失败时不会覆盖已有输出。

## 恢复可选生成增强

若以后存在允许使用的本地 Ollama 服务，显式改为 `mode: ollama`，即可使用原有服务
自动启动及生成问题流程。`on_failure: extractive` 是另一设置：它仅控制 Ollama 推理
失败后的降级，不能代替 `mode: extractive`，也不能跳过服务检查。

## 验证

```bash
python -m unittest discover -s tests -p test_extractive_mode.py -v
```

测试验证 HTTP 与服务启动完全未调用、原文来源保持不变、配置/CLI/环境变量模式选择、
原子输出保护，以及真实本地 Qdrant 接受摘录结果并可读回。嵌入计算使用测试替身，
不代表真实模型检索质量或公司电脑已经验收。
