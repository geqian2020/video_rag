# 建库进度与中断恢复

`ingest.index` 在完整快照建库中增加批次日志、完整块向量缓存和未发布快照恢复。
原来的 V7 调用和命令行参数不变，不需要修改 `.sh`。

## 日志

程序分别报告读取输入、模型加载、解析模型身份、BM25 统计、打开 Qdrant、各批次的
检查点校验、编码/复用向量、写入与读回校验、最终计数、元数据写入、发布 `CURRENT`。

每批打印序号、块范围、完成百分比、已复用块数和本次运行耗时。编码、写入等阶段较长时，
每 15 秒打印 `still running`；这只是等待提示，不声称计算已经前进。

示例（数字为格式示意）：

```text
[INDEX] Resume unpublished snapshot: generation=..., blocks=7124
[INDEX] Batch 100/891, blocks=793-800/7124: verified reuse, completed=11.2%
[INDEX] Batch 101/891, blocks=801-808/7124: encode/load saved vectors: start
[INDEX] Embedding checkpoint: sentences=8, unique=8, reused=0, pending=8, invalid=0
[INDEX] Batch 101/891, blocks=801-808/7124: saved, completed=11.3%, ...
```

## 保存位置

所有路径均相对于实际生效的 `storage.root`；不是相对于转录目录。

| 路径 | 内容 |
|---|---|
| `index_embeddings.sqlite3` | 完整块原文的向量，逐推理批次事务提交 |
| `builds/<任务哈希>.json` | 输入、配置、模型身份与未发布 generation 的对应关系 |
| `<generation>/checkpoints/batch-00000001.json` 等 | 批次输入哈希及 Qdrant 记录读回摘要 |
| `<generation>/qdrant/` | embedded 模式的当前工作库 |
| `<generation>/manifest.json`、`chunks.jsonl` | 完整构建通过校验后的快照元数据和原文 |
| `CURRENT` | 最终发布的可查询 generation |

服务端 Qdrant 模式使用相应 collection，批次记录和向量缓存仍保存在本地。它们包含
业务派生数据，应与转录和索引一样留在批准的数据目录，不提交 Git。

## 恢复规则

1. 使用输入完整内容及顺序、模型配置、实际模型身份、维度、批量大小、BM25 设置和
   分词器版本、Qdrant 地址与 collection 前缀，识别同一构建任务。
2. 同一任务若有未发布 generation，则继续该 generation，不删除已完成 collection。
3. 有批次记录时，从 Qdrant 读回全部预期点，检查 ID、原文 payload 与向量摘要。
4. 通过校验的批次直接跳过；丢点、记录损坏或未保存批次记录时，从块向量缓存补写。
5. 向量在写库之前先保存。所以即使 Qdrant 写入失败，已提交的向量仍可避免重新编码。
6. 只有确实缺失或缓存损坏的文本才需要重新推理；中断时未保存的最后一个推理批次可能重算。
7. 全部批次完成，点数检查、manifest 和原文文件写入成功并关闭 Qdrant 后，才原子切换
   `CURRENT`。中断前正在使用的旧索引保持有效。

Qdrant 的 cosine 向量可能在重启前后呈现不同归一化表示。批次摘要对 Dense 方向归一化，
按小数点后六位规范化，容忍 float32 数值差异；payload 和 BM25 内容仍参与摘要。
这是数值一致性检查，不是对原始向量字节逐位相等的承诺。

修改输入、批量大小或索引参数会建立新任务和新快照；同模型且内容相同的块仍可复用
向量缓存。模型或其身份变化时不会混用旧向量。旧未发布任务保留，不自动删除。

已发布的快照不会原地修改。再次运行完整建库会创建新的快照，并可复用保存的块向量；
这不等于增量更新已发布 collection。API 仍需在发布后重启加载新快照。

自定义注入的 embedder 在未提供可验证身份时，不跨调用恢复，避免错误复用。
Python 调用方可通过 `build_index(..., model_identity=...)` 显式提供真实、稳定的身份；
正常 CLI 会自动为项目 Embedder 解析模型身份，无须用户填写。

## 更新公司电脑

覆盖 `ingest/index.py`，确保已有 `ingest/embedding_cache.py`。新建库默认开启恢复，
原命令与配置保持不变。恢复时必须使用相同有效配置及相同索引根目录。

已运行的旧 Python 进程不会热更新。旧版没有保存完整块向量检查点，也没有任务对应
记录，不能安全推断它的未发布 collection 属于哪份输入，因此不会自动接管旧版半成品。
新检查点从更新后的首次运行开始积累；不会删除旧转录、分块、音轨或已有索引。

## 验证

```bash
python -m unittest discover -s tests -p test_index_resume.py -v
```

使用真实本地 Qdrant 和确定性模型替身，覆盖编码失败、强制进程退出、写库后未保存批次
标记、缺失记录、篡改 payload/向量、损坏批次标记、输入/模型变化、保留旧发布快照、
发布失败后恢复和额外点阻止发布。真实 BGE-M3、Windows/OneDrive 和 Qdrant 服务端
环境仍需在部署机验收。
