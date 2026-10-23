# 验证记录

## OCR/ASR 升级阶段（历史记录）

- 该阶段验证环境：Linux x86_64、Python 3.12.14；未安装真实 Whisper、embedding、reranker 权重或 Ollama 服务。OCR 使用真实本地 RapidOCR 模型做了独立验证。
- `python -m pytest -q`：**34 passed**，约 1.85 秒。Starlette 对其测试客户端使用 httpx 发出一条弃用提醒，不影响测试通过。
- 编译、CLI --help、示例配置、转录与 gold 格式校验通过。
- 原 RAG 使用真实 Qdrant embedded 完成 dense/sparse 写入、过滤、查询、替换索引、建库失败保护；语义模型为显式测试替身。
- Ollama 请求使用 httpx MockTransport 验证 JSON schema、关闭思考参数、增强输出和失败策略；没有调用真实 Ollama 推理。
- CrossEncoder 包装层验证排序与超长拒绝；上次另检查 sentence-transformers 5.7.0 wheel 中两个模型构造器支持使用的参数。

## 本次 OCR/ASR 测试

- 新增 13 项测试：重复帧不自动批准、同一帧去重、低置信度与非法值、相近词不合并、CSV 改正拼写与确认、视频/时间范围与 token 预算、配置校验、缺失 Whisper 权重禁止下载、本地权重复用、Whisper prompt 参数与原始输出/时间戳、失败保留已有输出、CER/术语误插入、真实 ffmpeg 裁剪抽帧。
- 真实 OCR 验证：用 Pillow 在本地生成清晰英文测试图 `Qdrant BM25 RadioTest`，调用 rapidocr-onnxruntime 1.4.4 包内 ONNX 模型，返回正确文本，模型分数约 0.993。此分数不是经校准的概率，也不代表模糊公司视频的表现。
- 检查了 rapidocr-onnxruntime 1.4.4 wheel 的构造器与三个内置 ONNX 文件，以及 openai-whisper 20250625 源码的 initial_prompt/carry_initial_prompt 接口和 prompt token 处理。
- 真实 Whisper 推理尚未执行；ASR 对接测试注入测试替身。没有声称专有名词准确率已经提高。需要目标音频与人工参考做 baseline/assisted 对照。

当前实际安装的关键测试依赖：qdrant-client 1.19.1、jieba 0.42.1、fastapi 0.159.1、httpx 0.28.1、pydantic 2.14.1、numpy 2.5.4、pytest 9.1.1、rapidocr-onnxruntime 1.4.4、onnxruntime 1.31.0、opencv-python 5.0.0.93。requirements 采用兼容范围；固定 OCR 适配器版本，不提供假定跨平台通用的 lock。

尚未执行：真实 RAG/Whisper/Ollama 全链路、公司模糊画面的 OCR 质量、GPU、Windows、Docker build/compose、真实检索指标、公司工作台联调。部署前使用批准的完整权重跑 README 与 OCR_ASSISTED_ASR.md 的流程，记录准确依赖版本与模型 SHA。

交付只包含通用源码、文档与虚构样例，不包含本地截图、词库、业务转录、旧 Chroma 数据库、下载的模型或测试环境；本次源码通过 GitHub 提交发布，不改写已有历史；发布结果以 main 提交记录为准。

## 最新：公开语料测试升级

新增 test/ 下载/字幕转换/CPU 基线和 eval.make_gold；扩展 eval.run_eval 的无答案、Hit、分组与逐题诊断。当前回归测试 **41 passed**（两条依赖弃用提醒）。另实际运行 4 个公开视频的完整字幕、624 块、60 道草稿题的真实轻量 embedding + cross-encoder 检索，详情见 ../test/CLOUD_RUN.md。这项真实推理区别于早期测试替身；默认生产模型与 OCR/Whisper 验收边界不变。

## 2026-10-23 恢复后验证

实际重跑：57 passed，约 3.32 秒，一条上游测试客户端弃用提醒。包含 Qdrant/ffmpeg 与模型测试替身。没有重新跑真实 BGE、Whisper 或完整五视频 80 题；历史数字见 test/OPTIMIZATION_RUN.md，原始扩充明细未完整恢复。
