# Ollama 自动启动

`python -m ingest.enrich` 现在会先检查配置地址的 Ollama 服务。现有 V7 脚本调用这个
入口，因此无须修改 `.sh`，也无须每次手动执行 `ollama serve`。

## 执行顺序

1. 请求 `enrichment.url` 的 `/api/version`，超时为 2 秒。
2. 服务已经运行时直接复用，不重启、不修改已有进程的配置。
3. 仅当连接被明确拒绝，且地址为本机 HTTP 回环地址时，查找 PATH 中已有的 `ollama`。
4. 后台启动 `ollama serve`，为新进程设置 `OLLAMA_NO_CLOUD=1`，将 `OLLAMA_HOST`
   绑定到配置中的本地地址和端口。保留已有 `OLLAMA_MODELS` 等环境设置。
5. 最多等待约 30 秒；服务就绪后，继续检查 `/api/show` 中指定模型是否存在、是否指向远程。
6. 服务与模型检查通过后才开始发送块原文进行增强。

服务进程的标准输入关闭、日志写入文件。Windows 使用 detached process，其他平台使用
独立 session；本次增强结束后服务继续运行，下一次复用。仅结束增强进程不会主动停止服务。

## 首次准备与更新

仍需有已安装的 Ollama 和准备好的本地模型。程序不自动安装软件、不调用 `pull`，也不
用联网替代缺失模型。未在 PATH 中找到 Ollama 时，检查已有安装位置，不要重复安装。

手动更新公司电脑：新增 `core/ollama.py`，覆盖 `ingest/enrich.py`。保持其余配置不变，
直接运行原 V7 脚本。已经完成分块时，也可以直接运行增强入口：

```bash
python -m ingest.enrich --input ../video_rag_data/chunks/chunks.jsonl \
  --output ../video_rag_data/enriched/enriched.jsonl
```

上面的相对路径要求代码目录与 `video_rag_data` 同级；其他部署使用实际路径。
无需重新转录或分块。

## 日志和配置边界

- 自动启动的服务日志保存在增强输出目录的 `ollama-server.log`。V7 通常为
  `DATA_IN/enriched/ollama-server.log`。
- 控制台会输出复用、启动、等待、就绪和模型检查信息。
- 配置为远程内网服务或 HTTPS 地址时，不自动启动本地服务；已有服务仍可按配置访问。
- 初次检查遇到超时、证书错误、权限错误、非 Ollama HTTP 服务等情况，不把它们当作
  “服务没启动”，不尝试接管或重启。
- 已运行服务的环境变量不会被客户端改变。它的本地模式仍需由原启动方式设置；
  程序保留原有本地模型检查，不自动重启其他应用在用的服务。
- 启动超时会报错并提供日志位置，不强杀仍可能启动中的进程。
- 可选参数 `--no-auto-start-ollama` 显式禁止自动启动。

这项修改仅处理服务生命周期，尚未给 `ingest.enrich` 增加逐块检查点。增强结果仍在全部
完成后原子写入；不能把自动启动等同于增强恢复功能。

## 验证

```bash
python -m unittest discover -s tests -p test_ollama_startup.py -v
```

测试覆盖现有服务复用、连接拒绝后的启动与等待、二进制缺失、禁用启动、自定义端口、
IPv6、非本机/HTTPS 限制、非拒绝连接错误、错误 HTTP 服务、启动退出和超时，以及
新进程环境与 Windows detached 参数。增强入口集成测试验证先检查服务和模型，再发送
原文；启动失败不会覆盖已有增强输出。

测试使用模拟 HTTP 服务和进程，不安装或启动真实 Ollama；Windows 公司电脑仍需实测。
