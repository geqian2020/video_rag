# 用 HTML 页面提问和检索视频原文

网页通过同一个 FastAPI 服务的 `/search` 检索已建好的索引。无需 Qwen/Ollama，无需
重新转录、分块或建库。返回的是候选原文证据，不自动生成自然语言答案。

## 手动更新公司电脑

按目录结构更新以下文件：

| 文件 | 操作 |
|---|---|
| `start_search_ui.bat` | 放在代码仓库根目录或旁边的 `video_rag_data`，日常双击启动 |
| `serve/api.py` | 覆盖现有文件；保留原 `/search` 契约，新增根路径 HTML 页面 |
| `serve/search.html` | 新增页面，包含全部样式和 JavaScript，无外部 CDN |
| `serve/run.py` | 新增本地启动入口 |
| `serve/__init__.py` | 保留此空文件，使 serve 明确作为本地 Python 包 |

## Windows 双击启动（推荐）

`start_search_ui.bat` 支持两种位置：代码仓库根目录，或与代码目录同级的
`video_rag_data`。日常直接双击此文件，无需先打开 Git Bash、激活虚拟环境或输入命令。

放在 `video_rag_data` 时，启动器使用 `../video_rag-experiment-post-rerank-context` 的代码
与 `.venv/Scripts/python.exe`，索引明确指向 `.bat` 所在目录的 `index`。代码目录名称与
此部署约定一致；自定义名称时需要调整批处理中的 `APP_REPO`。放在代码根目录时，
使用当前目录代码及旁边的 `../video_rag_data/index`。

无论 `.bat` 放在哪里，`serve/run.py`、`serve/api.py`、`serve/search.html` 和
`serve/__init__.py` 都应位于代码仓库的 `serve/` 下，不应复制到数据目录。

批处理直接使用代码目录的 `.venv/Scripts/python.exe`，自动切换到代码目录，并明确传入
此布局下的索引位置。直接使用 `python -m serve.run` 时，仍沿用 Python 入口的配置、
环境变量和默认部署路径查找逻辑。
索引不存在时会报错，不自动建库或创建空索引。

启动器会先检查 `config.yaml`、`core/config.py` 和 `serve/` 下必要文件是否齐全，再用
绝对路径执行 `serve/run.py`。该入口显式加入仓库根目录到模块搜索路径，兼容 Python
启用安全路径、未自动包含当前目录的情况。若提示缺文件，完整解压更新包并保留目录
结构；不要把 `run.py`、`api.py`、`search.html` 平铺到仓库根目录。

服务和模型加载完成后，默认浏览器自动打开本机检索页。将终端显示的 Access key
填入页面即可查询；访问密钥的使用方式保持不变。可以将终端最小化，保持它运行。
若希望从桌面启动，请创建快捷方式，原 `.bat` 继续留在上述支持的位置。

此入口不安装软件、不下载模型、不调用 Qwen/Ollama。模型仍会在每次新的服务进程启动
时从硬盘加载一次，服务运行期间供多个查询复用。启动失败时窗口暂停显示错误。
浏览器被系统策略阻止打开时，终端会提供可手动访问的地址。

批处理默认使用 8000 端口，也可从命令行附加 `--port 8001` 或自定义索引参数。
它不注册 Windows 开机服务或计划任务；双击启动和开机自启是不同功能。

## 启动已完成的索引

在仓库根目录、已激活 `.venv` 的 Git Bash 中执行：

```bash
python -m serve.run --index-root "../video_rag_data/index"
```

上例适用于代码目录与 `video_rag_data` 同级的部署。
通用用法是 `python -m serve.run --index-root "实际包含 CURRENT 的索引根目录"`。
传根目录，不要传单个 generation 子目录。`--index-root` 会覆盖配置中的 `storage.root`，
防止误读仓库下另一套默认数据。模型配置仍须与建库时匹配。

启动器先检查 `CURRENT` 文件，设置访问密钥，再以单 worker 启动本机服务。它在已有
`RAG_API_KEY` 环境变量非空时复用该值，否则生成一个本次进程使用的随机密钥；密钥只在
本机终端打印，不写进 HTML，不附加到网页 URL，也不保存到源码。

等待终端出现 `Application startup complete`，然后在浏览器访问：

```text
http://127.0.0.1:8000/
```

将终端打印的 `Access key` 粘贴到页面左侧，输入问题并点击“检索原文”。
模型在 API 启动时加载，后续查询复用。启动期间先等待，不能把模型加载时间误认为
网页无响应。保持终端开启；`Ctrl+C` 停止服务，但不会删除硬盘中的索引。

若 8000 端口已占用，可使用 `--port 8001`，随后访问对应端口。不要为了重复启动而
终止不明进程。默认只绑定 `127.0.0.1`，供本机浏览器使用。

仍可沿用原来的 uvicorn 启动方式，只要正确设置 `RAG_API_KEY`、索引根目录和配置。
独立双击 HTML 使用 `file://` 无法正常调用该同源 API，应通过上面的 HTTP 地址打开。

## 页面能力

- 问题输入；支持 Ctrl/Command + Enter 提交。
- 返回 5、10、20 或 30 条结果。
- 可按完整 `video_id` 限定视频，多个 ID 用逗号或换行分隔。
- 显示匹配排名、视频 ID、起止时间、原文及重排/RRF 分数。
- 复制完整证据文本、导出 JSON、清空结果与密钥。
- 本地服务就绪提示、检索等待计时、空结果与错误提示。
- 桌面及手机宽度布局，无需前端构建工具或其他静态服务器。

查询使用带 `X-API-Key` 的同源 `POST /search`；不会把密钥写进浏览器 localStorage 或
sessionStorage。转录原文用 DOM `textContent` 渲染，不能执行其中的 HTML 或脚本。
网页不访问外部图片、字体、脚本或模型服务，未开启跨域访问。

分数不是概率，不能将排名第一理解为已经验证正确。时间显示到秒，接口保留原始浮点
秒值。页面展示视频 ID 与时间范围，没有凭空生成 SharePoint 播放链接，也不能根据
哈希 ID 自动定位本机视频路径。

## 常见问题

| 现象 | 检查位置 |
|---|---|
| `Index CURRENT not found` | `--index-root` 是否为实际索引根目录 |
| 模型/索引配置不一致 | 使用建库时的 embedding、revision、token 等配置 |
| 启动时无法加载模型 | 已下载的模型缓存及实际 Python 环境 |
| 页面 401 | Access key 是否与本次服务终端一致 |
| 页面 422 | 问题长度、返回数量、视频 ID 设置与模型 token 上限 |
| 页面 503 | 运行 API 的终端中的服务状态 |
| 没有结果 | 视频 ID 限制是否正确，尝试具体产品词或术语 |
| 更新索引后仍是旧结果 | 重启 API，加载 `CURRENT` 指向的新快照 |

本地 embedded Qdrant 同一快照不要同时被多个查询进程打开。网页启动后，不要再对相同
快照并行启动 CLI 查询或第二个 API。需要同时查看多个网页时，复用同一个服务即可。

## 组件关系

`serve/search.html` 负责交互和结果展示；`serve/api.py` 负责鉴权与响应结构；
`retrieve/search.py` 执行 Dense、BM25、融合和重排；Qdrant 提供已发布索引。
`serve/run.py` 仅负责指定索引、准备本机访问密钥和启动服务，不参与建库。
