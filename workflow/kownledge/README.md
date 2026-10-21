# 会议视频转录工作流

递归扫描当前目录及所有子目录中的 `.mp4` 文件，使用 `ffmpeg` 提取音频，
调用本地 Whisper 模型进行中文语音识别，并在每个视频旁生成同名的
`.md` 转录文件。

## 1. 安装 ffmpeg

Windows 上推荐使用 winget 安装：

```powershell
winget install --id Gyan.FFmpeg -e --source winget
```

安装完成后，**必须重新打开一个新的终端窗口**（或重新登录），否则当前
终端仍会提示找不到 `ffmpeg`，因为 PATH 环境变量只在新启动的进程中生效。

验证安装：

```powershell
ffmpeg -version
```

## 2. 创建 Python 虚拟环境

项目要求 Python 3.9+（本项目在 Python 3.13 上验证通过）。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. 安装项目依赖

```powershell
pip install -r requirements.txt
```

### 依赖选型说明

本项目使用 **openai-whisper**（PyTorch 后端），而不是 faster-whisper
（CTranslate2 后端）。原因：

- faster-whisper 依赖 `av>=11`（PyAV），但截至本项目验证时，PyAV 尚未
  为 Python 3.13 + Windows 发布预编译 wheel，安装会触发源码编译，
  需要额外的编译器和 ffmpeg 开发库，失败风险高。
- openai-whisper 的完整依赖链（torch、numba、llvmlite、tiktoken、
  numpy）均已在 PyPI 上为 Python 3.13 + Windows 发布官方预编译 wheel，
  可以直接 `pip install` 成功，无需编译。

如果你使用的是 Python 3.11 或更早版本，且更看重 CPU 推理速度，可以
自行切换到 `faster-whisper`（届时 PyAV 大概率已有对应 wheel）。

## 4. 下载 / 配置中文语音识别模型

无需手动下载，首次运行时 `whisper.load_model()` 会自动从
`https://openaipublic.azureedge.net` 下载模型权重并缓存到
`~/.cache/whisper`（Windows 下为 `%USERPROFILE%\.cache\whisper`）。

模型选择建议（均支持中文，体积与速度的取舍）：

| 模型   | 参数量 | 相对速度 | 说明                     |
|--------|--------|----------|--------------------------|
| tiny   | 39M    | 最快     | 精度较低，仅适合快速预览 |
| base   | 74M    | 快       | 精度一般                |
| small  | 244M   | 中等     | **默认选择**，CPU 上精度与速度较均衡 |
| medium | 769M   | 慢       | 精度更高，CPU 上耗时明显增加 |
| large  | 1550M  | 很慢     | 精度最高，CPU 上不建议用于长会议 |

默认使用 `small`。可通过 `--model` 参数切换，例如：

```powershell
python transcribe_meetings.py --model medium
```

## 5. 运行工作流

在项目根目录下执行：

```powershell
python transcribe_meetings.py
```

常用参数：

| 参数              | 说明                                               |
|-------------------|----------------------------------------------------|
| `--root DIR`      | 指定扫描的根目录，默认当前目录                       |
| `--model NAME`    | Whisper 模型名称，默认 `small`                       |
| `--chunk-seconds N` | 音频分片长度（秒），默认 600，用于避免长会议一次性占用过多内存 |
| `--language CODE` | 转录语言代码，默认 `zh`（中文）                      |
| `--force`         | 忽略已存在的 `.md` 文件，强制重新转录并覆盖           |
| `--only STR`      | 仅处理路径中包含指定子字符串的文件，用于单文件测试     |

## 6. 重新处理已生成 Markdown 的视频

默认情况下，如果视频同名的 `.md` 文件已存在，会被跳过。如果需要强制
重新转录并覆盖已有文件：

```powershell
python transcribe_meetings.py --force
```

也可以先手动删除对应的 `.md` 文件，再正常运行脚本。

## 7. 查看失败日志

运行过程中的所有日志（包括每个文件的处理结果、失败原因）会：

1. 实时打印到终端；
2. 同时写入根目录下的 `transcribe_log.txt` 文件。

处理结束后终端和日志文件末尾都会输出统计信息：

```
处理完成统计:
  找到: N
  成功: N
  跳过: N
  失败: N
失败详情:
  - <文件路径>: <失败原因>
```

## 8. CPU 与 GPU 模式的区别

本工作流**当前固定使用 CPU 模式**（脚本中显式指定
`whisper.load_model(model_name, device="cpu")`），原因是本机未检测到
NVIDIA GPU（`nvidia-smi` 不存在）。

- **CPU 模式**：无需额外安装 CUDA / cuDNN，兼容性最好，但转录速度较慢，
  长会议（数小时）使用 `small` 模型可能需要数十分钟到数小时不等。
- **GPU 模式**：若未来更换到带 NVIDIA GPU 的机器，可安装支持 CUDA 的
  torch 版本，并将脚本中的 `device="cpu"` 改为 `device="cuda"`，速度
  通常可提升 5-10 倍以上。本项目未启用此模式，也未安装任何 CUDA 相关
  组件。

## 输出格式

每个视频生成的 `.md` 文件结构：

```markdown
# 会议转录：{视频文件名}

* 原始文件：`{视频相对路径}`
* 转录语言：中文
* 生成时间：{生成时间}

## 转录内容

[HH:MM:SS] 转录文本...
[HH:MM:SS] 转录文本...
```

无法识别的片段会标记为 `[无法识别]`，不会中断整体处理流程。

## 临时文件

转录过程中会在系统临时目录创建临时的 WAV 音频分片文件，每个分片处理
完成后立即删除；整个视频处理完毕后，其对应的临时目录也会自动清理。
不会在项目目录中残留任何临时音频文件。

## 中文文件名与特殊字符

脚本全程使用 `pathlib.Path` 处理文件路径，并在调用 `ffmpeg`/`ffprobe`
时以参数列表（而非字符串拼接）方式传递路径，因此可以正确处理包含
中文、空格、`&` 等特殊字符的文件名。
