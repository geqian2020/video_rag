#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_text.py

对 Whisper 转录文本做三步清洗，供 build_chunks.py 在切片前调用：

    1. 简繁体统一（繁体 -> 简体，使用 OpenCC）
    2. 术语修正替换（term_corrections.json 中列出的真正 ASR 错误，直接替换文本）
    3. 俚语/产品代号标注（glossary.json 中列出的内部俚语，不修改文本，
       只返回该片段命中了哪些俚语词条，供上层写入 chunk 的 jargon_terms 字段）

设计原则：
    - 俚语表（glossary.json）里的词条绝不替换原文——它们是说话人确实说出的词，
      只是在内部语境里另有含义（例如产品代号）。替换会篡改转录的真实性。
    - 术语修正表（term_corrections.json）里的词条是真正的语音识别错误，
      与说话人实际想表达的内容不一致，所以直接替换。
    - 两张配置表都设计为可持续增量维护：发现新的错误就加进 term_corrections.json，
      发现新的内部俚语就加进 glossary.json，不要求一次性列全。

用法（作为库被 build_chunks.py 导入）：
    from clean_text import TextCleaner
    cleaner = TextCleaner(config_dir=Path("video_kb/config"))
    cleaned_text, jargon_terms = cleaner.clean(raw_text)

也可以直接运行做快速自测：
    python clean_text.py
"""

import json
import re
import sys
from pathlib import Path

# Windows 终端默认代码页（如 cp1252/cp936）无法打印部分中文字符，
# 这里强制将 stdout/stderr 重新配置为 UTF-8，避免直接运行本脚本时崩溃。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from opencc import OpenCC
except ImportError:
    print(
        "错误: 未安装 opencc-python-reimplemented。"
        "请先运行: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class TextCleaner:
    """封装简繁转换、术语修正、俚语标注三步清洗逻辑。"""

    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR):
        self.config_dir = Path(config_dir)
        self._converter = OpenCC("t2s")  # 繁体 -> 简体
        self._corrections = self._load_corrections()
        self._glossary = self._load_glossary()

    def _load_corrections(self):
        path = self.config_dir / "term_corrections.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        corrections = []
        for item in data.get("corrections", []):
            pattern = item["pattern"]
            replacement = item["replacement"]
            match_type = item.get("match_type", "literal")
            if match_type == "case_insensitive_word":
                # \b 词边界在中英混排文本里对纯 ASCII 词仍然有效
                regex = re.compile(
                    r"(?<![A-Za-z0-9])" + re.escape(pattern) + r"(?![A-Za-z0-9])",
                    re.IGNORECASE,
                )
                corrections.append(("regex", regex, replacement))
            else:
                corrections.append(("literal", pattern, replacement))
        return corrections

    def _load_glossary(self):
        path = self.config_dir / "glossary.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        terms = []
        for item in data.get("terms", []):
            term = item["term"]
            aliases = item.get("aliases", [])
            terms.append((term, [term] + aliases))
        return terms

    def to_simplified(self, text: str) -> str:
        """繁体转简体。对已经是简体或纯英文的文本是无操作的（OpenCC 保持不变）。"""
        return self._converter.convert(text)

    def apply_corrections(self, text: str) -> str:
        """应用术语修正表，替换真正的 ASR 识别错误。"""
        for kind, pattern, replacement in self._corrections:
            if kind == "literal":
                text = text.replace(pattern, replacement)
            else:
                text = pattern.sub(replacement, text)
        return text

    def find_jargon_terms(self, text: str):
        """检测文本中出现了哪些俚语/产品代号词条，返回命中的标准词条名列表（不修改文本）。"""
        found = []
        for term, variants in self._glossary:
            for variant in variants:
                if variant in text:
                    found.append(term)
                    break
        return found

    def clean(self, text: str):
        """
        执行完整清洗流程，返回 (cleaned_text, jargon_terms)。

        注意：俚语检测在简体转换 + 术语修正之后进行，确保 glossary.json
        中登记的词条（应为简体）能够正确匹配。
        """
        text = self.to_simplified(text)
        text = self.apply_corrections(text)
        jargon_terms = self.find_jargon_terms(text)
        return text, jargon_terms


def _self_test(out_path: Path = None):
    """自测函数。若提供 out_path，结果以 UTF-8 写入该文件，避免 Windows 终端
    默认代码页（cp1252/936 等）无法显示中文导致的乱码或崩溃；否则打印到 stdout。"""
    cleaner = TextCleaner()
    samples = [
        "我们教准TX教准这边过一下",
        "我们wisb现在用fab以前用et software",
        "一份呢是伦敦的，一份是Whisper",
        "維傑裡面的記憶是什麼意思",  # 繁体样本
    ]
    lines = []
    for s in samples:
        cleaned, jargon = cleaner.clean(s)
        lines.append(f"原文: {s}")
        lines.append(f"清洗后: {cleaned}")
        lines.append(f"命中俚语: {jargon}")
        lines.append("-" * 40)
    report = "\n".join(lines)
    if out_path:
        Path(out_path).write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _self_test(out_path=sys.argv[1])
    else:
        _self_test()
