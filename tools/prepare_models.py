"""仅在获准联网的准备机运行；不读取公司数据。已有本地目录时完全复用。"""
import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-download', action='store_true', help='显式允许从 Hugging Face 下载模型')
    p.add_argument('--embedding', default='BAAI/bge-m3')
    p.add_argument('--embedding-revision', default='main')
    p.add_argument('--reranker', default='BAAI/bge-reranker-v2-m3')
    p.add_argument('--reranker-revision', default='main')
    a = p.parse_args()
    if not a.allow_download:
        p.error('需要 --allow-download；工作机可直接使用已批准的本地权重，不必运行此脚本')
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    from core.io import write_json
    metadata = {}
    for kind, model, revision in [('embedding', a.embedding, a.embedding_revision),
                                  ('reranker', a.reranker, a.reranker_revision)]:
        if Path(model).is_dir():
            metadata[kind] = {'path': str(Path(model).resolve()), 'source': 'existing-directory'}
            continue
        # 先解析成不可变 commit SHA；模型正文不含业务数据。
        sha = HfApi().model_info(model, revision=revision).sha
        path = snapshot_download(model, revision=sha,
            ignore_patterns=['*.onnx', 'onnx/*', '*.bin', '*.h5', '*.msgpack', '*.ot'])
        metadata[kind] = {'id': model, 'revision': sha, 'cache_path': path}
        print(f'{kind}: {model} revision={sha}')
    write_json('models/download-manifest.json', metadata)
    print('请依据 models/download-manifest.json 将模型 revision 写入 config.yaml；离线传输整个缓存或完整模型目录')


if __name__ == '__main__':
    main()
