"""真实 CPU 英文小模型基线；不将该结果冒充默认 BGE-M3/Ollama/OCR 效果。"""
import argparse
from pathlib import Path
import json
import time

import torch

from core.config import load_config
from core.io import read_rows,write_json,write_rows
from core.models import Embedder
from core.sparse import vector
from eval.run_eval import validate_gold,make_record,report,metrics
from ingest.chunk import build_chunks
from ingest.index import build_index
from retrieve.rerank import Reranker
from retrieve.search import Searcher,rrf


def payloads(points):
    return [{**p.payload,'retrieval_score':float(p.score)} for p in points]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='test/config.cpu.yaml')
    p.add_argument('--root',default='data/public_benchmark')
    p.add_argument('--gold',default='data/public_benchmark/gold.jsonl')
    p.add_argument('--build',action='store_true',help='仅首次或语料/配置变化时构建')
    a=p.parse_args();cfg=load_config(a.config);root=Path(a.root)
    torch.set_num_threads(4)
    embedding=Embedder(cfg)
    if a.build:
        chunks=build_chunks(read_rows(root/'transcripts.jsonl'),embedding,cfg)
        for c in chunks:
            c.update(summary=c['text'].split('\n')[0],questions=[],enrichment_status='extractive_fallback')
        write_rows(root/'chunks.jsonl',chunks)
        build_index(chunks,cfg,embedding)
        print(f'已建立真实向量/稀疏索引，{len(chunks)} 块；未启用生成问题增强',flush=True)
    rows=validate_gold(read_rows(a.gold));reranker=Reranker(cfg);searcher=Searcher(cfg,embedding,reranker)
    records=[];ablations={mode:[] for mode in ['dense','bm25','hybrid']};started=time.monotonic()
    try:
        for i,row in enumerate(rows,1):
            # 全部视频参与竞争，禁止用 gold 的 video_id 偷做检索过滤。
            dense=searcher.client.query_points(collection_name=searcher.collection,using='dense',
                query=embedding.encode([row['query']],query=True)[0].tolist(),limit=30,with_payload=True).points
            q=vector(row['query'],searcher.manifest['bm25'],query=True)
            sparse=searcher.client.query_points(collection_name=searcher.collection,using='bm25',query=q,
                limit=30,with_payload=True).points if q.indices else []
            fused=rrf([dense,sparse],cfg.retrieval.rrf_k,30)
            results=reranker.rerank(row['query'],fused,5)
            records.append(make_record(row,fused,results))
            if row['relevant']:
                for name,hits in [('dense',payloads(dense)),('bm25',payloads(sparse)),('hybrid',fused)]:
                    ablations[name].append({'at30':metrics(hits,row['relevant'],30),'at5':metrics(hits,row['relevant'],5)})
            print(f'已评估 {i}/{len(rows)}',flush=True)
    finally:searcher.close()
    result=report(records)
    result['ablations']={name:{stage:{metric:sum(r[stage][metric] for r in vals)/len(vals)
        for metric in ['recall','mrr','hit']} for stage in ['at30','at5']} for name,vals in ablations.items() if vals}
    result['elapsed_seconds']=time.monotonic()-started
    result['conditions']={'embedding':cfg.models.embedding,'reranker':cfg.models.reranker,
        'enrichment':'off; extractive_fallback','corpus':'downloaded English captions, all videos searched',
        'labels':'agent_draft unless explicitly reviewed; provisional, not acceptance score',
        'ocr_asr_tested':False}
    write_json(root/'report.json',result);write_rows(root/'details.jsonl',records)
    print(json.dumps(result['overall'],indent=2))


if __name__=='__main__':main()
