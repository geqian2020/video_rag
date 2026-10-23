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
    p.add_argument('--chunks',default=None,help='与 --build 同用，复用冻结分块')
    p.add_argument('--threads',type=int,choices=range(1,65),default=4)
    p.add_argument('--build',action='store_true',help='仅首次或语料/配置变化时构建')
    a=p.parse_args();cfg=load_config(a.config);root=Path(a.root)
    if a.chunks and not a.build:p.error('--chunks 仅在 --build 时使用')
    torch.set_num_threads(a.threads)
    embedding=Embedder(cfg)
    if a.build:
        chunks=read_rows(a.chunks) if a.chunks else build_chunks(read_rows(root/'transcripts.jsonl'),embedding,cfg)
        for c in chunks if not a.chunks else []:
            c.update(summary=c['text'].split('\n')[0],questions=[],enrichment_status='extractive_fallback')
        write_rows(root/'chunks.jsonl',chunks)
        build_index(chunks,cfg,embedding)
        print(f'已建立真实向量/稀疏索引，{len(chunks)} 块；本入口不调用增强模型，冻结分块字段沿用',flush=True)
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
            candidates=[searcher.context.expand(h,cfg,embedding.token_count) for h in fused] if searcher.context else fused
            results=reranker.rerank(row['query'],candidates,5)
            records.append(make_record(row,candidates,results))
            if row['relevant']:
                for name,hits in [('dense',payloads(dense)),('bm25',payloads(sparse)),('hybrid',fused)]:
                    ablations[name].append({'at30':metrics(hits,row['relevant'],30,row.get('evidence_policy','all')),'at5':metrics(hits,row['relevant'],5,row.get('evidence_policy','all'))})
            print(f'已评估 {i}/{len(rows)}',flush=True)
    finally:searcher.close()
    result=report(records)
    result['ablations']={name:{stage:{metric:sum(r[stage][metric] for r in vals)/len(vals)
        for metric in ['recall','mrr','hit']} for stage in ['at30','at5']} for name,vals in ablations.items() if vals}
    result['elapsed_seconds']=time.monotonic()-started
    result['conditions']={'embedding':cfg.models.embedding,'reranker':cfg.models.reranker,
        'enrichment':'frozen chunk fields preserved' if a.chunks else 'off; extractive_fallback','corpus':'downloaded English captions, all videos searched',
        'labels':'agent_draft unless explicitly reviewed; provisional, not acceptance score',
        'ocr_asr_tested':False,'fixed_chunks':a.chunks,'context_seconds':cfg.retrieval.context_seconds,
        'embedding_max_tokens':cfg.models.max_tokens,'rerank_tokens':cfg.models.rerank_tokens}
    write_json(root/'report.json',result);write_rows(root/'details.jsonl',records)
    print(json.dumps(result['overall'],indent=2))


if __name__=='__main__':main()
