"""固定逐题候选，单独比较标题或时间去重；不改写返回证据。"""
import argparse
from pathlib import Path
import time
from core.config import load_config
from core.io import read_rows, write_json, write_rows
from eval.run_eval import make_record, report
from eval.statistics import paired_compare
from retrieve.rerank import Reranker
from test.download import load_manifest


def with_titles(reranker, query, candidates, titles, top_k=5):
    originals={c['chunk_id']:c for c in candidates}
    inputs=[{**c,'text':f"Video title: {titles[c['video_id']]}\nTranscript:\n{c['text']}"} for c in candidates]
    ranked=reranker.rerank(query,inputs,top_k)
    return [{**originals[h['chunk_id']],'rerank_score':h['rerank_score']} for h in ranked]


def diverse_topk(ranked, top_k=5, threshold=.5):
    selected=[]
    for hit in ranked:
        redundant=False
        for old in selected:
            if hit['video_id']!=old['video_id']:continue
            overlap=max(0.,min(hit['end_ts'],old['end_ts'])-max(hit['start_ts'],old['start_ts']))
            shorter=min(hit['end_ts']-hit['start_ts'],old['end_ts']-old['start_ts'])
            if shorter>0 and overlap/shorter>=threshold:
                redundant=True;break
        if not redundant:selected.append(hit)
        if len(selected)==top_k:break
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--details',required=True)
    p.add_argument('--mode',choices=['titles','diversity'],default='titles')
    p.add_argument('--videos',default='test/videos.json')
    p.add_argument('--config',default='test/config.context.cpu.yaml')
    p.add_argument('--output',default='data/public_benchmark/replay')
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    rows=read_rows(a.details);titles={r['video_id']:r['title'] for r in load_manifest(a.videos)}
    cfg=load_config(a.config);reranker=Reranker(cfg);records=[];started=time.monotonic()
    for i,row in enumerate(rows,1):
        hits=(with_titles(reranker,row['query'],row['candidates'],titles) if a.mode=='titles'
              else diverse_topk(reranker.rerank(row['query'],row['candidates'],len(row['candidates']))))
        records.append(make_record(row,row['candidates'],hits))
        print(f'{a.mode} {i}/{len(rows)}',flush=True)
    out=report(records);out['paired']=paired_compare(rows,records)
    out['conditions']={'mode':a.mode,'reranker':cfg.models.reranker,'rerank_tokens':cfg.models.rerank_tokens,
                       'limits':'development only; fixed candidates; diversity threshold=.5; no query rewriting'}
    out['elapsed_seconds']=time.monotonic()-started
    write_json(Path(a.output)/'report.json',out);write_rows(Path(a.output)/'details.jsonl',records)
    print(out['overall'])


if __name__=='__main__':main()
