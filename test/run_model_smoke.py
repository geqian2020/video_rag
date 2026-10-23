"""本地真实 BGE 中文功能检查；不是召回率验收，不自动下载权重。"""
import argparse
import time
from core.config import Settings
from core.models import Embedder
from core.io import write_json
from retrieve.rerank import Reranker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--embedding',required=True);p.add_argument('--reranker',required=True)
    p.add_argument('--output',default='data/public_benchmark/model_smoke.json')
    a=p.parse_args()
    import torch
    torch.set_num_threads(3)
    cfg=Settings(models={'embedding':a.embedding,'reranker':a.reranker})
    started=time.monotonic();emb=Embedder(cfg);rerank=Reranker(cfg)
    query='Qdrant 如何保存视频片段的时间信息？'
    docs=['每个知识块写入 video_id、start_ts 和 end_ts；Qdrant 的 payload 保存这些字段。',
          '午餐菜单包含米饭和蔬菜。','Whisper 使用经人工确认的局部术语提示转录。']
    vec=emb.encode([query]+docs);scores=vec[1:]@vec[0]
    result=rerank.rerank(query,[{'chunk_id':str(i),'text':text} for i,text in enumerate(docs)],3)
    out={'fixture':'3 fictional Chinese passages; functional smoke only',
         'embedding':a.embedding,'reranker':a.reranker,'dimension':emb.dimension,
         'dense_scores':[float(s) for s in scores],'dense_best':int(scores.argmax()),
         'reranked_ids':[r['chunk_id'] for r in result],
         'passed':int(scores.argmax())==0 and result[0]['chunk_id']=='0',
         'elapsed_seconds':time.monotonic()-started}
    write_json(a.output,out);print(out)
    if not out['passed']:raise SystemExit(1)


if __name__=='__main__':main()
