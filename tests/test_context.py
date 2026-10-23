"""邻句扩展保留原命中与时间，严格受预算和空隙约束。"""
from core.config import Settings
from retrieve.context import SourceContext
from eval.run_eval import make_record,report


def fixture():
    rows=[{'video_id':'v','start':i,'end':i+5,'text':str(i)} for i in [0,5,10,50]]
    return [{'chunk_id':str(i),'video_id':'v','start_ts':r['start'],'end_ts':r['end'],
             'text':r['text'],'source_segments':[r]} for i,r in enumerate(rows)]


def test_expansion_limits():
    chunks=fixture();context=SourceContext(chunks)
    cfg=Settings(chunking={'min_seconds':0,'max_seconds':15},retrieval={'context_seconds':10})
    hit=context.expand(chunks[1],cfg,len)
    assert hit['text']=='0\n5\n10'
    assert (hit['start_ts'],hit['end_ts'])==(0,15)
    assert (hit['retrieved_start_ts'],hit['retrieved_end_ts'])==(5,10)
    cfg.chunking.max_seconds=5
    assert context.expand(chunks[1],cfg,len)['text']=='5'
    cfg.retrieval.context_seconds=0
    assert context.expand(chunks[1],cfg,len)==chunks[1]


def test_raw_and_expanded_recall_differ():
    chunks=fixture();cfg=Settings(retrieval={'context_seconds':10})
    hit={**SourceContext(chunks).expand(chunks[1],cfg,len),'rerank_score':1}
    row={'query':'test','relevant':[{'video_id':'v','start_ts':0,'end_ts':5}]}
    out=report([make_record(row,[hit],[hit])])['overall']['metrics']
    assert out['recall_stage_at30']['recall']==1
    assert out['raw_recall_stage_at30']['recall']==0


def test_unreconstructable_source_falls_back():
    chunks=fixture();hit={**chunks[1],'text':'unknown legacy text'}
    assert SourceContext(chunks).expand(hit,Settings(retrieval={'context_seconds':10}),len)==hit
