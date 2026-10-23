"""时间标签检索评估；支持无答案题、逐题诊断、按题型/视频汇总。"""
import argparse
from collections import defaultdict
import json
import math

from core.config import load_config
from core.io import read_rows, write_json, write_rows
from retrieve.search import Searcher
from eval.statistics import wilson, bootstrap_mean


def relevant(hit, gold):
    if 'chunk_id' in gold:
        return hit['chunk_id']==gold['chunk_id']
    if hit['video_id']!=gold['video_id']:
        return False
    overlap=max(0,min(hit['end_ts'],gold['end_ts'])-max(hit['start_ts'],gold['start_ts']))
    # 不采用“重叠 1 秒即可”规则；它会更宽松，反而增加偶然命中。
    return overlap/(gold['end_ts']-gold['start_ts'])>=.5


def metrics(hits,gold,k,evidence_policy='all'):
    if not gold:
        raise ValueError('无答案题不得混入 Recall/MRR 的分母')
    if evidence_policy not in {'all','alternatives'}: raise ValueError('未知 evidence_policy')
    matched={i for i,g in enumerate(gold) if any(relevant(h,g) for h in hits[:k])}
    rr=next((1/rank for rank,h in enumerate(hits[:k],1) if any(relevant(h,g) for g in gold)),0.)
    return {'recall':float(bool(matched)) if evidence_policy=='alternatives' else len(matched)/len(gold),'mrr':rr,'hit':float(bool(matched))}


def validate_gold(rows):
    if not rows:raise ValueError('gold set 不能为空')
    for row in rows:
        if not isinstance(row.get('query'),str) or not row['query'].strip() or 'relevant' not in row:
            raise ValueError('每条 gold 必须有 query 和 relevant')
        if row.get('type')=='unanswerable':
            if row['relevant']:raise ValueError('无答案题的 relevant 必须为空')
        elif not row['relevant']:
            raise ValueError('可回答题需要非空 relevant')
        if row.get('evidence_policy','all') not in {'all','alternatives'}: raise ValueError('未知 evidence_policy')
        for g in row['relevant']:
            if 'chunk_id' not in g:
                from core.schema import Segment
                Segment(video_id=g['video_id'],text='gold',start=g['start_ts'],end=g['end_ts'])
    return rows


def make_record(row,candidates,results,min_score=None):
    if min_score is not None and not math.isfinite(min_score):raise ValueError('min_score 必须为有限数')
    accepted=results if min_score is None else [h for h in results if h['rerank_score']>=min_score]
    record={'query':row['query'],'type':row.get('type','unspecified'),
        'video_id':row.get('video_id') or (row['relevant'][0].get('video_id','unknown') if row['relevant'] else 'unknown'),
        'reviewed':row.get('reviewed'), 'label_status':row.get('label_status','unspecified'),
        'split':row.get('split','unspecified'),'relevant':row['relevant'],
        'evidence_policy':row.get('evidence_policy','all'),
        'candidates':candidates,'results':results,'accepted_results':accepted,
        'max_rerank_score':max((h['rerank_score'] for h in results),default=None)}
    if row.get('type')=='unanswerable':
        record.update(answerable=False,false_accept=bool(accepted),failed=bool(accepted))
    else:
        a,b=metrics(candidates,row['relevant'],30,row.get('evidence_policy','all')),metrics(accepted,row['relevant'],5,row.get('evidence_policy','all'))
        raw=[{**h,'start_ts':h.get('retrieved_start_ts',h['start_ts']),'end_ts':h.get('retrieved_end_ts',h['end_ts'])} for h in candidates]
        record.update(answerable=True,recall_stage_at30=a,raw_recall_stage_at30=metrics(raw,row['relevant'],30,row.get('evidence_policy','all')),rerank_at5=b,failed=b['recall']<1)
    return record


def aggregate(records):
    yes=[r for r in records if r['answerable']];no=[r for r in records if not r['answerable']]
    means={stage:{key:sum(r[stage][key] for r in yes)/len(yes) for key in ['recall','mrr','hit']}
           for stage in ['recall_stage_at30','rerank_at5']} if yes else {}
    if yes and all('raw_recall_stage_at30' in r for r in yes):
        means['raw_recall_stage_at30']={key:sum(r['raw_recall_stage_at30'][key] for r in yes)/len(yes) for key in ['recall','mrr','hit']}
    return {'queries':len(records),'answerable':len(yes),'unanswerable':len(no),
            'metrics':means,'false_accept_rate':sum(r['false_accept'] for r in no)/len(no) if no else None,
            'failed_queries':sum(r['failed'] for r in records),
        'ci95':{stage:{'hit_wilson':wilson(sum(r[stage]['hit'] for r in yes),len(yes)),
                       'recall_bootstrap':bootstrap_mean([r[stage]['recall'] for r in yes]),
                       'mrr_bootstrap':bootstrap_mean([r[stage]['mrr'] for r in yes])} for stage in means},
        'false_accept_ci95':wilson(sum(r['false_accept'] for r in no),len(no)),
        'interval_note':'逐题抽样区间，不涵盖视频内相关性、标签错误或调参偏差。'}


def report(records,min_score=None):
    by_type=defaultdict(list);by_video=defaultdict(list);by_split=defaultdict(list)
    for r in records:
        by_type[r['type']].append(r);by_video[r['video_id']].append(r);by_split[r.get('split','unspecified')].append(r)
    return {'overall':aggregate(records),'by_type':{k:aggregate(v) for k,v in by_type.items()},
        'by_video':{k:aggregate(v) for k,v in by_video.items()},'by_split':{k:aggregate(v) for k,v in by_split.items()},'min_score':min_score,
        'unreviewed_or_unspecified_labels':sum(r['reviewed'] is not True for r in records),
        'note':'阈值应在独立验证集选择后锁定；不应在同一测试集调参。未设阈值时有返回的无答案题会计为误接受。'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',default='examples/gold.jsonl')
    p.add_argument('--config',default=None)
    p.add_argument('--split',action='append',help='只评估指定划分，可重复')
    p.add_argument('--lock',default=None,help='校验冻结 gold 哈希')
    p.add_argument('--min-score',type=float,default=None)
    p.add_argument('--output',default=None,help='本地汇总报告 JSON，勿上传真实评测数据')
    p.add_argument('--details',default=None,help='本地逐题候选/结果/失败记录 JSONL')
    a=p.parse_args()
    if a.lock:
        if not a.split:p.error('使用 --lock 必须显式指定 --split')
        from eval.freeze_gold import verify_lock
        verify_lock(a.gold,a.lock)
    rows=validate_gold(read_rows(a.gold))
    if a.split:rows=[r for r in rows if r.get('split') in a.split]
    if not rows:raise ValueError('所选划分没有题目')
    searcher=Searcher(load_config(a.config))
    records=[]
    try:
        for row in rows:
            out=searcher.search(row['query'],top_k=5,video_ids=row.get('video_ids'),include_candidates=True)
            records.append(make_record(row,out['candidates'],out['results'],a.min_score))
    finally:searcher.close()
    out=report(records,a.min_score)
    if a.output:write_json(a.output,out)
    if a.details:write_rows(a.details,records)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
