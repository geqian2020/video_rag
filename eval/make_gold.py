"""验证逐字证据并映射到稳定时间段；默认仅接纳人工 reviewed=true 的题。"""
import argparse
from collections import defaultdict
import re

from core.io import read_rows, write_json, write_rows
from core.schema import Segment


def norm(text):
    return re.sub(r'\s+', '', text)


def source_index(rows):
    groups=defaultdict(list)
    for raw in rows:
        seg=Segment.model_validate(raw)
        groups[seg.video_id].append(seg.model_dump())
    index={}
    for video,segs in groups.items():
        segs.sort(key=lambda s:(s['start'],s['end']))
        spans=[];full=''
        for seg in segs:
            text=norm(seg['text'])
            spans.append((len(full),len(full)+len(text),seg));full+=text
        index[video]=(full,spans)
    return index


def locate(evidence, indexed, hint=None, max_seconds=60):
    ev=norm(evidence)
    if not ev:
        raise ValueError('empty_evidence')
    full,spans=indexed
    positions=[];cursor=0
    while True:
        cursor=full.find(ev,cursor)
        if cursor<0:break
        positions.append(cursor);cursor+=1
    candidates=[]
    for pos in positions:
        hit=[s for a,b,s in spans if a<pos+len(ev) and b>pos]
        start,end=min(s['start'] for s in hit),max(s['end'] for s in hit)
        if end-start>max_seconds or any(b['start']-a['end']>10 for a,b in zip(hit,hit[1:])):
            continue
        if hint is not None and not start<=hint<=end:
            continue
        candidates.append((start,end))
    if len(candidates)!=1:
        raise ValueError('evidence_not_found_or_ambiguous')
    return candidates[0]


def make_gold(segments,drafts,allow_unreviewed=False):
    index=source_index(segments);kept=[];rejected=[];seen=set()
    for number,row in enumerate(drafts,1):
        try:
            query=row.get('query','').strip();video=row.get('video_id')
            if not query or video not in index:raise ValueError('missing_query_or_unknown_video')
            if norm(query).casefold() in seen:raise ValueError('duplicate_query')
            reviewed=row.get('reviewed') is True
            if not reviewed and not allow_unreviewed:raise ValueError('human_review_required')
            kind=row.get('type')
            if kind not in {'factual','howto','why','compare','unanswerable'}:raise ValueError('invalid_type')
            evidence=row.get('evidence',[])
            if isinstance(evidence,str):evidence=[evidence] if evidence.strip() else []
            relevant=[]
            if kind=='unanswerable':
                if evidence or row.get('answer'):raise ValueError('unanswerable_must_have_empty_evidence_and_answer')
            else:
                if not evidence or len(evidence)>3:raise ValueError('require_one_to_three_quotes')
                for quote in evidence:
                    start,end=locate(quote,index[video],row.get('evidence_start_hint'))
                    relevant.append({'video_id':video,'start_ts':start,'end_ts':end,'evidence':quote})
            kept.append({'query':query,'answer':row.get('answer',''),'type':kind,'video_id':video,
                'relevant':relevant,'reviewed':reviewed,'split':row.get('split','test'),
                'label_status':'human_reviewed' if reviewed else 'agent_draft'})
            seen.add(norm(query).casefold())
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            rejected.append({'line':number,'reason':str(exc)})
    return kept,rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segments',required=True)
    p.add_argument('--draft',required=True)
    p.add_argument('--output',default='data/public_benchmark/gold.jsonl')
    p.add_argument('--allow-unreviewed',action='store_true',help='仅限流程试跑，输出带 agent_draft 标记')
    a=p.parse_args()
    gold,rejects=make_gold(read_rows(a.segments),read_rows(a.draft),a.allow_unreviewed)
    if not gold:raise ValueError('无合格题；未覆盖已有 gold，请检查 reviewed/证据/拒绝原因：'+str(rejects[:5]))
    write_rows(a.output,gold);write_json(a.output+'.rejected.json',rejects)
    print({'kept':len(gold),'dropped':len(rejects),'draft_only':a.allow_unreviewed})


if __name__=='__main__':main()
