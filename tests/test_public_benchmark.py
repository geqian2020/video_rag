"""公开语料工具与评估边界；全部使用虚构正文，不依赖网络。"""
import pytest
from eval.make_gold import source_index,locate,make_gold
from eval.run_eval import validate_gold,metrics,make_record,report
from test.prepare_corpus import convert_json3
from test.download import load_manifest,command


def seg(text,start=0,end=5,video='a'):
    return {'video_id':video,'text':text,'start':start,'end':end}


def test_manifest_has_five_verified_hour_long_sources():
    videos=load_manifest('test/videos.json')
    assert len(videos)==5 and all(3600<=v['duration_seconds']<=7200 for v in videos)
    assert '--skip-download' in command(videos[0],'data','subtitles')
    assert '--compat-options' in command(videos[0],'data','both',True)
    assert '--no-check-certificates' not in command(videos[0],'data','both',True)


def test_json3_handles_rolling_line_times_and_blank_events():
    data={'events':[{'tStartMs':0,'dDurationMs':4000,'segs':[{'utf8':'Alpha'}]},
        {'tStartMs':1990,'segs':[{'utf8':'\n'}]},
        {'tStartMs':2000,'dDurationMs':4000,'segs':[{'utf8':'Beta'}]}]}
    rows=convert_json3(data,'a',5)
    assert [r['text'] for r in rows]==['Alpha','Beta']
    assert rows[0]['end']==2 and rows[1]['end']==5
    assert all(r['timestamp_quality']=='estimated' for r in rows)


def test_time_locator_never_crosses_videos_or_large_gaps():
    index=source_index([seg('Alpha'),seg('Beta',5,10),seg('Gamma',100,105),seg('Other',video='b')])
    assert locate('Alpha Beta',index['a'])==(0,10)
    with pytest.raises(ValueError):locate('Other',index['a'])
    with pytest.raises(ValueError):locate('BetaGamma',index['a'])


def test_ambiguous_quote_requires_hint():
    index=source_index([seg('same'),seg('same',10,15)])
    with pytest.raises(ValueError):locate('same',index['a'])
    assert locate('same',index['a'],hint=12)==(10,15)


def test_human_gate_and_unknown_evidence():
    rows=[seg('Alpha supports a tool.')]
    draft={'video_id':'a','query':'What can it use?','answer':'tool','evidence':['Alpha supports a tool.'],'type':'factual','reviewed':False}
    gold,bad=make_gold(rows,[draft]);assert not gold and bad[0]['reason']=='human_review_required'
    gold,_=make_gold(rows,[draft],True)
    assert gold[0]['label_status']=='agent_draft'
    assert gold[0]['relevant'][0]['start_ts']==0
    assert 'video_ids' not in gold[0]  # 不把答案视频当查询过滤器
    draft.update(reviewed=True,evidence=['invented fact'])
    assert not make_gold(rows,[draft])[0]


def test_unanswerable_allowed_but_not_in_recall_denominator():
    row={'query':'Unknown?','type':'unanswerable','video_id':'a','relevant':[]}
    assert validate_gold([row])
    with pytest.raises(ValueError):metrics([],[],5)
    hit={'chunk_id':'x','video_id':'a','start_ts':0,'end_ts':5,'text':'x','rerank_score':-2}
    baseline=make_record(row,[hit],[hit]);reject=make_record(row,[hit],[hit],min_score=0)
    assert report([baseline])['overall']['false_accept_rate']==1
    assert report([reject],0)['overall']['false_accept_rate']==0
    assert report([baseline])['overall']['metrics']=={}


def test_recall_deduplicates_evidence_and_does_not_accept_one_second_only():
    gold=[{'video_id':'a','start_ts':0,'end_ts':10}]
    hit={'video_id':'a','start_ts':9,'end_ts':11}
    assert metrics([hit],gold,5)['hit']==0
    assert metrics([dict(hit,start_ts=5,end_ts=10)]*2,gold,5)['recall']==1
