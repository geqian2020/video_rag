"""冻结校验、标签配对、替代来源与术语次数边界。"""
from pathlib import Path
import hashlib
import json
import pytest
from core.config import Settings
from core.vocabulary import Term,Vocabulary,build_prompt
from eval.statistics import wilson,paired_compare
from eval.run_asr_eval import evaluate
from ocr.vocab import candidates


def record(query,value):
    score={'recall':value,'mrr':value,'hit':value}
    return {'query':query,'type':'factual','relevant':[{'video_id':'a','start_ts':0,'end_ts':5}],
            'answerable':True,'recall_stage_at30':score,'rerank_at5':score}


def test_wilson_small_sample():
    lo,hi=wilson(42,48)
    assert .75<lo<.76 and .94<hi<.95
    assert wilson(0,0) is None and wilson(0,10)[0]==0


def test_pairing_validates_labels_and_duplicates():
    before=[record('a',0),record('b',1)];after=[record('b',1),record('a',1)]
    out=paired_compare(before,after,iterations=100)
    assert out['metrics']['rerank_at5']['hit']['improved']==1
    after[1]['relevant'][0]['end_ts']=6
    with pytest.raises(ValueError):paired_compare(before,after)
    with pytest.raises(ValueError):paired_compare(before,before+before)


def test_repeated_terms_counted_and_extra_videos_rejected():
    v=Vocabulary(entries=[Term(term='CrewAI',approved=True,scope='global')])
    ref=[{'video_id':'a','text':'CrewAI uses tools.'}]
    pred=[{'video_id':'a','text':'CrewAI CrewAI CrewAI uses tools.'}]
    out=evaluate(ref,pred,v)
    assert out['term_recall']==1 and out['excess_term_occurrences']==2 and out['extra_term_types']==0
    with pytest.raises(ValueError):evaluate(ref,pred+[{'video_id':'b','text':'surplus'}],v)


def test_confusable_candidates_remain_unapproved():
    v=Vocabulary(entries=[Term(term='CrewAI',approved=True,scope='global')])
    rows=[{'video_id':'a','timestamp':0,'frame_path':'fixture.png','confidence':.95,'text':s}
          for s in ['CrewAl','CrewA1']]
    out=candidates(rows,Settings(),v)
    assert len(out)==2 and len({r['confusable_group'] for r in out})==1
    assert all(r['confirmed_spelling_suggestions']=='CrewAI' and r['approved']=='' for r in out)


def test_alternatives_are_not_joint_requirements():
    from eval.run_eval import metrics,make_record
    labels=[{'video_id':v,'start_ts':0,'end_ts':5} for v in ['a','b']]
    h={'video_id':'b','start_ts':0,'end_ts':5,'rerank_score':1}
    assert metrics([h],labels,5)['recall']==.5
    assert metrics([h],labels,5,'alternatives')['recall']==1
    row={'query':'either','type':'factual','relevant':labels,'evidence_policy':'alternatives'}
    assert make_record(row,[h],[h])['rerank_at5']['recall']==1


def test_freeze_rejects_unreviewed_or_consumed_test():
    from eval.freeze_gold import freeze
    rows=[{'query':s,'type':'factual','relevant':[{'video_id':s,'start_ts':0,'end_ts':5}],
           'split':s,'reviewed':True} for s in ['dev','validation','test']]
    assert freeze(rows,set(),minimum=3)['questions']==3
    with pytest.raises(ValueError):freeze(rows,{'test'},minimum=3)
    rows[0]['reviewed']=False
    with pytest.raises(ValueError):freeze(rows,set(),minimum=3)


def test_cross_video_gold_keeps_policy():
    from eval.make_gold import make_gold
    segs=[{'video_id':v,'text':'same answer','start':0,'end':5} for v in ['a','b']]
    row={'video_id':'a','query':'a common fact','type':'factual','reviewed':True,
         'evidence_policy':'alternatives','evidence':[{'video_id':v,'quote':'same answer'} for v in ['a','b']]}
    gold,bad=make_gold(segs,[row])
    assert not bad and gold[0]['evidence_policy']=='alternatives'
    assert {r['video_id'] for r in gold[0]['relevant']}=={'a','b'}
    assert gold[0]['split']=='unspecified'


def test_frozen_file_mutation_rejected(tmp_path):
    from eval.freeze_gold import verify_lock
    gold=tmp_path/'gold.jsonl';gold.write_text('fixture')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'gold_sha256':hashlib.sha256(gold.read_bytes()).hexdigest()}))
    assert verify_lock(gold,lock)
    gold.write_text('changed')
    with pytest.raises(ValueError):verify_lock(gold,lock)


def test_leading_zero_ocr_suggests_correct_scope():
    v=Vocabulary(entries=[Term(term='Ollama',approved=True,scope='global')])
    row={'video_id':'a','timestamp':0,'frame_path':'fixture.png','confidence':.95,'text':'0llama'}
    out=candidates([row],Settings(),v)
    assert out[0]['ocr_term']=='0llama' and out[0]['confirmed_spelling_suggestions']=='Ollama'
    assert out[0]['approved']==''


def test_title_is_not_spoken_evidence():
    from test.replay_rerank import with_titles
    original={'chunk_id':'1','video_id':'v','text':'spoken','start_ts':0,'end_ts':5}
    class Fake:
        def rerank(self,query,candidates,top_k):
            assert 'Video title: Fixture' in candidates[0]['text']
            return [{**candidates[0],'rerank_score':2}]
    assert with_titles(Fake(),'q',[original],{'v':'Fixture'})[0]['text']=='spoken'
    assert original['text']=='spoken'


def test_diversity_preserves_other_sources():
    from test.replay_rerank import diverse_topk
    ranked=[{'chunk_id':i,'video_id':v,'start_ts':s,'end_ts':e}
            for i,v,s,e in [('a','v',0,10),('b','v',1,11),('c','w',0,10),('d','v',30,40)]]
    assert [r['chunk_id'] for r in diverse_topk(ranked,3)]==['a','c','d']


def test_english_terms_prompt_has_no_chinese_prefix():
    cfg=Settings(asr={'language':'en'})
    v=Vocabulary(entries=[Term(term='LangGraph',approved=True,scope='global')])
    assert build_prompt(v,'a',0,30,cfg,len)[0]=='LangGraph.'
    cfg.asr.prompt_style='labeled'
    assert build_prompt(v,'a',0,30,cfg,len)[0]=='Vocabulary: LangGraph.'


def test_window_quality_is_recorded(monkeypatch,tmp_path):
    import transcribe.run as run
    monkeypatch.setattr(run,'probe',lambda _:{'duration':10})
    monkeypatch.setattr(run,'extract_audio',lambda _v,p,_s,_e:p.write_bytes(b'stub'))
    class Model:
        def transcribe(self,*args,**kwargs):
            return {'segments':[{'start':1,'end':3,'text':'source','compression_ratio':3.0,
                                 'avg_logprob':-.2,'no_speech_prob':.01}]}
    out=tmp_path/'asr.jsonl'
    run.transcribe_video('stub','a',Vocabulary(),Settings(),Model(),len,out)
    audit=json.loads(Path(str(out)+'.audit.json').read_text())
    assert audit['review_required_windows']==1
    assert audit['windows'][0]['quality'][0]['compression_ratio']==3.0
