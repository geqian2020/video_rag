"""复用本地公开媒体与 Whisper 权重，对照真实 OCR、有/无词表转录；不联网。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from core.config import Settings
from core.io import read_rows,write_json,write_rows
from core.vocabulary import Term,Vocabulary
from eval.run_asr_eval import evaluate,contains
from ocr.extract import extract,load_engine
from transcribe.run import load_local_whisper,transcribe_video


def parse_srt(text):
    def seconds(value):
        h,m,s=value.replace(',','.').split(':')
        return int(h)*3600+int(m)*60+float(s)
    rows=[]
    for block in re.split(r'\n\s*\n',text):
        lines=block.strip().splitlines()
        if len(lines)>=3 and '-->' in lines[1]:
            start,end=lines[1].split(' --> ')
            rows.append({'start':seconds(start),'end':seconds(end),'text':' '.join(lines[2:])})
    if not rows:raise ValueError('没有有效 SRT')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',default='test/media_cases.json')
    p.add_argument('--video',required=True);p.add_argument('--subtitles',required=True)
    p.add_argument('--whisper',required=True,help='本地可信 .pt 权重，不自动下载')
    p.add_argument('--output',default='data/public_benchmark/media_smoke')
    a=p.parse_args()
    import torch
    from whisper.tokenizer import get_tokenizer
    torch.set_num_threads(3)
    cfg=Settings(asr={'model':a.whisper,'language':'en'},ocr={'interval_seconds':15})
    plan=json.loads(Path(a.cases).read_text(encoding='utf-8'))
    source=Path(a.video).resolve();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    subtitles=parse_srt(Path(a.subtitles).read_text(encoding='utf-8-sig'))
    engine=load_engine(cfg);model=load_local_whisper(cfg)
    tok=get_tokenizer(model.is_multilingual,num_languages=model.num_languages,language='en',task='transcribe')
    count=lambda text:len(tok.encode(' '+text.strip()))
    vocab=Vocabulary();reference=[];baseline=[];assisted=[];ocr_results=[];started=time.monotonic()
    for case in plan['cases']:
        vid=case['video_id'];clip=root/(vid+'.mp4')
        subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-ss',str(case['start']),
            '-protocol_whitelist','file,pipe','-i',str(source),'-t',str(case['end']-case['start']),
            '-c:v','libx264','-preset','ultrafast','-c:a','aac',str(clip)],check=True,capture_output=True)
        folder=extract(clip,vid,root/'ocr',cfg,engine);obs=read_rows(folder/'observations.jsonl')
        supported=[r for r in obs if contains(r['text'],case['term'])]
        # 仅对预先按画面核对的拼写，且当次 OCR 确实检出时启用；不批准其它词。
        if supported:
            vocab.entries.append(Term(term=case['term'],approved=True,video_id=vid,
                timestamps=[r['timestamp'] for r in supported],evidence=supported))
        for row in subtitles:
            if row['start']>=case['start'] and row['end']<=case['end']:
                reference.append({'video_id':vid,'start':row['start']-case['start'],
                                  'end':row['end']-case['start'],'text':row['text']})
        baseline.extend(transcribe_video(clip,vid,Vocabulary(),cfg,model,count,root/(vid+'.baseline.jsonl')))
        assisted.extend(transcribe_video(clip,vid,vocab,cfg,model,count,root/(vid+'.assisted.jsonl')))
        ocr_results.append({'video_id':vid,'observations':len(obs),'expected_term':case['term'],
                            'detected_frames':len({r['timestamp'] for r in supported})})
    write_rows(root/'reference.jsonl',reference);write_rows(root/'baseline.jsonl',baseline)
    write_rows(root/'assisted.jsonl',assisted);write_json(root/'vocabulary.json',vocab.model_dump())
    out={'source_page':plan['source_page'],'cases':plan['cases'],'whisper_model':a.whisper,
         'whisper_sha256':hashlib.sha256(Path(a.whisper).read_bytes()).hexdigest(),'ocr':ocr_results,
         'baseline':evaluate(reference,baseline,vocab),'assisted':evaluate(reference,assisted,vocab),
         'reference_quality':plan['reference_quality'],
         'limits':'2 English clips, official captions; boundary mismatch possible; no company acceptance',
         'elapsed_seconds':time.monotonic()-started}
    write_json(root/'report.json',out);print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
