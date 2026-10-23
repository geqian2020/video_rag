"""人工审核后冻结评测集：校验划分、既往曝光并保存哈希，不代替人工审核。"""
import argparse
import hashlib
import json
from pathlib import Path
from core.io import read_rows, write_json
from eval.run_eval import validate_gold


def freeze(rows, consumed_videos, minimum=150):
    validate_gold(rows)
    if len(rows)<minimum: raise ValueError(f'仅 {len(rows)} 题；正式冻结至少需要 {minimum} 题')
    if any(r.get('reviewed') is not True for r in rows): raise ValueError('存在未人工审核题')
    seen, assignments, counts = set(), {}, {}
    for row in rows:
        query=''.join(row['query'].casefold().split())
        if query in seen: raise ValueError('存在重复问题')
        seen.add(query)
        split=row.get('split')
        if split not in {'dev','validation','test'}: raise ValueError('必须显式指定 dev/validation/test')
        videos={g['video_id'] for g in row['relevant'] if 'video_id' in g}
        if row.get('video_id'): videos.add(row['video_id'])
        if not videos or any('chunk_id' in g for g in row['relevant']):
            raise ValueError('必须使用视频时间标签；无答案题也须有 video_id')
        for vid in videos:
            if vid in assignments and assignments[vid]!=split: raise ValueError(f'视频 {vid} 跨划分')
            if split=='test' and vid in consumed_videos: raise ValueError(f'视频 {vid} 已曝光，不能作未见测试')
            assignments[vid]=split
        counts[split]=counts.get(split,0)+1
    if set(counts)!={'dev','validation','test'}: raise ValueError('需要三个划分')
    return {'questions':len(rows),'by_split':counts,'video_assignments':assignments,'reviewed':True,
            'note':'此锁不验证语义或审核者身份；测试曝光后须重新分类。'}


def verify_lock(gold_path, lock_path):
    lock=json.loads(Path(lock_path).read_text(encoding='utf-8'))
    actual=hashlib.sha256(Path(gold_path).read_bytes()).hexdigest()
    if actual!=lock.get('gold_sha256'): raise ValueError('gold 内容已改变，与冻结哈希不一致')
    return lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',required=True); p.add_argument('--consumed',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError('冻结清单已存在，请另用版本名')
    out=freeze(read_rows(a.gold),{r['video_id'] for r in read_rows(a.consumed)})
    out['gold_sha256']=hashlib.sha256(Path(a.gold).read_bytes()).hexdigest()
    write_json(a.output,out)
    print(f"冻结 {out['questions']} 题，{out['by_split']}")


if __name__=='__main__': main()
