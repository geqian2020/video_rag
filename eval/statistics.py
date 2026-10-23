"""离线置信区间和严格逐题配对；不调用检索或模型。"""
import argparse
import hashlib
import json
import math
import random
from core.io import read_rows, write_json


def wilson(successes, n, z=1.959963984540054):
    if n == 0: return None
    p = successes/n
    d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]


def bootstrap_mean(values, iterations=5000, seed=42):
    if not values: return None
    rng, n = random.Random(seed), len(values)
    means = sorted(sum(rng.choices(values, k=n))/n for _ in range(iterations))
    return [means[int(.025*(iterations-1))], means[int(.975*(iterations-1))]]


def identity(row):
    value = [row['query'], row['type'], row['relevant'], row.get('evidence_policy','all')]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def paired_compare(before, after, iterations=5000, seed=42):
    def indexed(rows):
        out = {}
        for r in rows:
            key = identity(r)
            if key in out: raise ValueError('存在重复问题/标签，无法唯一配对')
            out[key] = r
        return out
    a, b = indexed(before), indexed(after)
    if not a or a.keys() != b.keys():
        raise ValueError('两轮问题与标签必须完全相同；不能静默丢弃不匹配题')
    out = {'paired_questions':len(a), 'metrics':{},
           'method':'query-paired percentile bootstrap; fixed seed; 95% CI; video correlation not modeled',
           'warning':'已查看的留出集应记为已消费；调参集结果不能作为独立验收。'}
    yes = [k for k in a if a[k]['answerable']]
    for stage in ['raw_recall_stage_at30','recall_stage_at30','rerank_at5']:
        if not yes or not all(stage in a[k] and stage in b[k] for k in yes): continue
        out['metrics'][stage] = {}
        for metric in ['recall','hit','mrr']:
            av = [a[k][stage][metric] for k in yes]
            bv = [b[k][stage][metric] for k in yes]
            diffs = [y-x for x,y in zip(av,bv)]
            summary = {'n':len(yes),'before':sum(av)/len(av),'after':sum(bv)/len(bv),
                       'delta':sum(diffs)/len(diffs),'delta_ci95':bootstrap_mean(diffs,iterations,seed)}
            if metric == 'hit':
                summary.update(before_ci95=wilson(sum(av),len(av)),after_ci95=wilson(sum(bv),len(bv)),
                    improved=sum(x==0 and y==1 for x,y in zip(av,bv)),
                    regressed=sum(x==1 and y==0 for x,y in zip(av,bv)))
            out['metrics'][stage][metric] = summary
    no = [k for k in a if not a[k]['answerable']]
    out['unanswerable'] = {'n':len(no),'before_accepted':sum(a[k]['false_accept'] for k in no),
                          'after_accepted':sum(b[k]['false_accept'] for k in no)}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',required=True); p.add_argument('--after',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=paired_compare(read_rows(a.before),read_rows(a.after))
    write_json(a.output,out)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
