"""小块命中后补充相邻原句；保留原命中时间，且遵守时间与 token 预算。"""
from bisect import bisect_left
from collections import defaultdict


class SourceContext:
    def __init__(self, chunks):
        grouped = defaultdict(dict)
        for chunk in chunks:
            for s in chunk['source_segments']:
                grouped[chunk['video_id']].setdefault((s['start'], s['end'], s['text']), s)
        self.units = {v: sorted(rows.values(), key=lambda s: (s['start'], s['end']))
                      for v, rows in grouped.items()}
        self.starts = {v: [s['start'] for s in rows] for v, rows in self.units.items()}

    def expand(self, hit, cfg, token_count):
        if cfg.retrieval.context_seconds == 0:
            return hit
        units = self.units[hit['video_id']]
        lo = bisect_left(self.starts[hit['video_id']], hit['start_ts'])
        hi = bisect_left(self.starts[hit['video_id']], hit['end_ts'])
        chosen = units[lo:hi]
        if not chosen or '\n'.join(s['text'] for s in chosen) != hit['text']:
            return hit
        lower = hit['start_ts'] - cfg.retrieval.context_seconds
        upper = hit['end_ts'] + cfg.retrieval.context_seconds
        left, right, blocked = lo-1, hi, set()
        while True:
            options = []
            if left >= 0 and 'left' not in blocked and units[left]['end'] > lower:
                options.append((hit['start_ts']-units[left]['end'], 'left', units[left]))
            if right < len(units) and 'right' not in blocked and units[right]['start'] < upper:
                options.append((units[right]['start']-hit['end_ts'], 'right', units[right]))
            if not options:
                break
            _, side, s = min(options, key=lambda x: (x[0], x[1]))
            proposal = [s]+chosen if side == 'left' else chosen+[s]
            gap = chosen[0]['start']-s['end'] if side == 'left' else s['start']-chosen[-1]['end']
            text = '\n'.join(x['text'] for x in proposal)
            if (gap > cfg.chunking.gap_seconds
                    or max(x['end'] for x in proposal)-min(x['start'] for x in proposal) > cfg.chunking.max_seconds
                    or token_count(text) > cfg.models.max_tokens):
                blocked.add(side)
                continue
            chosen = proposal
            if side == 'left': left -= 1
            else: right += 1
        return {**hit, 'retrieved_start_ts': hit['start_ts'], 'retrieved_end_ts': hit['end_ts'],
                'text': '\n'.join(s['text'] for s in chosen),
                'start_ts': min(s['start'] for s in chosen), 'end_ts': max(s['end'] for s in chosen),
                'source_segments': chosen}
