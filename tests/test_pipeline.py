"""真实 Qdrant embedded 集成；模型用显式测试替身，不下载权重。"""
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from core.config import Settings, load_config, local_endpoint
from core.schema import Chunk, Enhancement, Segment, validate_enhancement
from core.sparse import fit, tokenize, vector
from eval.run_eval import metrics
from ingest.chunk import build_chunks
from ingest.convert_legacy import convert
from ingest.enrich import enrich_one, check_local_model
from ingest.index import build_index
from retrieve.search import Searcher, rrf
from serve.api import create_app

class FakeEmbedder:
    dimension = 3
    def token_count(self, text):
        return len(text)
    def encode(self, texts, query=False):
        out = []
        for text in texts:
            v = np.array([1., float('校准' in text), float('水果' in text)])
            out.append(v / np.linalg.norm(v))
        return np.asarray(out)

class FakeReranker:
    def rerank(self, query, candidates, top_k):
        return [{**c, 'rerank_score': float('校准' in c['text'])} for c in
                sorted(candidates, key=lambda c: '校准' not in c['text'])[:top_k]]

@pytest.fixture
def cfg(tmp_path):
    c = Settings()
    c.storage.root = str(tmp_path / 'index')
    c.chunking.min_seconds = 0
    return c

def chunks(cfg):
    rows = build_chunks([{'video_id': 'a', 'start': 0, 'end': 10, 'text': '设备校准需要采集测试数据。'},
                         {'video_id': 'b', 'start': 0, 'end': 10, 'text': '水果每天需要冷藏保存。'}], FakeEmbedder(), cfg)
    for r in rows:
        r.update(summary=r['text'], enrichment_status='extractive_fallback')
    return rows

@pytest.mark.parametrize('start,end', [(-1, 2), (1, 1), (3, 2), (0, float('nan')), (0, float('inf'))])
def test_bad_time(start, end):
    with pytest.raises(ValueError):
        Segment(video_id='a', text='x', start=start, end=end)

def test_semantic_boundary_and_stable_ids(cfg):
    cfg.chunking.similarity_threshold = .99
    rows = [{'video_id': 'a', 'start': 0, 'end': 10, 'text': '设备校准。'},
            {'video_id': 'a', 'start': 10, 'end': 20, 'text': '水果冷藏。'}]
    out = build_chunks(rows, FakeEmbedder(), cfg)
    assert len(out) == 2 and out[-1]['end_ts'] == 20
    assert out[0]['chunk_id'] == build_chunks(rows, FakeEmbedder(), cfg)[0]['chunk_id']

def test_gap_time_and_token_limits(cfg):
    cfg.chunking.similarity_threshold = -1
    cfg.chunking.max_seconds = 15
    rows = [{'video_id': 'a', 'start': i, 'end': i + 10, 'text': '校准。'} for i in [0, 10, 40]]
    assert len(build_chunks(rows, FakeEmbedder(), cfg)) == 3
    with pytest.raises(ValueError, match='单句'):
        build_chunks([{'video_id': 'a', 'start': 0, 'end': 1, 'text': '词' * 1025}], FakeEmbedder(), cfg)

def test_bm25_exact_formula():
    texts = ['校准 校准', '水果']
    stats = fit(texts)
    d, q = vector(texts[0], stats), vector('校准', stats, query=True)
    score = sum(dict(zip(d.indices, d.values)).get(i, 0) * v for i, v in zip(q.indices, q.values))
    n = tokenize(texts[0]).count('校准')
    expected = stats['idf']['校准'] * n * 2.5 / (n + 1.5 * (.25 + .75 * len(tokenize(texts[0])) / stats['avgdl']))
    assert score == pytest.approx(expected)
    assert not vector('xyz', stats, query=True).indices

def test_rrf_duplicate_once():
    p = SimpleNamespace(id='a', payload={'chunk_id': 'a'})
    assert rrf([[p, p], [p]])[0]['rrf_score'] == pytest.approx(2 / 61)

def test_real_qdrant_roundtrip_filter_replacement(cfg):
    rows = chunks(cfg)
    build_index(rows, cfg, FakeEmbedder())
    s = Searcher(cfg, FakeEmbedder(), FakeReranker())
    try:
        assert len(s.search('校准')) == 2
        assert [h['video_id'] for h in s.search('校准', video_ids=['a'])] == ['a']
        assert s.search('校准', video_ids=['missing']) == []
        assert s.search('校准', video_ids=[]) == []
    finally:
        s.close()
    build_index(rows[:1], cfg, FakeEmbedder())
    s = Searcher(cfg, FakeEmbedder(), FakeReranker())
    try:
        assert s.search('水果', video_ids=['b']) == []
    finally:
        s.close()

def test_failure_keeps_current(cfg):
    rows = chunks(cfg)
    old = build_index(rows, cfg, FakeEmbedder())
    class Broken(FakeEmbedder):
        def encode(self, *args, **kwargs):
            raise RuntimeError('故障注入')
    with pytest.raises(RuntimeError):
        build_index(rows, cfg, Broken())
    assert (Path(cfg.storage.root) / 'CURRENT').read_text().strip() == old

def test_mismatched_model_rejected(cfg):
    build_index(chunks(cfg), cfg, FakeEmbedder())
    cfg.models.embedding = 'another-model'
    with pytest.raises(ValueError, match='配置不一致'):
        Searcher(cfg, FakeEmbedder(), FakeReranker())

def test_enrichment_rejects_invented_content(cfg):
    c = Chunk.model_validate(chunks(cfg)[0])
    e = Enhancement(summary='外部背景', questions=[{'question': f'问题{i}是什么？', 'evidence': c.text} for i in range(3)])
    with pytest.raises(ValueError, match='摘要'):
        validate_enhancement(c.text, e)
    bad = {'summary': c.text, 'questions': [{'question': '校准是什么？', 'evidence': '编造证据'}] * 3}
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'message': {'content': json.dumps(bad)}}))) as client:
        with pytest.raises(RuntimeError):
            enrich_one(c, client, cfg)
        cfg.enrichment.on_failure = 'extractive'
        out = enrich_one(c, client, cfg)
        assert out['enrichment_status'] == 'extractive_fallback' and not out['questions']

def test_ollama_schema_request_success(cfg):
    c = Chunk.model_validate(chunks(cfg)[0])
    payload = {'summary': c.text, 'questions': [{'question': q, 'evidence': c.text} for q in
               ['设备校准要做什么？', '需要采集哪种数据？', '测试数据用于什么过程？']]}
    def handler(request):
        body = json.loads(request.content)
        assert body['think'] is False and body['format']['type'] == 'object'
        assert c.text in body['messages'][1]['content']
        return httpx.Response(200, json={'message': {'content': json.dumps(payload)}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert enrich_one(c, client, cfg)['enrichment_status'] == 'validated'

def test_cloud_and_external_endpoints_rejected(cfg):
    for url in ['https://api.external.com', 'http://user:password@localhost']:
        with pytest.raises(ValueError):
            local_endpoint(url, cfg)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={'remote_host': 'cloud'}))) as client:
        with pytest.raises(ValueError):
            check_local_model(client, cfg)

def test_legacy_end_is_estimated():
    rows = convert('[00:00:00] 第一行\n[00:00:00] 同秒\n[00:00:05] 第二行', 'a', 12)
    assert rows[0]['end'] == 5 and rows[-1]['end'] == 12
    assert all(r['timestamp_quality'] == 'estimated' for r in rows)
    with pytest.raises(ValueError):
        convert('[00:00:05] 最后一行', 'a', 5)

def test_eval_recall_and_mrr():
    gold = [{'chunk_id': 'a'}, {'chunk_id': 'b'}]
    assert metrics([{'chunk_id': 'a'}] * 2, gold, 5) == {'recall': .5, 'mrr': 1, 'hit': 1.0}
    assert metrics([{'chunk_id': 'x'}, {'chunk_id': 'b'}], gold, 5)['mrr'] == .5

def test_api_auth_validation_contract(monkeypatch, cfg):
    monkeypatch.setenv('RAG_API_KEY', 'test-secret')
    rows = chunks(cfg)
    class Backend:
        def search(self, *args):
            return [{**rows[0], 'rrf_score': .1, 'rerank_score': 2}]
        def close(self):
            pass
    with TestClient(create_app(Backend)) as client:
        assert client.get('/health').status_code == 200
        assert client.post('/search', json={'query': '校准'}).status_code == 401
        headers = {'X-API-Key': 'test-secret'}
        assert client.post('/search', headers=headers, json={'query': ' '}).status_code == 422
        assert client.post('/search', headers=headers, json={'query': '校准', 'top_k': 31}).status_code == 422
        r = client.post('/search', headers=headers, json={'query': '校准'})
        assert r.status_code == 200 and r.json()['context_is_untrusted'] is True
        assert r.json()['results'][0]['end_ts'] == 10
        assert 'source_segments' not in r.json()['results'][0]

def test_config_env_offline(monkeypatch):
    monkeypatch.setenv('RAG_MODELS__DEVICE', 'cpu')
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    assert load_config().models.offline is True
    import os
    assert os.environ['HF_HUB_OFFLINE'] == '1'

def test_cross_encoder_wrapper_orders_and_refuses_truncation(monkeypatch, cfg):
    import sys
    from retrieve.rerank import Reranker
    class Cross:
        def __init__(self, name, **kwargs):
            assert kwargs['local_files_only'] is True and kwargs['trust_remote_code'] is False
            self.tokenizer = lambda q, t, **kw: {'input_ids': list(range(len(q) + len(t)))}
        def predict(self, pairs, **kwargs):
            return np.array([.1, 2.])
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(CrossEncoder=Cross))
    reranker = Reranker(cfg)
    candidates = [{'chunk_id': 'a', 'text': 'first'}, {'chunk_id': 'b', 'text': 'second'}]
    assert reranker.rerank('q', candidates, 1)[0]['chunk_id'] == 'b'
    with pytest.raises(ValueError, match='超过'):
        reranker.rerank('x' * 2048, candidates, 1)


def test_excerpt_must_be_complete_sentence(cfg):
    c = Chunk.model_validate(chunks(cfg)[0])
    e = Enhancement(summary='设备校准', questions=[{'question': f'第{i}个问题是什么？', 'evidence': c.text} for i in range(3)])
    with pytest.raises(ValueError, match='摘要'):
        validate_enhancement(c.text, e)
