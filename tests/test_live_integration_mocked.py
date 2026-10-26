"""Contract-style integration tests, with provider HTTP transport mocked.

These are NOT equivalent to validating the provider's live API behavior.
"""
import asyncio
from pathlib import Path
import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app, JevClient, WEIGHTS, parse_resume
from app.structured_parser import parse_structure

client=TestClient(app)
FIXTURES=Path(__file__).resolve().parents[1]/'fixtures'

@pytest.mark.parametrize('name,expected_exp,expected_projects',[
    ('01_strong_ai_engineer.pdf',2,1),
    ('01_strong_ai_engineer.docx',2,1),
    ('03_unrelated_marketing.pdf',2,1),
    ('03_unrelated_marketing.docx',2,1),
    ('08_nonstandard_headings.pdf',1,1),
    ('08_nonstandard_headings.docx',1,1),
    ('11_two_column_layout.pdf',1,1),
])
def test_real_pdf_docx_fixtures_parse(name,expected_exp,expected_projects):
    source=FIXTURES/name
    text,warnings=parse_resume(name,source.read_bytes())
    parsed=parse_structure(text)
    assert len(parsed.experience)==expected_exp
    assert len(parsed.projects)==expected_projects
    assert not parsed.warnings


def test_provider_contract_success_with_mocked_http(monkeypatch):
    questions={'summary':{'type':'score'},'quantified':{'type':'noul'}}
    async def fake_transport(request):
        assert request.url.path=='/v1/systemone'
        assert request.headers.get('Authorization')=='Bearer MOCK-KEY'
        assert b'jev-1.14-free' in request.content
        return httpx.Response(200,json={'model':'jev-1.14-free','answers':{
            'summary':{'type':'score','score':2.5},
            'quantified':{'type':'noul','noul':0.8}}})
    original=httpx.AsyncClient
    monkeypatch.setattr('app.main.httpx.AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(fake_transport), **kwargs))
    result=asyncio.run(JevClient('MOCK-KEY').ask({'resume':'fake'},questions))
    assert result['answers']['summary']['score']==2.5


def test_provider_rate_limit_and_missing_answer_with_mock(monkeypatch):
    original=httpx.AsyncClient
    async def rate_limit(request):
        return httpx.Response(429,headers={'Retry-After':'60'},json={'error':{'code':'limit'}})
    monkeypatch.setattr('app.main.httpx.AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(rate_limit),**kwargs))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(JevClient('MOCK-KEY').ask({}, {'relevance':{'type':'score'}}))
    assert exc.value.status_code==429

    async def malformed(request):
        return httpx.Response(200,json={'answers':{'irrelevant':{'type':'score','score':1}}})
    monkeypatch.setattr('app.main.httpx.AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(malformed),**kwargs))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(JevClient('MOCK-KEY').ask({}, {'relevance':{'type':'score'}}))
    assert exc.value.status_code==502


def test_api_review_with_stubbed_provider(monkeypatch):
    monkeypatch.setenv('BEATAPI_API_KEY','DO-NOT-USE-REAL-KEY')
    monkeypatch.delenv('APP_API_KEY',raising=False)
    async def fake_ask(self, state, questions):
        assert 'FastAPI' in state['resume']
        assert len(questions)==12
        return {'model':'jev-1.14-free','answers':{
            k: {'type':'score','score':2.4,'confidence':.72,'probabilities':{'3':.7}} 
                 if k in WEIGHTS else {'type':'noul','noul':.6}
                 for k in questions
        }}
    monkeypatch.setattr(JevClient,'ask',fake_ask)
    f=FIXTURES/'01_strong_ai_engineer.pdf'
    result=client.post('/api/review',files={'file':(f.name,f.read_bytes(),'application/pdf')},data={
        'target_role':'Python Backend Engineer',
        'job_description':'Required: Python FastAPI Docker REST APIs SQL. Build reliable backend services.'})
    assert result.status_code==200,result.text
    body=result.json()
    assert body['model']=='jev-1.14-free'
    assert body['job_match_score']>60
    assert body['structured_resume']['experience'][0]['title']=='Senior AI Engineer'
    assert body['missing_required_skills']==['SQL']
    assert body['bullet_reviews'][0]['job_priority_score']>=0
    assert len(body['criteria'])==9
