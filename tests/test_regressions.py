import asyncio
import math
import pytest
from fastapi import HTTPException
from app.main import extract_bullets, skill_details, score_answer, noul_answer, JevClient, local_review

MARKETING = '''EXPERIENCE
Brightway Consumer Brands | March 2023 – Present
• Led a seasonal marketing campaign that improved CTR from 2% to 3%.
SELECTED MARKETING PROJECTS
MULTICHANNEL PRODUCT LAUNCH — 2024
Responsibilities:
Created a launch communication plan covering audience groups and project milestones.
CUSTOMER LOYALTY EMAIL REFRESH — 2023
Responsibilities:
- Designed a simple A/B test comparing promotional subject lines and results.
EDUCATION
Bachelor of Arts
'''


def test_project_headings_are_not_bullets():
    bullets = extract_bullets(MARKETING)
    assert len(bullets) == 3
    assert all('PRODUCT LAUNCH —' not in x['text'] for x in bullets)
    assert all(x['role'] != 'Responsibilities:' for x in bullets)
    assert bullets[1]['role'].startswith('MULTICHANNEL PRODUCT LAUNCH')
    assert bullets[2]['role'].startswith('CUSTOMER LOYALTY EMAIL REFRESH')


def test_no_skill_leakage():
    jd = 'Python FastAPI Docker RAG LangGraph LangChain vector databases'
    result = skill_details('Experience\nMarketing Manager\n- Managed advertising campaigns.', 'AI Engineer', jd)
    assert all(x['status'] == 'not_demonstrated' for x in result)


def test_claim_denial():
    resume = 'No professional Python, Docker, RAG or LangGraph experience is claimed.\n- Did not develop machine learning services.'
    result = skill_details(resume, 'AI Engineer', 'Python Docker RAG LangGraph machine learning')
    assert not any(x['status'] == 'matched' for x in result)


def test_positive_evidence_citation():
    result = skill_details('Experience\nAI Engineer\n- Built REST APIs using Python and FastAPI.', 'AI Engineer', 'Python FastAPI REST APIs Docker')
    present = {x['skill'] for x in result if x['status'] == 'matched'}
    assert present == {'Python', 'FastAPI', 'REST APIs'}
    assert all(x['evidence'] for x in result if x['status'] == 'matched')


def test_nan_answer_rejected():
    with pytest.raises(ValueError):
        score_answer({'type': 'score', 'score': math.nan})
    with pytest.raises(ValueError):
        noul_answer({'type': 'noul', 'noul': math.inf})


def test_jev_bad_model_configuration(monkeypatch):
    monkeypatch.setenv('JEV_MODEL', 'malicious-model')
    with pytest.raises(HTTPException):
        JevClient('fake-key')


def test_marketing_review_low_fit():
    r = local_review(MARKETING, 'Senior AI Engineer', 'Python FastAPI Docker RAG LangGraph', extract_bullets(MARKETING), [])
    assert r.job_match_score < 10
    assert r.matching_skills == []
    assert r.resume_quality_score >= 0
