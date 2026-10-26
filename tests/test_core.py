from app.main import score_answer, extract_bullets, parse_resume, WEIGHTS
import pytest

def test_weights(): assert abs(sum(WEIGHTS.values())-1)<1e-9
def test_score(): assert score_answer({'type':'score','score':1.5}).score==50
def test_bad_score():
    with pytest.raises(ValueError): score_answer({'type':'score','score':4})
def test_bullets():
    b=extract_bullets('Experience\nSoftware Engineer\n• Built API\n- Reduced latency\nEducation\n• BSc')
    assert len(b)==2
def test_txt(): assert parse_resume('a.txt',b'Hello world')[0]=='Hello world'

from app.main import local_review, score_question, noul_question

def test_question_wire_format():
    assert score_question('Test')['instructions'] == 'Test'
    assert 'question' not in score_question('Test')
    assert noul_question('Test')['type'] == 'noul'

def test_offline_mode():
    result=local_review('Experience\nAI Engineer\n- Built a Python API for 500 users\nSkills\nPython', 'AI Engineer', 'Python API', extract_bullets('Experience\nAI Engineer\n- Built a Python API for 500 users\nSkills\nPython'), [])
    assert result.model == 'local-heuristic'
    assert 0 <= result.overall_score <= 100
    assert result.bullet_reviews

from app.main import skill_evidence, skill_details, combined_scores, Criterion

def test_negated_list_does_not_match():
    resume = 'No professional Python, FastAPI, Docker, LangChain, LangGraph, RAG, or vector database experience is claimed.'
    found, missing, required = skill_evidence(resume, 'AI Engineer', 'Python FastAPI Docker LangChain LangGraph RAG vector databases')
    assert found == []
    assert len(missing) == 7

def test_positive_evidence_and_no_jd_leakage():
    resume = 'Experience\nAI Engineer\n- Built production REST APIs using Python and FastAPI.'
    found, missing, required = skill_evidence(resume, 'AI Engineer', 'Python FastAPI Docker REST APIs')
    assert set(found) == {'Python', 'FastAPI', 'REST APIs'}
    assert 'Docker' in missing
    assert skill_details(resume, 'AI Engineer', 'Python FastAPI Docker REST APIs')[0]['evidence']

def test_low_relevance_is_not_inflated():
    criteria = {k: Criterion(score=2) for k in WEIGHTS}
    score, quality, found, missing = combined_scores(criteria, 'No Python, Docker or RAG experience.', 'AI Engineer', 'Python Docker RAG')
    assert score < 10
    assert found == []

def test_review_without_key_has_evidence():
    resume = 'Experience\nMarketing Manager\n- Managed ad campaigns\nSkills\nNo Python, Docker or RAG experience.'
    result = local_review(resume, 'AI Engineer', 'Python Docker RAG', extract_bullets(resume), [])
    assert result.matching_skills == []
    assert len(result.skill_evidence) == 3
    assert result.job_match_score == 0
