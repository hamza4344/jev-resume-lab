"""Regression tests for passive technology mentions and PDF soft wraps.

A technology discussed in meetings is NOT demonstrated proficiency, even if
PDF extraction splits the explicit disclaimer onto another physical line.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, skill_details, semantic_relevance, rank_bullets
from app.structured_parser import parse_structure

FIXTURE = (Path(__file__).resolve().parents[1] / 'fixtures' /
           '04_negation_trap.txt').read_text(encoding='utf-8')
TARGET = 'Data Engineer'
JD = 'Required: Python, Docker, Monitoring, SQL, Airflow, RAG, LangChain, LangGraph, REST APIs.'


def wrapped_fixture():
    return FIXTURE.replace('cloud deployment, without implementing these technologies.',
                           'cloud\ndeployment, without implementing these technologies.')


def statuses(text):
    return {item['skill']: item['status'] for item in skill_details(text, TARGET, JD)}


def test_wrapped_disclaimer_recovered_without_creating_metadata():
    parsed = parse_structure(wrapped_fixture())
    assert len(parsed.experience) == 1
    row = parsed.experience[0]
    assert row.title == 'Marketing Analyst'
    assert row.organization == 'Eastfield Retail'
    assert not row.metadata
    assert len(row.responsibilities) == 3
    meeting = next(e for e in row.responsibilities if 'planning meetings' in e.text)
    assert meeting.source_line == 14
    assert meeting.text.endswith('cloud deployment, without implementing these technologies.')


@pytest.mark.parametrize('resume', [FIXTURE, wrapped_fixture()])
def test_passive_exposure_is_not_a_technical_skill_match(resume):
    found = statuses(resume)
    assert all(status == 'not_demonstrated' for status in found.values()), found


@pytest.mark.parametrize('resume', [FIXTURE, wrapped_fixture()])
def test_passive_exposure_has_no_technical_bullet_relevance(resume):
    bullets = parse_structure(resume).bullets()
    rated = rank_bullets(bullets, TARGET, JD)
    assert len(rated) == 3
    assert all(entry['dimensions']['relevance'] == 0 for entry in rated)
    assert all(not entry['relevance_evidence']['required_skills'] for entry in rated)
    assert all(not entry['relevance_evidence']['transferable_capabilities'] for entry in rated)
    assert all(not entry['relevance_evidence']['matched_job_terms'] for entry in rated)


@pytest.mark.parametrize('text', [
    'Attended planning meetings about Docker, Python and RAG.',
    'Discussed Docker deployment in planning meetings.',
    'Observed Docker deployment in a technical demo.',
    'Joined workshops about Docker and Python.',
    'Attended planning meetings about Docker; implemented a marketing campaign.',
    'Documented user requirements for Docker but did not implement Docker.',
])
def test_technology_exposure_is_not_implementation(text):
    result = statuses(text)
    assert result['Docker'] == 'not_demonstrated'
    assert semantic_relevance(text, TARGET, JD, [])[0] == 0


def test_separate_genuine_implementation_still_counts():
    text = 'Attended meetings about Docker; later deployed Docker containers for a Python service.'
    details = statuses(text)
    assert details['Docker'] == 'matched'
    assert semantic_relevance(text, TARGET, JD, [])[0] > 0


def test_unrelated_implementation_does_not_credit_incidental_docker():
    text = 'Attended meetings about Docker; implemented Python REST APIs.'
    details = statuses(text)
    assert details['Docker'] == 'not_demonstrated'
    assert details['Python'] == 'matched'
    assert details['REST APIs'] == 'matched'


def test_api_wrap_and_evidence_results_agree(monkeypatch):
    monkeypatch.delenv('BEATAPI_API_KEY', raising=False)
    monkeypatch.delenv('APP_API_KEY', raising=False)
    client = TestClient(app)
    text = wrapped_fixture().encode('utf-8')
    parse_response = client.post('/api/parse', files={'file': ('bilal.txt', text, 'text/plain')})
    assert parse_response.status_code == 200, parse_response.text
    parsed = parse_response.json()['structured_resume']
    assert parsed['experience'][0]['metadata'] == []
    assert len(parsed['experience'][0]['responsibilities']) == 3
    response = client.post('/api/review',
                           files={'file': ('bilal.txt', text, 'text/plain')},
                           data={'target_role': TARGET, 'job_description': JD})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['matching_skills'] == []
    assert all(e['status'] == 'not_demonstrated' for e in result['skill_evidence'])
    assert all(b['dimensions']['relevance'] == 0 for b in result['bullet_reviews'])
    assert result['job_match_score'] <= 10


def test_comma_separated_exposure_is_not_an_unlabeled_skills_list():
    lines = [
        'Exposure to Docker, Python, SQL, and Airflow.',
        'Discussed Docker, SQL, and Airflow.',
        'Attended meetings about Docker, RAG, Python, and SQL.',
    ]
    for line in lines:
        assert not any(d['status'] == 'matched' for d in skill_details(line, TARGET, JD)), line


def test_compact_skills_list_without_a_heading_still_works():
    details = statuses('Python, Docker, Airflow, SQL')
    assert all(details[name] == 'matched' for name in ('Python', 'Docker', 'Airflow', 'SQL'))


def test_mocked_jev_cannot_reintroduce_docker_match(monkeypatch):
    from app.main import JevClient
    monkeypatch.setenv('BEATAPI_API_KEY', 'DUMMY-TEST-KEY')
    monkeypatch.delenv('APP_API_KEY', raising=False)

    async def fake_ask(self, state, questions):
        return {'model': 'jev-1.14-free',
                'answers': {name: ({'type': 'score', 'score': 0.3} if question['type'] == 'score'
                                   else {'type': 'noul', 'noul': 0.1})
                            for name, question in questions.items()}}

    monkeypatch.setattr(JevClient, 'ask', fake_ask)
    response = TestClient(app).post(
        '/api/review',
        files={'file': ('sample.txt', wrapped_fixture().encode(), 'text/plain')},
        data={'target_role': TARGET, 'job_description': JD},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['matching_skills'] == []
    assert payload['model'] == 'jev-1.14-free'
    assert payload['structured_resume']['experience'][0]['metadata'] == []
    assert all(item['dimensions']['relevance'] == 0 for item in payload['bullet_reviews'])
