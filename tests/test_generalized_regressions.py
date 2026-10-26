"""Cross-profession safety & regression checks for role-aware ranking."""
import pytest
from app.main import rank_bullets, semantic_relevance, skill_details, active_capabilities
from app.structured_parser import parse_structure

BACKEND_ROLE = 'Python Backend Engineer'
BACKEND_JD = 'Required: Python, FastAPI, REST APIs, SQL, Docker. Build reliable backend services.'

@pytest.mark.parametrize('bullet,needed', [
    ('Reduced median latency from 3.2 seconds to 2.1 seconds after batching requests.', 'performance'),
    ('Developed regression tests, latency metrics, and monitoring dashboards.', 'testing_observability'),
    ('Created evaluation pipelines and traced latency over production requests.', 'testing_observability'),
    ('Deployed API services on Kubernetes with Docker containers.', 'deployment'),
    ('Designed LangGraph agent workflows with tool execution and tracing.', 'workflow'),
])
def test_backend_transfer_without_specific_jd_terms(bullet,needed):
    score, direct, related = semantic_relevance(bullet, BACKEND_ROLE, BACKEND_JD, [])
    assert score >= 20, (bullet,score,related)
    assert needed in related

@pytest.mark.parametrize('bullet', [
    'Increased email newsletter click-through rate from 2.4% to 3.1%.',
    'Designed an A/B test comparing promotional subject lines for marketing.',
    'Managed relationships with vendors for annual brand events.',
    'Published social media campaigns and developed brand messaging.',
    'Designed a marketing poster for a product launch.',
])
def test_marketing_bullets_not_backend_evidence(bullet):
    score,direct,related=semantic_relevance(bullet,BACKEND_ROLE,BACKEND_JD,[])
    assert score == 0, (bullet,score,related)
    assert direct == [] and related == []

@pytest.mark.parametrize('role,jd,bullet,capability', [
    ('Marketing Manager','Lead digital campaigns, email marketing and conversion performance',
     'Increased email click-through rate from 2.4% to 3.1%.','marketing_analytics'),
    ('Financial Analyst','Prepare budgets, financial forecasts, and cash-flow reports',
     'Prepared monthly cash flow forecasts and reconciled quarterly expense budgets.','financial_analysis'),
    ('Sales Manager','Lead sales and manage prospect pipelines',
     'Managed CRM prospect pipeline and closed enterprise sales deals.','sales_business_development'),
    ('Frontend Developer','Build frontend React applications and responsive web interfaces',
     'Built a responsive React frontend and deployed an accessible browser UI.','frontend_development'),
    ('Operations Manager','Improve supply chain and inventory operations',
     'Reduced warehouse inventory discrepancies by 12%.','supply_chain_operations'),
    ('AI Engineer','Build retrieval augmented generation services and evaluate models',
     'Designed RAG pipelines with embeddings and model inference.','ml_engineering'),
    ('Data Engineer','Manage SQL ETL pipelines and warehouse data quality',
     'Built scheduled SQL ETL data pipelines feeding a data warehouse.','data_integration'),
    ('QA Engineer','Implement test automation and regression validation',
     'Built automated regression tests using pytest.','testing_observability'),
    ('UX Designer','Conduct user research and usability testing',
     'Created Figma prototypes and ran user research interviews.','ux_design'),
    ('Customer Success Manager','Provide customer support and resolve tickets',
     'Resolved customer support tickets within SLA response times.','customer_support'),
])
def test_cross_profession_positive_relevance(role,jd,bullet,capability):
    score,_,related=semantic_relevance(bullet,role,jd,[])
    assert score>0, (role,score,related)
    assert capability in related

@pytest.mark.parametrize('role,jd,bullet', [
    ('Financial Analyst','Prepare budgets and financial forecasts','Built FastAPI APIs in Docker.'),
    ('Marketing Manager','Plan campaigns and brand strategy','Reduced median API latency through batching.'),
    ('Sales Manager','Lead a sales pipeline and prospecting','Designed a LangGraph agent workflow.'),
    ('Python Backend Engineer','FastAPI SQL Docker REST APIs','No experience with FastAPI APIs or Docker.'),
])
def test_unrelated_or_explicitly_denied(role,jd,bullet):
    score,direct,related=semantic_relevance(bullet,role,jd,[])
    assert score==0, (role,bullet,score,related)
    assert direct==[] and related==[]


def test_unknown_profession_can_match_grounded_jd_terms():
    score,direct,related=semantic_relevance(
        'Analyzed soil erosion and watershed sedimentation patterns.',
        'Watershed Hydrologist',
        'Study soil erosion and watershed sedimentation patterns.', [])
    assert score>0
    assert direct==[] and related==[]
    assert semantic_relevance('Managed warehouse shipping labels.',
        'Watershed Hydrologist','Study soil erosion and watershed sedimentation.',[])[0]==0


def test_role_with_no_description_still_has_safe_defaults():
    assert 'performance' in active_capabilities('Backend Engineer','')
    assert 'testing_observability' in active_capabilities('Backend Engineer','')
    assert 'workflow' not in active_capabilities('Marketing Manager','Manage campaigns and content.')


def test_job_priority_not_generic_impact_score():
    resume='''EXPERIENCE
Senior AI Engineer | Northstar | Jan 2022 - Present
- Improved retrieval precision by 18 percentage points on a 600-query benchmark.
- Built production FastAPI REST APIs for question answering.
- Containerized REST APIs with Docker.
- Developed regression tests and monitoring dashboards.
- Designed marketing posters.
'''
    scores=rank_bullets(parse_structure(resume).bullets(), BACKEND_ROLE, BACKEND_JD)
    assert scores[1]['job_priority_position']==1
    assert scores[1]['job_priority_score']>scores[0]['job_priority_score']
    assert scores[3]['dimensions']['relevance']>0
    assert scores[4]['dimensions']['relevance']==0
    assert scores[0]['achievement_strength_score'] > scores[1]['achievement_strength_score']
    assert scores[0]['achievement_position'] < scores[1]['achievement_position']
    assert all(s['ordering_score']==s['job_priority_score'] for s in scores)
    assert all(s['suggested_position']==s['job_priority_position'] for s in scores)
    assert all('capability_sources' in s['relevance_evidence'] for s in scores)


def test_missing_sql_stays_missing_and_common_sibling_tools_detected():
    text='''SUMMARY
Engineer shipping Python microservices.
SKILLS
Python, FastAPI, Redis, React, AWS, Git.
EXPERIENCE
- Built Python FastAPI services with Redis caching.'''
    jd='Required: Python, SQL, FastAPI, Redis, React, AWS, Git.'
    result={d['skill']:d for d in skill_details(text,'Backend Engineer',jd)}
    assert result['SQL']['status']=='not_demonstrated'
    for key in ['Python','FastAPI','Redis','React','AWS','Git']:
        assert result[key]['status']=='matched', key


def test_skill_negation_never_becomes_relevance():
    resume='No professional Python, FastAPI, Docker or SQL experience is claimed.'
    details=skill_details(resume,BACKEND_ROLE,BACKEND_JD)
    assert all(x['status']=='not_demonstrated' for x in details)
    score,direct,related=semantic_relevance(resume,BACKEND_ROLE,BACKEND_JD,[])
    assert score==0 and direct==[] and related==[]


def test_full_third_party_skill_negation_fixture():
    from pathlib import Path
    resume=(Path(__file__).resolve().parents[1]/'fixtures'/'04_negation_trap.txt').read_text()
    jd='Python, FastAPI, Docker, RAG, LangChain, LangGraph, SQL, REST APIs'
    details=skill_details(resume, 'Senior AI Engineer', jd)
    assert not any(x['status']=='matched' for x in details), details


def test_denial_of_other_skill_does_not_remove_positive_skill():
    details=skill_details('- Built FastAPI APIs in Python; did not use SQL.',
                          'Backend Engineer', 'FastAPI Python SQL')
    status={d['skill']:d['status'] for d in details}
    assert status['FastAPI']=='matched'
    assert status['Python']=='matched'
    assert status['SQL']=='not_demonstrated'


def test_known_role_avoids_generic_lexical_fallback():
    # Generic word 'performance' in a JD must not turn a marketing report
    # into evidence of API performance engineering.
    score, _, related = semantic_relevance('Prepared a campaign performance report.',
       'Python Backend Engineer','Build Python APIs, monitor performance and reliability.',[])
    assert score == 0 and related==[]
