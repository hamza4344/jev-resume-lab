from app.main import rank_bullets, active_capabilities, semantic_relevance
from app.structured_parser import parse_structure

ROLE = 'Python Backend Engineer'
JD = 'Required Python, FastAPI, REST APIs, Docker and SQL. Build reliable backend services and optimize API latency.'

def test_two_rankings_and_job_priority():
    resume = '''EXPERIENCE
Senior AI Engineer | Northstar | Jan 2022 - Present
- Improved retrieval precision at five by 18 percentage points on a 600-query benchmark using hybrid search.
- Built production FastAPI REST APIs for document question answering.
- Reduced API latency from 3.2 seconds to 2.1 seconds after batching requests.
- Designed marketing posters for a product launch.
'''
    scored = rank_bullets(parse_structure(resume).bullets(), ROLE, JD)
    assert scored[1]['job_priority_position'] == 1
    assert scored[1]['dimensions']['relevance'] > scored[0]['dimensions']['relevance']
    assert scored[0]['achievement_strength_score'] == 100
    assert scored[0]['achievement_position'] < scored[1]['achievement_position']
    assert scored[2]['dimensions']['relevance'] > 0
    assert scored[3]['dimensions']['relevance'] == 0
    assert all(x['suggested_position'] == x['job_priority_position'] for x in scored)

def test_unrelated_metrics_do_not_win_tailored_ranking():
    resume = '''EXPERIENCE
Marketing Manager | Example | 2021 - Present
- Increased email clicks by 50%.
- Managed social content scheduling.
'''
    scored = rank_bullets(parse_structure(resume).bullets(), ROLE, JD)
    assert all(x['dimensions']['relevance'] == 0 for x in scored)
    assert all('job_priority_score' in x and 'achievement_strength_score' in x for x in scored)

def test_no_unrequested_capability_award():
    assert 'workflow' not in active_capabilities('Marketing Manager', 'Manage campaigns and content.')
    score, direct, related = semantic_relevance('Designed LangGraph workflows', 'Marketing Manager', 'Manage campaigns and content.', [])
    assert score == 0 and not related
