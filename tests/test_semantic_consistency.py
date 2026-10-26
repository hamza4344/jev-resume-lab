from app.structured_parser import parse_structure
from app.main import rank_bullets

ROLE="Python Backend Engineer"
JD="Required: Python, FastAPI, REST APIs, SQL and Docker. Build reliable backend services."

def test_separate_responsibilities_achievements_outcomes():
    resume="""EXPERIENCE
Backend Engineer | Acme | Jan 2022 - Present
- Built production FastAPI REST APIs.
- Improved throughput through caching.
- Reduced API latency from 3.2 seconds to 2.1 seconds.
PROJECTS
Cloud Integration | 2024
Objective: Improve service availability.
Responsibilities: Implemented queue-based workflows.
Outcome: Reduced request processing time by 20%.
"""
    parsed=parse_structure(resume)
    work=parsed.experience[0]
    assert [x.kind for x in work.responsibilities]==['responsibility']
    assert len(work.achievements)==1
    assert len(work.outcomes)==1
    assert parsed.projects[0].responsibilities[0].text=='Implemented queue-based workflows.'
    assert len(parsed.projects[0].outcomes)==1
    assert len(parsed.bullets())==5

def test_semantic_backend_transfer_and_evidence():
    resume="""EXPERIENCE
AI Engineer | Beacon | Jan 2022 - Present
- Built FastAPI REST APIs.
- Designed LangGraph agent workflows with tool execution and tracing.
- Reduced median answer latency from 3.2 seconds to 2.1 seconds after batching requests.
- Designed a marketing flyer.
"""
    bullet_scores=rank_bullets(parse_structure(resume).bullets(),ROLE,JD)
    assert bullet_scores[0]['dimensions']['relevance']>0
    assert bullet_scores[1]['dimensions']['relevance']>0
    assert 'workflow' in bullet_scores[1]['relevance_evidence']['transferable_capabilities']
    assert bullet_scores[2]['dimensions']['relevance']>0
    assert 'performance' in bullet_scores[2]['relevance_evidence']['transferable_capabilities']
    assert bullet_scores[3]['dimensions']['relevance']==0
    assert all(0 <= x['dimensions']['relevance'] <= 100 for x in bullet_scores)

def test_unrelated_marketing_does_not_gain_backend_relevance():
    text="""EXPERIENCE
Marketing Lead | Sample | Jan 2022 - Present
- Managed social media publishing and promotional campaigns.
"""
    scores=rank_bullets(parse_structure(text).bullets(),ROLE,JD)
    assert scores[0]['dimensions']['relevance']==0
