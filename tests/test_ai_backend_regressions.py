from app.structured_parser import parse_structure
from app.main import skill_details

ROLE = 'Python Backend Engineer'
JD = 'Required: Python, FastAPI, REST APIs, SQL, Docker.'
TEXT = '''Samira Qureshi
Senior AI Engineer
PROFESSIONAL SUMMARY
AI engineer with six years of experience shipping Python services, retrieval systems and AI workflows.
TECHNICAL SKILLS
Python, FastAPI, LangChain, LangGraph, RAG, Docker, AWS, REST APIs.
PROFESSIONAL EXPERIENCE
Senior AI Engineer | Northstar Systems | Jan 2022 - Present
- Built FastAPI REST APIs.
- Improved retrieval with hybrid search and
reranking.
AI Engineer | Beacon Labs | Jul 2020 - Dec 2021
- Integrated LangChain retrievers.
PROJECTS
Enterprise Support Copilot | 2024
Objective: Provide cited answers.
Responsibilities: Implemented ingestion and REST API endpoints.
Outcome: Reduced lookup time by 22%.
'''

def test_multiple_employment_headers_and_wrap():
    p=parse_structure(TEXT)
    assert len(p.experience)==2
    assert p.experience[0].title=='Senior AI Engineer'
    assert p.experience[0].organization=='Northstar Systems'
    assert p.experience[1].title=='AI Engineer'
    assert p.experience[1].organization=='Beacon Labs'
    assert 'and reranking.' in ' '.join(x.text for x in p.experience[0].responsibilities + p.experience[0].achievements + p.experience[0].outcomes)

def test_labeled_project_responsibility():
    p=parse_structure(TEXT)
    assert len(p.projects)==1
    assert p.projects[0].title=='Enterprise Support Copilot | 2024'
    assert p.projects[0].responsibilities[0].text=='Implemented ingestion and REST API endpoints.'

def test_python_skills_and_no_sql_hallucination():
    details={d['skill']:d for d in skill_details(TEXT, ROLE, JD)}
    assert details['Python']['status']=='matched'
    assert details['SQL']['status']=='not_demonstrated'

def test_negative_skill_list_does_not_match():
    resume='TECHNICAL SKILLS\nNo professional Python, FastAPI, REST API, Docker or SQL experience is claimed.'
    details=skill_details(resume, ROLE, JD)
    assert not any(d['status']=='matched' for d in details)
