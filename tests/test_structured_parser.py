from pathlib import Path
from app.structured_parser import parse_structure
from app.main import extract_bullets, Review

SAMPLE = Path(__file__).resolve().parents[1] / 'fixtures' / 'marketing_resume.txt'

def test_marketing_resume_project_metadata():
    data = parse_structure(SAMPLE.read_text(encoding='utf-8'))
    assert len(data.experience) == 3
    assert len(data.projects) == 4
    assert [p.title for p in data.projects] == [
        'MULTICHANNEL PRODUCT LAUNCH — 2024',
        'CUSTOMER LOYALTY EMAIL REFRESH — 2023',
        'REGIONAL RETAIL PROMOTION OPERATIONS — 2022',
        'CUSTOMER FEEDBACK AND MESSAGING REVIEW — 2021',
    ]
    assert all(len(p.objectives)==1 and len(p.outcomes)==1 for p in data.projects)
    assert all(p.title for p in data.projects)
    assert len(data.bullets()) == 49
    assert not any('Project objective:' in b['role'] or b['role']=='Responsibilities:' for b in data.bullets())
    assert not any(b['text'].startswith(('Project objective:', 'Responsibilities:')) for b in data.bullets())

def test_project_heading_is_not_scored():
    data = parse_structure('PROJECTS\n1. MY CUSTOM LAUNCH — 2024\nProject objective: Launch the thing.\nResponsibilities:\n- Built a test campaign.\nOutcome: Improved conversion.\nEDUCATION\nBSc')
    assert len(data.projects)==1
    assert data.projects[0].title == 'MY CUSTOM LAUNCH — 2024'
    assert len(data.bullets())==2
    assert data.projects[0].objectives[0].text=='Launch the thing.'

def test_employment_multiple_entries_and_sections():
    doc = 'EXPERIENCE\nSoftware Engineer\nACME | January 2022 – Present\n- Built a Python service.\nData Engineer\nBeta | 2020 – 2022\n- Automated reports.\nEDUCATION\n- Bachelor Degree'
    data=parse_structure(doc)
    assert len(data.experience)==2
    assert [r.title for r in data.experience]==['Software Engineer','Data Engineer']
    assert data.experience[0].organization=='ACME'
    assert len(data.bullets())==2
    assert 'education' in data.other_sections

def test_objective_not_role_if_title_missing():
    doc='PROJECTS\nProject objective: Improve adoption.\nResponsibilities:\n- Built onboarding materials.'
    data=parse_structure(doc)
    assert data.projects[0].title is None
    assert len(data.projects[0].objectives)==1
    assert data.bullets()[0]['role']=='Unspecified project'
    assert data.warnings

def test_py39_pydantic_review_schema():
    assert 'structured_resume' in Review.model_fields
