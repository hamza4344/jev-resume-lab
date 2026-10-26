from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import io, os, re, json, asyncio, math
from typing import Any, Optional, Dict, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from pydantic import BaseModel, Field
import httpx
from pypdf import PdfReader
from docx import Document
from app.structured_parser import parse_structure, logical_resume_lines

app = FastAPI(title='Jev Resume Match API', version='0.5.0')
MAX_FILE = 5 * 1024 * 1024
WEIGHTS = {'summary':.10,'achievement':.25,'impact':.30,'ownership':.10,'relevance':.10,'ordering':.05,'context':.05,'structure':.03,'skills':.02}
RUBRICS = {
 'summary':'How clearly does the resume summary demonstrate value for the target role?',
 'achievement':'How often does the experience describe achievements rather than duties?',
 'impact':'How credible and specific is the evidence of outcomes and impact?',
 'ownership':'How clearly does the resume show ownership and initiative?',
 'relevance':'How well does documented work experience directly satisfy core job requirements? Score 0 if no job-specific role experience, even when candidate has years of unrelated experience. Do not infer from generic words.',
 'ordering':'Are the most relevant accomplishments presented prominently?',
 'context':'Is the context of each employer and role understandable?',
 'structure':'How readable and logically organized is the extracted resume text?',
 'skills':'How many REQUIRED job skills are evidenced through resume projects or employment? Score 0 for none; do not count a skill that is explicitly denied or only mentioned in a different context.'
}
LEVELS = ['0: no documented evidence for this specific criterion', '1: weak or indirect evidence only', '2: some direct and credible evidence', '3: strong direct evidence across multiple examples']
class Criterion(BaseModel):
    score: float = Field(ge=0,le=100)
    confidence: Optional[float] = Field(default=None,ge=0,le=1)
    probabilities: dict[str,float] = Field(default_factory=dict)
class Review(BaseModel):
    overall_score: float
    job_match_score: float = 0
    resume_quality_score: float = 0
    matching_skills: list[str] = Field(default_factory=list)
    partial_required_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    unmet_required_skills: list[str] = Field(default_factory=list)
    skill_coverage_score: float = Field(default=0, ge=0, le=100)
    skill_evidence: list[dict[str, Any]] = Field(default_factory=list)
    criteria: dict[str,Criterion]
    quick_checks: dict[str,float]
    bullet_reviews: list[dict[str,Any]]
    warnings: list[str]
    model: str
    structured_resume: dict[str, Any] = Field(default_factory=dict)
    note: str = 'Measures evidence in resume text, not candidate ability or hiring probability.'

def parse_resume(filename: str, data: bytes) -> tuple[str,list[str]]:
    suffix = filename.lower().rsplit('.',1)[-1]
    warnings=[]
    try:
        if suffix=='pdf':
            reader=PdfReader(io.BytesIO(data))
            if len(reader.pages)>30: raise ValueError('PDF exceeds 30 pages')
            text='\n'.join(p.extract_text() or '' for p in reader.pages)
            if len(text.strip())<100: warnings.append('PDF contains little extractable text; OCR may be needed.')
        elif suffix=='docx':
            doc=Document(io.BytesIO(data))
            text='\n'.join([p.text for p in doc.paragraphs]+[c.text for t in doc.tables for r in t.rows for c in r.cells])
        elif suffix=='txt': text=data.decode('utf-8-sig')
        else: raise ValueError('Only PDF, DOCX and TXT files are accepted')
    except ValueError: raise
    except Exception as e: raise ValueError('Invalid or unreadable document') from e
    text=text.replace('\x00', '').strip()
    if not text: raise ValueError('No extractable text; run OCR first')
    if len(text)>40000: raise ValueError('Extracted text exceeds 40,000 characters')
    return text,warnings

SECTION_HEADINGS = re.compile(r"^(?:professional\s+|work\s+|relevant\s+)?(?:experience|employment|work history|internships|projects|selected projects|selected marketing projects|education|skills|technical skills|certifications|awards|publications|references|summary|profile)(?:\s*:)?$", re.I)
EXPERIENCE_HEADINGS = {'experience', 'professional experience', 'work experience', 'relevant experience', 'employment', 'work history', 'internships'}
PROJECT_HEADINGS = {'projects', 'selected projects', 'selected marketing projects'}
END_HEADINGS = {'education', 'skills', 'technical skills', 'certifications', 'awards', 'publications', 'references', 'summary', 'profile'}
LABEL_ONLY = re.compile(r"^(?:responsibilities|key responsibilities|achievements|key achievements|duties|highlights|selected achievements|project overview|technologies used|contributions)\s*:?$", re.I)
BULLET_MARKER = re.compile(r"^(?:[•*\-–—]\s+|\d+[.)]\s+)")
DATE_RANGE = re.compile(r"(?:20\d{2}|19\d{2})\s*(?:[-–—]|to)\s*(?:present|current|20\d{2}|19\d{2})", re.I)
ACTION_START = re.compile(r"^(?:built|developed|designed|implemented|delivered|improved|reduced|increased|led|created|launched|automated|optimized|managed|deployed|scaled|planned|analyzed|consolidated|coordinated|reviewed|prepared|supported|maintained|drafted|organized|assisted|presented|worked|conducted|produced|wrote|helped|ensured|segmented|grouped|collected|held|monitored|integrated|engineered|migrated|introduced|established)\b", re.I)

def extract_bullets(text: str) -> list[dict]:
    """Backwards-compatible adapter for the structured resume parser."""
    return parse_structure(text).bullets()


def score_answer(answer:dict) -> Criterion:
    if answer.get('type')!='score': raise ValueError('Unexpected Jev answer type')
    raw=float(answer['score'])
    if not math.isfinite(raw) or not 0<=raw<=3: raise ValueError('Jev score outside 0-3')
    return Criterion(score=round(raw/3*100,2),confidence=answer.get('confidence'),probabilities=answer.get('probabilities',{}))

def noul_answer(answer:dict)->float:
    if answer.get('type')!='noul': raise ValueError('Unexpected Jev yes/no answer')
    p=float(answer['noul'])
    if not math.isfinite(p) or not 0<=p<=1: raise ValueError('Jev probability outside 0-1')
    return round(p*100,2)

def score_question(prompt:str)->dict:
    return {'type':'score','instructions':prompt,'criteria':LEVELS}

def noul_question(prompt:str)->dict:
    return {'type':'noul','instructions':prompt,'criteria':{'true':'Clear evidence in the supplied resume','false':'Not evidenced or contradicted by the supplied resume'}}


# Requirement detection and candidate evidence are deliberately separate.
# Job-description text NEVER counts as evidence of a candidate skill.
SKILL_ALIASES = {
  'Python': ['python'], 'FastAPI': ['fastapi'],
  'RAG': ['rag', 'retrieval augmented generation', 'retrieval-augmented generation'],
  'LangChain': ['langchain'], 'LangGraph': ['langgraph'], 'Docker': ['docker'],
  'Kubernetes': ['kubernetes', 'k8s'],
  'Vector databases': ['vector database', 'vector databases', 'pinecone', 'weaviate', 'chroma', 'faiss'],
  'REST APIs': ['rest api', 'rest apis', 'restful api', 'restful apis'],
  'LLMs': ['llm', 'llms', 'large language model', 'large language models'],
  'AI agents': ['ai agents', 'ai agent', 'agentic ai', 'multi-agent systems'],
  'Cloud deployment': ['cloud deployment', 'aws', 'azure', 'gcp'],
  'Evaluation pipelines': ['evaluation pipeline', 'evaluation pipelines', 'llm evaluation'],
  'Monitoring': ['production monitoring', 'observability', 'langfuse', 'prometheus'],
  'Machine learning': ['machine learning', 'ml models'],
  'SQL': ['sql', 'postgresql', 'mysql', 'sql server', 'sqlite'],
  # Common software/data/marketing tool requirements. Extend this vocabulary
  # through explicit aliases; unseen niche technologies are NOT inferred.
  'JavaScript': ['javascript'], 'TypeScript': ['typescript'],
  'React': ['reactjs', 'react.js', 'react'], 'Node.js': ['node.js', 'nodejs'],
  'Django': ['django'], 'Flask': ['flask'], 'Java': ['java'],
  'C#': ['c#', 'c sharp'], 'Go': ['golang', 'go programming language'],
  'AWS': ['aws', 'amazon web services'], 'Azure': ['microsoft azure', 'azure'],
  'GCP': ['gcp', 'google cloud platform'], 'Terraform': ['terraform'],
  'CI/CD': ['ci/cd', 'ci-cd', 'continuous integration and continuous delivery',
            'continuous integration and continuous deployment',
            'continuous integration and delivery', 'continuous integration and deployment'],
  'Git': ['git', 'github', 'gitlab'],
  'PyTorch': ['pytorch'], 'TensorFlow': ['tensorflow'],
  'Airflow': ['apache airflow', 'airflow'], 'Spark': ['apache spark', 'pyspark'],
  'MongoDB': ['mongodb', 'mongo db'], 'Redis': ['redis'],
  'Power BI': ['power bi', 'powerbi'], 'Tableau': ['tableau'],
  'Excel': ['microsoft excel', 'excel'], 'Figma': ['figma'],
  'Salesforce': ['salesforce'], 'HubSpot': ['hubspot'],
  'Google Analytics': ['google analytics', 'ga4'], 'SEO': ['seo', 'search engine optimization'],
  'Google Ads': ['google ads', 'google adwords'],
}
# These expressions cover explicit lack of *experience*, not generic nouns like
# "no code required" (which can be part of a positive project statement).
NEGATION = re.compile(
    r'\b(?:no|not|without|never|lack(?:s|ed|ing)?|did\s+not|does\s+not|'
    r'don\x27t|cannot|can\x27t|neither|nor|unfamiliar\s+with|'
    r'no\s+(?:professional|hands-on|practical)\s+experience)\b', re.I)
NEGATIVE_SCOPE = re.compile(r'\b(?:no|not|without|never|lack(?:s|ed|ing)?|did\s+not|does\s+not)\b', re.I)
NEGATIVE_LIST_START = re.compile(
    r'\b(?:no|without|lack(?:s|ed|ing)?)\s+(?:(?:professional|hands-on|practical|working|commercial)\s+)*'
    r'(?:(?:experience|expertise|background|knowledge|skills|work|projects|development|training|coding|implementation)\s+(?:in|with|of|using|on)\s+)?', re.I)
POSITIVE_VERBS = re.compile(
    r'\b(?:built|developed|implemented|designed|deployed|integrated|maintained|created|optimized|'
    r'engineered|delivered|configured|configuring|set up|automated|used|using|worked with|hands-on|experienced with|proficient in|'
    r'expertise in|skilled in|implemented with|powered by)\b', re.I)


def _term_matches(text: str, phrase: str):
    # Prevent matching 'RAG' inside 'storage' or SQL inside PostgreSQL.
    pattern = r'(?<![a-z0-9])' + re.escape(phrase) + r'(?![a-z0-9])'
    return list(re.finditer(pattern, text, re.I))


def _term_present(text: str, phrase: str) -> bool:
    """Used for requirements only; never evidence of actual proficiency."""
    return bool(_term_matches(text, phrase))


def _evidence_lines(resume: str):
    # Split comma-delimited lists only for NEGATION logic, not for display.
    # Each resume line remains an auditable evidence quote.
    return [item.text.strip(' •-*\t') for item in logical_resume_lines(resume)
            if item.text.strip()]


def _evidence_with_context(resume: str):
    """Return exact candidate quotes with whether they belong to a skill list.

    Standalone technologies beneath a SKILLS heading are direct *claims*,
    while the same words in ordinary prose are not automatically proficiency.
    """
    in_skills = False
    skills_headers = {'skills', 'technical skills', 'tech stack', 'technologies',
                      'tools and technologies', 'core skills', 'technical competencies',
                      'core technologies', 'technology stack'}
    other_headers = {'summary', 'profile', 'experience', 'professional experience',
                     'work experience', 'employment', 'work history', 'career journey',
                     'projects', 'selected work', 'portfolio', 'education',
                     'certifications', 'awards', 'publications', 'references',
                     'volunteer experience', 'interests', 'languages'}
    for line in _evidence_lines(resume):
        heading = line.rstrip(':').strip().casefold()
        if heading in skills_headers:
            in_skills = True
            continue
        if heading in other_headers:
            in_skills = False
            continue
        yield line, in_skills


def _is_denied(line: str, match_start: int, alias: Optional[str] = None) -> bool:
    """Suppress explicit negative claims and descriptions of requirements only."""
    prefix = line[:match_start]
    clause = re.split(r'[.;]|\b(?:however|although|but)\b', prefix, flags=re.I)[-1]
    if NEGATIVE_LIST_START.search(clause):
        return True
    if NEGATIVE_SCOPE.search(clause[-90:]):
        return True
    tail = line[match_start:]
    if re.search(r'\b(?:experience\s+is\s+not\s+claimed|not\s+claimed|not\s+used|not\s+developed|not\s+experienced|no\s+experience)\b', tail, re.I):
        return True
    # Explicit downstream denial of the *same* skill overrides an earlier
    # incidental positive mention: 'used reports from Python services; did
    # not write Python code'. A denial of a *different* skill must not leak.
    if alias:
        for denial in NEGATIVE_SCOPE.finditer(tail):
            # Negation generally scopes only to the next clause, not an entire
            # paragraph or subsequent positive claims.
            scope = re.split(r'[.;]|\b(?:however|although|but)\b', tail[denial.end():], maxsplit=1, flags=re.I)[0]
            if _term_present(scope, alias):
                return True
        # Candidate did not implement this work; it belonged to other staff.
        if re.search(r'\b(?:developed|implemented|built|written|maintained|created)\s+by\s+(?:other|another|external|third[- ]party)\b',tail,re.I) and \
           not re.search(r'\b(?:i|my team|we)\s+(?:built|developed|implemented|used|wrote)\b',line,re.I):
            return True
    return False

# An incidental technology mention is not proof of hands-on use.  These
# sentences are common in nontechnical resumes and are especially risky when
# PDF extraction wraps the denial onto the next physical line.
PASSIVE_EXPOSURE = re.compile(
    r'\b(?:attended|observed|watched|discussed|heard about|read about|'
    r'participated in|sat in|joined|listened to|took notes (?:on|about)|'
    r'documented (?:user |business )?requirements? (?:for|about)|'
    r'worked alongside|collaborated with)\b.{0,110}\b'
    r'(?:meetings?|planning|workshops?|conferences?|seminars?|sessions?|presentations?|demos?|discussions?|'
    r'about|regarding|relating to|requirements?|engineers?|developers?)\b', re.I)
PASSIVE_DISCLAIMER = re.compile(
    r'\b(?:without\s+(?:personally\s+)?(?:implementing|building|coding|'
    r'developing|deploying|configuring|using|managing)|'
    r'(?:did not|not|never)\s+(?:implement|build|code|develop|deploy|configure|use|manage)|'
    r'(?:not|never)\s+responsible for|'
    r'(?:implemented|built|developed|maintained|deployed)\s+by\s+'
    r'(?:other|another|external|third[- ]party))\b', re.I)


def _non_owned_claim(line: str, alias: Optional[str] = None) -> bool:
    """Conservatively reject passive references without a hands-on claim.

    A later, independent clause with concrete implementation can still count.
    Ownership is not inferred merely from planning, requirements or meetings.
    """
    if not (PASSIVE_EXPOSURE.search(line) or PASSIVE_DISCLAIMER.search(line)):
        return False
    pieces = re.split(
        r'\s*[;.]\s*|\b(?:but|then|however|subsequently)\b|'
        r'\band\s+(?=(?:built|implemented|deployed|coded|programmed|configured|developed|wrote|engineered|maintained|containerized|automated)\b)',
        line, flags=re.I)
    concrete = re.compile(
        r'\b(?:built|implemented|deployed|coded|programmed|configured|'
        r'developed|wrote|engineered|maintained|containerized|automated)\b', re.I)
    return not any(concrete.search(piece) and
                   (alias is None or _term_present(piece, alias)) and
                   not PASSIVE_DISCLAIMER.search(piece) and
                   not PASSIVE_EXPOSURE.search(piece)
                   for piece in pieces)

META_LINE = re.compile(r'\b(?:required|requirements|job description|looking for|seeking|must have|should have|nice to have|missing skills|skills to learn|interested in learning|currently learning|no experience)\b', re.I)

def _positive_skill_line(line: str, alias: str, in_skills_section: bool = False) -> bool:
    matches = _term_matches(line, alias)
    if not matches:
        return False
    if _non_owned_claim(line, alias):
        return False
    if META_LINE.search(line) and not re.search(r'\b(?:developed|built|deployed|implemented|used|using|worked with|hands-on|experienced with|proficient in|expertise in)\b', line, re.I):
        return False
    if re.match(r'^\s*(?:skills|technical skills|technologies|tools|tech stack)\s*:', line, re.I):
        return any(not _is_denied(line, m.start(), alias) for m in matches)
    if in_skills_section and len(line) <= 350 and not NEGATION.search(line):
        # A single 'CI/CD' or 'Python' under a skills heading is a positive
        # resume claim; the same text under education or interests is not.
        return any(not _is_denied(line, m.start(), alias) for m in matches)
    # Standalone skills list under TECHNICAL SKILLS (without a colon) and
    # summary claims such as 'experience shipping Python services' count as
    # explicitly claimed experience, never as independently verified mastery.
    # Only compact, label-like comma lists count outside an explicit skills
    # section.  A long narrative containing three technology names (e.g.
    # "Attended meetings about Docker, RAG, Python") is not a skills list.
    first_list_item = line.split(',', 1)[0].strip(' •-*\t')
    is_technical_list = (',' in line and len(line) < 250 and
                         len(line.split(',')) >= 3 and
                         len(first_list_item.split()) <= 2 and
                         not re.search(r'\b(?:exposure|attended|discussed|observed|'
                                       r'watched|participated|joined|planning|'
                                       r'meeting|considered|interested|learning|'
                                       r'aware of|heard of|read about)\b', line, re.I) and
                         not META_LINE.search(line) and
                         not NEGATION.search(line))
    experiential_claim = bool(re.search(r'\b(?:experience|experienced)\b.{0,55}\b(?:shipping|building|developing|using|with|in)\b', line, re.I))
    if not (POSITIVE_VERBS.search(line) or ACTION_START.match(line) or is_technical_list or experiential_claim):
        return False
    return any(not _is_denied(line, m.start(), alias) for m in matches)


# Compound job requirements require EVIDENCE FOR EACH component. Mentioning
# one part does not mean that the candidate performed the entire activity.
# These requirements are activated ONLY when the JD names the compound concept,
# not when a JD happens to mention one of its components.
COMPOUND_REQUIREMENTS = {
    'CI/CD': {
        'requirement_aliases': SKILL_ALIASES['CI/CD'],
        'components': {
            'CI': ['ci/cd', 'ci-cd', 'continuous integration', 'ci pipeline',
                   'ci pipelines', 'ci workflow', 'ci workflows',
                   'automated build and test pipeline', 'automated build-and-test pipeline',
                   'github actions ci', 'jenkins ci', 'pull request build checks'],
            'CD': ['ci/cd', 'ci-cd', 'continuous delivery', 'continuous deployment',
                   'cd pipeline', 'cd pipelines', 'deployment pipeline',
                   'deployment pipelines', 'release pipeline', 'release pipelines',
                   'automated deployment pipeline', 'automated deployments',
                   'automated production releases', 'automated production deployments'],
        },
    },
    'Testing and monitoring': {
        'requirement_aliases': ['testing and monitoring', 'testing & monitoring',
                                'test automation and monitoring',
                                'testing and observability'],
        'components': {
            'Testing': ['automated tests', 'automated testing', 'regression tests',
                        'integration tests', 'unit tests', 'pytest', 'test automation',
                        'test suite', 'qa testing'],
            'Monitoring': ['monitoring alerts', 'monitoring dashboards',
                           'production monitoring', 'observability', 'prometheus',
                           'grafana', 'telemetry monitoring', 'service monitoring',
                           'error monitoring', 'application monitoring'],
        },
    },
    'Cloud architecture': {
        'requirement_aliases': ['cloud architecture', 'cloud solution architecture',
                                'cloud infrastructure architecture'],
        'components': {
            'Cloud platform': ['aws', 'azure', 'gcp', 'google cloud platform',
                               'amazon web services', 'cloud infrastructure'],
            'Architecture': ['architected cloud', 'cloud architecture',
                             'cloud infrastructure design', 'designed cloud infrastructure',
                             'designed cloud systems', 'cloud system design',
                             'designed distributed cloud', 'cloud solutions architect'],
        },
    },
}

def _named_requirements(target: str) -> list[str]:
    names = [skill for skill, aliases in SKILL_ALIASES.items()
             if any(_term_present(target, alias) for alias in aliases)]
    for skill, config in COMPOUND_REQUIREMENTS.items():
        if skill not in names and any(_term_present(target, term)
                                      for term in config['requirement_aliases']):
            names.append(skill)
    return names


def _component_explicitly_denied(line: str, aliases: list[str], part: str) -> bool:
    """Ignore explicit local component denials, including bare CI or CD tokens.

    'Built CI/CD workflows but did not implement CD' => only CI is credited.
    A denial separated by 'but' from a different positive claim must not
    propagate to the unrelated part.
    """
    probes = list(aliases) + ([part] if part in {'CI', 'CD'} else [])
    for negative in NEGATIVE_SCOPE.finditer(line):
        following = re.split(r'[.;]|\b(?:however|although|but)\b',
                             line[negative.end():], maxsplit=1, flags=re.I)[0]
        if any(_term_present(following, alias) for alias in probes):
            return True
    return False


def _third_party_pipeline_only(line: str) -> bool:
    # Merely consuming third-party build artifacts is not implementing a CI/CD
    # pipeline. A candidate-owned implementation in the same statement is
    # recognized; this guard is intentionally conservative.
    artifacts = re.search(
        r'\b(?:artifacts?|reports?|outputs?|results?)\s+'
        r'(?:generated|provided|produced|built|created)\s+by\s+'
        r'(?:other|another|external|third[- ]party)\b', line, re.I)
    outside_work = re.search(
        r'\b(?:pipelines?|workflows?|automation)\s+'
        r'(?:built|created|maintained|implemented|managed)\s+by\s+'
        r'(?:other|another|external|third[- ]party)\b', line, re.I)
    owned_work = re.search(
        r'\b(?:i|we|our team)\s+(?:built|configured|implemented|designed|'
        r'maintained|deployed|managed|automated)\s+(?:\w+\s+){0,5}'
        r'(?:ci/cd|ci|cd|pipeline|workflow)\b', line, re.I)
    return bool((artifacts or outside_work) and not owned_work)


def _compound_skill_details(resume: str, name: str) -> dict:
    config = COMPOUND_REQUIREMENTS[name]
    evidence_by_component = {}
    for part, aliases in config['components'].items():
        evidence = []
        for line, in_skills in _evidence_with_context(resume):
            if _non_owned_claim(line):
                continue
            # Both parts need their OWN positive evidence. A component-specific
            # disclaimer negates only that component, not the entire resume.
            if _third_party_pipeline_only(line) and name == 'CI/CD':
                continue
            if _component_explicitly_denied(line, aliases, part):
                continue
            if any(_positive_skill_line(line, alias, in_skills) for alias in aliases):
                evidence.append(line[:350])
            if len(evidence) == 3:
                break
        evidence_by_component[part] = evidence
    demonstrated = [part for part, items in evidence_by_component.items() if items]
    unverified = [part for part in config['components'] if part not in demonstrated]
    status = ('matched' if not unverified else
              'partially_demonstrated' if demonstrated else 'not_demonstrated')
    # De-duplicate quote strings while preserving resume source order.
    evidence = list(dict.fromkeys(item for part in config['components']
                                  for item in evidence_by_component[part]))
    return {'skill': name, 'status': status, 'evidence': evidence,
            'demonstrated_components': demonstrated,
            'unverified_components': unverified,
            'component_evidence': evidence_by_component}

def skill_details(resume: str, role: str, description: str):
    target = role + '\n' + description
    details = []
    for skill in _named_requirements(target):
        if skill in COMPOUND_REQUIREMENTS:
            details.append(_compound_skill_details(resume, skill))
            continue
        evidence = []
        for line, in_skills in _evidence_with_context(resume):
            for alias in SKILL_ALIASES[skill]:
                if _positive_skill_line(line, alias, in_skills):
                    # A positive mention is claimed evidence, never a
                    # verified qualification or certification.
                    evidence.append(line[:350])
                    break
            if len(evidence) >= 3:
                break
        details.append({'skill': skill,
                        'status': 'matched' if evidence else 'not_demonstrated',
                        'evidence': evidence})
    return details


def skill_evidence(resume: str, role: str, description: str):
    """Compatibility adapter: partial requirements are NOT full matches.

    The 'missing' tuple member means unfulfilled (partial or missing), as
    older callers expect. The API presents partial and missing separately.
    """
    details = skill_details(resume, role, description)
    return ([x['skill'] for x in details if x['status'] == 'matched'],
            [x['skill'] for x in details if x['status'] != 'matched'],
            [x['skill'] for x in details])


def combined_scores(criteria, resume: str, role: str, description: str, offline=False):
    details = skill_details(resume, role, description)
    matched = [x['skill'] for x in details if x['status'] == 'matched']
    partial = [x['skill'] for x in details if x['status'] == 'partially_demonstrated']
    missing = [x['skill'] for x in details if x['status'] == 'not_demonstrated']
    relevance = criteria['relevance'].score
    model_skills = criteria['skills'].score
    if details:
        # A partial requirement earns 1/2 coverage; it NEVER becomes a
        # complete hard-skill match. Display partials separately in the API.
        coverage = 100 * (len(matched) + 0.5 * len(partial)) / len(details)
        if offline:
            match = .65 * coverage + .35 * min(coverage, relevance)
        else:
            match = .45 * coverage + .35 * relevance + .20 * model_skills
        if not offline and relevance < 10 and model_skills < 10:
            match = min(match, 10.0 + .15 * coverage)
    else:
        match = relevance
    match = round(max(0, min(100, match)), 2)
    quality_weights = {'summary': .10, 'achievement': .25, 'impact': .25,
                       'ownership': .12, 'ordering': .08, 'context': .08, 'structure': .12}
    quality = round(sum(criteria[k].score * w for k, w in quality_weights.items()), 2)
    return match, quality, matched, missing

# Explainable competency taxonomy shared across software, AI, business,
# marketing, operations and other job descriptions.
from app.relevance import CAPABILITIES, active_capabilities, demand_weights, lexical_overlap, known_role_family


def semantic_relevance(text: str, target_role: str, description: str, required: list) -> tuple:
    """Score *documented relevance of one bullet*, not hiring likelihood.

    Direct skills use skill_details (negation-aware); related competencies must
    be both demonstrated by this bullet and relevant to the role or JD.
    Target-specific lexical overlap is a small, auditable fallback for roles
    outside the capability catalog. It never creates a hard-skill match.
    """
    if not text.strip() or _non_owned_claim(text):
        return 0.0, [], []
    details = skill_details(text, target_role, description)
    direct = [d['skill'] for d in details if d['status'] == 'matched']
    partial = [d['skill'] for d in details if d['status'] == 'partially_demonstrated']
    demands = demand_weights(target_role, description)
    related = []
    support = 0.0
    for name, weight in demands.items():
        pattern = CAPABILITIES[name].evidence
        hits = list(re.finditer(pattern, text, re.I))
        # Avoid awarding a competency for a negated claim about that competency.
        if any(not _non_owned_claim(text, match.group(0)) and
               not _is_denied(text, match.start(), match.group(0)) for match in hits):
            related.append(name)
            support += 34.0 * weight
    # Direct evidence remains the strongest indicator; transferable evidence
    # can support a relevant bullet without falsely filling required skills.
    direct_signal = min(100.0, 60.0 * len(direct) + 25.0 * len(partial))
    if direct or partial:
        relevance = min(100.0, direct_signal + min(18.0, support * .38))
    else:
        relevance = min(65.0, support)
    # Exact domain-specific terms help arbitrary jobs not covered by the
    # curated taxonomy; avoid counting generic resume verbs or stopwords.
    overlap = (lexical_overlap(text, target_role, description)
               if not known_role_family(target_role) else [])
    if overlap and not direct:
        relevance = min(70.0, relevance + min(12.0, 4.0 * len(overlap)))
    return round(min(100.0, relevance), 2), direct, related


def rank_bullets(bullets, target_role: str, description: str):
    """Return TWO auditable rankings: achievement strength and job-tailored priority.

    ``ordering_score`` and ``suggested_position`` remain as compatibility aliases
    for the job-tailored ranking. Use ``achievement_position`` to reorder for a
    general-purpose resume instead. All rankings are within the employment or
    project entry, never across different jobs.
    """
    _, _, required = skill_evidence('', target_role, description)
    result = []
    for b in bullets:
        t = b['text']
        action = bool(ACTION_START.search(t))
        outcome = bool(re.search(r'\b(?:increased|decreased|reduced|improved|saved|grew|cut|accelerated|raised|boosted|lowered)\b', t, re.I))
        metric = bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:%|percent|percentage points?|hours?|days?|users?|customers?|ms|milliseconds?|seconds?|queries?|requests?|x)\b|\d+(?:\.\d+)?\s*%', t, re.I))
        relevance, direct, related = semantic_relevance(t, target_role, description, required)
        partial_direct = [d['skill'] for d in skill_details(t, target_role, description)
                          if d['status'] == 'partially_demonstrated']
        accomplishment = 100 if outcome and metric else 65 if outcome else 40 if action else 15
        impact = 100 if outcome and metric else 60 if outcome else 15 if metric else 0
        transferable = (round(min(75, 15 + 15 * len(related)), 2)
                        if related else (50 if re.search(r'\b(?:led|coordinated|collaborated|managed|designed|organized|presented)\b', t, re.I) else 15))
        dimensions = {'accomplishment': accomplishment, 'impact': impact,
                      'transferable': transferable, 'relevance': relevance}
        # Quality is independent of the target role.
        achievement_strength = round(.55 * accomplishment + .45 * impact, 2)
        # In job-tailoring, alignment is the primary factor. The achievement
        # component is deliberately capped so unrelated metrics cannot dominate.
        job_priority = round(.80 * relevance + .20 * achievement_strength, 2)
        result.append({**b, 'dimensions': dimensions,
                       'achievement_strength_score': achievement_strength,
                       'job_priority_score': job_priority,
                       'ordering_score': job_priority,
                       'relevance_evidence': {'required_skills': direct,
                                              'partial_required_skills': partial_direct,
                                              'transferable_capabilities': related,
                                              'capability_sources': {name: ('explicit_job_description'
                                                  if demand_weights(target_role, description).get(name) == 1.0
                                                  else 'role_family') for name in related},
                                              'matched_job_terms': lexical_overlap(t, target_role, description)
                                                  if not direct and not known_role_family(target_role) else []}})
    for entry_id in set(b['entry_id'] for b in result):
        subset = [b for b in result if b['entry_id'] == entry_id]
        for position, b in enumerate(sorted(subset, key=lambda x: (-x['job_priority_score'], x['id'])), 1):
            b['job_priority_position'] = position
            b['suggested_position'] = position
        for position, b in enumerate(sorted(subset, key=lambda x: (-x['achievement_strength_score'], x['id'])), 1):
            b['achievement_position'] = position
    return result

class JevClient:
    """JEV Decisions API client with safe, actionable upstream errors."""
    ENDPOINT = 'https://api.beatapi.io/v1/systemone'

    def __init__(self, key: str):
        self.key = key
        self.model = os.getenv('JEV_MODEL', 'jev-1.14-free')
        if self.model not in {'jev-1.14-free','jev-1.14','jev-1.13-free','jev-1.13'}:
            raise HTTPException(500, 'Unsupported JEV_MODEL configuration')

    async def ask(self, state: dict, questions: dict) -> dict:
        payload = {
            'model': self.model,
            'state': state,
            'questions': questions,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(75.0, connect=15.0), follow_redirects=False) as client:
                response = await client.post(
                    self.ENDPOINT,
                    headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail='JEV request timed out. Try a shorter resume or retry later.') from exc
        except httpx.RequestError as exc:
            # Never expose request headers or an API key.
            raise HTTPException(status_code=502, detail='Cannot connect to BeatAPI. Check internet, proxy, DNS, and TLS settings.') from exc

        if response.status_code != 200:
            # BeatAPI usually returns an {error:{code,message}} envelope, but
            # support both nested and flat formats without exposing secrets.
            code = 'upstream_error'
            message = ''
            try:
                error_body = response.json()
                if isinstance(error_body, dict):
                    err = error_body.get('error', error_body)
                    if isinstance(err, dict):
                        code = str(err.get('code') or error_body.get('code') or code)
                        message = str(err.get('message') or error_body.get('message') or '')
                    elif isinstance(err, str):
                        message = err
            except (ValueError, TypeError):
                pass
            # Status and provider code are sufficient for troubleshooting;
            # supplier messages might echo private resume data, so omit them.
            safe_code = re.sub(r'[^a-zA-Z0-9_-]', '', code)[:70] or 'upstream_error'
            suffix = ' (' + safe_code + ')' if safe_code != 'upstream_error' else ''
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                retry_after = retry_after if retry_after.isdigit() else '60'
                raise HTTPException(status_code=429, detail='JEV rate limit reached' + suffix + '. Retry after ' + retry_after + ' seconds.', headers={'Retry-After': retry_after})
            if response.status_code in (401, 403):
                raise HTTPException(status_code=503, detail='BeatAPI authentication or model permission failed' + suffix + '. Verify BEATAPI_API_KEY and JEV_MODEL.')
            if response.status_code == 400:
                raise HTTPException(status_code=502, detail='BeatAPI rejected the request (HTTP 400)' + suffix + '. Check request schema, model and input length.')
            if response.status_code == 402:
                raise HTTPException(status_code=503, detail='BeatAPI credits are insufficient' + suffix + '. Select jev-1.14-free.')
            if response.status_code in (500, 502, 503, 504):
                raise HTTPException(status_code=503, detail='BeatAPI service is temporarily unavailable (HTTP ' + str(response.status_code) + ')' + suffix + '.')
            raise HTTPException(status_code=502, detail='BeatAPI returned HTTP ' + str(response.status_code) + suffix + '.')

        try:
            body = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail='BeatAPI returned invalid JSON.') from exc
        if not isinstance(body, dict) or not isinstance(body.get('answers'), dict):
            raise HTTPException(status_code=502, detail='BeatAPI response is missing the answers object.')
        missing = sorted(set(questions) - set(body['answers']))
        if missing:
            raise HTTPException(status_code=502, detail='BeatAPI omitted required answers: ' + ', '.join(missing[:8]))
        return body


def _skill_state_lists(resume: str, role: str, description: str) -> dict:
    details = skill_details(resume, role, description)
    partial = [d['skill'] for d in details if d['status'] == 'partially_demonstrated']
    missing = [d['skill'] for d in details if d['status'] == 'not_demonstrated']
    all_unmet = [d['skill'] for d in details if d['status'] != 'matched']
    coverage = round(100 * (len(details) - len(all_unmet) + 0.5 * len(partial))
                     / len(details), 2) if details else 0.0
    return {'partial_required_skills': partial,
            'missing_required_skills': missing,
            'unmet_required_skills': all_unmet,
            'skill_coverage_score': coverage,
            'skill_evidence': details}

def local_review(resume: str, role: str, description: str, bullets: list[dict], warnings: list[str]) -> Review:
    """Clearly labeled, rule-based offline assessment; never masquerades as JEV."""
    words = lambda t: set(re.findall(r'[a-z][a-z0-9+#.-]{2,}', t.lower()))
    resume_words = words(resume)
    target_words = words(role + ' ' + description) - {'and','the','with','for','are','you','will','our','this','that','have','from'}
    coverage = len(target_words & resume_words) / max(1, len(target_words))
    lines = [b['text'] for b in bullets]
    numerical = sum(bool(re.search(r'\b\d+(?:[.,]\d+)?\s*(?:%|x|k|m|ms|hours|users|customers)?\b', x, re.I)) for x in lines) / max(1,len(lines))
    verb_pattern = r'^(?:built|developed|designed|implemented|delivered|improved|reduced|increased|led|created|launched|automated|optimized|managed|deployed|scaled)\b'
    action = sum(bool(re.search(verb_pattern,x,re.I)) for x in lines) / max(1,len(lines))
    sections = sum(bool(re.search(r'(?im)^\s*'+x+r'\b',resume)) for x in ('experience','education','skills','projects')) / 4
    values = {'summary': 100 if re.search(r'(?im)^\s*(summary|profile)\b',resume) else 20,
      'achievement':action*100,'impact':numerical*100,'ownership':action*100,
      'relevance':coverage*100,'ordering':sections*100,'context':sections*100,
      'structure':sections*100,'skills':coverage*100}
    criteria={k: Criterion(score=round(min(100,max(0,v)),2)) for k,v in values.items()}
    reviews=rank_bullets(bullets,role,description)
    warnings=warnings+['Offline heuristic mode: scores are basic text indicators, NOT JEV model judgments. Obtain a BeatAPI key to use JEV.']
    if not bullets: warnings.append('No recognizable experience bullets found.')
    match, quality, found, missing = combined_scores(criteria, resume, role, description, offline=True)
    return Review(overall_score=match, job_match_score=match, resume_quality_score=quality, matching_skills=found, **_skill_state_lists(resume, role, description),
      criteria=criteria,quick_checks={'quantified':round(numerical*100,2),'company_context':round(sections*100,2),'action_verbs':round(action*100,2)},
      bullet_reviews=reviews,warnings=warnings,model='local-heuristic',structured_resume=parse_structure(resume).to_dict())

async def require_access(x_api_key:Optional[str]=Header(default=None)):
    configured=os.getenv('APP_API_KEY')
    if configured:
        import secrets
        if not x_api_key or not secrets.compare_digest(x_api_key,configured): raise HTTPException(401,'Unauthorized')

@app.get('/health')
def health(): return {'status':'ok'}

@app.post('/api/review',response_model=Review,dependencies=[Depends(require_access)])
async def review(file:UploadFile=File(...),target_role:str=Form(...),job_description:str=Form('')):
    if not 1<=len(target_role.strip())<=200 or len(job_description)>12000: raise HTTPException(422,'Invalid role or job description length')
    if file.filename is None or file.filename.lower().rsplit('.',1)[-1] not in {'pdf','docx','txt'}: raise HTTPException(415,'Unsupported file format')
    data=await file.read(MAX_FILE+1)
    await file.close()
    if len(data)>MAX_FILE: raise HTTPException(413,'File too large (5MB max)')
    try: resume,warnings=parse_resume(file.filename,data)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    key=os.getenv('BEATAPI_API_KEY')
    structured = parse_structure(resume)
    bullets=structured.bullets()
    warnings.extend(structured.warnings)
    if not key:
        return local_review(resume, target_role.strip(), job_description, bullets, warnings)
    full_questions={k:score_question(v) for k,v in RUBRICS.items()}
    full_questions.update({'quantified':noul_question('Do most achievement bullets contain credible quantitative measures?'), 'company_context':noul_question('Do experience entries identify meaningful role or employer context?'), 'action_verbs':noul_question('Do most experience bullets begin with strong action verbs?')})
    client=JevClient(key)
    full_state={'resume':resume,'target_role':target_role.strip(),'job_description':job_description or 'Not provided'}
    # Do not ask dozens of generic bullet questions in a single JEV request.
    # They produced near-zero, poorly differentiated values and inflated token usage.
    # Bullet ordering is calculated transparently from textual evidence below.
    response = await client.ask(full_state,full_questions)
    answers=response['answers']
    try:
        criteria={k:score_answer(answers[k]) for k in WEIGHTS}
        checks={k:noul_answer(answers[k]) for k in ('quantified','company_context','action_verbs')}
        bullet_reviews = rank_bullets(bullets, target_role, job_description)
    except (KeyError,ValueError,TypeError,OverflowError) as exc: raise HTTPException(502,'Jev returned incomplete or malformed answers') from exc
    if not bullets: warnings.append('No recognizable experience bullets found; bullet scoring skipped.')
    if len(bullets)==150: warnings.append('At most 150 experience bullets are reviewed.')
    match, quality, found, missing = combined_scores(criteria, resume, target_role, job_description)
    warnings.append('overall_score measures job compatibility; resume_quality_score is separate. Skill matches are text evidence, not independently verified proficiency.')
    return Review(overall_score=match, job_match_score=match, resume_quality_score=quality, matching_skills=found, **_skill_state_lists(resume, target_role, job_description), criteria=criteria, quick_checks=checks, bullet_reviews=bullet_reviews, warnings=warnings, model=response.get('model','unknown'), structured_resume=structured.to_dict())

@app.post('/api/parse', dependencies=[Depends(require_access)])
async def parse_uploaded_resume(file: UploadFile = File(...)):
    """Inspect structured extraction without making a JEV model request."""
    if not file.filename or file.filename.lower().rsplit('.', 1)[-1] not in {'pdf', 'docx', 'txt'}:
        raise HTTPException(415, 'Unsupported file format')
    data = await file.read(MAX_FILE + 1)
    await file.close()
    if len(data) > MAX_FILE:
        raise HTTPException(413, 'File too large (5MB max)')
    try:
        text, warnings = parse_resume(file.filename, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    parsed = parse_structure(text)
    return {'structured_resume': parsed.to_dict(),
            'bullet_count': len(parsed.bullets()),
            'warnings': warnings + parsed.warnings}
