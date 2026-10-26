"""Conservative document structure parser.

Uses explicit document section state, typed records and source line references.
Never guesses a job/project title from an objective or achievement sentence.
Unknown information is left unknown instead of invented.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

BULLET = re.compile(r'^\s*(?:[•*\-–—]\s+|\d+[.)]\s+)(.+)$')
NUMBERED_TITLE = re.compile(r'^\s*\d+[.)]\s+(.+)$')
DATES = re.compile(r'\b(?:19|20)\d{2}\b|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(?:19|20)\d{2}\b', re.I)
RANGE = re.compile(r'\b(?:19|20)\d{2}\s*(?:[-–—]|to)\s*(?:present|current|(?:19|20)\d{2})\b', re.I)
LABEL = re.compile(r'^(?:responsibilities|key responsibilities|achievements|key achievements|highlights|contributions|duties|technologies used|work)\s*:?$', re.I)
INLINE_LABEL = re.compile(r'^(?:responsibilities|key responsibilities|achievements|key achievements|highlights|contributions|duties|work)\s*:\s*(.+)$', re.I)
OBJECTIVE = re.compile(r'^(?:project\s+objective|objective|goal|scope)\s*:\s*(.+)$', re.I)
OUTCOME = re.compile(r'^(?:outcome|results?|impact)\s*:\s*(.+)$', re.I)
ACTION = re.compile(r'^(?:built|developed|designed|implemented|delivered|improved|reduced|increased|led|created|launched|automated|optimized|managed|deployed|scaled|planned|analyzed|consolidated|coordinated|reviewed|prepared|supported|maintained|drafted|organized|assisted|presented|worked|conducted|produced|wrote|helped|ensured|segmented|grouped|collected|held|monitored|integrated|engineered|migrated|introduced|established)\b', re.I)
HEADINGS = {
 'experience': ('experience','work experience','professional experience','employment','employment history','work history','relevant experience','career history','career journey','history','internships'),
 'projects': ('projects','selected projects','selected marketing projects','professional projects','personal projects','key projects','project experience','portfolio projects','selected work','portfolio'),
 'skills': ('skills','technical skills','core skills','core competencies','tools and platforms','technologies','technical expertise','core capabilities','technical scope'),
 'education': ('education','academic background','qualifications'),
 'summary': ('summary','professional summary','profile','professional profile','career summary','about'),
 'certifications': ('certifications','licenses','professional development','training'),
 'other': ('references','awards','publications','interests','career interests','additional information','additional responsibilities','volunteer experience')
}
SECTION_LOOKUP = {v:k for k, names in HEADINGS.items() for v in names}

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip()).strip(' :').lower()

def is_heading(s: str) -> Optional[str]:
    return SECTION_LOOKUP.get(normalize(s))

def is_title(s: str) -> bool:
    """Conservative title identification only inside a relevant section."""
    s=s.strip()
    if not s or OBJECTIVE.match(s) or OUTCOME.match(s) or LABEL.match(s):
        return False
    if ACTION.match(s) or s.endswith(('.', ';', ',')) or len(s)>130:
        return False
    if not 1 <= len(s.split()) <= 14:
        return False
    if s.isupper(): return True
    # Project headings often contain an abbreviated acronym and a year.
    # e.g. 'Basic FAQ Search Prototype | 2025' and 'Migration (2025)'.
    if re.search(r'(?:\|\s*(?:19|20)\d{2}\s*$|\((?:19|20)\d{2}\)\s*$)', s) and len(s.split()) <= 12:
        return True
    # Short standalone names without sentence punctuation, often project title case.
    return s.istitle() and len(s.split()) <= 9

@dataclass(frozen=True)
class LogicalLine:
    """Text recovered from one or more physical lines; first-line provenance."""
    text: str
    source_line: int
    end_line: int
    raw_start: str


def logical_resume_lines(text: str) -> List[LogicalLine]:
    """Join soft-wrapped prose without crossing section/entry boundaries.

    Plain-text PDF extraction often wraps a bullet mid-phrase without preserving
    indentation (e.g. ``cloud\n deployment, without implementing ...``).  That
    split changes the meaning of negative claims. Preserve the original first
    physical line for attribution. Do not merge independent bullets, headings,
    employer/date headers or completed sentences.
    """
    result: List[LogicalLine] = []
    section: Optional[str] = None
    for line_no, raw in enumerate(text.splitlines(), 1):
        content = raw.strip()
        if not content:
            continue
        heading = is_heading(content)
        if heading:
            section = heading
        previous = result[-1] if result else None
        boundary = (heading is not None or BULLET.match(content) or
                    NUMBERED_TITLE.match(content) or LABEL.fullmatch(content) or
                    OBJECTIVE.match(content) or OUTCOME.match(content) or
                    INLINE_LABEL.match(content) or
                    (section == 'experience' and DATES.search(content) and '|' in content) or
                    (section == 'projects' and is_title(content)))
        if previous and section in ('experience', 'projects', 'summary', 'skills') and not boundary:
            prev_text = previous.text.rstrip()
            if (len(prev_text) >= 48 and
                not prev_text.endswith(('.', '!', '?')) and
                not is_heading(prev_text) and
                not (section == 'projects' and is_title(prev_text)) and
                not (section == 'experience' and DATES.search(prev_text) and '|' in prev_text) and
                (re.match(r'^[a-z,(]', content) or
                 re.search(r'\b(?:and|or|with|using|for|by|to|from|of|in|the|a|an)\s*$', prev_text, re.I)) and
                not ACTION.match(content)):
                result[-1] = LogicalLine(prev_text + ' ' + content,
                                         previous.source_line, line_no,
                                         previous.raw_start)
                continue
        result.append(LogicalLine(content, line_no, line_no, raw))
    return result


@dataclass
class Evidence:
    text: str
    source_line: int
    kind: str = 'responsibility'

@dataclass
class Entry:
    id: str
    title: Optional[str] = None
    organization: Optional[str] = None
    dates: Optional[str] = None
    objectives: List[Evidence] = field(default_factory=list)
    responsibilities: List[Evidence] = field(default_factory=list)
    achievements: List[Evidence] = field(default_factory=list)
    outcomes: List[Evidence] = field(default_factory=list)
    metadata: List[Evidence] = field(default_factory=list)

@dataclass
class ParsedResume:
    experience: List[Entry] = field(default_factory=list)
    projects: List[Entry] = field(default_factory=list)
    other_sections: Dict[str,List[Evidence]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str,Any]:
        return asdict(self)

    def bullets(self, limit: int = 150) -> List[dict]:
        output=[]
        for section, entries in (('experience',self.experience),('projects',self.projects)):
            for entry in entries:
                context = entry.title or entry.organization or ('Unspecified project' if section=='projects' else 'Unspecified role')
                for e in sorted(entry.responsibilities+entry.achievements+entry.outcomes, key=lambda x:x.source_line):
                    output.append({'id':len(output),'role':context,'section':section,
                                   'entry_id':entry.id,'source_line':e.source_line,
                                   'evidence_kind':e.kind,'text':e.text})
                    if len(output)>=limit: return output
        return output


RESULT_PATTERN = re.compile(r'\b(?:increased|decreased|reduced|improved|saved|cut|accelerated|raised|boosted|grew|achieved|delivered|lowered|shortened)\b', re.I)
METRIC_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent|percentage points?|ms|milliseconds?|seconds?|hours?|days?|users?|requests?|queries?|x)\b',re.I)

def statement_kind(text: str, label: Optional[str] = None) -> str:
    """Explicit evidence outcome cues take priority over section labels."""
    if RESULT_PATTERN.search(text) and (METRIC_PATTERN.search(text) or re.search(r'\b(?:from .+ to |by \d|fewer|faster|lower|higher)\b',text,re.I)):
        return 'outcome'
    if RESULT_PATTERN.search(text):
        return 'achievement'
    if label and any(tag in label for tag in ('achievement','highlight','contribution')):
        return 'achievement'
    return 'responsibility'

def add_statement(entry: Entry, text: str, index: int, label: Optional[str] = None) -> None:
    kind=statement_kind(text,label)
    evidence=Evidence(text,index,kind)
    if kind=='outcome': entry.outcomes.append(evidence)
    elif kind=='achievement': entry.achievements.append(evidence)
    else: entry.responsibilities.append(evidence)

def last_statement(entry: Entry) -> Evidence:
    return max(entry.responsibilities+entry.achievements+entry.outcomes,key=lambda e:e.source_line)

def parse_structure(text: str) -> ParsedResume:
    result=ParsedResume()
    section=None
    entry=None
    group=None
    recent_role=None
    lines=logical_resume_lines(text)
    active_label=None

    def create_entry(group_name: str) -> Entry:
        collection=result.experience if group_name=='experience' else result.projects
        item=Entry(id='%s-%d'%(group_name,len(collection)+1))
        collection.append(item)
        return item

    for physical in lines:
        index=physical.source_line
        raw=physical.raw_start
        line=physical.text
        if not line: continue
        heading=is_heading(line)
        if heading:
            section=heading
            entry=None
            recent_role=None
            continue
        if section not in ('experience','projects'):
            result.other_sections.setdefault(section or 'unclassified',[]).append(Evidence(line,index,'text'))
            continue
        if LABEL.fullmatch(line):
            active_label=normalize(line)
            continue
        labeled=INLINE_LABEL.match(line)
        if labeled:
            if entry is None: entry=create_entry(section)
            content=labeled.group(1).strip()
            if content:
                add_statement(entry, content, index, active_label)
            active_label='responsibilities'
            continue
        objective=OBJECTIVE.match(line)
        outcome=OUTCOME.match(line)
        if objective or outcome:
            if entry is None:
                entry=create_entry(section)
                result.warnings.append('Metadata before entry title at line %d'%index)
            kind='objective' if objective else 'outcome'
            ev=Evidence((objective or outcome).group(1).strip(),index,kind)
            if objective: entry.objectives.append(ev)
            else: entry.outcomes.append(ev)
            continue
        numbered=NUMBERED_TITLE.match(line)
        # Numbered headings are titles in project sections, not experience bullets.
        if section=='projects' and numbered and is_title(numbered.group(1)):
            entry=create_entry(section)
            entry.title=numbered.group(1).strip()
            continue
        bullet=BULLET.match(line)
        if bullet:
            content=bullet.group(1).strip()
            if not content: continue
            if entry is None: entry=create_entry(section)
            add_statement(entry, content, index, active_label)
            continue
        # Single-line header: Role | Employer | dates.  Also handles independent
        # roles from the same employer without merging them into prior records.
        if section=='experience' and ('|' in line or ' — ' in line or ' – ' in line) and DATES.search(line):
            parts=[part.strip() for part in (line.split('|') if '|' in line else re.split(r'\s+[—–]\s+', line))]
            if len(parts) >= 3 and parts[0] and parts[1] and not ACTION.match(parts[0]):
                entry=create_entry(section)
                entry.title=parts[0]
                entry.organization=parts[1]
                entry.dates=' | '.join(parts[2:])
                continue
        # Employer line with date range is metadata, and should not replace title.
        if section=='experience' and (RANGE.search(line) or ('|' in line and DATES.search(line))):
            if entry is None or entry.responsibilities or entry.achievements or entry.outcomes:
                entry=create_entry(section)
            segments=[x.strip() for x in line.split('|')]
            if len(segments)>=2:
                entry.organization=segments[0]
                entry.dates=' | '.join(segments[1:])
            else:
                entry.dates=line
            continue
        # In an experience section titles generally precede employer/date lines.
        if section=='experience' and is_title(line):
            if entry is None or entry.responsibilities or entry.achievements or entry.outcomes:
                entry=create_entry(section)
            if entry.title is None: entry.title=line
            else: entry.metadata.append(Evidence(line,index,'heading'))
            continue
        if section=='projects' and is_title(line):
            entry=create_entry(section)
            entry.title=line
            continue
        # Wrapped lines from PDF extraction can lose indentation. Join obvious
        # continuations before classifying standalone metadata.
        if entry is not None and (entry.responsibilities or entry.achievements or entry.outcomes) and (last_statement(entry).text.rstrip().endswith((' and',' or',',',';',' with',' using',' by',' for'))):
            last_statement(entry).text += ' ' + line
            continue
        # Standalone action statements are evidence, but never promoted into titles.
        if ACTION.match(line):
            if entry is None: entry=create_entry(section)
            add_statement(entry, line, index, active_label)
        elif entry is not None and (entry.responsibilities or entry.achievements or entry.outcomes) and raw[:1].isspace() and len(line.split())>3:
            # Continuation only if original line was indented.
            last_statement(entry).text += ' ' + line
        else:
            if entry is None: entry=create_entry(section)
            entry.metadata.append(Evidence(line,index,'unclassified'))
    # Normalize types after line continuation merges; preserve original source line.
    for collection in (result.experience, result.projects):
        for item in collection:
            statements=sorted(item.responsibilities+item.achievements+item.outcomes, key=lambda ev: ev.source_line)
            item.responsibilities=[]
            item.achievements=[]
            item.outcomes=[]
            for ev in statements:
                # Explicit outcome labels stay outcomes even without numerical evidence.
                kind = 'outcome' if ev.kind=='outcome' else statement_kind(ev.text)
                ev.kind=kind
                if kind=='outcome': item.outcomes.append(ev)
                elif kind=='achievement': item.achievements.append(ev)
                else: item.responsibilities.append(ev)
    for group in (result.experience,result.projects):
        for item in group:
            if not item.title: result.warnings.append('%s has no confidently identified title'%item.id)
    return result
