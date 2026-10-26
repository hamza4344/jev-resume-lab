"""Explainable, role-aware transferable competency matching.

Requirements, role-family defaults, and resume evidence are intentionally
separate. A transferable capability is NEVER proof of a named hard skill.
Unknown roles can still use explicit job-description demands. This is a
transparent lexical/rule-based model, not learned semantic entailment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Capability:
    # Phrases in source resume evidence; patterns should describe specific work.
    evidence: str
    # Explicit signals in the requested job description.
    demand: str


# Deliberately distinct from matching a skill mentioned in a summary or JD.
# Patterns use word boundaries and inflections to reduce substring matches.
CAPABILITIES: Dict[str, Capability] = {
    'api_development': Capability(
        r'\b(?:api\b|apis\b|api[- ]endpoints?\b|restful\b|fastapi\b|flask\b|django\b|microservices?\b|http[- ](?:services?|servers?)\b)',
        r'\b(?:api\b|apis\b|endpoint\w*\b|backend\b|back[- ]?end\b|microservices?\b|restful\b|http\b|fastapi\b|flask\b|django\b)'),
    'performance': Capability(
        r'\b(?:latenc\w*|throughput|profil\w*|cach(?:e|ed|ing)\b|batch(?:ing|ed)?\b|response[- ]time|request[- ]rate|requests[- ]per[- ]second|load[- ]test\w*|p95\b|p99\b|speed(?:up)?\b|scalab\w*|query[- ]optim\w*)',
        r'\b(?:latenc\w*|throughput|performance|optimis\w*|optimiz\w*|scalab\w*|efficient\w*|response[- ]time|cach\w*|reliab\w*|high[- ]availability)'),
    'workflow': Capability(
        r'\b(?:workflows?\b|orchestrat\w*|agentic\b|agents?\b|job[- ]queues?\b|background[- ]jobs?\b|task[- ]queues?\b|asynchron\w*|async\b|tool[- ]execution|scheduled[- ]tasks?\b)',
        r'\b(?:workflows?\b|orchestrat\w*|automation|async\b|asynchron\w*|queues?\b|background\b|integrations?\b|agents?\b)'),
    'deployment': Capability(
        r'\b(?:deploy\w*|docker\b|kubernetes\b|k8s\b|cloud[- ]infra\w*|ci/cd\b|containeriz\w*|terraform\b|helm\b)',
        r'\b(?:deploy\w*|docker\b|container\w*|cloud\b|kubernetes\b|k8s\b|devops\b|ci/cd\b|production\b|terraform\b)'),
    'data_integration': Capability(
        r'\b(?:databases?\b|sql\b|postgres\w*|pgvector\b|redis\b|retriev\w*|ingest\w*|data[- ]pipelines?\b|pinecone\b|etl\b|elt\b|warehouse\b|data[- ]migration\b)',
        r'\b(?:databases?\b|sql\b|postgres\w*|data[- ]pipelines?\b|data[- ]integrat\w*|storage\b|retriev\w*|ingest\w*|etl\b|warehouse\b)'),
    'testing_observability': Capability(
        r'\b(?:unit[- ]tests?\b|integration[- ]tests?\b|regression[- ]tests?\b|regression\b|api[- ]tests?\b|smoke[- ]tests?\b|pytest\b|test[- ]suite\b|automated[- ]tests?\b|evaluat\w*|monitor\w*|trac\w*|observab\w*|latency[- ]metrics?\b|metrics?[- ]dashboards?\b|incident[- ]response\b|alerting\b)' ,
        r'\b(?:test\w*|qa\b|quality[- ]assurance\b|regression\b|monitor\w*|observab\w*|trac\w*|metrics?\b|reliab\w*|evaluat\w*|alert\w*|incident\b)'),
    'security_compliance': Capability(
        r'\b(?:oauth\b|jwt\b|rbac\b|authoriz\w*|authenticat\w*|encrypt\w*|access[- ]controls?\b|security[- ]audit\w*|compliance\b|threat[- ]model\w*|vulnerab\w*|soc2\b|iso27001\b)',
        r'\b(?:security\b|auth\w*\b|access[- ]controls?\b|compliance\b|privacy\b|encrypt\w*|rbac\b|oauth\b|audit\w*)'),
    'ml_engineering': Capability(
        r'\b(?:machine[- ]learning\b|model[- ]train\w*|fine[- ]tun\w*|embedding\w*|pytorch\b|tensorflow\b|llms?\b|rag\b|transformers?\b|inference\b|feature[- ]engineer\w*|mlops\b)',
        r'\b(?:machine[- ]learning\b|ai\b|llms?\b|rag\b|pytorch\b|model\w*\b|mlops\b|inference\b|embeddings?\b)'),
    'frontend_development': Capability(
        r'\b(?:react\b|vue\b|angular\b|javascript\b|typescript\b|frontend\b|front[- ]end\b|html\b|css\b|responsive[- ]design\b|browser[- ]ui\b)',
        r'\b(?:react\b|vue\b|angular\b|frontend\b|front[- ]end\b|javascript\b|typescript\b|html\b|css\b|web[- ]ui\b)'),
    'ux_design': Capability(
        r'\b(?:usability\b|user[- ]research\b|wirefram\w*|prototyp\w*|figma\b|interaction[- ]design\b|user[- ]journey\b|accessib\w*|design[- ]system\b)',
        r'\b(?:ux\b|ui/ux\b|user[- ]experience\b|usability\b|wirefram\w*|prototyp\w*|figma\b|design[- ]system\b|accessib\w*)'),
    'marketing_campaigns': Capability(
        r'\b(?:campaigns?\b|email[- ]marketing\b|paid[- ]social\b|seo\b|sem\b|content[- ]marketing\b|brand[- ]launch\b|media[- ]buy\w*|ad[- ]creatives?\b)',
        r'\b(?:marketing\b|campaigns?\b|seo\b|sem\b|brand\b|paid[- ]media\b|content[- ]strateg\w*|social[- ]media\b)'),
    'marketing_analytics': Capability(
        r'\b(?:click[- ]through\b|ctr\b|conversion[- ]rate\b|google[- ]analytics\b|attribution\b|a/b[- ]test\w*|return[- ]on[- ]ad[- ]spend\b|roas\b|lead[- ]generat\w*|customer[- ]acquisition\b)',
        r'\b(?:conversion\b|engagement\b|analytics\b|marketing[- ]roi\b|a/b[- ]test\w*|experimentation\b|ctr\b|campaign[- ]perform\w*|customer[- ]acquisition\b)'),
    'sales_business_development': Capability(
        r'\b(?:sales[- ]pipeline\b|prospect\w*|cold[- ]outreach\b|lead[- ]qualif\w*|quota\b|crm\b|account[- ]manag\w*|deal[- ]clos\w*|negotiat\w*|revenue[- ]grow\w*)',
        r'\b(?:sales\b|pipeline\b|prospect\w*|lead[- ]generat\w*|quota\b|account[- ]manage\w*|business[- ]develop\w*|crm\b|revenue\b)'),
    'financial_analysis': Capability(
        r'\b(?:forecast\w*|budget\w*|cash[- ]flow\b|financ\w*[- ]model\w*|variance[- ]analysis\b|reconcil\w*|balance[- ]sheet\b|audit\w*|gaap\b|accounts[- ]payable\b|financial[- ]report\w*)',
        r'\b(?:finance\b|financial\b|account\w*|audit\w*|budget\w*|forecast\w*|cash[- ]flow\b|bookkeep\w*|variance\b|tax\b)'),
    'analytics_reporting': Capability(
        r'\b(?:dashboards?\b|business[- ]intelligence\b|power[- ]bi\b|tableau\b|kpis?\b|data[- ]visualiz\w*|statistic\w*|experiment[- ]analysis\b|analytical[- ]report\w*)',
        r'\b(?:analytics?\b|data[- ]analys\w*|dashboards?\b|report\w*|kpis?\b|power[- ]bi\b|tableau\b|business[- ]intelligence\b)'),
    'project_delivery': Capability(
        r'\b(?:project[- ]plan\w*|milestones?\b|roadmaps?\b|stakeholder\w*|agile\b|sprints?\b|delivery[- ]track\w*|cross[- ]functional\b|project[- ]deliv\w*)',
        r'\b(?:project[- ]manage\w*|program[- ]manage\w*|agile\b|sprints?\b|roadmaps?\b|stakeholders?\b|delivery\b|milestones?\b)'),
    'customer_support': Capability(
        r'\b(?:ticket\w*|help[- ]desk\b|customer[- ]support\b|customer[- ]success\b|sla\b|customer[- ]satisf\w*|issue[- ]resol\w*|support[- ]requests?\b|knowledge[- ]base\b)',
        r'\b(?:support\b|customer[- ]success\b|help[- ]desk\b|sla\b|ticket\w*|customer[- ]satisf\w*|service[- ]desk\b)'),
    'supply_chain_operations': Capability(
        r'\b(?:inventory\b|procure\w*|shipments?\b|warehouse\b|supply[- ]chain\b|logistics\b|fulfil\w*|vendor[- ]sourc\w*|stock[- ]levels?\b)',
        r'\b(?:inventory\b|procure\w*|supply[- ]chain\b|logistics\b|warehouse\b|operations\b|fulfil\w*|vendor[- ]manage\w*)'),
}

# The role family activates related but not explicitly named competencies with
# a smaller weight than an explicit JD requirement. Domain signals are NEVER
# added to hard-skill coverage.
ROLE_FAMILIES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    'backend': (r'\b(?:back[- ]?end|server[- ]side|api engineer|python developer|software engineer|software developer)\b',
                ('api_development','performance','workflow','deployment','data_integration','testing_observability','security_compliance')),
    'ai_ml': (r'\b(?:ai engineer|ml engineer|machine[- ]learning|llm engineer|nlp engineer|data scientist|computer[- ]vision)\b',
              ('ml_engineering','data_integration','performance','testing_observability','deployment','api_development')),
    'data': (r'\b(?:data engineer|data analyst|analytics engineer|bi analyst|analytics specialist)\b',
             ('data_integration','analytics_reporting','performance','testing_observability')),
    'frontend': (r'\b(?:front[- ]end|web developer|react developer|ui engineer)\b',
                 ('frontend_development','ux_design','testing_observability','performance','api_development')),
    'devops': (r'\b(?:devops|site reliability|sre\b|platform engineer|cloud engineer|infrastructure engineer)\b',
               ('deployment','testing_observability','performance','security_compliance','workflow')),
    'qa': (r'\b(?:qa engineer|sqa engineer|test engineer|software tester|quality assurance)\b',
            ('testing_observability','api_development','workflow','performance')),
    'security': (r'\b(?:cybersecurity|security engineer|appsec|security analyst)\b',
                  ('security_compliance','testing_observability','deployment')),
    'marketing': (r'\b(?:marketing|growth manager|brand manager|seo specialist|content strategist|social media)\b',
                   ('marketing_campaigns','marketing_analytics','analytics_reporting','project_delivery')),
    'sales': (r'\b(?:sales|business development|account executive|account manager)\b',
               ('sales_business_development','project_delivery','analytics_reporting')),
    'finance': (r'\b(?:finance|financial analyst|accountant|bookkeep|auditor)\b',
                 ('financial_analysis','analytics_reporting','project_delivery')),
    'design': (r'\b(?:designer|ux researcher|graphic design|product design)\b',
                ('ux_design','frontend_development','project_delivery')),
    'operations': (r'\b(?:operations|project manager|program manager|supply chain|logistics|procurement)\b',
                    ('supply_chain_operations','project_delivery','analytics_reporting')),
    'support': (r'\b(?:customer support|customer success|technical support|support specialist|help desk)\b',
                 ('customer_support','project_delivery','testing_observability')),
}

# Neutral verbs & organization words should not confer semantic relevance.
STOPWORDS = set('a an and are as at be been being by can could do does each for from has have in into is it its may must of on or our should so than that the their these this to use using via was were will with you your required requirements experience experienced strong ability skills skill knowledge role team teams work working candidate candidates good great build built implement implemented develop developed manage managed support services service including responsible responsibilities various multiple ability essential preferred musthave must'.split())


def demand_weights(target_role: str, job_description: str) -> Dict[str, float]:
    """1.0 = explicit JD demand, 0.70 = plausible family competence."""
    result: Dict[str, float] = {}
    role = target_role.lower()
    jd = job_description.lower()
    for pattern, defaults in ROLE_FAMILIES.values():
        if re.search(pattern, role, re.I):
            for name in defaults:
                result[name] = max(result.get(name, 0), 0.70)
    for name, spec in CAPABILITIES.items():
        if re.search(spec.demand, jd, re.I):
            result[name] = 1.0
    return result


def active_capabilities(target_role: str, job_description: str) -> List[str]:
    return list(demand_weights(target_role, job_description))


def known_role_family(target_role: str) -> bool:
    """Lexical fallback should not dilute curated scoring for known roles."""
    return any(re.search(pattern, target_role, re.I)
               for pattern, _ in ROLE_FAMILIES.values())


def lexical_overlap(text: str, target_role: str, description: str) -> List[str]:
    """Conservative exact-term fallback for professions beyond the catalog.

    A shared token is not full semantic equivalence or a hard skill match.
    Only include meaningful long terms mentioned in both work evidence and JD.
    No overlap is credited on an explicit negative experience claim.
    """
    if re.search(r'\b(?:no|not|without|never|lack|lacks|lacking|did not|does not|have not|cannot)\b', text, re.I):
        return []
    def keywords(value: str):
        return {w for w in re.findall(r'\b[a-z][a-z0-9-]{4,}\b', value.lower())
                if w not in STOPWORDS and not w.endswith('ly')}
    return sorted((keywords(text) & keywords(target_role + ' ' + description)))[:8]
