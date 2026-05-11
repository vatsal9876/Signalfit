import json
import re
from pathlib import Path
import faiss
import numpy as np

try:
    from .embeddings import get_model
except ImportError:
    from embeddings import get_model

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "shl_catalog.json"
INDEX_PATH = ROOT / "data" / "data_index"

catalog = json.load(open(CATALOG_PATH, "r", encoding="utf-8"))
index = faiss.read_index(str(INDEX_PATH))


KEY_TO_TEST_TYPE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

SENIORITY_TO_JOB_LEVELS = {
    "entry-level": {"Entry-Level", "Graduate", "General Population"},
    "entry level": {"Entry-Level", "Graduate", "General Population"},
    "junior": {"Entry-Level", "Graduate", "General Population"},
    "graduate": {"Entry-Level", "Graduate", "General Population"},
    "mid-level": {
        "Mid-Professional",
        "Professional Individual Contributor",
        "General Population",
    },
    "mid level": {
        "Mid-Professional",
        "Professional Individual Contributor",
        "General Population",
    },
    "senior": {
        "Professional Individual Contributor",
        "Manager",
        "Front Line Manager",
        "Supervisor",
        "General Population",
    },
    "lead": {
        "Professional Individual Contributor",
        "Manager",
        "Front Line Manager",
        "Supervisor",
        "General Population",
    },
    "manager": {
        "Manager",
        "Front Line Manager",
        "Supervisor",
        "General Population",
    },
    "director": {"Director", "Executive", "General Population"},
    "executive": {"Executive", "Director", "General Population"},
}

DOMAIN_TERMS = {
    "software_engineering": [
        "software",
        "developer",
        "engineer",
        "coding",
        "programming",
        "java",
        "python",
        "sql",
        "spring",
        "rest",
        "aws",
        "docker",
    ],
    "customer_service": [
        "customer",
        "service",
        "contact center",
        "call center",
        "support",
        "phone",
        "chat",
        "retail",
    ],
    "sales": ["sales", "seller", "account", "business development"],
    "finance": [
        "finance",
        "financial",
        "accounting",
        "accounts payable",
        "accounts receivable",
    ],
    "healthcare": ["healthcare", "medical", "hospital", "clinic", "hipaa", "patient"],
    "leadership": ["leadership", "leader", "manager", "executive", "director"],
    "safety_manufacturing": [
        "safety",
        "manufacturing",
        "industrial",
        "warehouse",
        "dependability",
    ],
    "office_productivity": ["excel", "word", "powerpoint", "office", "microsoft"],
    "language": ["language", "spoken", "written", "english", "spanish", "svar"],
    "general_cognitive": [
        "cognitive",
        "reasoning",
        "numerical",
        "verbal",
        "inductive",
        "deductive",
        "g+",
    ],
    "data_analytics": ["data", "analytics", "statistics", "tableau", "power bi"],
    "operations": ["operations", "administration", "process", "workflow"],
}


def build_query(state):
    requirements = state.get("requirements") or []
    signals = []

    if state.get("personality_required"):
        signals.append("personality behavior OPQ")
    if state.get("leadership_required"):
        signals.append("leadership management executive competencies")
    if state.get("technical_required"):
        signals.append("technical knowledge skills coding")
    if state.get("development_use"):
        signals.append("development coaching 360 report")
    if state.get("selection_use"):
        signals.append("selection hiring screening")
    if state.get("domain"):
        signals.extend(DOMAIN_TERMS.get(state["domain"], []))
    if state.get("test_types"):
        signals.append(" ".join(_test_type_labels(state["test_types"])))
    if state.get("soft_test_types"):
        signals.append("optional complementary " + " ".join(
            _test_type_labels(state["soft_test_types"])
        ))

    return f"""
    Role:
    {state.get("role", "")}

    Seniority:
    {state.get("seniority", "")}

    Requirements:
    {", ".join(requirements)}

    Assessment Signals:
    {", ".join(signals)}

    Domain:
    {state.get("domain", "")}

    Desired Test Types:
    {", ".join(state.get("test_types") or [])}

    Optional Complementary Test Types:
    {", ".join(state.get("soft_test_types") or [])}
    """


def _distance_to_score(distance):
    return round(1 / (1 + float(distance)), 4)


def _format_item(item, distance):
    return {
        "name": item.get("name", ""),
        "url": item.get("link", ""),
        "description": item.get("description", ""),
        "job_levels": item.get("job_levels", []),
        "keys": item.get("keys", []),
        "duration": item.get("duration", ""),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
        "score": _distance_to_score(distance),
    }


def _test_type_labels(test_types):
    labels = {
        "A": "ability aptitude cognitive reasoning",
        "B": "biodata situational judgment scenarios",
        "C": "competencies behaviors",
        "D": "development 360 coaching report",
        "E": "assessment exercises role play presentation",
        "K": "knowledge skills technical",
        "P": "personality behavior OPQ",
        "S": "simulations work sample",
    }

    return [
        labels[test_type]
        for test_type in test_types
        if test_type in labels
    ]


def _candidate_test_types(candidate):
    return {
        KEY_TO_TEST_TYPE[key]
        for key in candidate.get("keys", [])
        if key in KEY_TO_TEST_TYPE
    }


def _expected_test_types(state):
    test_types = state.get("test_types") or []

    if test_types:
        return set(test_types)

    inferred = set()

    if state.get("technical_required"):
        inferred.update(["K", "S"])
    if state.get("personality_required"):
        inferred.add("P")
    if state.get("leadership_required"):
        inferred.update(["C", "P", "D", "E"])
    if state.get("development_use"):
        inferred.add("D")

    requirements = " ".join(state.get("requirements") or []).lower()

    if any(word in requirements for word in ["cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal", "inductive", "deductive"]):
        inferred.add("A")
    if any(word in requirements for word in ["situational", "judgment", "judgement", "scenario"]):
        inferred.add("B")
    if any(word in requirements for word in ["simulation", "call handling", "live coding", "phone", "contact center"]):
        inferred.add("S")
    if any(word in requirements for word in ["spoken", "written", "language", "english", "spanish"]):
        inferred.update(["K", "S"])

    return inferred


def _seniority_levels(state):
    seniority = state.get("seniority")

    if not seniority:
        return set()

    return SENIORITY_TO_JOB_LEVELS.get(str(seniority).strip().lower(), set())


def _matches_test_type(candidate, state):
    expected = _expected_test_types(state)

    if not expected:
        return True

    candidate_types = _candidate_test_types(candidate)

    return bool(candidate_types & expected)


def _matches_seniority(candidate, state):
    expected_levels = _seniority_levels(state)

    if not expected_levels:
        return True

    levels = set(candidate.get("job_levels", []))

    if not levels:
        return True

    return bool(levels & expected_levels)


def _metadata_matches(candidate, state):
    return (
        _matches_test_type(candidate, state)
        and _matches_seniority(candidate, state)
    )


def _domain_score(candidate, state):
    domain = state.get("domain")

    if not domain:
        return 0

    terms = DOMAIN_TERMS.get(domain, [])
    text = _candidate_text(candidate)

    if not terms:
        return 0

    matches = sum(1 for term in terms if term in text)

    return min(matches * 0.035, 0.18)


def _metadata_boost(candidate, state):
    boost = 0

    if _matches_test_type(candidate, state):
        if _expected_test_types(state):
            boost += 0.16

    if _matches_seniority(candidate, state):
        if _seniority_levels(state):
            boost += 0.08

    boost += _domain_score(candidate, state)

    return boost


def _rule_boost(candidate, state):
    name = candidate.get("name", "").lower()
    text = " ".join(
        [
            candidate.get("name", ""),
            candidate.get("description", ""),
            " ".join(candidate.get("job_levels", [])),
            " ".join(candidate.get("keys", [])),
        ]
    ).lower()

    boost = 0

    if state.get("personality_required") and (
        "personality" in text or "behavior" in text or "behaviour" in text
    ):
        boost += 0.20

        if "opq" in text or "occupational personality questionnaire" in text:
            boost += 0.35

    if state.get("leadership_required") and (
        "leadership" in text or "manager" in text or "executive" in text
    ):
        boost += 0.18

    if state.get("technical_required") and (
        "knowledge & skills" in text or "coding" in text or "technical" in text
    ):
        boost += 0.16

    if state.get("development_use") and (
        "development" in text or "360" in text or "coaching" in text
    ):
        boost += 0.14

    if state.get("selection_use") and (
        "selection" in text or "screen" in text or "assessment" in text
    ):
        boost += 0.08

    soft_types = set(state.get("soft_test_types") or [])
    candidate_types = _candidate_test_types(candidate)

    if "P" in soft_types and "P" in candidate_types:
        boost += 0.05
    if "A" in soft_types and "A" in candidate_types:
        boost += 0.05

    if (
        state.get("leadership_required")
        and state.get("personality_required")
        and state.get("selection_use")
    ):
        if "occupational personality questionnaire" in name:
            boost += 0.70
        if "leadership report" in name:
            boost += 0.55
        if "universal competency report" in name:
            boost += 0.50

        requirements = " ".join(state.get("requirements") or []).lower()

        if "sales" in name and state.get("domain") != "sales":
            boost -= 0.35
        if "development" in name and not state.get("development_use"):
            boost -= 0.30
        if "team" in name and "team" not in requirements:
            boost -= 0.20

    if state.get("remote_required") and candidate.get("remote") == "yes":
        boost += 0.05

    if state.get("adaptive_required") and candidate.get("adaptive") == "yes":
        boost += 0.05

    return boost + _metadata_boost(candidate, state)


def _artifact_penalty(candidate, state):
    name = candidate.get("name", "").lower()
    requirements = " ".join(state.get("requirements") or []).lower()

    if state.get("development_use") or "report" in requirements:
        return 0

    artifact_terms = [
        "candidate report",
        "ability test report",
        "narrative report",
        "profile report",
    ]

    if any(term in name for term in artifact_terms):
        return 40

    return 0


def _candidate_text(item):
    return " ".join(
        [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("job_levels", [])),
            " ".join(item.get("keys", [])),
        ]
    ).lower()


def _tokens(text):
    return {
        token
        for token in re.findall(r"[a-z0-9+#.]+", text.lower())
        if len(token) > 2 or token in {"g+", "c#", "c++", "f#"}
    }


def _specific_query_tokens(state):
    parts = list(state.get("requirements") or [])

    generic = {
        "assessment",
        "assessments",
        "candidate",
        "candidates",
        "hiring",
        "selection",
        "screening",
        "skills",
        "technical",
        "knowledge",
        "role",
        "level",
    }

    return {
        token
        for token in _tokens(" ".join(parts))
        if token not in generic
    }


def _broad_requirement(term):
    return term in {
        "personality",
        "behavior",
        "behaviour",
        "leadership",
        "communication",
        "competency",
        "competencies",
        "problem",
        "solving",
        "cognitive",
        "handling",
    }


def _specific_token_score(item_tokens, specific_tokens):
    score = 0

    for token in specific_tokens:
        if token in item_tokens:
            score += 4 if _broad_requirement(token) else 12

    return score


def _name_token_score(item, specific_tokens):
    name_tokens = _tokens(item.get("name", ""))
    score = 0

    for token in specific_tokens:
        if token in name_tokens:
            score += 4 if _broad_requirement(token) else 10

    return score


def _phrase_score(item, state):
    text = _candidate_text(item)
    name = item.get("name", "").lower()
    score = 0

    for phrase in state.get("requirements") or []:
        phrase = phrase.lower()

        if len(phrase) <= 2 or phrase not in text:
            continue

        phrase_tokens = _tokens(phrase)

        if phrase_tokens and all(_broad_requirement(token) for token in phrase_tokens):
            score += 6
        else:
            score += 18

        if phrase in name:
            score += 20

    return score


def lexical_retrieve(state, k=10, candidates=None, keep_rank_score=False):
    query_tokens = _tokens(build_query(state))
    specific_tokens = _specific_query_tokens(state)
    results = []
    searchable = candidates or catalog

    for item in searchable:
        item_tokens = _tokens(_candidate_text(item))
        overlap = len(query_tokens & item_tokens)
        specific_score = _specific_token_score(item_tokens, specific_tokens)
        name_score = _name_token_score(item, specific_tokens)
        phrase_score = _phrase_score(item, state)

        if overlap == 0 and specific_score == 0 and name_score == 0 and phrase_score == 0:
            continue

        result = _format_item(item, 1 / max(overlap, 1))
        result["_rank_score"] = (
            overlap * 0.2
            + specific_score
            + name_score
            + phrase_score
            + _rule_boost(result, state)
            - _artifact_penalty(result, state)
        )
        results.append(result)

    results.sort(key=lambda item: item["_rank_score"], reverse=True)

    if not keep_rank_score:
        for item in results:
            item.pop("_rank_score", None)

    return results[:k]


def _diversify_by_test_type(results, state, k):
    expected = _expected_test_types(state)

    if len(expected) <= 1:
        return results[:k]

    selected = []
    selected_names = set()

    for test_type in sorted(expected):
        for item in results:
            name = item.get("name", "")

            if name in selected_names:
                continue

            if test_type in _candidate_test_types(item):
                selected.append(item)
                selected_names.add(name)
                break

    for item in results:
        if len(selected) >= k:
            break

        name = item.get("name", "")

        if name not in selected_names:
            selected.append(item)
            selected_names.add(name)

    return selected[:k]


def retrieve(state, k=10, pool_size=35):

    query = build_query(state)
    metadata_names = {
        item.get("name", "")
        for item in catalog
        if _metadata_matches(item, state)
    }
    embedding_model = get_model()

    if embedding_model is None:
        return lexical_retrieve(state, k=k, candidates=catalog)

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    search_k = min(max(k, pool_size), index.ntotal)

    distances, indices = index.search(
        query_embedding,
        search_k
    )

    results = []
    seen = set()

    def add_result(result):
        name = result.get("name", "")

        if not name or name in seen:
            return

        seen.add(name)
        results.append(result)

    lexical_results = lexical_retrieve(
        state,
        k=pool_size,
        candidates=catalog,
        keep_rank_score=True,
    )

    for rank, result in enumerate(lexical_results):
        result["_rank_score"] = (
            result.get("_rank_score", 0)
            + (0.35 if result.get("name", "") in metadata_names else 0)
            + (pool_size - rank) / max(pool_size, 1) * 0.25
        )
        add_result(result)

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        item = catalog[idx]
        result = _format_item(item, distance)
        result["_rank_score"] = (
            result["score"]
            + _rule_boost(result, state)
            + (0.35 if item.get("name", "") in metadata_names else 0)
            - _artifact_penalty(result, state)
        )

        add_result(result)

    results.sort(key=lambda item: item["_rank_score"], reverse=True)
    results = _diversify_by_test_type(results, state, k)

    for item in results:
        item.pop("_rank_score", None)

    return results[:k]


if __name__ == "__main__":

    state = {
        "role": "executive leadership",
        "seniority": "director",
        "requirements": [
            "benchmarking",
            "personality evaluation"
        ]
    }

    results = retrieve(state)

    for i, item in enumerate(results, start=1):

        print(f"\n{i}. {item['name']}")
        print(item["keys"])
        print(item["url"])
