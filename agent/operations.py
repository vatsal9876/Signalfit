import re


ALIASES = {
    "opq": [
        "Occupational Personality Questionnaire OPQ32r",
        "OPQ Candidate Report 2.0",
        "OPQ Profile Report",
    ],
    "opq32r": [
        "Occupational Personality Questionnaire OPQ32r",
    ],
    "gsa": [
        "Global Skills Assessment",
        "Global Skills Development Report",
    ],
}


def conversation_text(messages):
    return "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in messages
    )


def _normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _tokens(text):
    return set(_normalise(text).split())


def _fuzzy_score(text_tokens, item):
    name_tokens = _tokens(item.get("name", ""))
    if not name_tokens:
        return 0

    overlap = len(text_tokens & name_tokens)
    return overlap / len(name_tokens)


def extract_mentioned_assessments(messages, catalog):
    text = _normalise(conversation_text(messages))
    text_tokens = set(text.split())
    mentioned = []
    seen = set()

    by_name = {
        item.get("name", ""): item
        for item in catalog
    }

    for alias, preferred_names in ALIASES.items():
        if not re.search(rf"\b{re.escape(alias)}\b", text):
            continue

        for name in preferred_names:
            item = by_name.get(name)
            if item and name not in seen:
                mentioned.append(item)
                seen.add(name)
                break

    if len(mentioned) >= 2:
        return mentioned[:4]

    for item in catalog:
        name = item.get("name", "")
        normalised_name = _normalise(name)

        if not name:
            continue

        exact = normalised_name in text
        fuzzy = _fuzzy_score(text_tokens, item) >= 0.65

        if (exact or fuzzy) and name not in seen:
            mentioned.append(item)
            seen.add(name)

    return mentioned[:4]
