import re

try:
    from .operations import extract_mentioned_assessments
    from .reranker import rerank_candidates
    from .state import extract_state
except ImportError:
    from operations import extract_mentioned_assessments
    from reranker import rerank_candidates
    from state import extract_state

from retrieval.search import catalog, retrieve


# =========================================================
# RESPONSE SCHEMA
# =========================================================

TEST_TYPE_BY_KEY = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

REMOVAL_WORDS = [
    "drop",
    "remove",
    "exclude",
    "without",
    "skip",
]

ADDITION_WORDS = [
    "add",
    "also",
    "include",
    "replace",
    "instead",
    "with",
]

SELECTION_WORDS = [
    "right fit",
    "final list",
]

CHALLENGE_WORDS = [
    "do we need",
    "really need",
    "redundant",
    "difference",
    "different",
    "why",
]

OMISSION_WORDS = REMOVAL_WORDS + [
    "replace",
    "instead",
    "only",
]


def make_response(
    reply,
    recommendations=None,
    end_of_conversation=False,
):
    """
    Problem-statement response schema:
    {
      "reply": string,
      "recommendations": [
        {"name": string, "url": string, "test_type": string}
      ],
      "end_of_conversation": boolean
    }
    """

    return {
        "reply": str(reply or ""),
        "recommendations": normalise_recommendations(recommendations),
        "end_of_conversation": bool(end_of_conversation),
    }


def normalise_recommendations(recommendations):
    if not recommendations:
        return []

    normalised = []

    for item in recommendations[:10]:
        normalised.append({
            "name": str(item.get("name", "")),
            "url": str(item.get("url", "")),
            "test_type": str(item.get("test_type", "")),
        })

    return normalised


def has_retrievable_signal(state):
    requirements = state.get("requirements") or []

    return any([
        state.get("role"),
        state.get("domain") and state.get("domain") != "other",
        state.get("test_types"),
        requirements,
        state.get("personality_required"),
        state.get("leadership_required"),
        state.get("technical_required"),
    ])


# =========================================================
# GROUNDED OUTPUT HELPERS
# =========================================================

def _latest_user_text(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _catalog_items_mentioned_in_text(text):
    text = text.lower()
    mentioned = []

    for item in catalog:
        name = item.get("name", "")
        url = item.get("link", "")

        if not name:
            continue

        name_pos = text.find(name.lower())
        url_pos = text.find(url.lower()) if url else -1
        positions = [pos for pos in [name_pos, url_pos] if pos >= 0]

        if positions:
            mentioned.append((min(positions), item))

    mentioned.sort(key=lambda pair: pair[0])

    return [item for _, item in mentioned]


def extract_prior_shortlist(messages):
    """
    Reconstruct the latest working shortlist from assistant message text.

    The API is stateless, so refinements must recover prior recommendations
    from the conversation history. The latest emitted shortlist is the source
    of truth; scanning the full history would resurrect items removed in a
    later refinement.
    """

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue

        content = message.get("content", "")
        marker = "Shortlist:"

        if marker not in content:
            continue

        shortlist_text = content.rsplit(marker, 1)[-1]
        mentioned = _catalog_items_mentioned_in_text(shortlist_text)

        if mentioned:
            return mentioned

    return []


def _name_aliases(item):
    name = item.get("name", "")
    aliases = [name.lower()]

    compact = (
        name.lower()
        .replace("(new)", "")
        .replace("  ", " ")
        .strip()
    )
    aliases.append(compact)

    for token in [
        "opq",
        "opq32r",
        "rest",
        "aws",
        "docker",
        "excel",
        "word",
        "g+",
        "verify",
        "8.0",
    ]:
        if token in name.lower():
            aliases.append(token)

    if "restful web services" in name.lower():
        aliases.append("rest")
    if "occupational personality questionnaire" in name.lower():
        aliases.extend(["opq", "opq32r"])
    if "shl verify interactive g+" in name.lower():
        aliases.extend(["g+", "verify g+"])
    if "safety & dependability 8.0" in name.lower():
        aliases.extend(["8.0 bundle", "safety and dependability 8.0"])

    return aliases


def apply_user_removals(candidates, messages):
    latest = _latest_user_text(messages).lower()

    if not any(word in latest for word in REMOVAL_WORDS):
        return candidates

    filtered = []

    for item in candidates:
        aliases = _name_aliases(item)
        should_remove = False

        for alias in aliases:
            if not alias:
                continue

            escaped_alias = re.escape(alias)

            if any(
                re.search(rf"\b{word}\b[^.?!;]{{0,50}}\b{escaped_alias}\b", latest)
                for word in REMOVAL_WORDS
            ):
                should_remove = True
                break

            if re.search(rf"\b{escaped_alias}\b[^.?!;]{{0,30}}\b(out|off)\b", latest):
                should_remove = True
                break

        if not should_remove:
            filtered.append(item)

    return filtered


def apply_user_positive_selection(candidates, messages):
    latest = _latest_user_text(messages).lower()

    if not any(word in latest for word in SELECTION_WORDS):
        return candidates

    selected = []

    for item in candidates:
        aliases = _name_aliases(item)

        if any(alias and re.search(rf"\b{re.escape(alias)}\b", latest) for alias in aliases):
            selected.append(item)

    return selected or candidates


def user_mentions_item(item, messages):
    latest = _latest_user_text(messages).lower()

    for alias in _name_aliases(item):
        if alias and re.search(rf"\b{re.escape(alias)}\b", latest):
            return True

    return False


def can_omit_prior_item(item, messages):
    latest = _latest_user_text(messages).lower()

    return (
        user_mentions_item(item, messages)
        and any(word in latest for word in OMISSION_WORDS)
    )


def should_retrieve_for_turn(state, operation, prior_shortlist, messages):
    if not has_retrievable_signal(state):
        return False

    if state.get("final_confirmation"):
        return False

    latest = _latest_user_text(messages).lower()

    if not prior_shortlist:
        return True

    if operation == "refine":
        return any(word in latest for word in ADDITION_WORDS)

    if operation == "recommend":
        if any(word in latest for word in CHALLENGE_WORDS):
            return False

        return True

    return False


def build_allowed_candidate_pool(retrieved, prior, operation="recommend", k=25):
    allowed = []
    seen = set()
    pools = prior + retrieved if operation == "refine" else retrieved + prior

    for item in pools:
        name = item.get("name", "")
        if name and name not in seen:
            allowed.append(item)
            seen.add(name)

    return allowed[:k]


def select_from_rerank(
    candidates,
    state,
    messages,
    operation="recommend",
    prior_shortlist=None,
    retrieved=None,
    k=10,
):
    rerank = rerank_candidates(
        state,
        candidates[:15],
        messages=messages,
        operation=operation,
        previous_shortlist=prior_shortlist or [],
        new_candidates=retrieved or [],
    )
    selected_names = rerank.get("selected_names", [])
    omitted_previous = rerank.get("omitted_previous", [])

    by_name = {item["name"]: item for item in candidates}

    selected_items = [
        by_name[name]
        for name in selected_names[:k]
        if name in by_name
    ]

    return selected_items, rerank.get("reply", ""), omitted_previous


def approved_llm_omissions(omitted_previous, prior_shortlist):
    prior_names = {
        item.get("name", "")
        for item in prior_shortlist
        if item.get("name")
    }
    approved = []

    for item in omitted_previous or []:
        name = item.get("name", "")
        reason = item.get("reason", "")

        if name in prior_names and len(str(reason).strip()) >= 12:
            approved.append(name)

    # Keep deletions bounded unless the user explicitly removes more in text.
    max_omissions = max(1, len(prior_names) // 2)

    return set(approved[:max_omissions])


def preserve_unaffected_prior_items(
    selected_items,
    candidates,
    prior_shortlist,
    messages,
    omitted_previous=None,
):
    """
    The LLM is allowed to rank and add, but not silently rewrite history.
    Previous shortlist items survive unless the latest user message explicitly
    names that item in a removal/replacement/narrowing instruction.
    """

    if not prior_shortlist:
        return selected_items[:10]

    candidate_names = {item.get("name", "") for item in candidates}
    approved_omissions = approved_llm_omissions(
        omitted_previous or [],
        prior_shortlist,
    )
    selected_by_name = {
        item.get("name", ""): item
        for item in selected_items
        if item.get("name")
    }
    preserved = []
    seen = set()

    for item in prior_shortlist:
        name = item.get("name", "")

        if (
            not name
            or name not in candidate_names
            or can_omit_prior_item(item, messages)
            or name in approved_omissions
        ):
            continue

        preserved.append(selected_by_name.get(name, item))
        seen.add(name)

    for item in selected_items:
        name = item.get("name", "")

        if name and name not in seen:
            preserved.append(item)
            seen.add(name)

        if len(preserved) >= 10:
            break

    return preserved[:10]


def build_recommendation_reply(state, operation):
    prefix = "Updated shortlist" if operation == "refine" else "Here are relevant SHL assessments"
    details = []

    if state.get("role"):
        details.append(state["role"])
    if state.get("seniority"):
        details.append(state["seniority"])
    if state.get("personality_required"):
        details.append("personality")
    if state.get("leadership_required"):
        details.append("leadership")
    if state.get("technical_required"):
        details.append("technical skills")
    if state.get("domain") and state.get("domain") != "other":
        details.append(str(state["domain"]).replace("_", " "))

    if details:
        return f"{prefix} for {', '.join(details)}."

    return f"{prefix} based on your current requirements."


def append_shortlist_to_reply(reply, recommendations):
    if not recommendations:
        return reply

    names = "; ".join(item["name"] for item in recommendations)
    return f"{reply} Shortlist: {names}."


def catalog_item_to_recommendation(item):
    test_types = [
        TEST_TYPE_BY_KEY[key]
        for key in item.get("keys", [])
        if key in TEST_TYPE_BY_KEY
    ]

    return {
        "name": item.get("name", ""),
        "url": item.get("link", item.get("url", "")),
        "test_type": ",".join(test_types) or "Unknown",
    }


def compare_assessments(items):
    lines = ["Here is a grounded comparison of the assessments you mentioned:"]

    for item in items:
        categories = ", ".join(item.get("keys", [])) or "not listed"
        levels = ", ".join(item.get("job_levels", [])) or "not listed"
        duration = item.get("duration") or "not listed"

        lines.append(
            f"{item['name']}: categories: {categories}; "
            f"job levels: {levels}; duration: {duration}."
        )

    lines.append(
        "I can only compare fields present in the catalog, so names and URLs "
        "come directly from the stored SHL metadata."
    )

    return " ".join(lines)


# =========================================================
# MAIN AGENT PIPELINE
# =========================================================

def agent(messages):
    # ---------------------------------------------
    # STEP 1 — EXPLICIT OPERATION + STATE
    # ---------------------------------------------

    state = extract_state(messages)
    operation = state.get("operation", "recommend")

    if state.get("out_of_scope"):
        return make_response(
            (
                "I can help with SHL assessment recommendations and grounded "
                "comparisons, but I cannot help with that request."
            ),
        )

    # ---------------------------------------------
    # STEP 2 — COMPARE
    # ---------------------------------------------

    if operation == "compare":
        mentioned = extract_mentioned_assessments(messages, catalog)

        if len(mentioned) < 2:
            return make_response(
                (
                    "Which two or more SHL assessments should I compare? "
                    "Please use their catalog names."
                ),
            )

        return make_response(
            compare_assessments(mentioned),
        )

    if operation == "clarify" and state.get("clarification_question"):
        return make_response(
            state["clarification_question"],
        )

    # ---------------------------------------------
    # STEP 3 — RECONSTRUCT WORKING SHORTLIST
    # ---------------------------------------------
    # The latest assistant-emitted shortlist is the conversation state.
    # Retrieval can propose additions, but it should not replace this state.
    prior_shortlist = extract_prior_shortlist(messages)

    if state.get("final_confirmation") and prior_shortlist:
        selected_items = apply_user_positive_selection(
            apply_user_removals(prior_shortlist, messages),
            messages,
        )[:10]
        recommendations = [
            catalog_item_to_recommendation(item)
            for item in selected_items
        ]
        reply = append_shortlist_to_reply(
            "Confirmed. Final shortlist locked in.",
            recommendations,
        )

        return make_response(
            reply,
            recommendations,
            end_of_conversation=bool(recommendations),
        )

    # ---------------------------------------------
    # STEP 4 — BOUNDED CANDIDATE PROPOSAL
    # ---------------------------------------------
    # First turns and additive/replacement refinements retrieve from the
    # current state. Removal/challenge/final turns reuse the working shortlist.
    should_retrieve = should_retrieve_for_turn(
        state,
        operation,
        prior_shortlist,
        messages,
    )
    retrieved = retrieve(state, k=15) if should_retrieve else []
    candidates = build_allowed_candidate_pool(
        retrieved,
        prior_shortlist,
        operation=operation,
    )

    if not candidates:
        return make_response(
            "I need a little more role or assessment context before I can recommend a catalog-grounded shortlist.",
        )

    # ---------------------------------------------
    # STEP 5 — BOUNDED LLM SHORTLIST EDITOR
    # ---------------------------------------------
    # The reranker can only choose from candidates, which are previous
    # shortlist items plus any retrieved additions for this turn.
    selected_items, rerank_reply, omitted_previous = select_from_rerank(
        candidates,
        state,
        messages,
        operation=operation,
        prior_shortlist=prior_shortlist,
        retrieved=retrieved,
        k=10,
    )
    selected_items = preserve_unaffected_prior_items(
        selected_items,
        candidates,
        apply_user_removals(prior_shortlist, messages),
        messages,
        omitted_previous=omitted_previous,
    )
    recommendations = [
        catalog_item_to_recommendation(item)
        for item in selected_items
    ]

    reply = rerank_reply or build_recommendation_reply(state, operation)
    reply = append_shortlist_to_reply(
        reply,
        recommendations,
    )

    return make_response(
        reply,
        recommendations,
        end_of_conversation=False,
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": "We need a solution for senior leadership."
        }
    ]

    response = agent(messages)

    print("\nAGENT RESPONSE:")
    print(response)
