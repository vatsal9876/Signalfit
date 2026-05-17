import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError

try:
    from .prompts import state_system_prompt
except ImportError:
    from prompts import state_system_prompt


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=True)

MODEL_NAME = os.getenv(
    "GROQ_STATE_MODEL",
    os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
)

DEFAULT_STATE = {
    "operation": "recommend",
    "out_of_scope": False,
    "refusal_reason": None,
    "clarification_intent": "none",
    "clarification_question": None,
    "role": None,
    "seniority": None,
    "domain": None,
    "test_types": [],
    "soft_test_types": [],
    "requirements": [],
    "personality_required": False,
    "leadership_required": False,
    "technical_required": False,
    "development_use": False,
    "selection_use": False,
    "remote_required": False,
    "adaptive_required": False,
    "final_confirmation": False,
}


def _client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for LLM state extraction.")
    timeout = float(os.getenv("GROQ_TIMEOUT", "20"))
    return Groq(api_key=api_key, timeout=timeout)


def _create_chat_completion(client, **kwargs):
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            if attempt == 2:
                raise

            message = str(exc)
            match = re.search(r"try again in ([0-9.]+)s", message)
            delay = float(match.group(1)) if match else 2 * (attempt + 1)
            time.sleep(min(delay + 1, 45))
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429 and attempt < 2:
                message = str(exc)
                match = re.search(r"try again in ([0-9.]+)s", message)
                delay = float(match.group(1)) if match else 2 * (attempt + 1)
                time.sleep(min(delay + 1, 45))
                continue

            raise RuntimeError(f"Groq state extraction failed: {exc}") from exc


def _conversation_text(messages):
    return "\n".join(
        f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages
    )


def _latest_user_text(messages):
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message.get("content", "")

    return ""


def _repair_short_clarification_answer(state, messages):
    latest = _latest_user_text(messages).strip().lower()
    compact_latest = re.sub(r"[^a-z.]+", "", latest).strip(".")
    conversation = _conversation_text(messages).lower()

    language_answers = {
        "us",
        "u.s.",
        "usa",
        "english",
        "uk",
        "u.k.",
        "australian",
        "indian",
        "spanish",
        "hybrid",
    }

    if (
        state.get("operation") == "clarify"
        and state.get("clarification_intent") == "language_constraint"
        and (
            latest in language_answers
            or compact_latest in language_answers
            or "functionally bilingual" in latest
            or "go with the hybrid" in latest
        )
    ):
        state["operation"] = "recommend"
        state["clarification_intent"] = "none"
        state["clarification_question"] = None

        if "contact center" in conversation or "contact centre" in conversation:
            state["domain"] = state.get("domain") or "customer_service"
        if "healthcare" in conversation or "patient records" in conversation:
            state["domain"] = state.get("domain") or "healthcare"

    return state


def _parse_json_object(content):
    text = content.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])

        raise


def _repair_state(client, conversation, raw_state, error):
    repair_prompt = f"""
    The previous JSON did not satisfy the required state schema.

    Validation error:
    {error}

    Conversation:
    {conversation}

    Previous JSON:
    {json.dumps(raw_state, indent=2)}

    Return corrected JSON only. Preserve correct extracted fields where
    possible. If operation is clarify, choose exactly one valid
    clarification_intent and include exactly one concise clarification_question.
    If operation is not clarify, clarification_intent must be "none" and
    clarification_question must be null.
    """

    response = _create_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": state_system_prompt},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return _parse_json_object(response.choices[0].message.content)


def _normalise_state(raw_state):
    state = DEFAULT_STATE.copy()

    if isinstance(raw_state, dict):
        state.update({key: raw_state.get(key, state[key]) for key in state})

    if not isinstance(state["requirements"], list):
        state["requirements"] = []

    if not isinstance(state["test_types"], list):
        state["test_types"] = []

    if not isinstance(state["soft_test_types"], list):
        state["soft_test_types"] = []

    state["test_types"] = _normalise_test_types(state["test_types"])
    state["soft_test_types"] = _normalise_test_types(state["soft_test_types"])
    state["soft_test_types"] = [
        test_type
        for test_type in state["soft_test_types"]
        if test_type in {"A", "P"} and test_type not in state["test_types"]
    ]

    if state.get("operation") not in {
        "clarify",
        "compare",
        "refine",
        "recommend",
    }:
        state["operation"] = "recommend"

    state["out_of_scope"] = bool(state.get("out_of_scope"))

    if not state["out_of_scope"]:
        state["refusal_reason"] = None

    if state.get("clarification_intent") not in {
        "none",
        "role_missing",
        "seniority_missing",
        "assessment_purpose",
        "role_focus",
        "skill_priority",
        "language_constraint",
        "assessment_mix",
        "constraint_conflict",
    }:
        state["clarification_intent"] = "none"

    for key in [
        "personality_required",
        "leadership_required",
        "technical_required",
        "development_use",
        "selection_use",
        "remote_required",
        "adaptive_required",
        "final_confirmation",
    ]:
        state[key] = _normalise_bool(state.get(key))

    if _clarification_answered(state):
        state["clarification_intent"] = "none"
        state["clarification_question"] = None
        if state["operation"] == "clarify":
            state["operation"] = "recommend"

    has_clarification = state.get("clarification_intent") != "none" and bool(
        state.get("clarification_question")
    )

    if state["operation"] != "clarify" and has_clarification:
        state["operation"] = "clarify"

    if state["operation"] == "clarify" and state["clarification_intent"] == "none":
        raise ValueError(f"Clarify operation missing clarification intent: {raw_state}")

    if state["operation"] != "clarify":
        state["clarification_intent"] = "none"
        state["clarification_question"] = None
    elif state["clarification_intent"] == "none":
        state["clarification_question"] = None
    elif not state.get("clarification_question"):
        raise ValueError(f"Clarification intent missing question: {raw_state}")

    state["domain"] = _normalise_domain(state.get("domain"))

    if not state["test_types"]:
        state["test_types"] = _infer_test_types_from_state(state)
        state["soft_test_types"] = [
            test_type
            for test_type in state["soft_test_types"]
            if test_type in {"A", "P"} and test_type not in state["test_types"]
        ]

    return state


def _normalise_test_types(values):
    allowed = {"A", "B", "C", "D", "E", "K", "P", "S"}
    aliases = {
        "ability": "A",
        "aptitude": "A",
        "cognitive": "A",
        "biodata": "B",
        "situational": "B",
        "sjt": "B",
        "competency": "C",
        "competencies": "C",
        "development": "D",
        "360": "D",
        "assessment exercise": "E",
        "exercise": "E",
        "knowledge": "K",
        "skills": "K",
        "technical": "K",
        "personality": "P",
        "behavior": "P",
        "behaviour": "P",
        "simulation": "S",
        "simulations": "S",
    }
    result = []

    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        code = text.upper()

        if code in allowed:
            mapped = code
        else:
            mapped = aliases.get(text.lower())

        if mapped and mapped not in result:
            result.append(mapped)

    return result


def _normalise_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}

    return bool(value)


def _clarification_answered(state):
    intent = state.get("clarification_intent")

    if intent == "role_missing":
        return bool(state.get("role") or state.get("domain"))

    if intent == "seniority_missing":
        return bool(state.get("seniority"))

    if intent == "assessment_purpose":
        return bool(state.get("selection_use") or state.get("development_use"))

    if intent == "role_focus":
        return bool(state.get("requirements"))

    if intent == "skill_priority":
        return bool(state.get("requirements") or state.get("test_types"))

    if intent == "language_constraint":
        requirements = " ".join(state.get("requirements") or []).lower()
        return any(
            word in requirements
            for word in ["language", "english", "spanish", "french", "german"]
        )

    if intent == "assessment_mix":
        return len(state.get("test_types") or []) > 1

    return False


def _normalise_domain(value):
    if not value:
        return None

    domain = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    aliases = {
        "software": "software_engineering",
        "software_development": "software_engineering",
        "engineering": "software_engineering",
        "technology": "software_engineering",
        "tech": "software_engineering",
        "contact_center": "customer_service",
        "call_center": "customer_service",
        "support": "customer_service",
        "sales_reskilling": "sales",
        "financial": "finance",
        "accounting": "finance",
        "medical": "healthcare",
        "health": "healthcare",
        "safety": "safety_manufacturing",
        "manufacturing": "safety_manufacturing",
        "industrial": "safety_manufacturing",
        "office": "office_productivity",
        "productivity": "office_productivity",
        "excel_word": "office_productivity",
        "languages": "language",
        "cognitive": "general_cognitive",
        "analytics": "data_analytics",
        "data": "data_analytics",
    }

    return aliases.get(domain, domain or None)


def _infer_test_types_from_state(state):
    test_types = []

    if state.get("technical_required"):
        test_types.extend(["K", "S"])
    if state.get("personality_required"):
        test_types.append("P")
    if state.get("leadership_required"):
        test_types.extend(["C", "P", "D", "E"])
    if state.get("development_use"):
        test_types.append("D")

    requirements = " ".join(state.get("requirements") or []).lower()

    if any(
        word in requirements
        for word in [
            "cognitive",
            "ability",
            "aptitude",
            "reasoning",
            "numerical",
            "verbal",
            "inductive",
            "deductive",
        ]
    ):
        test_types.append("A")
    if any(
        word in requirements
        for word in ["situational", "judgment", "judgement", "scenario"]
    ):
        test_types.append("B")
    if any(
        word in requirements
        for word in [
            "simulation",
            "call handling",
            "live coding",
            "phone",
            "contact center",
        ]
    ):
        test_types.append("S")
    if any(
        word in requirements
        for word in ["spoken", "written", "language", "english", "spanish"]
    ):
        test_types.extend(["K", "S"])
    if any(
        word in requirements
        for word in ["communication", "competency", "competencies", "stakeholder"]
    ):
        test_types.append("C")

    return _normalise_test_types(test_types)


def extract_state(messages):
    conversation = _conversation_text(messages)
    client = _client()

    response = _create_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": state_system_prompt},
            {"role": "user", "content": conversation},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    raw_state = _parse_json_object(content)

    try:
        return _repair_short_clarification_answer(
            _normalise_state(raw_state),
            messages,
        )
    except ValueError as exc:
        repaired_state = _repair_state(
            client,
            conversation,
            raw_state,
            str(exc),
        )
        return _repair_short_clarification_answer(
            _normalise_state(repaired_state),
            messages,
        )


if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": (
                "Hiring a mid-level Java "
                "backend engineer with "
                "stakeholder communication"
            ),
        }
    ]

    state = extract_state(messages)

    print(json.dumps(state, indent=2))
