import os
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, RateLimitError

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=True)

MODEL_NAME = os.getenv(
    "GROQ_RERANKER_MODEL",
    os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
)


SYSTEM_PROMPT = """
You are an SHL assessment selection assistant.

Your job:
- select ONLY the relevant assessments from the provided candidates
- avoid redundant assessments
- prefer complementary assessments
- ONLY choose from provided candidates
- return between 1 and 10 assessment names
- do NOT pad the list to 10
- fewer recommendations is better when only a few candidates are truly relevant
- write a natural conversational reply that answers the user's latest request
  in the context of the latest user message

Rules:
- NEVER invent products
- NEVER invent URLs
- NEVER select an assessment just to fill the list
- preserve exact assessment names from the candidates
- If the operation is refine, treat the previous shortlist as the working
  shortlist. Preserve previous items unless the latest user message or updated
  requirements make a prior item no longer fit.
- If the operation is refine and retrieved candidates are supplied, use them
  only as possible additions/replacements. Do not let them wipe out unaffected
  previous shortlist items.
- If you omit a previous shortlist item, you MUST list it in omitted_previous
  with a concise reason grounded in the latest user message, updated
  requirements, or redundancy against a better selected item.
- Do not omit prior items silently.
- If the latest user removes an item, do not include it.
- If the latest user asks whether an item is redundant or needed, answer that
  question. Keep it if useful, or omit it only if you list a grounded reason in
  omitted_previous.
- soft_test_types in the extracted state are optional considerations, not
  requirements. Include those assessment types only when they clearly
  complement the user's explicit role needs and do not crowd out more direct
  matches.
- Do not add aptitude or personality solely because they appear in
  soft_test_types.
- Ground any explanation only in the provided candidates and extracted state.
- Keep the reply concise but not robotic.
- return ONLY valid JSON

Output schema:

{
  "reply": "short reason for the selected set",
  "selected_names": ["exact assessment name"],
  "omitted_previous": [
    {
      "name": "exact previous shortlist name",
      "reason": "why this prior item should be removed now"
    }
  ]
}
"""


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

            raise RuntimeError(f"Groq reranker failed: {exc}") from exc


def _latest_user_text(messages):
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message.get("content", "")

    return ""


def _names(items):
    return [item.get("name", "") for item in items or [] if item.get("name")]


def _candidate_block(items, title):
    if not items:
        return f"{title}:\n[]"

    text = f"{title}:\n"

    for idx, item in enumerate(items, start=1):
        text += f"""
        {title} Candidate {idx}

        Name:
        {item["name"]}

        Description:
        {item["description"]}

        Categories:
        {", ".join(item["keys"])}

        Job Levels:
        {", ".join(item["job_levels"])}
        """

    return text


def _compact_candidate_block(items, title):
    if not items:
        return f"{title}:\n[]"

    lines = [f"{title}:"]

    for idx, item in enumerate(items, start=1):
        description = re.sub(r"\s+", " ", item["description"]).strip()[:180]
        lines.append(
            (
                f"{idx}. {item['name']} | "
                f"Categories: {', '.join(item['keys'])} | "
                f"Levels: {', '.join(item['job_levels'])} | "
                f"Description: {description}"
            )
        )

    return "\n".join(lines)


def rerank_candidates(
    state,
    candidates,
    messages=None,
    operation="recommend",
    previous_shortlist=None,
    new_candidates=None,
):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for assessment selection.")

    timeout = float(os.getenv("GROQ_TIMEOUT", "20"))
    client = Groq(api_key=api_key, timeout=timeout)

    prompt = f"""
    Latest User Message:

    {_latest_user_text(messages or [])}

    Operation:

    {operation}

    User Requirements:

    {json.dumps(state, indent=2)}

    Previous Shortlist Names:

    {json.dumps(_names(previous_shortlist), indent=2)}

    Current Retrieved Candidate Names For This Turn:

    {json.dumps(_names(new_candidates), indent=2)}

    Previous Shortlist Candidate Details:

    {_compact_candidate_block(previous_shortlist or [], "Previous")}

    Current Retrieved Candidate Details:

    {_compact_candidate_block(new_candidates or [], "Allowed")}

    Select only the relevant final assessment set.
    Choose selected_names only from the previous shortlist or current retrieved candidates.
    If Previous Shortlist Names is non-empty, use it as the working shortlist.
    Keep previous shortlist items by default.
    You may remove previous shortlist items, but only when they no longer fit
    the latest user message or updated requirements, or when they are redundant
    with a better selected item.
    Every removed previous shortlist item must appear in omitted_previous with
    a specific grounded reason. Do not silently drop prior items.
    Add Current retrieved candidates only when they satisfy a new explicit
    request in the latest user message.
    Return 1 to 10 selected_names.
    Do not select weak matches just to reach 10.
    The top-level reply should directly answer the latest user message and
    explain the shortlist in this conversation's context.
    """

    response = _create_chat_completion(
        client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    result = _parse_json_object(content)

    valid_names = {item["name"] for item in candidates}
    selected = result.get("selected_names") or result.get("selected", [])

    valid_selected = []

    for item in selected:
        if isinstance(item, str):
            name = item
        else:
            name = item.get("name", "")

        if name in valid_names and name not in valid_selected:
            valid_selected.append(name)

    result["selected_names"] = valid_selected[:10]
    omitted_previous = []

    for item in result.get("omitted_previous", []):
        if isinstance(item, str):
            name = item
            reason = ""
        else:
            name = item.get("name", "")
            reason = item.get("reason", "")

        if name in valid_names and reason:
            omitted_previous.append(
                {
                    "name": name,
                    "reason": reason,
                }
            )

    result["omitted_previous"] = omitted_previous

    if not result["selected_names"]:
        raise ValueError(f"No valid assessments selected by reranker: {content}")

    return result
