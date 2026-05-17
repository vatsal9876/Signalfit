import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "shl_catalog.json"
DEFAULT_TRACES_DIR = ROOT.parent / "GenAI_SampleConversations"


@dataclass
class Turn:
    number: int
    user: str
    agent: str
    expected_names: list[str]


@dataclass
class EvalCase:
    trace_id: str
    turn_number: int
    messages: list[dict[str, str]]
    expected_names: list[str]


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as file:
        catalog = json.load(file)

    by_url = {}
    by_name = {}

    for item in catalog:
        name = item.get("name", "")
        url = item.get("link", "")

        if name:
            by_name[normalise_name(name)] = name
        if url:
            by_url[normalise_url(url)] = name

    return catalog, by_url, by_name


def normalise_url(url):
    return url.strip().strip("<>").rstrip("/").lower()


def normalise_name(name):
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def clean_block(body):
    body = body.strip()
    lines = []

    for line in body.splitlines():
        line = line.strip()

        if line.startswith(">"):
            line = line[1:].strip()

        lines.append(line)

    return "\n".join(lines).strip()


def canonical_expected_names(agent_text, by_url, by_name):
    expected = []
    seen = set()

    for line in agent_text.splitlines():
        if not line.strip().startswith("|"):
            continue
        if "shl.com/products/product-catalog/view/" not in line:
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

        if len(cells) < 2:
            continue

        raw_name = cells[1]
        url_match = re.search(r"<([^>]+)>", line)
        catalog_name = None

        if url_match:
            catalog_name = by_url.get(normalise_url(url_match.group(1)))

        if not catalog_name:
            catalog_name = by_name.get(normalise_name(raw_name), raw_name)

        if catalog_name and catalog_name not in seen:
            expected.append(catalog_name)
            seen.add(catalog_name)

    return expected


def parse_trace(path, by_url, by_name):
    text = path.read_text(encoding="utf-8")
    chunks = re.split(r"\n### Turn\s+", text)
    turns = []

    for chunk in chunks[1:]:
        number_match = re.match(r"(\d+)", chunk.strip())
        if not number_match:
            continue

        number = int(number_match.group(1))
        user_marker = "**User**"
        agent_marker = "**Agent**"
        user_start = chunk.find(user_marker)
        agent_start = chunk.find(agent_marker)

        if user_start < 0 or agent_start < 0 or agent_start < user_start:
            continue

        user_body = chunk[user_start + len(user_marker) : agent_start]
        agent_body = chunk[agent_start + len(agent_marker) :]

        user = clean_block(user_body)
        agent = clean_block(agent_body)
        expected_names = canonical_expected_names(agent, by_url, by_name)

        turns.append(Turn(number, user, agent, expected_names))

    return turns


def assistant_context(turn):
    text = re.sub(r"\n\|.*", "", turn.agent, flags=re.DOTALL).strip()

    if turn.expected_names:
        text = f"{text} Shortlist: {'; '.join(turn.expected_names)}."

    return text or "No recommendations this turn."


def build_eval_cases(traces_dir):
    _, by_url, by_name = load_catalog()
    cases = []

    for path in sorted(Path(traces_dir).glob("C*.md")):
        if path.name.endswith(":Zone.Identifier"):
            continue

        turns = parse_trace(path, by_url, by_name)
        history = []

        for turn in turns:
            history.append(
                {
                    "role": "user",
                    "content": turn.user,
                }
            )

            if turn.expected_names:
                cases.append(
                    EvalCase(
                        trace_id=path.stem,
                        turn_number=turn.number,
                        messages=list(history),
                        expected_names=turn.expected_names,
                    )
                )

            history.append(
                {
                    "role": "assistant",
                    "content": assistant_context(turn),
                }
            )

    return cases


def run_local_agent(messages):
    sys.path.insert(0, str(ROOT))

    from agent.controller import agent

    return agent(messages)


def run_endpoint(endpoint, messages):
    payload = json.dumps({"messages": messages}).encode("utf-8")
    req = request.Request(
        endpoint.rstrip("/") + "/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def canonical_prediction_names(response, by_url, by_name):
    names = []
    seen = set()

    for item in response.get("recommendations", [])[:10]:
        catalog_name = None
        url = item.get("url", "")
        name = item.get("name", "")

        if url:
            catalog_name = by_url.get(normalise_url(url))

        if not catalog_name and name:
            catalog_name = by_name.get(normalise_name(name), name)

        if catalog_name and catalog_name not in seen:
            names.append(catalog_name)
            seen.add(catalog_name)

    return names


def validate_schema(response):
    problems = []

    if not isinstance(response, dict):
        return ["response is not an object"]

    if not isinstance(response.get("reply"), str):
        problems.append("reply is not a string")
    if not isinstance(response.get("recommendations"), list):
        problems.append("recommendations is not a list")
    if not isinstance(response.get("end_of_conversation"), bool):
        problems.append("end_of_conversation is not a boolean")

    allowed_response_keys = {"reply", "recommendations", "end_of_conversation"}
    extra_response_keys = set(response) - allowed_response_keys

    if extra_response_keys:
        problems.append(f"extra response keys: {sorted(extra_response_keys)}")

    for idx, item in enumerate(response.get("recommendations", []), start=1):
        if not isinstance(item, dict):
            problems.append(f"recommendation {idx} is not an object")
            continue

        allowed_item_keys = {"name", "url", "test_type"}
        extra_item_keys = set(item) - allowed_item_keys

        if extra_item_keys:
            problems.append(
                f"recommendation {idx} extra keys: {sorted(extra_item_keys)}"
            )

        for key in allowed_item_keys:
            if not isinstance(item.get(key), str):
                problems.append(f"recommendation {idx}.{key} is not a string")

    return problems


def score_case(case, response, by_url, by_name):
    predicted = canonical_prediction_names(response, by_url, by_name)
    expected = set(case.expected_names)
    hits = sorted(expected & set(predicted))
    recall = len(hits) / len(expected) if expected else 0.0

    return {
        "trace": case.trace_id,
        "turn": case.turn_number,
        "expected": case.expected_names,
        "predicted": predicted,
        "hits": hits,
        "recall_at_10": recall,
        "schema_problems": validate_schema(response),
    }


def summarise_by_trace(results):
    by_trace = {}

    for result in results:
        trace = result["trace"]
        by_trace.setdefault(trace, []).append(result)

    summary = {}

    for trace, trace_results in sorted(by_trace.items()):
        recalls = [result["recall_at_10"] for result in trace_results]
        schema_failures = sum(
            1 for result in trace_results if result["schema_problems"]
        )

        summary[trace] = {
            "cases": len(trace_results),
            "mean_recall_at_10": (sum(recalls) / len(recalls) if recalls else 0.0),
            "schema_failures": schema_failures,
        }

    return summary


def print_trace_summary(summary):
    print("\nPer-conversation mean Recall@10")
    print("--------------------------------")

    for trace, values in summary.items():
        print(
            f"{trace}: "
            f"{values['mean_recall_at_10']:.3f} "
            f"({values['cases']} recommendation checkpoints, "
            f"{values['schema_failures']} schema failures)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Signalfit on public SHL conversation traces."
    )
    parser.add_argument(
        "--traces-dir",
        default=str(DEFAULT_TRACES_DIR),
        help="Directory containing C1.md ... C10.md trace files.",
    )
    parser.add_argument(
        "--endpoint",
        help="Base URL of a running FastAPI service, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Call agent.controller.agent directly instead of HTTP. This is the default when --endpoint is not set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print parsed evaluation cases; do not call the agent.",
    )
    args = parser.parse_args()

    _, by_url, by_name = load_catalog()
    cases = build_eval_cases(args.traces_dir)

    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "trace": case.trace_id,
                        "turn": case.turn_number,
                        "expected": case.expected_names,
                        "messages": case.messages,
                    }
                    for case in cases
                ],
                indent=2,
            )
        )
        return

    use_local = args.local or not args.endpoint

    results = []

    for case in cases:
        if use_local:
            response = run_local_agent(case.messages)
        else:
            response = run_endpoint(args.endpoint, case.messages)

        result = score_case(case, response, by_url, by_name)
        results.append(result)

        print(
            f"{case.trace_id} turn {case.turn_number}: "
            f"recall@10={result['recall_at_10']:.3f} "
            f"hits={len(result['hits'])}/{len(case.expected_names)}"
        )

        if result["schema_problems"]:
            print(f"  schema: {result['schema_problems']}")

    mean_recall = (
        sum(result["recall_at_10"] for result in results) / len(results)
        if results
        else 0.0
    )
    schema_failures = sum(1 for result in results if result["schema_problems"])
    by_trace = summarise_by_trace(results)

    print_trace_summary(by_trace)

    print("\nOverall")
    print("-------")
    print(f"Cases: {len(results)}")
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print(f"Schema failures: {schema_failures}")

    print("\nJSON")
    print("----")
    print(
        json.dumps(
            {
                "cases": len(results),
                "mean_recall_at_10": mean_recall,
                "schema_failures": schema_failures,
                # "by_trace": by_trace,
                # "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
