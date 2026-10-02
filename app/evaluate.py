"""Evaluation harness: run the golden set, or measure the relevance floor."""

from __future__ import annotations

import argparse
import json

from app import assistant, config, embeddings, search_index

CASES_FILE = config.DATA_DIR / "evaluation.json"


def load_cases() -> list[dict]:
    try:
        cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"no evaluation set at {CASES_FILE}") from None
    except json.JSONDecodeError as error:
        raise SystemExit(f"{CASES_FILE.name} is not valid JSON: {error}") from None

    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"{CASES_FILE.name} should contain a non-empty list of cases")

    seen: set[str] = set()
    for position, case in enumerate(cases, start=1):
        case_id = case.get("id")
        if not case_id:
            raise SystemExit(f"case {position} has no id")
        if case_id in seen:
            raise SystemExit(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if "question" not in case:
            raise SystemExit(f"{case_id}: no question")
    return cases


# --- Assertions -------------------------------------------------------------
def check(case: dict, result: dict) -> list[str]:
    text = (result.get("answer") or "").lower()
    failures: list[str] = []

    required = case.get("must_contain", [])
    if required:
        missing = [phrase for phrase in required if phrase.lower() not in text]
        if case.get("any_of"):
            if len(missing) == len(required):
                failures.append(f"none of {required} appeared")
        elif missing:
            failures.append(f"missing {missing}")

    disclaimers = case.get("must_disclaim", [])
    if disclaimers and not any(phrase.lower() in text for phrase in disclaimers):
        failures.append(f"never disclaimed: none of {disclaimers} appeared")

    for forbidden in case.get("must_not_contain", []):
        if forbidden.lower() in text:
            failures.append(f"said {forbidden!r}")

    if case.get("expect_no_sources") and result.get("sources"):
        cited = [source.get("title") for source in result["sources"]]
        failures.append(f"cited {cited} for a question with no answer")

    if case.get("expect_sources") and not result.get("sources"):
        failures.append("cited nothing")

    if result.get("truncated"):
        failures.append("answer hit the token limit and was cut off")

    return failures


# --- Golden set -------------------------------------------------------------
def run_case(case: dict) -> tuple[dict, list[str]]:
    # A case can be asked partway through a conversation. "after" lists the
    # questions that came first; they are really asked, so the history this
    # case sees is the history a person would have produced.
    history: list[dict] = []
    for earlier in case.get("after", []):
        earlier_answer = assistant.answer(earlier, history)
        history.append({"role": "user", "content": earlier})
        history.append({"role": "assistant", "content": earlier_answer["answer"]})

    answer = assistant.answer(case["question"], history)
    return answer, check(case, answer)


def run_suite() -> bool:
    cases = load_cases()
    passed = 0

    for case in cases:
        try:
            answer, failures = run_case(case)
        except Exception as error:
            print(f"ERROR {case['id']}: {type(error).__name__}: {error}")
            print(f"       question: {case['question']}")
            continue

        if failures:
            print(f"FAIL {case['id']}: {'; '.join(failures)}")
            print(f"     question: {case['question']}")
            print(f"     answer:   {answer['answer'][:200]}")
        else:
            passed += 1
            print(f"pass {case['id']}")

    print(f"\n{passed}/{len(cases)} passed")
    return passed == len(cases)


# --- Threshold measurement --------------------------------------------------
def best_similarity(question: str, history) -> float:
    # The raw top score, with no relevance floor applied - this is what the
    # floor itself is measured against, so it cannot use search_policies, which
    # already applies one.
    text = assistant.search_text(question, history)
    vector = embeddings.embed([text])[0]
    chunks = search_index.search(vector)
    return max((chunk["similarity"] for chunk in chunks), default=0.0)


def _kind(case: dict) -> str:
    if case.get("small_talk") or case.get("id") == "small-talk":
        return "small talk"
    if case.get("trap") == "not in corpus":
        return "no answer in corpus"
    return "real question"


def run_thresholds() -> None:
    cases = load_cases()
    groups: dict[str, list[float]] = {
        "real question": [],
        "small talk": [],
        "no answer in corpus": [],
    }

    for case in cases:
        # search_text only looks at the role and content of earlier USER turns,
        # so there is no need to actually call the model for each "after"
        # question here.
        history = [
            {"role": "user", "content": earlier}
            for earlier in case.get("after", [])
        ]
        groups[_kind(case)].append(best_similarity(case["question"], history))

    if not groups["real question"]:
        raise SystemExit("no real questions in the evaluation set, nothing to measure")

    lowest_real = min(groups["real question"])
    highest_small_talk = max(groups["small talk"], default=0.0)
    highest_no_answer = max(groups["no answer in corpus"], default=0.0)

    print(f"{'kind':<30} best score   n")
    print(f"{'real question (lowest)':<30} {lowest_real:.3f}"
          f"        {len(groups['real question'])}")
    print(f"{'small talk (highest)':<30} {highest_small_talk:.3f}"
          f"        {len(groups['small talk'])}")
    print(f"{'no answer in corpus (highest)':<30} {highest_no_answer:.3f}"
          f"        {len(groups['no answer in corpus'])}")

    highest_negative = max(highest_small_talk, highest_no_answer)
    if highest_negative >= lowest_real:
        print(
            f"\n{highest_negative:.3f} is ABOVE the lowest real question "
            f"({lowest_real:.3f}), so no threshold separates those two."
        )
        return

    print(
        f"\nMIN_BEST_SIMILARITY can sit anywhere between {highest_negative:.3f} "
        f"and {lowest_real:.3f}."
    )
    current = getattr(config, "MIN_BEST_SIMILARITY", None)
    if current is not None:
        fits = highest_negative <= current <= lowest_real
        print(f"Currently {current:.3f} — {'inside' if fits else 'OUTSIDE'} that range.")


# --- CLI --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the golden set, or measure the relevance floor."
    )
    parser.add_argument(
        "--thresholds",
        action="store_true",
        help="measure the relevance floor instead of running the golden set",
    )
    args = parser.parse_args()

    if args.thresholds:
        run_thresholds()
        return

    raise SystemExit(0 if run_suite() else 1)


if __name__ == "__main__":
    main()