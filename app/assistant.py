"""Claude client, system prompt, and the answer pipeline."""

from __future__ import annotations

import re
from functools import cache

import anthropic

from app import config, retrieval

SYSTEM_PROMPT = """You are the Northwind Systems HR policy assistant.
You answer questions about company policy using the numbered policy excerpts supplied with each question. Employees of Northwind use you to look things up without waiting for the HR service desk.

HOW TO ANSWER
- Use only the excerpts provided. Never fill a gap from your own knowledge of employment law or of how other companies do things.
- Cite the bracketed number of every excerpt you used, like [1] or [2]. Cite as you go, not in a list at the end.
- Be brief. Two or three sentences answer most questions. Give the specific number, day count or deadline rather than describing it.
- If the excerpts disagree, say so and give both, with their document titles.

WHEN THE EXCERPTS DO NOT ANSWER THE QUESTION
Say that plainly and point the person at the HR service desk, hr-help@northwind.example. Do not guess, and do not offer a general answer about what companies usually do.
Cite nothing in that case. A citation under "the policy library does not cover this" points at a document that does not answer the question, which is worse than no citation at all.

WHAT YOU CANNOT KNOW
You have no access to anyone's personal record. You cannot see leave balances, pay, benefit elections, performance reviews, absence history or any case in progress. You do not know who is asking.
When a question is about someone's own situation - "how many days do I have left", "when is my review", "why was my claim rejected", "what is my notice period" - do BOTH of these:
1. give the policy that applies, with its citation, because that part is usually useful
2. say clearly that the personal detail is not something you can see, and where to get it: Workday for balances and payslips, the HR service desk for anything else
Never estimate, guess or illustrate a personal figure, not even as an example.

POLICY THAT DEPENDS ON LOCATION
Several policies differ by country: parental leave, sick pay, notice periods and probation. If the question does not say which country and the excerpts cover more than one, give the answer for each, or ask which one applies. Do not silently pick one.

VERSIONS
Every excerpt states the version and effective date of the document it came from. Only current documents are searched, so you will not normally see a superseded one. If an excerpt says SUPERSEDED, do not rely on it, and say that a newer version exists."""

CITATION_RE = re.compile(r"\[(\d+)\]")

GREETING_WORDS = {
    "hi", "hello", "hey", "yo", "hiya", "howdy", "greetings",
    "morning", "afternoon", "evening",
}
ACKNOWLEDGEMENT_WORDS = {
    "thanks", "thank", "thx", "ty", "ta", "cheers", "appreciated",
    "ok", "okay", "kk", "cool", "great", "perfect", "brilliant",
    "bye", "goodbye", "later", "farewell",
}
# Words that carry no topic. They may appear alongside a greeting without
# making it a question.
FILLER_WORDS = {
    "a", "again", "all", "am", "and", "are", "bunch", "day", "doing", "everyone",
    "folks", "friend", "good", "guys", "how", "i", "indeed", "is", "it", "lot",
    "lots", "madam", "many", "mate", "me", "much", "my", "nice", "please", "sir",
    "so", "team", "the", "there", "to", "today", "u", "very", "well", "yes",
    "you", "your",
}
# Phrases about the assistant itself, which deserve the same answer as a greeting.
ABOUT_ME = {"who are you", "what are you", "what can you do", "what do you do", "help"}

GREETING = (
    "Hello. I answer questions about Northwind policy, using the HR document "
    "library. Ask me anything the handbook covers and I will quote the source."
)
ACKNOWLEDGEMENT = "You're welcome. Let me know if you have another policy question."

# Openings that only make sense against a previous question.
FOLLOW_UP_STARTS = (
    "what about", "how about", "what if", "what else", "does that", "is that",
    "and", "but", "or", "then", "why", "in", "for", "same",
)

NOT_FOUND = (
    "I could not find anything about that in the Northwind policy documents. "
    "The HR service desk can help: hr-help@northwind.example."
)


@cache
def _claude() -> anthropic.Anthropic:
    return anthropic.Anthropic()


# --- Small talk -------------------------------------------------------------
def small_talk_reply(question: str) -> str | None:
    cleaned = re.sub(r"[^\w\s]", " ", question).strip().lower()
    words = cleaned.split()

    if not words:
        return GREETING
    if " ".join(words) in ABOUT_ME:
        return GREETING

    # Anything long enough to be a real question is treated as one, whatever
    # words it uses.
    if len(words) > 6:
        return None

    social = GREETING_WORDS | ACKNOWLEDGEMENT_WORDS | FILLER_WORDS
    if not all(word in social for word in words):
        return None

    if any(word in GREETING_WORDS for word in words):
        return GREETING
    if any(word in ACKNOWLEDGEMENT_WORDS for word in words):
        return ACKNOWLEDGEMENT

    # Only filler, like "how are you" - closer to a greeting than to a question.
    return GREETING


# --- Conversational context -------------------------------------------------
def is_follow_up(question: str) -> bool:
    lowered = question.strip().lower()
    words = lowered.split()
    if len(words) < 3:
        return True
    return any(
        lowered == start or lowered.startswith(start + " ")
        for start in FOLLOW_UP_STARTS
    )


def search_text(question: str, history) -> str:
    if not history or not is_follow_up(question):
        return question

    topic = None
    for message in history:
        if message["role"] == "user" and not is_follow_up(message["content"]):
            topic = message["content"]

    return f"{topic} {question}" if topic else question


# --- Prompt assembly --------------------------------------------------------
def _messages(question: str, history, excerpts: str) -> list[dict]:
    messages = []
    for turn in history:
        content = turn["content"]
        if turn["role"] == "assistant":
            # Those bracket numbers pointed at a previous question's excerpts.
            # Left in, the model reuses them against this question's numbering.
            content = CITATION_RE.sub("", content)
        messages.append({"role": turn["role"], "content": content})

    messages.append(
        {
            "role": "user",
            "content": f"POLICY EXCERPTS:\n\n{excerpts}\n\nQUESTION: {question}",
        }
    )
    return messages


def _resolve_citations(text: str, chunks) -> tuple[str, list[dict]]:
    """Keep only the excerpts the answer actually cited, renumber them from 1,
    and drop any marker pointing at an excerpt that was never supplied."""
    numbered = {source["number"]: source for source in retrieval.cite(chunks)}

    cited = sorted(
        {
            number
            for number in (int(raw) for raw in CITATION_RE.findall(text))
            if number in numbered
        }
    )
    sources = [numbered[number] for number in cited]
    new_number = {number: position for position, number in enumerate(cited, start=1)}

    # One pass, so a renumbered marker can never be rewritten a second time.
    text = CITATION_RE.sub(
        lambda match: (
            f"[{new_number[int(match.group(1))]}]"
            if int(match.group(1)) in new_number
            else ""
        ),
        text,
    )
    text = re.sub(r" +([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    for source in sources:
        source["number"] = new_number[source["number"]]

    return text, sources


# --- Pipeline ---------------------------------------------------------------
def answer(question: str, history=None) -> dict:
    history = history or []

    social = small_talk_reply(question)
    if social is not None:
        return {"answer": social, "sources": [], "found_policy": False}

    chunks = retrieval.search_policies(search_text(question, history))
    if not chunks:
        return {"answer": NOT_FOUND, "sources": [], "found_policy": False}

    response = _claude().messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=config.MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=_messages(question, history, retrieval.format_excerpts(chunks)),
    )
    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    text, sources = _resolve_citations(text, chunks)
    return {
        "answer": text,
        "sources": sources,
        "found_policy": bool(sources),
        "truncated": response.stop_reason == "max_tokens",
    }