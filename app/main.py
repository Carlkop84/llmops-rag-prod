"""Streamlit front end for the Northwind HR assistant."""

from __future__ import annotations

import logging

import streamlit as st

from app import assistant, config, conversations, search_index

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Northwind HR Assistant", page_icon="📄")

# st.session_state only survives as long as this browser session, so the same
# turns are written to Postgres as they happen. chat_id is the row they belong
# to, and is None until the first question starts a conversation. saving goes
# False if the history store fails, so one bad write doesn't retry every turn.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("chat_id", None)
st.session_state.setdefault("saving", True)

EXAMPLES = [
    "How many holiday days can I carry into next year?",
    "How long is parental leave?",
    "How much sick pay do I get?",
    "Is Juneteenth a company holiday?",
]

ERROR_TEXT = (
    "Something went wrong answering that. Please try again, or contact the "
    "HR service desk: hr-help@northwind.example."
)


# --- History ----------------------------------------------------------------
def _remember(role: str, content: str, sources=None) -> None:
    """Persist one turn. Losing the history must not cost the user their answer."""
    if not st.session_state.saving:
        return
    try:
        if st.session_state.chat_id is None:
            # The first turn of a new chat is always the question, which the
            # conversation is titled after.
            st.session_state.chat_id = conversations.start_chat(content)
        conversations.add_message(st.session_state.chat_id, role, content, sources)
    except Exception:
        logger.exception("could not save chat turn")
        st.session_state.saving = False
        st.session_state.chat_id = None
        st.toast("This conversation is no longer being saved.")


def _history(messages) -> list[dict]:
    """The recent turns to send as context, excluding the live question."""
    limit = max(getattr(config, "MAX_HISTORY_TURNS", 0), 0)
    recent = messages[:-1][-limit:] if limit else []

    # A fixed-size window can open on an assistant reply, which the Messages
    # API rejects. Drop turns until it starts with a question.
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]

    return [
        {"role": message["role"], "content": message["content"]}
        for message in recent
    ]


# --- Asking -----------------------------------------------------------------
def ask(question: str) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    _remember("user", question)

    # Send recent turns along with the question, so a follow-up like "what
    # about in the UK?" makes sense.
    history = _history(st.session_state.messages)

    with st.spinner("Looking through the policy documents..."):
        try:
            result = assistant.answer(question, history)
        except Exception:
            logger.exception("answering failed")
            st.session_state.messages.append(
                {"role": "assistant", "content": ERROR_TEXT, "sources": []}
            )
            # Deliberately not stored: a failure notice is not part of the
            # conversation, and would be replayed to the model as context.
            return

    answer_text = result["answer"]
    if result.get("truncated"):
        answer_text += "\n\n_(This answer was cut short. Ask for the rest.)_"

    sources = result.get("sources", [])
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
    _remember("assistant", answer_text, sources)


def open_chat(chat_id: int) -> None:
    st.session_state.chat_id = chat_id
    st.session_state.messages = conversations.get_messages(chat_id)
    st.session_state.saving = True


# --- Rendering --------------------------------------------------------------
def show_sources(sources) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sorted(sources, key=lambda item: item["number"]):
            header = f"**[{source['number']}] {source.get('title') or 'Untitled'}**"
            if source.get("section"):
                header += f" — {source['section']}"
            st.markdown(header)

            detail = []
            if source.get("filename"):
                detail.append(source["filename"])
            if source.get("version"):
                detail.append(f"version {source['version']}")
            if source.get("effective_date"):
                detail.append(f"effective {source['effective_date']}")
            if source.get("region") and source["region"] != "global":
                detail.append(f"{source['region'].upper()} only")
            if detail:
                st.caption(" · ".join(detail))


def sidebar() -> None:
    if st.sidebar.button("New chat", use_container_width=True):
        st.session_state.chat_id = None
        st.session_state.messages = []
        st.session_state.saving = True
        st.rerun()

    if st.session_state.chat_id is not None:
        if st.sidebar.button("Delete this chat", use_container_width=True):
            conversations.delete_chat(st.session_state.chat_id)
            st.session_state.chat_id = None
            st.session_state.messages = []
            st.rerun()

    try:
        chats = conversations.list_chats()
    except Exception:
        logger.exception("could not list chats")
        return

    if not chats:
        return

    st.sidebar.caption("Earlier chats")
    for chat in chats:
        current = chat["chat_id"] == st.session_state.chat_id
        # Streamlit needs a key per button, because the labels repeat.
        if st.sidebar.button(
            chat["title"],
            key=f"chat-{chat['chat_id']}",
            use_container_width=True,
            type="primary" if current else "secondary",
        ):
            open_chat(chat["chat_id"])
            st.rerun()


# --- Page -------------------------------------------------------------------
st.title("Northwind HR Assistant")

# An assistant that answers "I could not find anything" to every question
# usually has an empty index rather than a broken model, so say that plainly
# rather than leaving somebody to guess. When it is working, this says nothing.
database_ok = True
indexed = 0
try:
    # Cheap and idempotent: the two chat tables are created the first time the
    # page is opened after a fresh database, and skipped every time after that.
    conversations.create_tables()
    indexed = search_index.count()
except Exception as error:
    database_ok = False
    logger.exception("database unreachable")
    st.error(f"Cannot reach the database: {type(error).__name__}")

if database_ok and indexed == 0:
    st.error("No policy documents are indexed. Run: docker compose run --rm ingest")

if not st.session_state.messages:
    st.caption(
        "I answer questions about Northwind policy from the HR document library. "
        "I cannot see your own record: for your leave balance, pay or review, "
        "check Workday or ask the HR service desk."
    )
    columns = st.columns(2)
    for number, example in enumerate(EXAMPLES):
        if columns[number % 2].button(example, use_container_width=True):
            ask(example)
            st.rerun()

# Redraw the conversation. st.chat_message supplies the avatars.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("Ask about holiday, sick pay, parental leave, expenses...")
if question:
    ask(question)
    st.rerun()

sidebar()