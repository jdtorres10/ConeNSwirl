import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_community.document_loaders import TextLoader

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


def coerce_chain_answer_to_text(answer: Any) -> str:
    """
    Retrieval + stuff-documents chains may return a str or an AIMessage with string or
    block-list content. Session history and JSON responses must use plain strings only —
    storing an AIMessage inside AIMessage breaks later turns.
    """
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer.strip()
    content = getattr(answer, "content", None)
    if content is None:
        return str(answer).strip()
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                t = getattr(block, "text", None)
                parts.append(t if isinstance(t, str) else str(block))
        return "".join(parts).strip()
    return str(content).strip()


def sanitize_chat_history_messages(messages: list[Any]) -> list[BaseMessage]:
    """
    Force string-only message content before sending chat_history to the retriever/LLM.
    Older sessions may contain AIMessage list blocks or accidentally nested objects; some
    providers reject those on subsequent turns.
    """
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, AIMessage):
            out.append(AIMessage(content=coerce_chain_answer_to_text(m)))
        elif isinstance(m, HumanMessage):
            c = m.content
            if isinstance(c, str):
                out.append(HumanMessage(content=c.strip()))
            else:
                out.append(HumanMessage(content=coerce_chain_answer_to_text(c)))
        else:
            out.append(m)
    return out


SYSTEM_PROMPT = """You are a friendly and enthusiastic assistant for Cone N' Swirl, \
a unique ice cream truck in San Antonio, TX. We serve blended ice cream in \
fresh-baked Hungarian chimney cones — a one-of-a-kind experience!

Use the retrieved context to help with:
- This week's stops, locations, hours, and where schedule updates live
- Contact: email, phone, and social media (@conenswirl on Instagram, Facebook, TikTok)
- Catering and private events (basics, minimums, how to reach us)
- General menu education: what Cone N' Swirl, Cup N' Swirl, and Cone Only are, what's included, \
  pricing from the files, popular combinations, allergens and dietary notes — never invent items \
  we do not list

Order building does NOT happen in this chat: do not walk the customer step-by-step through \
cone/cup choices (no "pick your base", "what blend next?", etc.). If they want to build or place \
an order, tell them clearly to tap **Build My Order** in this same window and use the chip buttons — \
that flow matches our menu card. You may still explain at a high level how the three order types differ.

If chat history includes a structured order note from the button builder (bracket text with JSON), \
answer follow-ups about that order: what they picked, pricing from context, allergies, or suggest \
they confirm details at the truck.

Wording: Do not insist on exact spelling; map casual phrases and typos to the closest real info in \
the context. When you quote schedule or menu text, match the context's wording.

Output: Plain text only (no Markdown: no asterisks, hash headings, or backticks). Use normal sentences; \
for lists start each line with a hyphen and a space.

Pricing: Use figures from context. If you add a total, check the math. If unsure, give piece prices and \
say the crew rings the exact total at the truck.

Tone: Warm San Antonio ice cream truck energy, clear and concise. Not a legal disclaimer unless the \
topic is allergies or food safety.

If the answer is not in the context, say so briefly and point to @conenswirl on Instagram or \
(956) 324-8733.

{context}"""


def _load_documents():
    docs = []
    for filename in ["contact.txt", "faq.txt", "menu_details.txt", "schedule.txt"]:
        loader = TextLoader(os.path.join(KB_PATH, filename))
        docs.extend(loader.load())
    menu_csv_path = os.path.join(KB_PATH, "Menu.csv")
    with open(menu_csv_path, encoding="utf-8") as f:
        csv_text = f.read()
    docs.append(
        Document(
            page_content="Menu.csv (reference table of cone combinations):\n" + csv_text,
            metadata={"source": "Menu.csv"},
        )
    )
    return docs


def build_chain():
    docs = _load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = VoyageAIEmbeddings(model="voyage-3", voyage_api_key=os.getenv("VOYAGE_API_KEY"))
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.28,
        max_tokens=768,
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user message, produce ONE standalone "
            "line suitable for searching Cone N' Swirl knowledge (truck schedule and stops, "
            "contact email and phone, catering, FAQ, menu offerings and pricing, allergens, "
            "how cones and cups work).\n"
            "- If the latest message is already a clear question, return it unchanged.\n"
            "- If they mention a day, neighborhood, this week, rain, Instagram, email, or phone, "
            "include those cues in the search line.\n"
            "- If they sound like they want to build or place an order by typing picks in chat, "
            "rewrite into a line about using the on-screen Build My Order button flow instead "
            "of collecting toppings in chat.\n"
            "Do NOT answer the user. Output only the standalone line, no preamble."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, qa_chain)


_ORDER_SUMMARY_PROMPT = """You are Swirly, the Cone N' Swirl assistant. A customer just finished \
picking their build using the on-screen buttons (not free-typed chat). Their exact choices are \
in the JSON below — do not invent add-ons they did not pick.

Turn this into a friendly plain-English recap they can read at the window: what they're getting, \
in a natural order. The JSON uses order_type (cone_n_swirl, cup_n_swirl, or cone_only) and fields \
cone_type, base, filling, standard_blend, extra_blend, premium_blend, stick_em, drizzle — use \
only the keys that are non-null. If they chose no filling, no extra blend, no premium blend, \
no Stick'em, or no drizzle, say so briefly using the exact meaning of those choices. Mention that \
pricing is confirmed at the truck if you are not listing dollar totals.

Output rules: plain text only — no Markdown (no asterisks, no hash headings, no backticks). \
Short paragraphs; warm San Antonio ice-cream-truck energy.

Customer order JSON:
{order_json}"""


def summarize_build_order(order: dict) -> str:
    """Turn a validated structured order into conversational plain text."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.35,
        max_tokens=512,
    )
    msg = _ORDER_SUMMARY_PROMPT.format(order_json=json.dumps(order, indent=2))
    out = llm.invoke(msg)
    text = getattr(out, "content", None)
    if text is None:
        text = str(out)
    elif isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(getattr(block, "text", str(block)))
        text = "".join(parts)
    else:
        text = str(text)
    text = text.strip()
    if not text:
        raise ValueError("empty summary from model")
    return text
