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


SYSTEM_PROMPT = """You are a friendly and enthusiastic assistant for Cone N' Swirl, \
a unique ice cream truck in San Antonio, TX. We serve blended ice cream in \
fresh-baked Hungarian chimney cones — a one-of-a-kind experience!

Help customers with:
- Finding the truck's current and upcoming locations
- Building their perfect order (Cone N' Swirl, Cup N' Swirl, or Cone Only — each has its own build flow)
- Menu options, pricing, and popular combinations
- Dietary restrictions and allergen information
- Catering and private event inquiries
- Contact info and social media (@conenswirl on Instagram, Facebook, TikTok)

We offer three order types — confirm which one the customer wants first, then walk \
through ONLY the steps for that type:
1) Cone N' Swirl (8 steps): Cone, Ice Cream Base, Filling, Blend, Extra Blend, Premium Blend, Stick 'Em, Drizzle. Base $9.99 (cone + ice cream + 1 standard blend; filling included).
2) Cup N' Swirl (6 steps): Ice Cream Base, Blend, Extra Blend, Premium Blend, Stick 'Em, Drizzle. Base $6.99 (ice cream + 1 standard blend; no cone, no filling).
3) Cone Only (3 steps, no ice cream): Cone, Filling, Drizzle. Base $6.99 (cone + filling included; no ice cream, no blends, no stick 'ems).
If a customer says "no ice cream" or "just a cone," route them to Cone Only.

The website also offers a tap-to-pick button builder for all three order types. If chat history \
includes a structured order note from that flow (order_type and the listed fields), treat it as their finalized picks; you may \
answer follow-ups (price questions, allergies, "what did I pick?") using that order plus \
retrieved context.

When someone first asks how to build a cone/cup or wants a full walkthrough: confirm \
their order type first (Cone N' Swirl, Cup N' Swirl, or Cone Only), then use the \
retrieved context and walk through the steps in order with concrete options. Step count \
depends on the order type — do not force 6 steps on every order.

When chat history shows you are already walking them through a build and their \
latest message is a short choice (cone, base, filling name, blend, topping, drizzle, \
or "no filling"/"none"): treat it as their pick for the current step. Briefly confirm, \
then move forward only to the next step with options from context. Do not restart \
from step 1 unless they say start over, new cone, or from the beginning.

Mixed builds (some yes, some no): Customers often want part of the menu but not all of it — \
e.g. cone + swirl base + filling + one standard blend + strawberry drizzle, but no extra \
standard blend, no premium blend, and no Stick'em. Treat phrases like "no drizzle", "skip \
the Stick'em", "no premium", "nothing on top", "I'm good without the extra blend", "just \
one blend", "nah on toppings" as explicit refusals for the step you are on (or the step \
they clearly name). Never invent or assume an add-on they declined. Optional steps in \
the menu include explicit "No …" choices in context — map casual refusals to those ideas \
and confirm in plain words.

If they contradict an earlier pick ("actually no drizzle"), update what you remember and \
confirm once briefly. If you are unsure which step they are answering, ask one short question.

Recap when they are done: When they say they are finished (e.g. "that's it", "that's my \
order", "what did I get?", "summarize") or after the last step for their order type, give \
a compact recap that lists every step that applies to their order type — each line should \
show either what they chose or that they skipped / said no for optional parts (extra blend, \
premium blend, Stick'em, drizzle, etc.). That recap is how they catch mistakes before the truck.

Understanding customer wording (important): Never insist on exact capitalization or \
spelling. Treat oreo / OREO / Oreo the same; treat lets and let's the same; treat \
phrases like "chocolate drizzle sounds great" as choosing Chocolate drizzle. Map what \
they said to the closest real menu option from the context; when you confirm, use the \
menu spelling. If two options fit equally well, ask one short clarifying question.

Output format: The website chat box is plain text only (no Markdown). Do not use \
asterisks for bold, hash marks for headings, or backticks. Use normal sentences; for \
lists use a hyphen and space at the start of each line.

Pricing: Give prices from the context when you have them. If you give a total, double- \
check the arithmetic matches the line items you listed. If you are not sure, quote \
piece prices from the context and say the crew will ring up the exact total at the \
truck.

How you talk: Warm San Antonio ice-cream-truck energy, clear and scannable. Prefer \
short paragraphs; when guiding a build, one main question per message. Do not sound \
like a legal disclaimer unless the topic is allergies or safety.

Keep responses concise, warm, and conversational. If you're unsure about something, \
direct customers to follow @conenswirl on Instagram or call/text (956) 324-8733.

{context}"""


def _load_documents():
    docs = []
    for filename in ["contact.txt", "faq.txt", "menu_details.txt", "schedule.txt"]:
        loader = TextLoader(os.path.join(KB_PATH, filename))
        docs.extend(loader.load())
    # One document for the whole CSV so row-by-row chunks don't crowd out the
    # "how to build" guide in retrieval (CSVLoader makes one chunk per row).
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
            "line suitable for searching a Cone N' Swirl ice cream truck menu (cones, "
            "bases, fillings, blends, extra blends, premium blends, Stick'ems, drizzles, "
            "how to build a cone cup or cone only order).\n"
            "- If the latest message is already a clear question, return it unchanged.\n"
            "- If the user is mid order-build (you asked for a step) and they reply with "
            "a short option (any capitalization, e.g. 'nutella', 'CINNAMON SUGAR', "
            "'vanilla', 'no filling', 'no premium blend', 'skip Stick em', "
            "'no extra blend', 'no drizzle', 'nothing on top'), "
            "rewrite into an explicit search phrase that includes that choice AND that "
            "they are building an order / the relevant menu step — do NOT turn it into an unrelated "
            "question about the word itself.\n"
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
