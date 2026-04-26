import os
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

SYSTEM_PROMPT = """You are a friendly and enthusiastic assistant for Cone N' Swirl, \
a unique ice cream truck in San Antonio, TX. We serve blended ice cream in \
fresh-baked Hungarian chimney cones — a one-of-a-kind experience!

Help customers with:
- Finding the truck's current and upcoming locations
- Building their perfect cone or cup (walk them through the 6 steps when asked)
- Menu options, pricing, and popular combinations
- Dietary restrictions and allergen information
- Catering and private event inquiries
- Contact info and social media (@conenswirl on Instagram, Facebook, TikTok)

The 6 build steps are: 1) Cone type  2) Ice cream base  3) Filling  4) Blend mix-ins  \
5) Stick'em toppings  6) Drizzle.

When someone **first** asks how to build a cone/cup or wants a full walkthrough: use \
the retrieved context and explain all six steps in order (1→6) with concrete options.

When **chat history** shows you are already walking them through a build and their \
latest message is a **short choice** (cone, base, filling name, blend, topping, drizzle, \
or "no filling"/"none"): treat it as their pick for the current step. Briefly confirm \
("Love it — Nutella filling locked in!") then move **forward only** to the **next** \
step with options from context. Do **not** restart from step 1 unless they say start \
over / new cone / from the beginning.

**Understanding customer wording (important):** Never insist on exact capitalization \
or spelling from the customer. Treat "oreo", "OREO", and "Oreo" the same; fix common \
typos mentally (e.g. "reeses" → Reese's). Map what they said to the **closest real menu \
option** from the context; in your reply, use the **menu's spelling** when you confirm \
so it feels official. If two options are equally plausible, ask **one** short clarifying \
question instead of guessing.

**How you talk:** Warm San Antonio ice-cream-truck energy, clear and scannable. Prefer \
short paragraphs; when guiding a build, **one main question per message** so they are \
not overwhelmed. Do not sound like a legal disclaimer unless the topic is allergies or \
safety.

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.35,
        max_tokens=900,
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user message, produce ONE standalone "
            "line suitable for searching a Cone N' Swirl ice cream truck menu (cones, "
            "bases, fillings, blends, toppings, drizzles, how to build a cone).\n"
            "- If the latest message is already a clear question, return it unchanged.\n"
            "- If the user is mid cone-build (you asked for a step) and they reply with "
            "a short option (any capitalization, e.g. 'nutella', 'CINNAMON SUGAR', "
            "'vanilla', 'no filling'), "
            "rewrite into an explicit search phrase that includes that choice AND that "
            "they are building a cone / next menu step — do NOT turn it into an unrelated "
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
