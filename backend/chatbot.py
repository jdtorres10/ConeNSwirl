import os
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader, CSVLoader

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

Keep responses concise, warm, and conversational. If you're unsure about something, \
direct customers to follow @conenswirl on Instagram or call/text (956) 324-8733.

{context}"""


def _load_documents():
    docs = []
    for filename in ["contact.txt", "faq.txt", "menu_details.txt", "schedule.txt"]:
        loader = TextLoader(os.path.join(KB_PATH, filename))
        docs.extend(loader.load())
    csv_loader = CSVLoader(os.path.join(KB_PATH, "Menu.csv"))
    docs.extend(csv_loader.load())
    return docs


def build_chain():
    docs = _load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given the chat history and the latest user question, rewrite it as a "
            "standalone question that can be understood without the history. "
            "Do NOT answer it — only rewrite if needed, otherwise return it as-is."
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
