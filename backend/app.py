import json
import uuid
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import (
    build_chain,
    coerce_chain_answer_to_text,
    sanitize_chat_history_messages,
    summarize_build_order,
)
from build_menu import BUILD_MENU, validate_and_normalize_order

load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*"))

print("Loading knowledge base and building RAG chain...")
rag_chain = build_chain()
print("Chatbot ready.")

_sessions: dict[str, list] = {}

# Long threads + RAG chunks can exceed model context or slow the request; keep recent turns only.
_MAX_STORED_MESSAGES = 32
_MAX_HISTORY_FOR_MODEL = 24


def _tail_messages(messages: list, limit: int) -> list:
    if len(messages) <= limit:
        return messages
    return messages[-limit:]


def _rag_answer_raw(result: object) -> object:
    """Normalize chain output shape across LangChain versions."""
    if not isinstance(result, dict):
        return None
    if result.get("answer") is not None:
        return result["answer"]
    for key in ("output", "response", "text"):
        if result.get(key) is not None:
            return result[key]
    return None


@app.route("/", methods=["GET"])
def root():
    """Homepage is not the chat UI — the site is on GitHub Pages; this API serves /health and POST /chat."""
    return jsonify({
        "service": "Cone N' Swirl chatbot API",
        "health": "/health",
        "chat": "POST /chat (JSON body: message, optional session_id)",
        "build_menu": "GET /build-menu",
        "build_complete": "POST /build-complete (JSON: order, optional session_id)",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/build-menu", methods=["GET"])
def build_menu():
    return jsonify(BUILD_MENU)


@app.route("/build-complete", methods=["POST"])
def build_complete():
    data = request.get_json(silent=True)
    if not data or "order" not in data:
        return jsonify({"error": "order is required"}), 400

    normalized, err = validate_and_normalize_order(data["order"])
    if err:
        return jsonify({"error": err}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    history = _sessions.get(session_id, [])

    try:
        summary = summarize_build_order(normalized)
    except Exception:
        app.logger.exception("build-complete summary error")
        return jsonify({
            "error": "Could not generate your order recap. Try again.",
            "session_id": session_id,
            "success": False,
        }), 500

    order_note = (
        "[Customer used the button builder. Structured order: "
        + json.dumps(normalized)
        + "]"
    )
    history.append(HumanMessage(content=order_note))
    history.append(AIMessage(content=summary))
    if len(history) > _MAX_STORED_MESSAGES:
        history[:] = _tail_messages(history, _MAX_STORED_MESSAGES)
    _sessions[session_id] = history

    return jsonify({
        "success": True,
        "session_id": session_id,
        "order": normalized,
        "summary": summary,
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "message is required"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "message cannot be empty"}), 400

    session_id = data.get("session_id") or str(uuid.uuid4())
    history = _sessions.get(session_id, [])
    # Repair any legacy messages (nested AIMessage / block content) before this turn.
    history[:] = sanitize_chat_history_messages(list(history))

    try:
        history_for_model = _tail_messages(history, _MAX_HISTORY_FOR_MODEL)
        result = rag_chain.invoke(
            {"input": user_message, "chat_history": history_for_model}
        )
        answer = coerce_chain_answer_to_text(_rag_answer_raw(result))
        if not answer:
            raise ValueError("empty model answer")

        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=answer))
        if len(history) > _MAX_STORED_MESSAGES:
            history[:] = _tail_messages(history, _MAX_STORED_MESSAGES)
        _sessions[session_id] = history

        return jsonify({"response": answer, "session_id": session_id, "success": True})

    except Exception:
        app.logger.exception("Chat error")
        return jsonify({
            "response": (
                "Oops, something went wrong on my end! Try again, or follow "
                "@conenswirl on Instagram for the latest info."
            ),
            "session_id": session_id,
            "success": False,
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
