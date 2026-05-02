import json
import uuid
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import summarize_build_order
from build_menu import BUILD_MENU, validate_and_normalize_order

load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*"))

_sessions: dict[str, list] = {}

_MAX_STORED_MESSAGES = 32


def _tail_messages(messages: list, limit: int) -> list:
    if len(messages) <= limit:
        return messages
    return messages[-limit:]


@app.route("/", methods=["GET"])
def root():
    """API root — site is on GitHub Pages; this service exposes /health, /build-menu, /build-complete."""
    return jsonify({
        "service": "Cone N' Swirl order API",
        "health": "/health",
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
