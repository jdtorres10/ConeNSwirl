import uuid
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import build_chain

load_dotenv()

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*"))

print("Loading knowledge base and building RAG chain...")
rag_chain = build_chain()
print("Chatbot ready.")

_sessions: dict[str, list] = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


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

    try:
        result = rag_chain.invoke({"input": user_message, "chat_history": history})
        answer = result["answer"]

        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=answer))
        _sessions[session_id] = history

        return jsonify({"response": answer, "session_id": session_id, "success": True})

    except Exception as e:
        app.logger.error("Chat error: %s", e)
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
