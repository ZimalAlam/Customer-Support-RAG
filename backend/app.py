"""
app.py

Flask API server that exposes a RAG-based chatbot endpoint.

This service:
1. Receives user questions via HTTP POST
2. Sends the query to the RAG pipeline
3. Returns the generated answer + related follow-up questions

Acts as the bridge between frontend UI and the ML backend.
"""

from utils import generate_rag_response
from flask import Flask, request, jsonify
from flask_cors import CORS
import time


# =====================================================
# Flask App Initialization
# =====================================================

app = Flask(__name__)

# Enable Cross-Origin Resource Sharing
# Allows frontend apps (React, Streamlit, etc.) to call this API
CORS(app)


# Ollama LLM endpoint (used indirectly inside utils.py)
LLM_API_URL = "http://localhost:11434/api/generate"


# =====================================================
# Chat Endpoint
# =====================================================

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for RAG-based Q&A.

    Expected JSON payload:
    {
        "user_message": "your question here"
    }

    Returns:
    {
        "response": "Answer text + related questions"
    }
    """

    # Extract user message from request body
    user_message = request.json.get('user_message')

    # Validate input
    if user_message is None:
        return jsonify({'error': 'Missing parameter user_message'}), 400

    # ---- Call RAG Pipeline ----
    start_time = time.time()

    answer, related_questions = generate_rag_response(user_message)

    end_time = time.time()
    print(f"took {end_time - start_time:.2f} seconds total")

    # Clean output formatting
    answer = answer.strip()
    related_questions = related_questions.strip()

    # Combine answer and related questions
    response = f"{answer}\n\nRelated questions:\n{related_questions}"

    return jsonify({'response': response})


# =====================================================
# Run Server
# =====================================================

if __name__ == '__main__':
    # Default: runs on http://127.0.0.1:5000
    # For production, use gunicorn/uvicorn instead
    app.run()
