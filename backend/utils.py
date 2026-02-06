"""
utils.py

Utility functions and RAG (Retrieval-Augmented Generation) pipeline setup
for a question-answering system using:

- Ollama LLM
- HuggingFace sentence embeddings
- Chroma vector database
- LangChain RetrievalQA chain

This module:
1. Loads environment-based configuration
2. Initializes embeddings + vector DB retriever
3. Builds a RetrievalQA pipeline
4. Provides helper functions for:
   - Extracting context from retrieved documents
   - Generating related follow-up questions
   - Running the full RAG pipeline
"""

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import os
import time


# ==============================
# Configuration (via ENV vars)
# ==============================

# LLM model name used by Ollama
# You can override by setting MODEL in environment
model = os.environ.get("MODEL", "mistral-openorca")

# Sentence embedding model for vector search
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")

# Directory where Chroma vector DB is stored
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")

# Number of top chunks to retrieve per query
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 1))


# ==============================
# Embeddings + Vector Store
# ==============================

# Create embedding model (converts text → vector)
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Load persistent Chroma vector database
db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# Create retriever interface (used by RAG chain)
retriever = db.as_retriever(
    search_kwargs={"k": target_source_chunks}
)

# Callback list (can be used later for streaming / logging)
callbacks = []


# ==============================
# LLM + RAG Chain Setup
# ==============================

# Initialize Ollama LLM
llm = Ollama(model=model, callbacks=callbacks)

# RetrievalQA chain:
# - "stuff" chain type means all retrieved docs are stuffed into one prompt
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# ==========================================================
# Helper Functions
# ==========================================================

def generate_related_questions(parent_question, related_context):
    """
    Generates follow-up questions related to the original user query
    using the retrieved document context.

    Args:
        parent_question (str): The original question asked by the user.
        related_context (str): Combined text extracted from retrieved documents.

    Returns:
        str: LLM-generated numbered list of related questions.
    """
    query = f"""
    What are some related questions that can be asked for the parent question 
    '{parent_question}' based on the following related context:

    {related_context}

    Only include the related questions in a numbered format.
    """

    # Direct LLM call (no retrieval here, just generation)
    res = llm(query)
    return res


def extract_related_context(docs):
    """
    Converts retrieved documents into a formatted context string
    that includes source information.

    Args:
        docs (list): List of LangChain Document objects.

    Returns:
        str: Formatted context string for prompt injection.
    """
    related_context = ""

    for document in docs:
        # Each document contains:
        # - metadata (like source)
        # - page_content (actual text)
        source = document.metadata.get('source', 'Unknown Source')
        related_context += f"\n> {source}:\n{document.page_content}\n"

    return related_context


def generate_rag_response(query):
    """
    Runs the full RAG pipeline:
    1. Retrieves relevant documents
    2. Generates an answer using LLM + context
    3. Produces related follow-up questions

    Args:
        query (str): User question.

    Returns:
        tuple:
            - answer (str): RAG-generated answer
            - related_questions (str): LLM-generated related questions
    """

    # ---- Step 1: Run RetrievalQA ----
    start_time = time.time()
    res = qa(query)
    end_time = time.time()

    print(f"took {end_time - start_time:.2f} seconds to complete RAG based QA")

    answer = res['result']
    docs = res['source_documents']

    # ---- Step 2: Generate Related Questions ----
    start_time = time.time()

    related_context = extract_related_context(docs)
    related_questions = generate_related_questions(query, related_context)

    end_time = time.time()
    print(f"took {end_time - start_time:.2f} seconds to generate related_questions")

    return answer, related_questions
