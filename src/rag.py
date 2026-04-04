import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.auth_helper import login_required

from src.extensions import db
from src.models import PDFFile

from groq import Groq

load_dotenv()

rag = Blueprint('rag', __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vector_store = None

# ------------------ PDF PROCESSING ------------------
def process_pdf(filepath):
    print("📄 Processing PDF...")

    reader = PdfReader(filepath)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if text.strip() == "":
        raise ValueError("No text found in PDF")

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    print(f"✅ Total chunks: {len(chunks)}")

    # Using a reliable, free embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings)

    return vector_db


# ------------------ GROQ LLM (FIXED MODEL) ------------------
def ask_llm(context, question):
    print("🤖 Asking LLM with llama-3.3-70b-versatile...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are an AI assistant that answers strictly based on the provided context.
Do NOT use any outside knowledge.
If the answer cannot be found in the context below, respond with exactly: "Not in document"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (based only on context above):
"""

    try:
        response = client.chat.completions.create(
            # Using the current recommended model from Groq (as of April 2026)
            # llama-3.3-70b-versatile is stable and not deprecated
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise RAG assistant. Only answer from given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API error: {str(e)}")
        # Fallback in case of API issues
        return f"Error querying LLM: {str(e)}"


# ------------------ UPLOAD ENDPOINT ------------------
@rag.route('/upload', methods=['POST'])
@login_required
def upload_pdf():
    global vector_store

    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"})

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        print(f"📁 Saved: {filepath}")

        # Save to DB (optional, but keeps record)
        try:
            pdf = PDFFile(filename=filename, filepath=filepath, user_id=1)
            db.session.add(pdf)
            db.session.commit()
            print("✅ Saved to DB")
        except Exception as db_err:
            print(f"⚠️ DB save skipped (may not be initialized): {db_err}")
            # Continue anyway - DB is optional for RAG functionality

        # Process PDF and create vector store
        vector_store = process_pdf(filepath)

        return jsonify({"success": True})

    except Exception as e:
        print("❌ ERROR in upload:", str(e))
        return jsonify({"success": False, "error": str(e)})


# ------------------ ASK ENDPOINT ------------------
@rag.route('/ask', methods=['POST'])
@login_required
def ask_question():
    global vector_store

    try:
        if vector_store is None:
            return jsonify({"answer": "⚠️ No document uploaded. Please upload a PDF first."})

        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"answer": "❌ Invalid request: missing question"})

        question = data.get("question")
        print(f"❓ Question: {question}")

        # Retrieve top 3 most relevant chunks
        docs = vector_store.similarity_search(question, k=3)
        
        if not docs:
            return jsonify({"answer": "No relevant content found in the document."})

        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # Optional: Truncate context if too long (Groq has token limits)
        if len(context) > 12000:
            context = context[:12000] + "...(truncated)"

        answer = ask_llm(context, question)

        return jsonify({"answer": answer})

    except Exception as e:
        print("❌ ERROR in ask:", str(e))
        return jsonify({"answer": f"Error processing question: {str(e)}"})