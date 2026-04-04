import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from src.extensions import db
from src.models import PDFFile
from src.auth_helper import login_required


import requests
from groq import Groq
from huggingface_hub import InferenceClient

load_dotenv()

quiz = Blueprint('quiz', __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- HF CLIENT ----------------
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY,
)

# ---------------- PDF → TEXT ----------------
def extract_text(filepath):
    reader = PdfReader(filepath)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    if not text.strip():
        raise ValueError("No text found in PDF")

    return text[:3000]


# ---------------- HF QUIZ ----------------
def generate_quiz_hf(text):
    prompt = f"""
Generate 10 multiple choice questions from the text.

Each question must have:
- Question
- 4 options (A, B, C, D)
- Correct answer

Format strictly:

Q1: ...
A) ...
B) ...
C) ...
D) ...
Answer: ...

Text:
{text}
"""

    result = hf_client.text_generation(
        prompt,
        model="google/flan-t5-large",   # better for instruction tasks
        max_new_tokens=512
    )

    return result


# ---------------- GROQ FALLBACK ----------------
def generate_quiz_groq(text):
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
Generate 10 multiple choice questions from the text.

Each question must have:
- Question
- 4 options (A, B, C, D)
- Correct answer

Format strictly:

Q1: ...
A) ...
B) ...
C) ...
D) ...
Answer: ...

Text:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a quiz generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content


# ---------------- MAIN ROUTE ----------------
@quiz.route('/generate-quiz', methods=['POST'])
@login_required
def generate_quiz():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"})

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        print("📄 Extracting text...")
        text = extract_text(filepath)

        print("🧠 Trying Hugging Face...")
        try:
            quiz_text = generate_quiz_hf(text)
        except Exception as e:
            print("⚠️ HF failed → using Groq")
            quiz_text = generate_quiz_groq(text)
        #save into database
        try:
            pdf = PDFFile(filename=filename, filepath=filepath, user_id=1)
            db.session.add(pdf)
            db.session.commit()
            print("✅ Saved to DB")
        except Exception as db_err:
            print(f"⚠️ DB save skipped (may not be initialized): {db_err}")

        return jsonify({
            "success": True,
            "quiz": quiz_text
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"success": False, "error": str(e)})
    