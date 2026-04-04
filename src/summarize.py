import os
import requests
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq
from src.extensions import db
from src.models import PDFFile
from src.auth_helper import login_required

load_dotenv()

summarize = Blueprint('summarize', __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# HF NEW ROUTER API
HF_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

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

    return text[:3000]  # limit


# ---------------- HF SUMMARIZER ----------------
def summarize_hf(text):
    response = requests.post(
        HF_URL,
        headers=headers,
        json={"inputs": text}
    )

    if response.status_code == 200:
        data = response.json()
        return data[0]['summary_text']
    else:
        raise Exception("HF Failed")


# ---------------- GROQ FALLBACK ----------------
def summarize_groq(text):
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
Summarize the following text in simple and clear points:

{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# ---------------- MAIN ROUTE ----------------
@summarize.route('/summarize', methods=['POST'])
@login_required
def summarize_pdf():
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
            summary = summarize_hf(text)
        except:
            print("⚠️ HF failed → Using Groq...")
            summary = summarize_groq(text)

        #save into database
        try:
            pdf = PDFFile(filename=filename, filepath=filepath, user_id=1)
            db.session.add(pdf)
            db.session.commit()
            print("✅ Saved to DB")
        except Exception as db_err:
            print(f"⚠️ DB save skipped (may not be initialized): {db_err}")
            # Continue anyway - DB is optional for RAG functionality


        return jsonify({
            "success": True,
            "summary": summary
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"success": False, "error": str(e)})