import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from groq import Groq
from huggingface_hub import InferenceClient
from src.auth_helper import login_required

load_dotenv()

smartnote = Blueprint('smartnote', __name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# HF Client
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
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
        raise ValueError("No text found")

    return text[:3000]


# ---------------- HF SMART NOTES ----------------
def generate_notes_hf(text):
    prompt = f"""
Create smart study notes from the text.

Requirements:
- Use simple English
- Use bullet points
- Easy to understand
- Then translate into Bengali and Hindi

Format:

--- English Notes ---
...

--- Bengali Notes ---
...

--- Hindi Notes ---
...

Text:
{text}
"""

    result = hf_client.text_generation(
        prompt,
        model="google/flan-t5-large",
        max_new_tokens=700
    )

    return result


# ---------------- GROQ FALLBACK ----------------
def generate_notes_groq(text):
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
Create smart study notes from the text.

Requirements:
- Use simple English
- Use bullet points
- Easy to understand
- Then translate into Bengali and Hindi

Format:

--- English Notes ---
...

--- Bengali Notes ---
...

--- Hindi Notes ---
...

Text:
{text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a smart study assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# ---------------- MAIN ROUTE ----------------
@smartnote.route('/smart-notes', methods=['POST'])
@login_required
def smart_notes():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file"})

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
            notes = generate_notes_hf(text)
        except:
            print("⚠️ HF failed → using Groq")
            notes = generate_notes_groq(text)

        return jsonify({
            "success": True,
            "notes": notes
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"success": False, "error": str(e)})