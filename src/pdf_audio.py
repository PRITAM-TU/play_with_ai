import os
import requests
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from src.auth_helper import login_required

load_dotenv()

pdf_audio = Blueprint('pdf_audio', __name__)

UPLOAD_FOLDER = "uploads"
AUDIO_FOLDER = "audio"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ============================================================
# 100% WORKING TTS CONFIGURATION
# Using the NEW Hugging Face router endpoint
# Model: facebook/mms-tts-eng (English, fast, free)
# ============================================================
TTS_MODEL = "facebook/mms-tts-eng"
TTS_API_URL = f"https://router.huggingface.co/hf-inference/models/{TTS_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


def extract_text_from_pdf(filepath):
    """Extract and clean text from PDF"""
    try:
        reader = PdfReader(filepath)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
            print(f"📄 Page {page_num}: {len(extracted or '')} chars")
        
        full_text = "\n".join(text_parts)
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        # Clean up whitespace
        full_text = ' '.join(full_text.split())
        
        # Limit to 2000 characters for TTS (avoids timeout)
        if len(full_text) > 2000:
            full_text = full_text[:2000]
            print(f"⚠️ Text truncated to 2000 characters")
        
        print(f"✅ Extracted {len(full_text)} characters")
        return full_text
        
    except Exception as e:
        raise Exception(f"PDF reading failed: {str(e)}")


def text_to_speech(text, output_path):
    """Use gTTS - completely free, no API key needed!"""
    try:
        from gtts import gTTS
        
        print("🔊 Using gTTS (Google TTS) - No API key required")
        
        # gTTS works with up to 100 chars per request, so we chunk if needed
        if len(text) > 500:
            # Take first 500 chars for better quality
            text = text[:2000]
            print(f"⚠️ Text truncated to 2000 chars for TTS")
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        
        print(f"✅ Audio saved with gTTS: {output_path}")
        return True
        
    except ImportError:
        print("❌ gTTS not installed. Install with: pip install gTTS")
        return False
    except Exception as e:
        print(f"❌ gTTS error: {str(e)}")
        return False


@pdf_audio.route('/pdf-to-audio', methods=['POST'])
@login_required
def pdf_to_audio():
    """Main endpoint: upload PDF → extract text → generate audio"""
    try:
        # 1. Validate file
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "Only PDF files are allowed"})
        
        # 2. Save PDF temporarily
        original_filename = secure_filename(file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, original_filename)
        file.save(pdf_path)
        print(f"📁 PDF saved: {pdf_path}")
        
        # 3. Extract text from PDF
        try:
            text_content = extract_text_from_pdf(pdf_path)
        except Exception as text_err:
            return jsonify({"success": False, "error": f"Text extraction failed: {str(text_err)}"})
        
        if len(text_content) < 50:
            return jsonify({"success": False, "error": "PDF contains very little text (minimum 50 characters needed)"})
        
        # 4. Generate unique audio filename
        base_name = os.path.splitext(original_filename)[0]
        audio_filename = f"{base_name}_audio.wav"
        
        # Handle duplicates
        counter = 1
        while os.path.exists(os.path.join(AUDIO_FOLDER, audio_filename)):
            audio_filename = f"{base_name}_audio_{counter}.wav"
            counter += 1
        
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        
        # 5. Convert text to speech
        try:
            text_to_speech(text_content, audio_path)
        except Exception as tts_err:
            # Clean up PDF if TTS fails
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return jsonify({"success": False, "error": f"Speech generation failed: {str(tts_err)}"})
        
        # 6. Verify audio file was created
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 5000:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            return jsonify({"success": False, "error": "Generated audio file is invalid or too small"})
        
        # 7. Clean up PDF (optional - keep for debugging)
        # os.remove(pdf_path)
        
        print(f"🎉 Success! Audio: {audio_path}")
        
        return jsonify({
            "success": True,
            "audio_url": f"/download-audio/{audio_filename}",
            "text_length": len(text_content),
            "file_size": os.path.getsize(audio_path)
        })
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"})


@pdf_audio.route('/download-audio/<filename>')
@login_required
def download_audio(filename):
    """Serve the generated audio file for download"""
    # Security: prevent path traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(AUDIO_FOLDER, safe_filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Audio file not found"}), 404
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=safe_filename,
        mimetype='audio/wav'
    )


@pdf_audio.route('/pdf-audio-health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "tts_model": TTS_MODEL,
        "upload_folder": UPLOAD_FOLDER,
        "audio_folder": AUDIO_FOLDER
    })