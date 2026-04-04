import os
from flask import Blueprint, request, jsonify,session
from dotenv import load_dotenv

from groq import Groq
from huggingface_hub import InferenceClient
from src.auth_helper import login_required

load_dotenv()

chatbot = Blueprint('chatbot', __name__)

HF_API_KEY = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# HF client
hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

# 🔥 MEMORY (important)
chat_history = []

# ---------------- HF ----------------
def ask_hf(messages):
    prompt = "You are a helpful AI assistant.\n\n"

    for msg in messages:
        prompt += f"{msg['role']}: {msg['content']}\n"

    prompt += "assistant:"

    result = hf_client.text_generation(
        prompt,
        model="google/flan-t5-large",
        max_new_tokens=300
    )

    return result


# ---------------- GROQ ----------------
def ask_groq(messages):
    client = Groq(api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4
    )

    return response.choices[0].message.content


# ---------------- ROUTE ----------------
@chatbot.route('/chat', methods=['POST'])
@login_required
def chat():
    global chat_history

    try:
        data = request.get_json()
        user_msg = data.get("message")

        if not user_msg:
            return jsonify({"reply": "⚠️ Empty message"})

        # add user message
        chat_history.append({"role": "user", "content": user_msg})

        # keep last 10 messages only (memory limit)
        chat_history = chat_history[-10:]

        print("💬 History:", chat_history)

        # try HF
        try:
            reply = ask_hf(chat_history)
        except:
            print("⚠️ HF failed → using Groq")
            reply = ask_groq(chat_history)

        # add assistant reply
        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"reply": reply})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"reply": "Error: " + str(e)})