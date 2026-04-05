from flask import Flask, render_template,redirect, session,url_for
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os
import mysql.connector
from src.auth import auth
from src.extensions import db    
from src.rag import rag
from src.pdf_audio import pdf_audio
from src.summarize import summarize
from src.QuizGenerator import quiz
from src.smartnote import smartnote
from src.chatbot import chatbot
from src.auth_helper import login_required




# Load env
load_dotenv()

app = Flask(__name__,template_folder='templates')
app.register_blueprint(auth)
app.register_blueprint(rag)
app.register_blueprint(pdf_audio)
app.register_blueprint(summarize)
app.register_blueprint(quiz)
app.register_blueprint(smartnote)
app.register_blueprint(chatbot)

# ==============================
# ENV VARIABLES
# ==============================
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

# ==============================
# STEP 1: CREATE DATABASE IF NOT EXISTS
# ==============================
def create_database():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=os.getenv("DB_PORT", 3306)
        )

        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        print("✅ Database checked/created successfully!")

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"❌ Error creating database: {e}")

# Call it before SQLAlchemy
create_database()



# ==============================
# STEP 2: CONNECT SQLALCHEMY
# ==============================

app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False




# INIT DB
db.init_app(app)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test-db")
def test_db():
    try:
        with app.app_context():
            db.create_all()
        return "✅ DB Connected!"
    except Exception as e:
        return str(e)
@login_required
@app.route("/dashboard")
@login_required
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("dashboard.html")


@app.route('/rag')
@login_required
def rag():
    return render_template('rag.html')

@app.route('/pdf-audio')
@login_required
def pdf_audio_page():
    return render_template('pdf_audio.html')

@app.route('/summarizer')
@login_required
def summarize_page():   
    return render_template('summarize.html')

@app.route('/quiz')
@login_required
def quiz_page():
    return render_template('QuizGenerator.html')

@app.route('/notes')
@login_required
def notes_page():
    return render_template('smartnote.html')

@app.route('/chatbot')
@login_required
def chatbot_page():
    return render_template('chatbot.html')    

@app.route("/logout")
@login_required
def logout():
    session.pop('user_id', None)
    return redirect(url_for('auth.login'))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)