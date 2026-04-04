from flask import Blueprint, render_template, request, redirect, session,url_for, flash, jsonify
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

from src.extensions import db
from src.models import User

auth = Blueprint("auth", __name__)

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper



# ---------------- REGISTER ----------------
@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("User already exists!", "error")
            return redirect(url_for('auth.register'))

        hashed_password = generate_password_hash(password)

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('auth.login'))

    return render_template('register.html')



# ---------------- LOGIN ----------------
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id   # ✅ IMPORTANT LINE
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password!", "error")
            return redirect(url_for('auth.login'))

    return render_template('login.html')

# ---------------- LOGOUT ----------------
