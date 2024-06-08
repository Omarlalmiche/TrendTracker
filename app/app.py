import os
from flask import Flask, flash, render_template, request, session, url_for, redirect, send_file
from controller import fetch_and_analyze_posts
from model import User
from services import UserService
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

app = Flask(__name__)
app.secret_key = 'sqdsqdq'  # Assurez-vous de définir une clé secrète pour la gestion des sessions

user_service = UserService()

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/connect')
def connect():
    return render_template('connect.html')

@app.route('/register', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']
    user = User(email=email, password=password)
    user_service.add(user)
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    user = user_service.authenticate_user(email, password)
    if user:
        session['user'] = user.email
        return redirect(url_for('menu'))
    else:
        return "Login failed. Please check your credentials."

@app.route('/menu')
def menu():
    if 'user' in session:
        return render_template('menu.html')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        results = fetch_and_analyze_posts()
        return render_template('dashboard.html', results=results)
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/profile')
def profile():
    if 'user' in session:
        # Fetch user profile information from the database
        user = user_service.get_user_by_email(session['user'])
        return render_template('profile.html', user=user)
    return redirect(url_for('home'))

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user' in session:
        last_name = request.form['last_name']
        first_name = request.form['first_name']
        email = request.form['email']
        user_service.update_user(session['user'], last_name, first_name, email)
        return redirect(url_for('profile'))
    return redirect(url_for('home'))

@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user' in session:
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        if user_service.change_password(session['user'], current_password, new_password):
            flash('Mot de passe changé avec succès.')
        else:
            flash('Échec du changement de mot de passe. Veuillez vérifier votre mot de passe actuel.')
        return redirect(url_for('profile'))
    return redirect(url_for('home'))

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user' in session:
        confirm_delete = request.form['confirm_delete']
        if confirm_delete == "SUPPRIMER":
            user_service.delete_user(session['user'])
            session.pop('user', None)
            flash('Compte supprimé avec succès.')
            return redirect(url_for('home'))
        else:
            flash('Veuillez taper "SUPPRIMER" pour confirmer la suppression de votre compte.')
            return redirect(url_for('profile'))
    return redirect(url_for('home'))


@app.route('/fundme')
def fund_me():
    return render_template('fundme.html')


@app.route('/download_report', methods=['POST'])
def download_report():
    if 'user' in session:
        results = fetch_and_analyze_posts()
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        def draw_page_header(p, title):
            p.setFont("Helvetica-Bold", 16)
            p.drawString(100, height - 40, title)
            p.setFont("Helvetica", 12)

        def draw_text(p, text, x, y):
            text_object = p.beginText(x, y)
            lines = text.split('\n')
            for line in lines:
                text_object.textLine(line)
            p.drawText(text_object)
            return y - 12 * len(lines)

        def add_new_page(p, title):
            p.showPage()
            draw_page_header(p, title)

        title = "Report Summary"
        draw_page_header(p, title)
        y = height - 80

        for subreddit, result in results.items():
            if y < 100:
                add_new_page(p, title)
                y = height - 80
            
            p.drawString(100, y, f"Subreddit: {subreddit}")
            y -= 20
            p.drawString(100, y, f"Positive: {result['sentiments']['Positive']}")
            y -= 20
            p.drawString(100, y, f"Neutral: {result['sentiments']['Neutral']}")
            y -= 20
            p.drawString(100, y, f"Negative: {result['sentiments']['Negative']}")
            y -= 20
            
            p.drawString(100, y, "Trending Topics:")
            y -= 20
            for topic, keywords in result['topics'].items():
                p.drawString(120, y, f"{topic}: {', '.join(keywords)}")
                y -= 20
            
            y -= 40

            # Generate interpretations based on the current results
            interpretations = generate_interpretations(result)

            for interpretation in interpretations:
                if y < 100:
                    add_new_page(p, title)
                    y = height - 80
                y = draw_text(p, interpretation, 100, y - 20)

        p.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')


def generate_interpretations(result):
    interpretations = []
    
    sentiments = result['sentiments']
    positive = sentiments['Positive']
    neutral = sentiments['Neutral']
    negative = sentiments['Negative']

    # Sentiment Analysis Interpretation
    interpretations.append(f"Sentiment Analysis Interpretation:\n"
                           f"The sentiment analysis shows {positive} positive, {neutral} neutral, and {negative} negative posts. "
                           f"This indicates that the majority of discussions are {max(sentiments, key=sentiments.get).lower()}.")

    # Trending Topics Interpretation
    trending_topics = result['topics']
    if trending_topics:
        most_discussed_topic = max(trending_topics, key=lambda t: len(trending_topics[t]))
        interpretations.append(f"Trending Topics Interpretation:\n"
                               f"The most discussed topic is '{most_discussed_topic}' with key terms "
                               f"{', '.join(trending_topics[most_discussed_topic])}. This suggests that users are highly engaged with this topic.")

    # Anomalies Interpretation
    anomalies = result['anomalies']
    if anomalies:
        interpretations.append(f"Anomalies Interpretation:\n"
                               f"Detected {len(anomalies)} anomalies, indicating high engagement on certain posts.")

    # Regression Plot Interpretation
    train_score = result.get('train_score')
    test_score = result.get('test_score')
    if train_score and test_score:
        interpretations.append(f"Regression Plot Interpretation:\n"
                               f"The regression model shows a training score of {train_score:.2f} and a test score of {test_score:.2f}. "
                               f"This indicates {'good' if test_score > 0 else 'poor'} generalization of the model to unseen data.")

    return interpretations



if __name__ == "__main__":
    app.run(debug=True)
