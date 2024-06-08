import mysql.connector
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw
from dal import DatabaseDAL, RedditDAL
from model import User

class RedditService:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id='tDem2PpJUX0IgE_eq9Z1ww',
            client_secret='c6QBvgdJVS_ESBzPyFChYEvekQ1LeA',
            user_agent='OkInterest7517'
        )
        self.dal = RedditDAL(self.reddit)

    def get_posts(self, subreddit_name):
        return self.dal.fetch_posts(subreddit_name)

    def get_hot_posts(self):
        return self.reddit.subreddit('all').hot()

    def get_top_trending_subreddits(self, limit=2):
        trending_subreddits = [sub.display_name for sub in self.reddit.subreddits.popular(limit=limit)]
        return trending_subreddits

class SentimentAnalysisService:
    def analyze_sentiment(self, text):
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        if sentiment_scores['compound'] >= 0.05:
            return 'Positive'
        elif sentiment_scores['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

class DatabaseService:
    def __init__(self):
        self.dal = DatabaseDAL()

    def save_post(self, post, sentiment):
        self.dal.insert_post(post, sentiment)

class UserService:
    def __init__(self):
        self.database = mysql.connector.connect(
            host="localhost", user="root", password="", database="scrapping", charset='utf8'
        )
        self.connect = self.database.cursor()

    def add(self, user: User) -> None:
        try:
            sql = "INSERT INTO t_user (email, password) VALUES (%s, %s)"
            val = (user.email, user.password)
            self.connect.execute(sql, val)
            self.database.commit()
        except Exception as e:
            print("Erreur lors de l'ajout de l'utilisateur :", e)

    def authenticate_user(self, email: str, password: str) -> User:
        sql = "SELECT * FROM t_user WHERE email = %s AND password = %s"
        val = (email, password)
        self.connect.execute(sql, val)
        user_data = self.connect.fetchone()
        if user_data:
            return User(email=user_data[0], password=user_data[1])
        else:
            return None

    def get_user_by_email(self, email: str) -> User:
        sql = "SELECT * FROM t_user WHERE email = %s"
        val = (email,)
        self.connect.execute(sql, val)
        user_data = self.connect.fetchone()
        if user_data:
            return User(email=user_data[0], password=user_data[1])
        else:
            return None

    def update_user(self, email: str, last_name: str, first_name: str, new_email: str) -> None:
        sql = "UPDATE t_user SET last_name = %s, first_name = %s, email = %s WHERE email = %s"
        val = (last_name, first_name, new_email, email)
        self.connect.execute(sql, val)
        self.database.commit()

    def change_password(self, email: str, current_password: str, new_password: str) -> bool:
        user = self.authenticate_user(email, current_password)
        if user:
            sql = "UPDATE t_user SET password = %s WHERE email = %s"
            val = (new_password, email)
            self.connect.execute(sql, val)
            self.database.commit()
            return True
        return False

    def delete_user(self, email: str) -> None:
        sql = "DELETE FROM t_user WHERE email = %s"
        val = (email,)
        self.connect.execute(sql, val)
        self.database.commit()
