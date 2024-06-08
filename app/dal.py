import mysql.connector

class RedditDAL:
    def __init__(self, reddit_instance):
        self.reddit = reddit_instance

    def fetch_posts(self, subreddit_name, limit=100):
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        for submission in subreddit.hot(limit=limit):
            post_data = {
                'title': submission.title,
                'score': submission.score,
                'id': submission.id,
                'url': submission.url,
                'num_comments': submission.num_comments,
                'created': submission.created,
                'body': submission.selftext,
                'upvotes': submission.ups,
                'downvotes': submission.downs
            }
            posts.append(post_data)
        return posts

class DatabaseDAL:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="scrapping",
            charset='utf8'
        )
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id VARCHAR(255) PRIMARY KEY,
                title TEXT,
                score INT,
                url TEXT,
                num_comments INT,
                created DATETIME,
                body TEXT,
                sentiment VARCHAR(255),
                upvotes INT,
                downvotes INT
            )
        ''')
        self.conn.commit()

    def insert_post(self, post, sentiment):
        sql = '''
            INSERT INTO posts (id, title, score, url, num_comments, created, body, sentiment, upvotes, downvotes)
            VALUES (%s, %s, %s, %s, %s, FROM_UNIXTIME(%s), %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            title = VALUES(title), score = VALUES(score), url = VALUES(url), 
            num_comments = VALUES(num_comments), created = VALUES(created), 
            body = VALUES(body), sentiment = VALUES(sentiment), 
            upvotes = VALUES(upvotes), downvotes = VALUES(downvotes)
        '''
        values = (
            post['id'], post['title'], post['score'], post['url'],
            post['num_comments'], post['created'], post['body'], sentiment,
            post['upvotes'], post['downvotes']
        )
        self.cursor.execute(sql, values)
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()
