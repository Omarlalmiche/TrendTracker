import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from services import RedditService, SentimentAnalysisService, DatabaseService

#Cette fonction prend en entrée un texte, le met en minuscules, enlève les mots d'arrêt, et lemmatise les mots restants
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.lower().split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    return filtered_words

def perform_topic_modeling(posts):
    texts = [post['title'] for post in posts]
    preprocessed_texts = [' '.join(preprocess_text(text)) for text in texts]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(preprocessed_texts)
    num_topics = 1
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', random_state=42)
    lda.fit(X)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    return topics

def detect_anomalies(posts):
    comments = [post['num_comments'] for post in posts]
    mean_comments = np.mean(comments)
    std_comments = np.std(comments)
    anomalies = [post for post in posts if post['num_comments'] > mean_comments + 3 * std_comments]
    return anomalies

def plot_linear_regression(X_train, y_train, X_test, y_test, model):
    X_train_flat = np.array(X_train).flatten()
    X_test_flat = np.array(X_test).flatten()
    y_train_flat = np.array(y_train).flatten()
    y_test_flat = np.array(y_test).flatten()

    trace_train = go.Scatter(
        x=X_train_flat,
        y=y_train_flat,
        mode='markers',
        name='Training Data'
    )
    trace_test = go.Scatter(
        x=X_test_flat,
        y=y_test_flat,
        mode='markers',
        name='Test Data'
    )
    trace_line = go.Scatter(
        x=np.concatenate([X_train_flat, X_test_flat]),
        y=model.predict(np.concatenate([X_train, X_test])).flatten(),
        mode='lines',
        name='Regression Line'
    )
    layout = go.Layout(
        title='Linear Regression: Post Score vs. Number of Comments',
        xaxis=dict(title='Post Score', range=[X_train_flat.min() - 1, X_train_flat.max() + 1]),
        yaxis=dict(title='Number of Comments', range=[y_train_flat.min() - 1, y_train_flat.max() + 1])
    )
    fig = go.Figure(data=[trace_train, trace_test, trace_line], layout=layout)
    return fig.to_json()


def train_linear_regression(posts):
    X = [[post['score']] for post in posts]
    y = [[post['num_comments']] for post in posts]

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Les données de caractéristiques ou les cibles sont vides.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)
    mean_cv_score = np.mean(cv_scores)

    regression_plot_json = plot_linear_regression(X_train, y_train, X_test, y_test, model)

    return model, train_score, test_score, mean_cv_score, regression_plot_json

def generate_trending_topics_plotly(trending_topics):
    preprocessed_topics = []
    for topic in trending_topics:
        preprocessed_topics.extend(preprocess_text(topic))
    
    top_trending_topics = Counter(preprocessed_topics).most_common(10)
    if not top_trending_topics:
        return None

    topics, counts = zip(*top_trending_topics)

    df = pd.DataFrame({
        'Topics': topics,
        'Counts': counts
    })

    fig = px.bar(df, x='Topics', y='Counts', title='Top Trending Topics in Subreddit')
    fig.update_layout(xaxis_title='Topic', yaxis_title='Frequency', xaxis_tickangle=-45)

    return fig.to_json()

def perform_pca(posts, n_components=2):
    df = pd.DataFrame(posts)
    features = ['score', 'num_comments']
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)
    
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    final_df = pd.concat([principal_df, df[['title']]], axis=1)
    
    return final_df, pca.explained_variance_ratio_.tolist()

def plot_pca(principal_df):
    fig = px.scatter(principal_df, x='PC1', y='PC2', text='title', title='Projection des données sur les deux premières composantes principales')
    fig.update_layout(showlegend=False, xaxis=dict(range=[principal_df['PC1'].min() - 1, principal_df['PC1'].max() + 1]), yaxis=dict(range=[principal_df['PC2'].min() - 1, principal_df['PC2'].max() + 1]))
    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
    return fig.to_json()

def generate_time_series_plot(posts, frequency='D'):
    df = pd.DataFrame(posts)
    df['created'] = pd.to_datetime(df['created'], unit='s')
    df.set_index('created', inplace=True)
    resampled_df = df.resample(frequency).size()
    
    if frequency == 'D':
        title = 'Number of Posts Per Day'
    
    fig = px.line(resampled_df, title=title, labels={'value': 'Number of Posts', 'created': 'Date'})
    return fig.to_json()

def get_post_scores(posts):
    return [post['score'] for post in posts]
def plot_post_scores_histogram(post_scores):
    histogram_data = [
        go.Histogram(
            x=post_scores,
            marker=dict(color='rgba(0, 0, 255, 0.7)'),
        )
    ]
    layout = go.Layout(
        title='Post Scores Distribution',
        xaxis=dict(title='Post Score'),
        yaxis=dict(title='Count')
    )
    fig = go.Figure(data=histogram_data, layout=layout)
    return fig.to_json()

def generate_correlation_heatmap(posts):
    # Create a DataFrame from posts
    df = pd.DataFrame(posts)
    
    # Select relevant features for the correlation matrix
    features = ['score', 'num_comments']
    correlation_matrix = df[features].corr()
    
    # Create a heatmap using Plotly
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                    title='Correlation Matrix Heatmap')
    fig.update_layout(
        xaxis=dict(title='Features', tickmode='array', tickvals=[0, 1], ticktext=features),
        yaxis=dict(title='Features', tickmode='array', tickvals=[0, 1], ticktext=features),
        autosize=True,
        width=500,  # Adjust width to fit your needs
        height=500  # Adjust height to fit your needs
    )
    
    # Add annotations to improve readability
    fig.update_traces(texttemplate="%{z:.2f}", textfont_size=12)
    
    return fig.to_json()

def fetch_and_analyze_posts():
    reddit_service = RedditService()
    sentiment_analysis_service = SentimentAnalysisService()
    db_service = DatabaseService()

    subreddit_name = reddit_service.get_top_trending_subreddits(limit=1)[0]

    posts = reddit_service.get_posts(subreddit_name)

    sentiments = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    trending_topics = []
    for post in posts:
        sentiment = sentiment_analysis_service.analyze_sentiment(post['title'])
        sentiments[sentiment] += 1
        db_service.save_post(post, sentiment)
        trending_topics.append(post['title'])

    topics = perform_topic_modeling(posts)
    anomalies = detect_anomalies(posts)

    if len(posts) > 1:
        model, train_score, test_score, mean_cv_score, regression_plot_json = train_linear_regression(posts)
    else:
        model, train_score, test_score, mean_cv_score, regression_plot_json = (None, 0, 0, 0, None, None)

    sentiment_plot_json = generate_trending_topics_plotly(sentiments)
    trending_topics_plotly = generate_trending_topics_plotly(trending_topics)

    principal_df, explained_variance_ratio = perform_pca(posts)
    pca_plot_json = plot_pca(principal_df)

    daily_posts_plot_json = generate_time_series_plot(posts, 'D')

    post_scores = get_post_scores(posts)
    post_scores_histogram_json = plot_post_scores_histogram(post_scores)

    correlation_heatmap_json = generate_correlation_heatmap(posts)

    results = {
        'sentiments': sentiments,
        'sentiment_plot_json': sentiment_plot_json,
        'trending_topics_plotly': trending_topics_plotly,
        'topics': topics,
        'anomalies': anomalies,
        'train_score': train_score,
        'test_score': test_score,
        'cv_score': mean_cv_score,
        'regression_plot_json': regression_plot_json,
        'pca_plot_json': pca_plot_json,
        'explained_variance_ratio': explained_variance_ratio,
        'daily_posts_plot_json': daily_posts_plot_json,
        'post_scores_histogram_json': post_scores_histogram_json,
        'correlation_heatmap_json': correlation_heatmap_json
    }

    return {subreddit_name: results}
