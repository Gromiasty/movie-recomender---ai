import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import Config

# Inicjalizacja aplikacji
app = Flask(__name__)
app.config.from_object(Config)

# Inicjalizacja rozszerzeń
db = SQLAlchemy(app)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Modele bazy danych
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    ratings = db.relationship('Rating', backref='user', lazy=True)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, nullable=False)
    mood = db.Column(db.String(50), nullable=True)

# Model rekomendacji
class MovieRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def recommend(self, movies, user_ratings, top_n=5):
        if not movies or not user_ratings:
            return movies[:top_n]
        
        try:
            # Przygotowanie danych
            rated_movies = [r['movie_id'] for r in user_ratings]
            ratings = [r['rating'] for r in user_ratings]
            
            # Ekstrakcja cech
            movie_descriptions = [m.get('overview', '') for m in movies]
            movie_ids = [m['id'] for m in movies]
            
            # Obliczenia podobieństwa
            tfidf_matrix = self.vectorizer.fit_transform(movie_descriptions)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Rekomendacje
            user_profile = np.zeros(len(movies))
            for i, movie_id in enumerate(movie_ids):
                if movie_id in rated_movies:
                    idx = rated_movies.index(movie_id)
                    user_profile[i] = ratings[idx]
            
            predicted_ratings = similarity_matrix.T.dot(user_profile)
            predicted_ratings /= similarity_matrix.sum(axis=1)
            
            # Sortowanie wyników
            ranked_indices = np.argsort(predicted_ratings)[::-1]
            return [movies[idx] for idx in ranked_indices if movies[idx]['id'] not in rated_movies][:top_n]
        
        except Exception as e:
            app.logger.error(f"Błąd rekomendacji: {e}")
            return movies[:top_n]

# Inicjalizacja komponentów
recommender = MovieRecommender()

# Funkcje pomocnicze
@cache.memoize(timeout=3600)
def fetch_movies_from_tmdb(mood, genre, year_from, year_to, min_rating):
    mood_to_genre = {
        'happy': 35, 'sad': 18, 'excited': 28, 
        'romantic': 10749, 'neutral': None
    }
    
    try:
        url = f"{app.config['TMDB_BASE_URL']}/discover/movie"
        params = {
            'api_key': app.config['TMDB_API_KEY'],
            'language': 'pl-PL',
            'sort_by': 'popularity.desc'
        }
        genre_ids = []
        if genre:
            genre_ids.append(str(genre))
        if mood_genre := mood_to_genre.get(mood):
            if str(mood_genre) not in genre_ids:
                genre_ids.append(str(mood_genre))
        if genre_ids:
            params['with_genres'] = ','.join(genre_ids)
        if year_from:
            params['primary_release_date.gte'] = f"{year_from}-01-01"
        if year_to:
            params['primary_release_date.lte'] = f"{year_to}-12-31"
        if min_rating:
            params['vote_average.gte'] = min_rating
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        base_url = "https://image.tmdb.org/t/p/w500"
        for movie in data.get('results', []):
            movie['poster_url'] = f"{base_url}{movie['poster_path']}" if movie.get('poster_path') else None
            movie['backdrop_url'] = f"{base_url}{movie['backdrop_path']}" if movie.get('backdrop_path') else None
        return data.get('results', [])[:10]
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Błąd TMDB: {e}")
        return []

def get_movie_details(movie_id):
    try:
        url = f"{app.config['TMDB_BASE_URL']}/movie/{movie_id}"
        params = {
            'api_key': app.config['TMDB_API_KEY'],
            'language': 'pl-PL'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def add_rating(user_id, movie_id, rating_value, mood):
    if not (0.5 <= rating_value <= 5.0):
        raise ValueError("Ocena musi być między 0.5 a 5.0")
    
    new_rating = Rating(
        user_id=user_id,
        movie_id=movie_id,
        rating=rating_value,
        mood=mood
    )
    db.session.add(new_rating)
    db.session.commit()
    return new_rating

def get_user_ratings(user_id):
    ratings = Rating.query.filter_by(user_id=user_id).all()
    return [{
        'movie_id': r.movie_id,
        'rating': r.rating,
        'mood': r.mood
    } for r in ratings]

# Widoki aplikacji
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        mood = request.form.get('mood')
        genre = request.form.get('genre')
        year_from = request.form.get('year_from')
        year_to = request.form.get('year_to')
        min_rating = request.form.get('min_rating')

        # Przechowywanie parametrów w sesji
        session['mood'] = mood
        session['genre'] = genre
        session['year_from'] = year_from
        session['year_to'] = year_to
        session['min_rating'] = min_rating

        return redirect(url_for('recommend'))
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    mood = session.get('mood', 'happy')
    genre = session.get('genre')
    year_from = session.get('year_from')
    year_to = session.get('year_to')
    min_rating = session.get('min_rating')
    movies = fetch_movies_from_tmdb(mood, genre, year_from, year_to, min_rating)
    if 'user_id' in session:
        user_ratings = get_user_ratings(session['user_id'])
        movies = recommender.recommend(movies, user_ratings)
    return render_template('recommend.html', movies=movies, mood=mood)

@app.route('/search')
def search():
    query = request.args.get('query', '').strip()
    if not query:
        return redirect(url_for('index'))
    
    try:
        page = request.args.get('page', 1, type=int)
        params = {
            'api_key': app.config['TMDB_API_KEY'],
            'query': query,
            'language': 'pl-PL',
            'page': page
        }
        
        if year := request.args.get('year'):
            params['year'] = year
        if genre := request.args.get('genre'):
            params['with_genres'] = genre
        
        response = requests.get(f"{app.config['TMDB_BASE_URL']}/search/movie", params=params)
        response.raise_for_status()
        data = response.json()
        
        # Przygotowanie URL obrazów
        base_url = "https://image.tmdb.org/t/p/w500"
        for movie in data.get('results', []):
            movie['poster_url'] = f"{base_url}{movie['poster_path']}" if movie.get('poster_path') else None
        
        return render_template('search_results.html',
                           movies=data.get('results', []),
                           query=query,
                           page=page,
                           total_pages=data.get('total_pages', 1))
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Błąd wyszukiwania: {e}")
        return render_template('error.html', message="Problem z wyszukiwaniem")

@app.route('/rate/<int:movie_id>', methods=['GET', 'POST'])
def rate(movie_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            rating = float(request.form.get('rating'))
            add_rating(session['user_id'], movie_id, rating, session.get('mood', 'neutral'))
            return redirect(url_for('recommend'))
        except (ValueError, TypeError) as e:
            abort(400, str(e))
    
    if not (movie := get_movie_details(movie_id)):
        abort(404, "Film nie znaleziony")
    
    return render_template('rate.html', movie=movie)

@app.route('/api/suggest')
def suggest_movies():
    query = request.args.get('q', '').strip()
    if len(query) < 3:
        return jsonify([])
    
    try:
        params = {
            'api_key': app.config['TMDB_API_KEY'],
            'query': query,
            'language': 'pl-PL'
        }
        response = requests.get(f"{app.config['TMDB_BASE_URL']}/search/movie", params=params)
        response.raise_for_status()
        
        suggestions = [{
            'title': movie['title'],
            'id': movie['id'],
            'year': movie.get('release_date', '')[:4] if movie.get('release_date') else ''
        } for movie in response.json().get('results', [])[:5]]
        
        return jsonify(suggestions)
    
    except requests.exceptions.RequestException:
        return jsonify([])

@app.route('/api/search_suggestions')
def search_suggestions():
    query = request.args.get('q', '').strip()
    if len(query) < 3:
        return jsonify([])
    
    try:
        url = f"{app.config['TMDB_BASE_URL']}/search/movie"
        params = {
            'api_key': app.config['TMDB_API_KEY'],
            'query': query,
            'language': 'pl-PL'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        suggestions = [{
            'title': movie['title'],
            'id': movie['id'],
            'year': movie.get('release_date', '')[:4] if movie.get('release_date') else ''
        } for movie in response.json().get('results', [])[:5]]
        
        return jsonify(suggestions)
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Search suggestion error: {e}")
        return jsonify([])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if not username:
            return render_template('login.html', error="Nazwa użytkownika jest wymagana")
        
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username)
            db.session.add(user)
            db.session.commit()
        session['user_id'] = user.id
        return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('mood', None)
    return redirect(url_for('index'))

@app.route('/my_ratings')
def my_ratings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_ratings = get_user_ratings(session['user_id'])
    rated_movies = []
    for r in user_ratings:
        movie = get_movie_details(r['movie_id'])
        if movie:
            movie['user_rating'] = r['rating']
            movie['user_mood'] = r['mood']
            rated_movies.append(movie)
    return render_template('my_ratings.html', movies=rated_movies)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', message=str(error)), 404

@app.errorhandler(400)
def bad_request_error(error):
    return render_template('error.html', message=str(error)), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)