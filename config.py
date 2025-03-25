import os

class Config:
    # TMDB API Configuration
    TMDB_API_KEY = 'c6669df507add879b41744fee3f41617'  # Replace with your actual TMDB API key
    TMDB_BASE_URL = "https://api.themoviedb.org/3"
    
    # Flask Configuration
    SECRET_KEY = os.getenv('d2f8a1b3c4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0', 'dev-secret-key')  # Change for production
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = 'sqlite:///movies.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False