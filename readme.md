# Movie Mood Recommender 🎬🍿

**Aplikacja rekomendująca filmy na podstawie nastroju użytkownika z wykorzystaniem technik AI**

##  Opis projektu

Movie Mood Recommender to inteligentny system rekomendacji filmów, który:
- Analizuje aktualny nastrój użytkownika
- Sugeruje dopasowane filmy z bazy TMDB
- Uczy się preferencji na podstawie ocen
- Oferuje personalizowane propozycje

##  Zaimplementowane modele AI

### 1. Hybrydowy system rekomendacji
Połączenie dwóch podejść:

#### a) Content-Based Filtering
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

self.vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = self.vectorizer.fit_transform(movie_descriptions)
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
```
- **Działanie**: Analizuje opisy filmów (NLP)
- **Technika**: TF-IDF + podobieństwo kosinusowe
- **Cel**: Znajduje filmy o podobnej tematyce

#### b) Collaborative Filtering (uproszczony)
```python
user_profile = np.zeros(len(movies))
for i, movie_id in enumerate(movie_ids):
    if movie_id in rated_movies:
        user_profile[i] = ratings[rated_movies.index(movie_id)]
```
- **Działanie**: Uwzględnia historię ocen użytkownika
- **Technika**: Weighted user profile
- **Cel**: Personalizacja rekomendacji

### 2. Mood-Based Filtering
```python
mood_to_genre = {
    'happy': 35,    # Komedia
    'sad': 18,      # Dramat
    'excited': 28,  # Akcja
    'romantic': 10749 # Romans
}
```
- **Działanie**: Mapuje nastrój na gatunki filmowe
- **Technika**: Rule-based classification
- **Cel**: Dopasowanie do aktualnego stanu emocjonalnego

## 🛠 Wymagania techniczne

- Python 3.10+
- Wymagane pakiety (zobacz `requirements.txt`):
  ```
  flask==2.0.3
  scikit-learn==1.2.2
  requests==2.31.0
  numpy==1.24.3
  ```

## 🚀 Jak uruchomić?

1. Sklonuj repozytorium:
```bash
git clone https://github.com/TwojaNazwa/movie-recommender.git
cd movie-recommender
```

2. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

3. Uzyskaj klucz API TMDB:
- Zarejestruj się na [TMDB](https://www.themoviedb.org/)
- Włącz konto API
- W pliku `config.py` dodaj:
```python
TMDB_API_KEY = 'twój_klucz'
```

4. Uruchom aplikację:
```bash
python app.py
```

5. Otwórz w przeglądarce:
```
http://localhost:5000
```
