import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import joblib
import os

# File paths for the saved model and encoders
MODEL_PATH = 'movie_recommendation_model.h5'
MOVIE_ENCODER_PATH = 'movie_encoder.pkl'
GENRE_ENCODER_PATH = 'genre_encoder.pkl'

# Load the dataset
df = pd.read_csv(r'D:\python\imdb_genres.csv')

# Assuming df contains the dataset with movie titles and combined genres

# Label Encoding for movies and genres
movie_encoder = LabelEncoder()
df['movie_id'] = movie_encoder.fit_transform(df['movie title - year'])

genre_encoder = LabelEncoder()
df['genre_id'] = genre_encoder.fit_transform(df['Combined_Genre'])

# Split the dataset
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Create TensorFlow Datasets
def create_tf_dataset(dataframe, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((
        {"movie_id": dataframe['movie_id'].values, "genre_id": dataframe['genre_id'].values},
        dataframe['genre_id'].values
    ))
    return dataset.shuffle(1000).batch(batch_size).prefetch(1)

train_dataset = create_tf_dataset(train)
test_dataset = create_tf_dataset(test)

# Build the model
class MovieRecommendationModel(tf.keras.Model):
    def __init__(self, num_movies, num_genres, embedding_dim=50):
        super(MovieRecommendationModel, self).__init__()
        # Embedding for movies
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_dim)
        # Embedding for genres
        self.genre_embedding = tf.keras.layers.Embedding(num_genres, embedding_dim)
        
        # Dense network for predicting genre
        self.genre_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_genres, activation='softmax')
        ])

    def call(self, inputs):
        movie_id = inputs["movie_id"]
        genre_id = inputs["genre_id"]
        
        # Embedding for movie
        movie_emb = self.movie_embedding(movie_id)
        
        # Embedding for genre
        genre_emb = self.genre_embedding(genre_id)
        
        # Combine movie and genre embeddings
        combined_emb = movie_emb + genre_emb
        
        # Predict genre (or movie recommendation) based on combined embedding
        return self.genre_dense(combined_emb)

# Function to train the model if not already trained
def train_model():
    # Number of unique movies and genres
    num_movies = df['movie_id'].nunique()
    num_genres = df['genre_id'].nunique()

    # Create the model
    model = MovieRecommendationModel(num_movies, num_genres)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)

    # Save the model
    model.save(MODEL_PATH)

    # Save encoders
    joblib.dump(movie_encoder, MOVIE_ENCODER_PATH)
    joblib.dump(genre_encoder, GENRE_ENCODER_PATH)

    return model

# Load the model if it exists, otherwise train it
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        model = train_model()
        print("Model trained and saved successfully.")
    return model

# Load the pre-trained or freshly trained model
model = load_or_train_model()

# Predict Recommendations
def recommend_movies(input_movie_name, top_n=5):
    try:
        # Get the movie ID and genre ID
        input_movie_id = movie_encoder.transform([input_movie_name])[0]
        input_genre_id = df[df['movie title - year'] == input_movie_name]['genre_id'].values[0]

        # Predict genre probabilities
        genre_probs = model.predict({"movie_id": tf.constant([input_movie_id]), "genre_id": tf.constant([input_genre_id])})

        # Get the predicted genre ID
        predicted_genre_id = np.argmax(genre_probs, axis=1)[0]

        # Filter movies with the same genre
        same_genre_movies = df[df['genre_id'] == predicted_genre_id]['movie title - year'].unique()

        return same_genre_movies[:top_n]
    
    except Exception as e:
        print(f"Error: {e}")
        return []

