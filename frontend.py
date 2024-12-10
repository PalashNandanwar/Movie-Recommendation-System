import streamlit as st
from tempCodeRunnerFile import recommend_movies  # Import the backend function

# Streamlit App Configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎥",
    layout="centered",
)

# Title and Description
st.title("🎬 Movie Recommendation System")
st.write("Welcome to the Movie Recommendation System! Enter a movie title to find similar movies.")

# Input box for the movie title
input_movie = st.text_input("Enter a Movie Title (with Year)", "")

# Display recommendations
if input_movie:
    st.write("Fetching recommendations...")
    recommendations = recommend_movies(input_movie)

    # Handle results
    if not recommendations or (isinstance(recommendations[0], str) and "Error" in recommendations[0]):
        st.error("No recommendations found. Please try another movie title or check your input.")
    else:
        st.success(f"Movies similar to '{input_movie}':")
        for idx, movie in enumerate(recommendations, 1):
            st.write(f"{idx}. {movie}")
