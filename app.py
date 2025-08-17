import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    # Load dataset from local file in the repo
    data = pd.read_csv("data.csv.txt", sep=",")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    sim = cosine_similarity(count_matrix)
    return data, sim

def recommend_movie(movie, data, sim):
    movie = movie.lower()
    if movie not in data['movie_title'].unique():
        return ["This movie is not in our database. Please check spelling."]
    else:
        i = data.loc[data['movie_title'] == movie].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:6]  # top 5 recommendations
        return [data['movie_title'][a] for a, _ in lst]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name below to get recommendations!")

data, sim = load_data()

movie = st.text_input("Enter a movie name:")
if st.button("Recommend"):
    recs = recommend_movie(movie, data, sim)
    if "not in our database" in recs[0]:
        st.error(recs[0])
    else:
        st.success(f"Movies similar to **{movie.upper()}**:")
        for r in recs:
            st.write("👉", r)
