import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    data = pd.read_csv("data.csv.txt", sep=",")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    sim = cosine_similarity(count_matrix)
    return data, sim

def recommend_movie(movie, data, sim):
    movie = movie.lower()
    if movie not in data['movie_title'].unique():
        return ["❌ This movie is not in our database. Please check spelling."]
    else:
        i = data.loc[data['movie_title'] == movie].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:6]
        return [data['movie_title'][a] for a, _ in lst]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Movie Recommender 🎬", page_icon="🎥", layout="centered")

# Custom CSS for aesthetics
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #FF4B4B;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #666666;
        }
        .recommend-box {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin: 5px 0px;
            box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find movies similar to your favorites!</div>', unsafe_allow_html=True)
st.write("")

# Load data
data, sim = load_data()

# Input field
movie = st.text_input("Enter a movie name:", placeholder="e.g. Inception, Avatar, Titanic")

if st.button("🔎 Recommend"):
    if movie.strip() == "":
        st.warning("⚠️ Please enter a movie name.")
    else:
        recs = recommend_movie(movie, data, sim)
        if "not in our database" in recs[0]:
            st.error(recs[0])
        else:
            st.success(f"Movies similar to **{movie.upper()}**:")
            for r in recs:
                st.markdown(f"<div class='recommend-box'>👉 {r}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")

