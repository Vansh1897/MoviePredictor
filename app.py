import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv.txt", sep=",")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    sim = cosine_similarity(count_matrix)
    return data, sim

def recommend_movie(movie, data, sim):
    movie = movie.lower()
    if movie not in data['movie_title'].str.lower().values:
        return ["❌ This movie is not in our database. Please check spelling."]
    else:
        i = data[data['movie_title'].str.lower() == movie].index[0]
        lst = list(enumerate(sim[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:6]
        return [data['movie_title'][a] for a, _ in lst]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Movie Recommender 🎬", page_icon="🎥", layout="centered")

st.markdown("""
    <style>
        .title {text-align:center;font-size:42px;font-weight:bold;color:#FF4B4B;}
        .subtitle {text-align:center;font-size:20px;color:#777;}
        .recommend-box {
            background-color:#ffffff;
            color:#000000;
            padding:15px;
            border-radius:12px;
            margin:10px 0;
            font-size:18px;
            font-weight:500;
            box-shadow:0 2px 6px rgba(0,0,0,0.2);
        }
        .recommend-box:hover {
            background-color:#FF4B4B;
            color:white;
            transition:0.3s;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find movies similar to your favorites!</div>', unsafe_allow_html=True)
st.write("")

data, sim = load_data()

movie = st.text_input("🎥 Enter a movie name:", placeholder="e.g. Inception, Avatar, Titanic")

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

st.markdown("---")
