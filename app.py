import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the dataset
@st.cache_data
def load_data():
    # Load the dataset (update path to the correct one)
    data = pd.read_csv('Coursera.csv')
    data['Course Description'] = data['Course Description'].fillna('')
    data = data.dropna(subset=['Course Name'])  # Drop rows with missing Course Titles
    return data

data = load_data()

# Streamlit UI
st.title("Course Recommendation System")
st.write("Select a course to find similar ones.")

# Dropdown to select a course
course_titles = data['Course Name'].values
selected_course = st.selectbox("Choose a Course", course_titles)
# Build the KNN model
st.cache_data
def build_knn_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Course Description'])
    
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(tfidf_matrix)
    
    return knn, tfidf_matrix

knn, tfidf_matrix = build_knn_model(data)

# Get KNN recommendations
def get_knn_recommendations(title, knn_model, tfidf_matrix, data):
    idx = data[data['Course Name'] == title].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    similar_courses = indices.flatten()[1:]
    return data['Course Name'].iloc[similar_courses]

# Display recommendations
if selected_course:
    recommendations = get_knn_recommendations(selected_course, knn, tfidf_matrix, data)
    st.write(f"Top 5 recommended courses similar to '{selected_course}':")
    for i, course in enumerate(recommendations):
        st.write(f"{i+1}. {course}")
