import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Loading the dataset
def load_data():
    data = pd.read_csv('Coursera.csv')
    data['Course Description'] = data['Course Description'].fillna('')
    data = data.dropna(subset=['Course Name'])  # This Drops rows with missing Course Titles
    return data

data = load_data()

# Creating the Streamlit UI

st.title("Course Recommendation System")
st.write("Select a course to find similar ones.")
#Making Dropdown to select a course
course_titles = data['Course Name'].values
selected_course = st.selectbox("Choose a Course", course_titles)

# Building the KNN model
@st.cache_data
def build_knn_model(data):
    tfidf = TfidfVectorizer(stop_words='english') #gives weight to topics
    tfidf_matrix = tfidf.fit_transform(data['Course Description'])
    
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(tfidf_matrix)
    
    return knn, tfidf_matrix

knn, tfidf_matrix = build_knn_model(data)

# Using the knn model to Get recommendations
def get_knn_recommendations(title, knn_model, tfidf_matrix, data):
    idx = data[data['Course Name'] == title].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=6)
    similar_courses = indices.flatten()[1:]
    return data['Course Name'].iloc[similar_courses]

# Displaying the recommendations
if selected_course:
    recommendations = get_knn_recommendations(selected_course, knn, tfidf_matrix, data)
    st.write(f"Top 5 recommended courses similar to '{selected_course}':")
    for i, course in enumerate(recommendations):
        st.write(f"{i+1}. {course}")
