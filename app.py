from flask import Flask, jsonify
from flask_cors import CORS
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import os
import spacy
#Text Preprocessing
#Word2Vec Model
#Similarity Measurement
app = Flask(__name__)
CORS(app)


# Base URL for .NET API
BASE_URL = "https://careerfusionbackend.smartwaveeg.com/api/ForRecommend"
# Load Spacy English model
nlp = spacy.load("en_core_web_sm")
def fetch_user_skills(user_id):
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}")
        response.raise_for_status()
        user_data = response.json()
        print(f"API Response: {user_data}")  # Debug line
        print(f"API Response: {user_data}")  # Debug line
        skills = user_data.get('combinedSkills', '')
        return skills.split(', ') if skills else []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching user skills: {e}")
        return []

# Load CSV data
def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Preprocess skills
def preprocess_skill(skill):
    doc = nlp(skill.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def preprocess_skills(skills):
    return [preprocess_skill(skill) for skill in skills]

# Generate Word2Vec Model
def generate_word2vec_model(skills):
    tokenized_skills = [skill.split() for skill in skills]
                                                 #dimensionality of the word vectors ,  smaller window size focuses on a narrower context.Ignores all words with a total frequency lower than this
    model = Word2Vec(sentences=tokenized_skills, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Get Word2Vec Similarity
def get_word2vec_similarity(skill1, skill2, model):
    vec1 = model.wv[skill1.split()]
    vec2 = model.wv[skill2.split()]
    return cosine_similarity([vec1.mean(axis=0)], [vec2.mean(axis=0)])[0][0]

# Compare skills using combined techniques
def compare_skills(user_skills, roadmap_data, word2vec_model):
    roadmap_skills = roadmap_data['preprocessed_skill'].tolist()
    roadmap_links = roadmap_data['Roadmap URL'].tolist()
    all_skills = user_skills + roadmap_skills
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_skills)
    
    user_tfidf = tfidf_matrix[:len(user_skills)]
    roadmap_tfidf = tfidf_matrix[len(user_skills):]
    
    similarities = cosine_similarity(user_tfidf, roadmap_tfidf)
    
    recommended_roadmaps = set()
    for i, user_skill in enumerate(user_skills):
        for j, roadmap_skill in enumerate(roadmap_skills):
            tfidf_score = similarities[i, j]
            word2vec_score = get_word2vec_similarity(user_skill, roadmap_skill, word2vec_model)
            combined_score = (tfidf_score + word2vec_score) / 2
            
            common_skills_count = sum([1 for us in user_skill.split() if any(us_part in roadmap_skill for us_part in us.split())])
            if combined_score > 0.1 and common_skills_count >= 2:
                recommended_roadmaps.add(roadmap_links[j])
    
    return list(recommended_roadmaps)

@app.route('/recommend_roadmaps/<user_id>', methods=['GET'])
def recommend_roadmaps(user_id):
    try:
        csv_file = os.path.join(os.path.dirname(__file__), 'skills_and_roadmaps.csv')
        
        user_skills = fetch_user_skills(user_id)
        print(f"User Skills: {user_skills}")
        
        if not user_skills:
            return jsonify({"error": "No skills found for the user."}), 404
        
        preprocessed_user_skills = preprocess_skills(user_skills)
        print(f"Preprocessed User Skills: {preprocessed_user_skills}")
        
        roadmap_data = load_csv_data(csv_file)
        
        print(f"CSV Columns: {roadmap_data.columns.tolist()}")
        
        if 'Skills' in roadmap_data.columns and 'Roadmap URL' in roadmap_data.columns:
            roadmap_data['preprocessed_skill'] = preprocess_skills(roadmap_data['Skills'])
        else:
            return jsonify({"error": "'Skills' or 'Roadmap URL' column not found in CSV file."}), 400
        
        all_skills = preprocessed_user_skills + roadmap_data['preprocessed_skill'].tolist()
        word2vec_model = generate_word2vec_model(all_skills)
        
        recommended_roadmaps = compare_skills(preprocessed_user_skills, roadmap_data[['preprocessed_skill', 'Roadmap URL']], word2vec_model)
        print(f"Recommended Roadmaps: {recommended_roadmaps}")
        
        return jsonify({"recommended_roadmaps": recommended_roadmaps})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
