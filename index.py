from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)

# --- Sample course dataset ---
data = {
    "course_id": [1, 2, 3, 4],
    "title": [
        "Introduction to AI",
        "Advanced Web Development",
        "Data Science Basics",
        "Machine Learning with Python"
    ],
    "description": [
        "Learn AI concepts including machine learning, neural networks, and NLP.",
        "Learn modern web development with React, Node.js, and databases.",
        "Introduction to data science concepts with Python and statistics.",
        "Hands-on machine learning in Python using scikit-learn and TensorFlow."
    ]
}
df = pd.DataFrame(data)

# --- Prepare TF-IDF vectorizer ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# --- Home route ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ AI Course Recommendation API is running!"})

# --- Recommendation endpoint ---
@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json.get("query", "")
    if not user_input:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    user_vec = tfidf.transform([user_input])
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-3:][::-1]
    recommendations = df.iloc[top_indices][['course_id', 'title', 'description']].to_dict('records')
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
