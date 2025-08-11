import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Sample course data - can be replaced with database
data = {
    "course_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "title": [
        "Introduction to AI",
        "Advanced Web Development",
        "Data Science Basics",
        "Machine Learning with Python",
        "Deep Learning Fundamentals",
        "Natural Language Processing",
        "Computer Vision Applications",
        "Big Data Analytics"
    ],
    "description": [
        "Learn AI concepts including machine learning, neural networks, and NLP. Covers supervised and unsupervised learning techniques.",
        "Learn modern web development with React, Node.js and databases. Build scalable web applications.",
        "Introduction to data science concepts with Python and statistics. Learn data visualization and analysis.",
        "Hands-on machine learning in Python using scikit-learn and TensorFlow. Build predictive models.",
        "Deep dive into neural networks, CNNs, RNNs, and advanced architectures. Practical implementations included.",
        "Learn to process and analyze text data using NLTK, spaCy, and transformer models. Build chatbots and sentiment analyzers.",
        "Explore image processing, object detection, and image classification using OpenCV and deep learning frameworks.",
        "Master big data technologies like Hadoop, Spark, and cloud platforms. Handle large-scale data processing."
    ],
    "category": ["AI", "Web Development", "Data Science", "ML", "Deep Learning", "NLP", "Computer Vision", "Big Data"],
    "level": ["Beginner", "Intermediate", "Beginner", "Intermediate", "Advanced", "Advanced", "Intermediate", "Advanced"],
    "duration_hours": [40, 60, 35, 50, 45, 40, 55, 70]
}

df = pd.DataFrame(data)

# Vectorize course descriptions
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create course index mapping
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    """Get course recommendations based on course title"""
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4]  # Get top 3 similar courses
        course_indices = [i[0] for i in sim_scores]
        return df.iloc[course_indices].to_dict('records')
    except KeyError:
        return []

def get_recommendations_by_query(query, top_n=3):
    """Get course recommendations based on user query"""
    user_vec = tfidf.transform([query])
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices].to_dict('records')
    
    # Add similarity scores
    for idx, rec in enumerate(recommendations):
        rec['similarity_score'] = float(sim_scores[top_indices[idx]])
    
    return recommendations

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "AI Recommendation Backend is running!",
        "endpoints": {
            "/courses": "GET - Get all courses",
            "/recommend/title": "POST - Get recommendations by course title",
            "/recommend/query": "POST - Get recommendations by user query",
            "/health": "GET - Health check"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "AI Recommendation Backend"})

@app.route('/courses', methods=['GET'])
def get_all_courses():
    """Get all available courses"""
    return jsonify(df.to_dict('records'))

@app.route('/recommend/title', methods=['POST'])
def recommend_by_title():
    """Get recommendations based on course title"""
    data = request.json
    title = data.get('title', '')
    
    if not title:
        return jsonify({"error": "Please provide a course title"}), 400
    
    recommendations = get_recommendations(title)
    
    if not recommendations:
        return jsonify({"error": "Course not found"}), 404
    
    return jsonify({
        "input_course": title,
        "recommendations": recommendations
    })

@app.route('/recommend/query', methods=['POST'])
def recommend_by_query():
    """Get recommendations based on user query"""
    data = request.json
    query = data.get('query', '')
    top_n = data.get('top_n', 3)
    
    if not query:
        return jsonify({"error": "Please provide a query"}), 400
    
    recommendations = get_recommendations_by_query(query, top_n)
    
    return jsonify({
        "query": query,
        "recommendations": recommendations
    })

@app.route('/search', methods=['POST'])
def search_courses():
    """Search courses by keyword"""
    data = request.json
    keyword = data.get('keyword', '').lower()
    
    if not keyword:
        return jsonify({"error": "Please provide a search keyword"}), 400
    
    # Search in title and description
    mask = df['title'].str.lower().str.contains(keyword) | \
           df['description'].str.lower().str.contains(keyword) | \
           df['category'].str.lower().str.contains(keyword)
    
    results = df[mask].to_dict('records')
    
    return jsonify({
        "keyword": keyword,
        "results": results,
        "count": len(results)
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
