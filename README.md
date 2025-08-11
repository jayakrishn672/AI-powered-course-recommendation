# AI Recommendation Backend

A Flask-based AI recommendation system that provides personalized course recommendations using machine learning techniques.

## Features

- **Course Recommendations**: Get personalized course recommendations based on:
  - Course titles
  - User queries/keywords
  - Content similarity using TF-IDF and cosine similarity
- **Search Functionality**: Search courses by keywords in title, description, or category
- **RESTful API**: Clean and well-documented API endpoints
- **CORS Support**: Ready for frontend integration

## Tech Stack

- **Backend**: Flask (Python)
- **ML Libraries**: scikit-learn (TF-IDF, cosine similarity)
- **Data Processing**: pandas, numpy
- **API**: RESTful endpoints with JSON responses

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## API Endpoints

### Base URL
- `GET /` - API information and available endpoints

### Courses
- `GET /courses` - Get all available courses
- `GET /health` - Health check endpoint

### Recommendations
- `POST /recommend/title` - Get recommendations by course title
  ```json
  {
    "title": "Introduction to AI"
  }
  ```

- `POST /recommend/query` - Get recommendations by user query
  ```json
  {
    "query": "I want to learn about neural networks",
    "top_n": 3
  }
  ```

### Search
- `POST /search` - Search courses by keyword
  ```json
  {
    "keyword": "machine learning"
  }
  ```

## Example Usage

### Get recommendations by query:
```bash
curl -X POST http://localhost:5000/recommend/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to learn about AI and machine learning", "top_n": 3}'
```

### Get recommendations by course title:
```bash
curl -X POST http://localhost:5000/recommend/title \
  -H "Content-Type: application/json" \
  -d '{"title": "Introduction to AI"}'
```

### Search courses:
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"keyword": "python"}'
```

## Response Format

All responses are in JSON format. Recommendation responses include:
- Input query/title
- Recommended courses with full details
- Similarity scores (for query-based recommendations)

## Development

To add more courses or modify the recommendation logic:
1. Update the course data in `app.py`
2. Modify the vectorization parameters in the TF-IDF setup
3. Adjust similarity thresholds as needed

## Future Enhancements

- Database integration (MongoDB/PostgreSQL)
- User preference tracking
- Collaborative filtering
- Real-time recommendation updates
- Advanced NLP techniques (BERT embeddings)
