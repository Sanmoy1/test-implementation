# User Behavior Prediction API

A Flask-based REST API that predicts user behavior classes using machine learning. This project demonstrates the implementation of a production-ready ML model serving system with proper error handling, data preprocessing, and API documentation.

## 🚀 Project Overview

This API serves a pre-trained machine learning model that predicts user behavior classes based on various user metrics and demographics. The system is designed to be scalable, maintainable, and production-ready with comprehensive error handling and input validation.

## 🛠️ Technology Stack

- **Backend Framework**: Flask 3.1.1
- **Machine Learning**: scikit-learn 1.6.1
- **Data Processing**: pandas 2.2.3, numpy 2.0.2
- **Production Server**: gunicorn 21.2.0
- **Model Persistence**: Pickle files (model.pkl, scaler.pkl)

## 📋 Features

### Core Functionality

- **User Behavior Prediction**: Predicts user behavior classes based on input features
- **RESTful API**: Clean, well-documented API endpoints
- **Input Validation**: Comprehensive validation for all input parameters
- **Error Handling**: Robust error handling with meaningful error messages
- **Data Preprocessing**: Automatic feature scaling and encoding

### API Endpoints

#### 1. Home Endpoint (`GET /`)

Returns API documentation and usage examples.

**Response:**

```json
{
  "message": "Welcome to the User Behavior Prediction API",
  "usage": {
    "endpoint": "/predict",
    "method": "POST",
    "example_input": {
      "App Usage Time (min/day)": 250,
      "Screen On Time (hours/day)": 5.2,
      "Battery Drain (mAh/day)": 1500,
      "Number of Apps Installed": 50,
      "Data Usage (MB/day)": 800,
      "Age": 28,
      "Gender": "Male"
    }
  }
}
```

#### 2. Prediction Endpoint (`POST /predict`)

Accepts user data and returns behavior predictions.

**Input Features:**

- `App Usage Time (min/day)`: Daily app usage in minutes
- `Screen On Time (hours/day)`: Daily screen time in hours
- `Battery Drain (mAh/day)`: Daily battery consumption
- `Number of Apps Installed`: Total installed applications
- `Data Usage (MB/day)`: Daily data consumption
- `Age`: User age
- `Gender`: "Male" or "Female"

**Response:**

```json
{
  "predicted_user_behavior_class": 1
}
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd flask
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

### Production Deployment

For production deployment using gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app
```

## 📊 Model Details

### Features Used

The model uses 7 features to predict user behavior:

1. **App Usage Time**: Daily application usage duration
2. **Screen On Time**: Daily screen activity duration
3. **Battery Drain**: Daily battery consumption
4. **Number of Apps Installed**: Total installed applications
5. **Data Usage**: Daily data consumption
6. **Age**: User demographic information
7. **Gender**: Binary categorical feature (Male/Female)

### Data Preprocessing

- **Feature Scaling**: StandardScaler for numerical features
- **Categorical Encoding**: Binary encoding for gender (Male=1, Female=0)
- **Input Validation**: Comprehensive validation for all input parameters

## 🔧 API Usage Examples

### Using curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "App Usage Time (min/day)": 250,
    "Screen On Time (hours/day)": 5.2,
    "Battery Drain (mAh/day)": 1500,
    "Number of Apps Installed": 50,
    "Data Usage (MB/day)": 800,
    "Age": 28,
    "Gender": "Male"
  }'
```

### Using Python requests

```python
import requests
import json

url = "http://localhost:5000/predict"
data = {
    "App Usage Time (min/day)": 250,
    "Screen On Time (hours/day)": 5.2,
    "Battery Drain (mAh/day)": 1500,
    "Number of Apps Installed": 50,
    "Data Usage (MB/day)": 800,
    "Age": 28,
    "Gender": "Male"
}

response = requests.post(url, json=data)
print(response.json())
```

## 🛡️ Error Handling

The API includes comprehensive error handling for:

- **Missing model files**: Returns 500 error if model.pkl or scaler.pkl are not found
- **Invalid JSON**: Returns 400 error for malformed JSON input
- **Missing features**: Returns 400 error if required features are missing
- **Invalid gender values**: Returns 400 error for non-binary gender inputs
- **Prediction errors**: Returns 500 error for model prediction failures

## 📁 Project Structure

```
flask/
├── app.py              # Main Flask application
├── wsgi.py             # WSGI entry point for production
├── requirements.txt    # Python dependencies
├── model.pkl          # Trained ML model
├── scaler.pkl         # Feature scaler
├── .gitignore         # Git ignore rules
└── readme.md          # This file
```

## 🔍 Code Quality Features

### Error Handling

- Comprehensive try-catch blocks
- Meaningful error messages
- Proper HTTP status codes
- Input validation and sanitization

### Code Organization

- Clean separation of concerns
- Well-documented functions and endpoints
- Consistent code formatting
- Proper variable naming conventions

### Production Readiness

- Environment variable support for port configuration
- WSGI compatibility for production deployment
- Proper logging and debugging information
- Scalable architecture

## 🧪 Testing

To test the API endpoints:

1. **Test home endpoint:**

   ```bash
   curl http://localhost:5000/
   ```

2. **Test prediction endpoint:**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"App Usage Time (min/day)": 250, "Screen On Time (hours/day)": 5.2, "Battery Drain (mAh/day)": 1500, "Number of Apps Installed": 50, "Data Usage (MB/day)": 800, "Age": 28, "Gender": "Male"}'
   ```

## 📈 Performance Considerations

- **Model Loading**: Models are loaded once at startup for optimal performance
- **Memory Efficiency**: Uses numpy arrays for efficient data processing
- **Scalability**: Designed to handle multiple concurrent requests
- **Error Recovery**: Graceful handling of model loading failures

## 🔮 Future Enhancements

Potential improvements for production deployment:

- Add authentication and authorization
- Implement rate limiting
- Add request/response logging
- Include model versioning
- Add health check endpoints
- Implement caching mechanisms
- Add comprehensive unit tests
- Include API documentation with Swagger/OpenAPI


