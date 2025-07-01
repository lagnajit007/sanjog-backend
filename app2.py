from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import logging
from collections import deque
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app2 = Flask(__name__)
CORS(app2)  # Enable CORS for all routes

# Load trained model
MODEL_PATH = 'backend/models/model.p'

# Global variables
model = None
labels_dict = {}

# Performance optimization settings
ENABLE_CACHING = True  # Set to False to disable caching
PREDICTION_CACHE_SIZE = 100  # Number of recent predictions to cache

# Prediction smoothing
HISTORY_SIZE = 5
prediction_history = deque(maxlen=HISTORY_SIZE)

# Map indices to sign language labels
# Use this as a fallback if labels_dict is missing or for additional mappings
SIGN_LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z",
    26: "0", 27: "1", 28: "2", 29: "3", 30: "4", 31: "5", 32: "6", 33: "7", 34: "8", 35: "9"
}

# Performance stats
request_times = deque(maxlen=100)  # Keep track of recent request processing times
last_request_time = 0  # Rate limiting
request_interval = 0.05  # 50ms minimum between requests

# Load the model at startup
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    start_time = time.time()
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        labels_dict = model_data['labels_dict']
    load_time = time.time() - start_time
    logger.info(f"Model and labels loaded successfully in {load_time:.2f}s.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    labels_dict = {}

# Create a prediction cache decorator
if ENABLE_CACHING:
    @lru_cache(maxsize=PREDICTION_CACHE_SIZE)
    def cached_predict(landmarks_tuple):
        """Cache predictions for identical landmark inputs"""
        landmarks_array = np.array(landmarks_tuple).reshape(1, -1)
        return model.predict(landmarks_array)[0]
else:
    def cached_predict(landmarks_tuple):
        """Non-cached prediction function"""
        landmarks_array = np.array(landmarks_tuple).reshape(1, -1)
        return model.predict(landmarks_array)[0]

# Create a function to process landmarks consistently
def process_landmarks(landmarks):
    """Process landmarks to ensure consistent format"""
    # Basic validation
    if not isinstance(landmarks, list):
        return None, "Landmarks must be a list"
        
    if len(landmarks) != 42:  # 21 landmarks with x,y coordinates
        return None, f"Expected 42 landmarks, got {len(landmarks)}"
    
    # Check for invalid values
    if any(not isinstance(v, (int, float)) or np.isnan(v) or np.isinf(v) for v in landmarks):
        return None, "Landmarks contain invalid values (NaN or infinity)"
    
    # Convert to numpy array and reshape
    try:
        landmarks_tuple = tuple(float(x) for x in landmarks)  # Make hashable for caching
        return landmarks_tuple, None
    except Exception as e:
        return None, f"Error processing landmarks: {str(e)}"
    

@app2.route('/', methods=['GET'])
def health_check():
    """Health check endpoint to verify if the server is running"""
    return jsonify({"status": "ok", "message": "Sign language recognition server is running"})

@app2.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive hand landmarks and return sign language prediction.
    Expected input: JSON with "landmarks" key containing flattened x,y coordinates.
    """
    global last_request_time
    
    # Simple rate limiting to prevent overload
    now = time.time()
    elapsed = now - last_request_time
    if elapsed < request_interval:
        time.sleep(request_interval - elapsed)
    last_request_time = time.time()
    
    start_time = time.time()
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get data from request
        data = request.get_json()
        
        # Basic validation
        if not data or "landmarks" not in data:
            return jsonify({"error": "Invalid request format. 'landmarks' field is required."}), 400
        
        landmarks = data['landmarks']
        
        # Process landmarks 
        landmarks_tuple, error = process_landmarks(landmarks)
        if error:
            return jsonify({'error': error}), 400
            
        # Make prediction using the cached function
        prediction = cached_predict(landmarks_tuple)
        
        # Convert prediction to label
        if isinstance(prediction, np.integer):
            prediction = int(prediction)
        
        # Try getting label from loaded labels_dict, fallback to SIGN_LABELS
        label = labels_dict.get(prediction, SIGN_LABELS.get(prediction, str(prediction)))
        
        # Add to history for smoothing
        prediction_history.append(label)
        
        # Simple smoothing: return most common prediction in history
        if len(prediction_history) >= 3:
            from collections import Counter
            counter = Counter(prediction_history)
            smoothed_label = counter.most_common(1)[0][0]
        else:
            smoothed_label = label
            
        # Confidence (robust): Use predict_proba if available, else fallback to 1.0 for match, 0.0 for mismatch
        try:
            input_array = np.array(landmarks_tuple).reshape(1, -1)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_array)[0]
                # If prediction is a label, get its index
                if hasattr(model, 'classes_') and prediction in model.classes_:
                    pred_idx = list(model.classes_).index(prediction)
                else:
                    pred_idx = int(prediction) if isinstance(prediction, (int, np.integer, float)) else 0
                confidence = float(proba[pred_idx])
            else:
                # Fallback: 1.0 if predicted label matches smoothed_label, else 0.0
                confidence = 1.0 if label == smoothed_label else 0.0
        except Exception as e:
            logger.warning(f"Couldn't calculate confidence: {e}")
            # Fallback: 1.0 if predicted label matches smoothed_label, else 0.0
            confidence = 1.0 if label == smoothed_label else 0.0
        
        # Track request time
        end_time = time.time()
        processing_time = end_time - start_time
        request_times.append(processing_time)
        
        # Occasionally log performance stats
        if len(request_times) % 20 == 0:
            avg_time = sum(request_times) / len(request_times)
            logger.info(f"Average prediction time: {avg_time*1000:.2f}ms over {len(request_times)} requests")
            
        return jsonify({
            "prediction": smoothed_label,
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time * 1000, 2)  # Include processing time in response
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app2.route('/gestures', methods=['GET'])
def list_gestures():
    """Return a list of all available gestures"""
    if not labels_dict:
        # Use SIGN_LABELS as fallback
        gestures = list(set(SIGN_LABELS.values()))
    else:
        # Get gestures from loaded labels
        gestures = list(set(labels_dict.values()))
        
    return jsonify({'gestures': gestures}), 200

@app2.route('/stats', methods=['GET'])
def get_stats():
    """Return server statistics"""
    if not request_times:
        return jsonify({
            'status': 'ok',
            'model_loaded': model is not None,
            'cache_enabled': ENABLE_CACHING,
            'request_count': 0
        })
        
    avg_time = sum(request_times) / len(request_times)
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'average_request_time_ms': round(avg_time * 1000, 2),
        'cache_enabled': ENABLE_CACHING,
        'request_count': len(request_times),
        'history_size': len(prediction_history)
    })

if __name__ == '__main__':
    # Run the Flask app
    app2.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 