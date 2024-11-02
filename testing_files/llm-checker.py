from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import CrossEncoder
import torch

app = Flask(__name__)
CORS(app)

# Load CrossEncoder model with sigmoid activation
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())

@app.route('/')
def home():
    return 'Root URL is working'

@app.route('/similarity', methods=['POST'])
def get_similarity():
    try:
        data = request.get_json()
        sentence1 = data.get('sentence1', '')
        sentence2 = data.get('sentence2', '')

        # Ensure both sentences are provided
        if not sentence1 or not sentence2:
            return jsonify({'error': 'Both sentence1 and sentence2 are required'}), 400

        # Predict similarity
        scores = model.predict([(sentence1, sentence2)])
        similarity_percentage = scores[0] * 100

        return jsonify({'similarity': f"{similarity_percentage:.2f}"})
    except Exception as e:
        # Log and return any errors that occur
        app.logger.error(f"Error in similarity calculation: {e}")
        return jsonify({'error': 'An error occurred during similarity calculation'}), 500

if __name__ == '__main__':
    app.run(debug=True)
